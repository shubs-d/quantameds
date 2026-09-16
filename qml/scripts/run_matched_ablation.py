#!/usr/bin/env python3
"""Matched-Capacity Student Ablation & Quantum Gap Diagnosis.

Trains and evaluates three capacity-matched architectures across all 5 folds:
  1. Classical-only head (ClassicalMLPStudent, ~309 params)
  2. Quantum-only head   (QuantumOnlyStudent, ~257 params)
  3. Hybrid student      (HybridRobustQuantaStudent, ~305 params)

Tracks quantum gradient health (barren plateau detection), analytic simulator mode,
and computes fold-level mean ± std for:
  - Accuracy
  - AUROC
  - Brier score
  - Expected Calibration Error (ECE)

Usage:
  .venv/bin/python scripts/run_matched_ablation.py --epochs 15 --batch-size 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.matched_ablation_models import (
    ClassicalMLPStudent,
    HybridRobustQuantaStudent,
    QuantumOnlyStudent,
)
from quantum_kc.training.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class CachedTabularDataset(Dataset):
    """Memory-mapped dataset pairing 5 base tabular features with cached teacher latents."""

    def __init__(
        self,
        base_features: np.ndarray,
        teacher_latents: torch.Tensor,
        labels: np.ndarray,
        mode: str = "train",
        p_dropout: float = 0.5,
        rng: np.random.RandomState = None,
    ) -> None:
        self.base_features = base_features.astype(np.float32)
        self.teacher_latents = teacher_latents.float()
        self.labels = labels.astype(np.float32)
        self.mode = mode
        self.p_dropout = p_dropout
        self.rng = rng or np.random.RandomState(42)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.base_features[idx]
        if self.mode == "tier1":
            pachy = 0.0
            mask = 0.0
        elif self.mode == "tier2":
            pachy = float(base[4])
            mask = 1.0
        elif self.mode == "train":
            is_present = (self.rng.rand() >= self.p_dropout)
            pachy = float(base[4]) if is_present else 0.0
            mask = 1.0 if is_present else 0.0
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        x_6d = torch.tensor(
            [base[0], base[1], base[2], base[3], pachy, mask],
            dtype=torch.float32,
        )
        z_teacher = self.teacher_latents[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_6d, z_teacher, label


def train_and_eval_fold(
    model_name: str,
    fold: int,
    train_ds: CachedTabularDataset,
    val_ds: CachedTabularDataset,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.5,
    beta: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train 1 model on 1 fold, logging quantum diagnostics per epoch."""
    set_seed(seed + fold)

    # Initialize model and optimizer
    if model_name == "classical":
        model = ClassicalMLPStudent().to(device)
        optimizer = Adam(model.parameters(), lr=lr)
    elif model_name == "quantum_only":
        model = QuantumOnlyStudent(n_layers=4).to(device)
        optimizer = Adam(model.parameters(), lr=1e-2)
    elif model_name == "hybrid":
        model = HybridRobustQuantaStudent(n_layers=2).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
    elif model_name == "hybrid_dual_rate":
        model = HybridRobustQuantaStudent(n_layers=2).to(device)
        classical_params = [p for n, p in model.named_parameters() if "vqc" not in n]
        quantum_params = [p for n, p in model.named_parameters() if "vqc" in n]
        optimizer = Adam([
            {"params": classical_params, "lr": lr},
            {"params": quantum_params, "lr": 1e-2},
        ])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Confirm simulator analytic mode for quantum layers
    if hasattr(model, "vqc_layer"):
        # PennyLane QNode device verification
        qnode_dev = model.vqc_layer.qnode.device
        assert (qnode_dev.shots.total_shots is None or not bool(qnode_dev.shots)), "Simulator must run in analytic statevector mode (shots=None)"
        logger.debug("Confirmed analytic simulator mode: shots=None")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [], "loss_distill": [], "loss_task": [],
        "quantum_grad_norm": [], "quantum_grad_var": [],
        "classical_grad_norm": [], "classical_grad_var": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_ds.set_mode("train")
        ep_tot_loss = 0.0
        ep_mse_loss = 0.0
        ep_bce_loss = 0.0
        n_batches = 0

        last_q_norm = 0.0
        last_q_var = 0.0
        last_c_norm = 0.0
        last_c_var = 0.0

        for x_batch, z_teacher_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            z_teacher_batch = z_teacher_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            z_pred, logits = model(x_batch, return_latent=True)

            loss_distill = mse_fn(z_pred, z_teacher_batch)
            loss_task = bce_fn(logits, y_batch)
            loss_total = alpha * loss_distill + beta * loss_task

            loss_total.backward()

            # Separate quantum and classical gradient diagnostics
            if hasattr(model, "vqc_layer") and model.vqc_layer.weights.grad is not None:
                q_grad = model.vqc_layer.weights.grad.detach().view(-1)
                last_q_norm = float(q_grad.norm().item())
                last_q_var = float(torch.var(q_grad).item()) if len(q_grad) > 1 else 0.0

            c_grads = [p.grad.detach().view(-1) for n, p in model.named_parameters() if "vqc" not in n and p.grad is not None]
            if c_grads:
                all_c = torch.cat(c_grads)
                last_c_norm = float(all_c.norm().item())
                last_c_var = float(torch.var(all_c).item())

            optimizer.step()

            ep_tot_loss += loss_total.item()
            ep_mse_loss += loss_distill.item()
            ep_bce_loss += loss_task.item()
            n_batches += 1

        history["train_loss"].append(ep_tot_loss / n_batches)
        history["loss_distill"].append(ep_mse_loss / n_batches)
        history["loss_task"].append(ep_bce_loss / n_batches)
        history["quantum_grad_norm"].append(last_q_norm)
        history["quantum_grad_var"].append(last_q_var)
        history["classical_grad_norm"].append(last_c_norm)
        history["classical_grad_var"].append(last_c_var)

    # Validation evaluation (both Tier 1 and Tier 2)
    def evaluate_mode(mode: str) -> Dict[str, Any]:
        model.eval()
        val_ds.set_mode(mode)
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for x_v, _, y_v in val_loader:
                x_v = x_v.to(device)
                logits = model(x_v)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_v.numpy().astype(int))

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        acc = float(accuracy_score(y_true, y_pred))
        try:
            auc_val = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc_val = 0.5
        brier = float(brier_score_loss(y_true, y_prob))
        ece = compute_ece(y_true, y_prob, n_bins=10)
        cm = confusion_matrix(y_true, y_pred).tolist()

        return {
            "accuracy": acc,
            "auroc": auc_val,
            "brier": brier,
            "ece": ece,
            "cm": cm,
        }

    tier1_metrics = evaluate_mode("tier1")
    tier2_metrics = evaluate_mode("tier2")

    return {
        "model": model_name,
        "fold": fold,
        "params": model.count_parameters(),
        "tier1": tier1_metrics,
        "tier2": tier2_metrics,
        "history": history,
        "final_q_grad_var": history["quantum_grad_var"][-1],
        "final_c_grad_var": history["classical_grad_var"][-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Run Matched-Capacity Student Ablation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="MSE distillation weight")
    parser.add_argument("--beta", type=float, default=0.5, help="BCE task loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--models", nargs="+", default=["classical", "hybrid", "hybrid_dual_rate", "quantum_only"], help="Models to test")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    print("=" * 70)
    print("STEP 2: MATCHED-CAPACITY ABLATION ACROSS 5 FOLDS")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Threads: {args.threads}")
    print(f"Models: {args.models}")
    print(f"Loss weights: alpha={args.alpha}, beta={args.beta}")
    print("=" * 70)

    # 1. Load cached teacher latents
    cache_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached latents not found at {cache_path}. Run cache_teacher_latents.py first.")

    cached_data = torch.load(cache_path, map_location="cpu")
    train_latents = cached_data["latents_train"]
    print(f"Loaded cached teacher latents: {train_latents.shape}")

    # 2. Load clinical tabular data
    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)
    assert len(train_df) == len(train_latents)

    models_to_test = args.models
    results_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models_to_test}

    # Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp_name = "matched-capacity-ablation"
    mlflow.set_experiment(exp_name)

    t0_start = time.time()

    for fold in range(1, 6):
        print(f"\n--- Processing Fold {fold}/5 ---")
        train_mask = (train_df["fold"] != fold).values
        val_mask = (train_df["fold"] == fold).values

        fold_train_df = train_df[train_mask].reset_index(drop=True)
        fold_val_df = train_df[val_mask].reset_index(drop=True)

        fold_train_latents = train_latents[train_mask].clone()
        fold_val_latents = train_latents[val_mask].clone()

        # Fit tabular pipeline on train fold only
        pipe = RobustStudentTabularPipeline()
        pipe.fit(fold_train_df)
        train_base = pipe.transform_base(fold_train_df)
        val_base = pipe.transform_base(fold_val_df)

        train_labels = fold_train_df["label"].values.astype(int)
        val_labels = fold_val_df["label"].values.astype(int)

        for model_name in models_to_test:
            rng_train = np.random.RandomState(args.seed + fold)
            rng_val = np.random.RandomState(args.seed + fold + 100)

            train_ds = CachedTabularDataset(
                train_base, fold_train_latents, train_labels,
                mode="train", p_dropout=0.5, rng=rng_train
            )
            val_ds = CachedTabularDataset(
                val_base, fold_val_latents, val_labels,
                mode="tier2", rng=rng_val
            )

            t_fold_start = time.time()
            res = train_and_eval_fold(
                model_name=model_name,
                fold=fold,
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                alpha=args.alpha,
                beta=args.beta,
                device="cpu",
                seed=args.seed,
            )
            t_fold_elapsed = time.time() - t_fold_start

            results_by_model[model_name].append(res)
            t2 = res["tier2"]
            print(
                f"  [{model_name:<12} Fold {fold}] ({t_fold_elapsed:.1f}s) "
                f"Tier 2 -> Acc: {t2['accuracy']:.4f} | AUROC: {t2['auroc']:.4f} | "
                f"Brier: {t2['brier']:.4f} | ECE: {t2['ece']:.4f} | "
                f"Q-GradVar: {res['final_q_grad_var']:.2e}"
            )

            # Log to MLflow
            with mlflow.start_run(run_name=f"{model_name}_fold{fold}"):
                mlflow.log_params({
                    "model_name": model_name,
                    "fold": fold,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "alpha": args.alpha,
                    "beta": args.beta,
                    "lr": args.lr,
                    "seed": args.seed,
                    **{f"param_{k}": v for k, v in res["params"].items()},
                })
                mlflow.log_metrics({
                    "tier2_accuracy": t2["accuracy"],
                    "tier2_auroc": t2["auroc"],
                    "tier2_brier": t2["brier"],
                    "tier2_ece": t2["ece"],
                    "tier1_accuracy": res["tier1"]["accuracy"],
                    "tier1_auroc": res["tier1"]["auroc"],
                    "final_q_grad_var": res["final_q_grad_var"],
                    "final_c_grad_var": res["final_c_grad_var"],
                    "train_time_sec": t_fold_elapsed,
                })

    total_time = time.time() - t0_start
    print(f"\nCompleted all 5 folds in {total_time:.1f} seconds.")

    # 3. Compute 5-fold Mean ± Std
    summary_data = []
    print("\n" + "=" * 70)
    print("5-FOLD MATCHED-CAPACITY ABLATION SUMMARY (MEAN ± STD)")
    print("=" * 70)

    for model_name in models_to_test:
        fold_results = results_by_model[model_name]
        params = fold_results[0]["params"]["total"]

        # Tier 2 metrics (pachymetry available)
        accs_t2 = [r["tier2"]["accuracy"] for r in fold_results]
        aurocs_t2 = [r["tier2"]["auroc"] for r in fold_results]
        briers_t2 = [r["tier2"]["brier"] for r in fold_results]
        eces_t2 = [r["tier2"]["ece"] for r in fold_results]

        # Tier 1 metrics (pachymetry masked)
        accs_t1 = [r["tier1"]["accuracy"] for r in fold_results]
        aurocs_t1 = [r["tier1"]["auroc"] for r in fold_results]

        # Quantum gradient variance
        q_vars = [r["final_q_grad_var"] for r in fold_results]

        row = {
            "Model": model_name,
            "Params": params,
            "T2_Acc": f"{np.mean(accs_t2):.4f} ± {np.std(accs_t2):.4f}",
            "T2_AUROC": f"{np.mean(aurocs_t2):.4f} ± {np.std(aurocs_t2):.4f}",
            "T2_Brier": f"{np.mean(briers_t2):.4f} ± {np.std(briers_t2):.4f}",
            "T2_ECE": f"{np.mean(eces_t2):.4f} ± {np.std(eces_t2):.4f}",
            "T1_AUROC": f"{np.mean(aurocs_t1):.4f} ± {np.std(aurocs_t1):.4f}",
            "Q_Grad_Var": f"{np.mean(q_vars):.2e} ± {np.std(q_vars):.2e}",
        }
        summary_data.append(row)
        print(f"Model: {model_name:<14} (Params: {params})")
        print(f"  Tier 2 Accuracy:  {row['T2_Acc']}")
        print(f"  Tier 2 AUROC:     {row['T2_AUROC']}")
        print(f"  Tier 2 Brier:     {row['T2_Brier']}")
        print(f"  Tier 2 ECE:       {row['T2_ECE']}")
        print(f"  Tier 1 AUROC:     {row['T1_AUROC']}")
        print(f"  Q-Grad Variance:  {row['Q_Grad_Var']}")
        print("-" * 50)

    # Save summary markdown and json
    out_md = config.BASE_DIR / "results" / "matched_capacity_ablation_summary.md"
    out_json = config.BASE_DIR / "results" / "matched_capacity_ablation_results.json"
    os.makedirs(config.BASE_DIR / "results", exist_ok=True)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_markdown(out_md, index=False)
    with open(out_json, "w") as f:
        json.dump(results_by_model, f, indent=2, default=str)

    print(f"\nWrote summary table to: {out_md}")
    print(f"Wrote raw fold metrics to: {out_json}")


if __name__ == "__main__":
    main()
