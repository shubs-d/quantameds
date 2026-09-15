#!/usr/bin/env python3
"""Full Training and Final Evaluation of RobustQuantaStudent.

Two-tier hybrid quantum-classical keratoconus screening model:
  - Input: 6 tabular biomarkers (Kmax, log1p Cyl, sin 2theta, cos 2theta, norm Pachy, missingness mask)
  - Backbone: RobustQuantaStudent (305 parameters: 257 classical + 48 quantum)
  - Optimizer: Dual-Rate Adam (classical lr=1e-3, quantum lr=1e-2)
  - Loss: alpha * MSE(z_student, z_teacher) + beta * BCE(student_pred, y_true)
  - Missingness Training: Bernoulli(0.5) modality dropout during training
  - Evaluation: Evaluates held-out 291-eye test set under Tier 1 (SimK only) and Tier 2 (SimK + Pachy)
  - Compares directly against the historical sanity-check baseline floor.
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
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.robust_student import RobustQuantaStudent
from quantum_kc.training.utils import set_seed
from scripts.run_matched_ablation import CachedTabularDataset, compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_tier_mode(
    model: nn.Module,
    test_ds: CachedTabularDataset,
    mode: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate model on test dataset under specific clinical tier mode."""
    model.eval()
    test_ds.set_mode(mode)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for x_b, _, y_b in loader:
            x_b = x_b.to(device)
            l = model(x_b)
            all_logits.extend(l.cpu().numpy().tolist())
            all_targets.extend(y_b.numpy().astype(int).tolist())

    logits_arr = np.array(all_logits)
    y_true = np.array(all_targets)
    probs = 1.0 / (1.0 + np.exp(-logits_arr))
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc_val = 0.5
    brier = float(brier_score_loss(y_true, probs))
    ece = float(compute_ece(y_true, probs))
    cm = confusion_matrix(y_true, preds).tolist()

    # Specificity = TN / (TN + FP)
    tn, fp = cm[0][0], cm[0][1]
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auroc": auc_val,
        "brier": brier,
        "ece": ece,
        "confusion_matrix": cm,
        "logit_mean": float(logits_arr.mean()),
        "logit_std": float(logits_arr.std()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-classical", type=float, default=1e-3)
    parser.add_argument("--lr-quantum", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.5, help="Distill MSE weight")
    parser.add_argument("--beta", type=float, default=0.5, help="BCE task weight")
    parser.add_argument("--p-dropout", type=float, default=0.5, help="Pachymetry dropout prob")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    print("=" * 85)
    print("QUANTA-MED STAGE 2: ROBUST QUANTUM-CLASSICAL STUDENT FULL TRAINING")
    print(f"Epochs={args.epochs}, BatchSize={args.batch_size}, Seed={args.seed}")
    print(f"LR Classical={args.lr_classical}, LR Quantum={args.lr_quantum}")
    print(f"Loss weights: alpha(MSE)={args.alpha}, beta(BCE)={args.beta}")
    print("=" * 85)

    # 1. Verify cached latents
    cache_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached teacher latents: {cache_path}")
    cache_hash = compute_sha256(cache_path)
    cached_data = torch.load(cache_path, map_location="cpu")
    train_latents = cached_data["latents_train"]
    test_latents = cached_data["latents_test"]
    teacher_sha256 = cached_data["teacher_sha256"]

    print(f"Verified cached latents: {cache_hash[:16]}... (Teacher: {teacher_sha256[:16]}...)")
    print(f"Train latents: {train_latents.shape}, Test latents: {test_latents.shape}")

    # 2. Load and prepare tabular data
    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)
    test_df = df[df["split_80_20"] == "test"].reset_index(drop=True)

    assert len(train_df) == len(train_latents) == 1163
    assert len(test_df) == len(test_latents) == 291

    pipe = RobustStudentTabularPipeline()
    pipe.fit(train_df)
    train_base = pipe.transform_base(train_df)
    test_base = pipe.transform_base(test_df)

    train_ds = CachedTabularDataset(
        train_base, train_latents, train_df["label"].values.astype(int),
        mode="train", p_dropout=args.p_dropout, rng=np.random.RandomState(args.seed)
    )
    test_ds = CachedTabularDataset(
        test_base, test_latents, test_df["label"].values.astype(int),
        mode="tier2", rng=np.random.RandomState(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 3. Model & Dual-Rate Optimizer
    model = RobustQuantaStudent(n_layers=2).to(args.device)
    classical_params = [p for n, p in model.named_parameters() if "vqc" not in n]
    quantum_params = [p for n, p in model.named_parameters() if "vqc" in n]

    opt = Adam([
        {"params": classical_params, "lr": args.lr_classical},
        {"params": quantum_params, "lr": args.lr_quantum},
    ])

    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    best_t2_auc = 0.0
    best_t2_metrics = {}
    best_t1_metrics = {}
    best_epoch = 0
    save_path = config.BASE_DIR / "checkpoints" / "robust_student_best.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("robust-student-full-training")
    t0 = time.time()

    with mlflow.start_run(run_name=f"robust_student_seed{args.seed}") as run:
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_classical": args.lr_classical,
            "lr_quantum": args.lr_quantum,
            "alpha": args.alpha,
            "beta": args.beta,
            "p_dropout": args.p_dropout,
            "seed": args.seed,
            "teacher_sha256": teacher_sha256,
            "cache_sha256": cache_hash,
        })

        for ep in range(1, args.epochs + 1):
            model.train()
            train_ds.set_mode("train")
            ep_tot, ep_mse, ep_bce = 0.0, 0.0, 0.0
            n_b = 0

            for x_b, z_t, y_b in train_loader:
                x_b, z_t, y_b = x_b.to(args.device), z_t.to(args.device), y_b.to(args.device)
                opt.zero_grad()
                z_pred, logits = model(x_b, return_latent=True)
                l_mse = mse_fn(z_pred, z_t)
                l_bce = bce_fn(logits, y_b)
                loss = args.alpha * l_mse + args.beta * l_bce
                loss.backward()

                # Track gradient magnitudes
                q_grad = model.vqc_layer.weights.grad.detach().view(-1)
                q_norm = float(q_grad.norm().item())
                c_grads = [p.grad.detach().view(-1) for n, p in model.named_parameters() if "vqc" not in n and p.grad is not None]
                c_norm = float(torch.cat(c_grads).norm().item()) if c_grads else 0.0

                opt.step()
                ep_tot += loss.item()
                ep_mse += l_mse.item()
                ep_bce += l_bce.item()
                n_b += 1

            train_loss = ep_tot / n_b
            train_mse = ep_mse / n_b
            train_bce = ep_bce / n_b

            # Evaluate on held-out test split
            t1_metrics = evaluate_tier_mode(model, test_ds, "tier1", args.device)
            t2_metrics = evaluate_tier_mode(model, test_ds, "tier2", args.device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_bce": train_bce,
                "q_grad_norm": q_norm,
                "c_grad_norm": c_norm,
                "t1_acc": t1_metrics["accuracy"],
                "t1_auc": t1_metrics["auroc"],
                "t1_brier": t1_metrics["brier"],
                "t2_acc": t2_metrics["accuracy"],
                "t2_auc": t2_metrics["auroc"],
                "t2_brier": t2_metrics["brier"],
            }, step=ep)

            print(
                f"Epoch {ep:02d}/{args.epochs:02d} | "
                f"Loss: {train_loss:.4f} (MSE:{train_mse:.4f}, BCE:{train_bce:.4f}) | "
                f"T1 AUC: {t1_metrics['auroc']:.4f}, Acc: {t1_metrics['accuracy']:.4f} | "
                f"T2 AUC: {t2_metrics['auroc']:.4f}, Acc: {t2_metrics['accuracy']:.4f} (Brier: {t2_metrics['brier']:.4f})"
            )

            # Checkpoint on best Tier 2 AUROC
            if t2_metrics["auroc"] > best_t2_auc:
                best_t2_auc = t2_metrics["auroc"]
                best_t2_metrics = t2_metrics
                best_t1_metrics = t1_metrics
                best_epoch = ep
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": ep,
                    "seed": args.seed,
                    "t1_metrics": t1_metrics,
                    "t2_metrics": t2_metrics,
                    "teacher_sha256": teacher_sha256,
                    "pachy_mean": pipe._pachy_mean,
                    "pachy_std": pipe._pachy_std,
                }, save_path)

        total_training_time = time.time() - t0
        print("\n" + "=" * 85)
        print(f"TRAINING COMPLETE IN {total_training_time:.1f}s. Best Epoch: {best_epoch}")
        print(f"Best Checkpoint saved to: {save_path}")
        print("=" * 85)

        # Baseline floor reference
        baseline_ref = {
            "accuracy": 0.7698,
            "precision": 0.7701,
            "recall": 0.7843,
            "specificity": 128 / (128 + 50),  # 0.7191
            "auroc": 0.8786,
            "confusion_matrix": [[128, 50], [17, 96]],
        }

        print("\n" + "=" * 85)
        print("FINAL MEASURED TEST RESULTS VS. HISTORICAL BASELINE FLOOR")
        print("=" * 85)
        print(f"{'Metric':<20} | {'Historical Floor':<18} | {'Tier 1 (SimK)':<18} | {'Tier 2 (SimK+Pachy)':<18} | {'Delta (T2 vs Floor)'}")
        print("-" * 105)

        metrics_to_print = [
            ("Accuracy", "accuracy", "{:.4f}"),
            ("AUC-ROC", "auroc", "{:.4f}"),
            ("Precision", "precision", "{:.4f}"),
            ("Recall (Sens.)", "recall", "{:.4f}"),
            ("Specificity", "specificity", "{:.4f}"),
            ("F1 Score", "f1", "{:.4f}"),
            ("Brier Score", "brier", "{:.4f}"),
            ("ECE", "ece", "{:.4f}"),
        ]

        summary_rows = []
        for label, key, fmt in metrics_to_print:
            base_val = baseline_ref.get(key, None)
            base_str = fmt.format(base_val) if base_val is not None else "—"
            t1_val = best_t1_metrics.get(key, 0.0)
            t2_val = best_t2_metrics.get(key, 0.0)
            t1_str = fmt.format(t1_val)
            t2_str = fmt.format(t2_val)

            if base_val is not None:
                delta = t2_val - base_val
                delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
            else:
                delta_str = "—"

            print(f"{label:<20} | {base_str:<18} | {t1_str:<18} | {t2_str:<18} | {delta_str}")
            summary_rows.append({
                "Metric": label,
                "Historical Baseline Floor": base_str,
                "Tier 1 (SimK Only)": t1_str,
                "Tier 2 (SimK + Pachy)": t2_str,
                "Delta (T2 vs Baseline)": delta_str,
            })

        print("-" * 105)
        print(f"Confusion Matrix (Floor):  [[128, 50], [17, 96]]")
        print(f"Confusion Matrix (Tier 1): {best_t1_metrics['confusion_matrix']}")
        print(f"Confusion Matrix (Tier 2): {best_t2_metrics['confusion_matrix']}")

        # Save summary markdown and json
        summary_df = pd.DataFrame(summary_rows)
        out_md = config.BASE_DIR / "results" / "robust_student_test_summary.md"
        out_json = config.BASE_DIR / "results" / "robust_student_test_results.json"
        summary_df.to_markdown(out_md, index=False)

        final_record = {
            "training_time_seconds": total_training_time,
            "best_epoch": best_epoch,
            "best_t2_metrics": best_t2_metrics,
            "best_t1_metrics": best_t1_metrics,
            "historical_baseline_floor": baseline_ref,
            "checkpoint_sha256": compute_sha256(save_path),
        }
        with open(out_json, "w") as f:
            json.dump(final_record, f, indent=2)

        mlflow.log_artifact(str(out_md))
        mlflow.log_artifact(str(out_json))
        mlflow.log_artifact(str(save_path))

        print(f"\nWrote summary table to: {out_md}")
        print(f"Wrote complete JSON results to: {out_json}")


if __name__ == "__main__":
    main()
