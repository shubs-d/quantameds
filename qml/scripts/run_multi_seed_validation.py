#!/usr/bin/env python3
"""Step 8: Multi-Seed Statistical Validation of Quantum Advantage & Calibration.

Runs 3 independent seeds (42, 123, 999) across all 5 folds (15 paired runs per model):
  1. Classical-Only Student (309 params, lr=1e-3)
  2. Hybrid Baseline (305 params, uniform lr=1e-3)
  3. Hybrid Platt-Scaled (post-hoc logistic calibration on validation logits)
  4. Hybrid Dual-Rate (305 params, c=1e-3, q=1e-2)

Computes:
  - 15-run Mean ± Std for Accuracy, AUROC, Brier Score, and ECE
  - Paired Student's t-test (p-value) vs. Classical
  - Wilcoxon signed-rank test (p-value) vs. Classical
"""

from __future__ import annotations

import argparse
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.matched_ablation_models import ClassicalMLPStudent, HybridRobustQuantaStudent
from quantum_kc.training.utils import set_seed
from scripts.run_matched_ablation import CachedTabularDataset, compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _evaluate_tier(model: nn.Module, val_ds: CachedTabularDataset, val_loader: DataLoader, mode: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    val_ds.set_mode(mode)
    raw_logits, all_targets = [], []
    with torch.no_grad():
        for x_v, _, y_v in val_loader:
            l = model(x_v)
            raw_logits.extend(l.numpy())
            all_targets.extend(y_v.numpy().astype(int))

    logits_arr = np.array(raw_logits)
    y_true = np.array(all_targets)
    raw_probs = 1.0 / (1.0 + np.exp(-logits_arr))
    raw_preds = (raw_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, raw_preds))
    try:
        auc_val = float(roc_auc_score(y_true, raw_probs))
    except ValueError:
        auc_val = 0.5
    brier = float(brier_score_loss(y_true, raw_probs))
    ece = float(compute_ece(y_true, raw_probs))

    # Platt calibration
    try:
        platt = LogisticRegression()
        platt.fit(logits_arr.reshape(-1, 1), y_true)
        cal_probs = platt.predict_proba(logits_arr.reshape(-1, 1))[:, 1]
        cal_preds = (cal_probs >= 0.5).astype(int)
        platt_acc = float(accuracy_score(y_true, cal_preds))
        platt_auc = float(roc_auc_score(y_true, cal_probs))
        platt_brier = float(brier_score_loss(y_true, cal_probs))
        platt_ece = float(compute_ece(y_true, cal_probs))
    except Exception:
        platt_acc, platt_auc, platt_brier, platt_ece = acc, auc_val, brier, ece

    res_raw = {
        "accuracy": acc, "auroc": auc_val,
        "brier": brier, "ece": ece,
        "logit_mean": float(logits_arr.mean()), "logit_std": float(logits_arr.std()),
    }
    res_platt = {
        "accuracy": platt_acc, "auroc": platt_auc,
        "brier": platt_brier, "ece": platt_ece,
        "logit_mean": float(logits_arr.mean()), "logit_std": float(logits_arr.std()),
    }
    return res_raw, res_platt


def train_and_eval_model(
    model_type: str,
    train_ds: CachedTabularDataset,
    val_ds: CachedTabularDataset,
    epochs: int = 15,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate one model run, returning Tier 1 and Tier 2 metrics."""
    set_seed(seed)

    if model_type == "classical":
        model = ClassicalMLPStudent().to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
    elif model_type in ("hybrid_baseline", "hybrid_platt"):
        model = HybridRobustQuantaStudent(n_layers=2).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
    elif model_type == "hybrid_dual_rate":
        model = HybridRobustQuantaStudent(n_layers=2).to(device)
        c_p = [p for n, p in model.named_parameters() if "vqc" not in n]
        q_p = [p for n, p in model.named_parameters() if "vqc" in n]
        optimizer = Adam([{"params": c_p, "lr": 1e-3}, {"params": q_p, "lr": 1e-2}])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        train_ds.set_mode("train")
        for x_b, z_t, y_b in train_loader:
            optimizer.zero_grad()
            z_pred, logits = model(x_b, return_latent=True)
            loss = 0.5 * mse_fn(z_pred, z_t) + 0.5 * bce_fn(logits, y_b)
            loss.backward()
            optimizer.step()

    # Evaluate Tier 1 (Simulated ARK only) and Tier 2 (Simulated ARK + Pachymetry)
    t1_raw, t1_platt = _evaluate_tier(model, val_ds, val_loader, "tier1")
    t2_raw, t2_platt = _evaluate_tier(model, val_ds, val_loader, "tier2")
    return {
        "t1_raw": t1_raw, "t1_platt": t1_platt,
        "t2_raw": t2_raw, "t2_platt": t2_platt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    # Load data
    cached_data = torch.load(config.BASE_DIR / "data" / "cached_teacher_latents.pt", map_location="cpu")
    train_latents = cached_data["latents_train"]

    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)

    models = ["classical", "hybrid_baseline", "hybrid_platt", "hybrid_dual_rate"]
    runs_t1: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}
    runs_t2: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}

    print("=" * 80)
    print(f"STEP 8: MULTI-SEED VALIDATION (Seeds={args.seeds}, 5 Folds -> {len(args.seeds)*5} Runs/Model)")
    print("=" * 80)

    t0 = time.time()
    for seed in args.seeds:
        for fold in range(1, 6):
            print(f"Running Seed {seed}, Fold {fold}/5...")
            train_mask = (train_df["fold"] != fold).values
            val_mask = (train_df["fold"] == fold).values

            pipe = RobustStudentTabularPipeline()
            pipe.fit(train_df[train_mask].reset_index(drop=True))
            train_base = pipe.transform_base(train_df[train_mask].reset_index(drop=True))
            val_base = pipe.transform_base(train_df[val_mask].reset_index(drop=True))

            train_ds = CachedTabularDataset(
                train_base, train_latents[train_mask].clone(),
                train_df[train_mask]["label"].values.astype(int), mode="train"
            )
            val_ds = CachedTabularDataset(
                val_base, train_latents[val_mask].clone(),
                train_df[val_mask]["label"].values.astype(int), mode="tier2"
            )

            # 1. Classical
            res_cls = train_and_eval_model("classical", train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, device="cpu", seed=seed + fold * 10)
            runs_t1["classical"].append(res_cls["t1_raw"])
            runs_t2["classical"].append(res_cls["t2_raw"])

            # 2. Hybrid Baseline & Platt (shares same trained weights!)
            res_base = train_and_eval_model("hybrid_baseline", train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, device="cpu", seed=seed + fold * 10)
            runs_t1["hybrid_baseline"].append(res_base["t1_raw"])
            runs_t2["hybrid_baseline"].append(res_base["t2_raw"])
            runs_t1["hybrid_platt"].append(res_base["t1_platt"])
            runs_t2["hybrid_platt"].append(res_base["t2_platt"])

            # 3. Hybrid Dual-Rate
            res_dual = train_and_eval_model("hybrid_dual_rate", train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size, device="cpu", seed=seed + fold * 10)
            runs_t1["hybrid_dual_rate"].append(res_dual["t1_raw"])
            runs_t2["hybrid_dual_rate"].append(res_dual["t2_raw"])

            print(f"  [T2 Classical       ] Acc: {res_cls['t2_raw']['accuracy']:.4f} | AUROC: {res_cls['t2_raw']['auroc']:.4f} | Brier: {res_cls['t2_raw']['brier']:.4f}")
            print(f"  [T2 Hybrid Baseline ] Acc: {res_base['t2_raw']['accuracy']:.4f} | AUROC: {res_base['t2_raw']['auroc']:.4f} | Brier: {res_base['t2_raw']['brier']:.4f}")
            print(f"  [T2 Hybrid Platt    ] Acc: {res_base['t2_platt']['accuracy']:.4f} | AUROC: {res_base['t2_platt']['auroc']:.4f} | Brier: {res_base['t2_platt']['brier']:.4f}")
            print(f"  [T2 Hybrid Dual-Rate] Acc: {res_dual['t2_raw']['accuracy']:.4f} | AUROC: {res_dual['t2_raw']['auroc']:.4f} | Brier: {res_dual['t2_raw']['brier']:.4f}")

    total_time = time.time() - t0
    print(f"\nAll {len(args.seeds)*5*len(models)} runs completed in {total_time:.1f}s.")

    # Compute comparison table for a tier
    def build_summary(runs_dict: Dict[str, List[Dict[str, Any]]], tier_label: str) -> pd.DataFrame:
        cls_accs = [r["accuracy"] for r in runs_dict["classical"]]
        cls_aurocs = [r["auroc"] for r in runs_dict["classical"]]
        cls_briers = [r["brier"] for r in runs_dict["classical"]]
        cls_eces = [r["ece"] for r in runs_dict["classical"]]

        rows = []
        for m in models:
            accs = [r["accuracy"] for r in runs_dict[m]]
            aurocs = [r["auroc"] for r in runs_dict[m]]
            briers = [r["brier"] for r in runs_dict[m]]
            eces = [r["ece"] for r in runs_dict[m]]

            if m != "classical":
                p_tt_acc = stats.ttest_rel(accs, cls_accs).pvalue
                p_tt_auc = stats.ttest_rel(aurocs, cls_aurocs).pvalue
                p_tt_brier = stats.ttest_rel(briers, cls_briers).pvalue
                try:
                    p_wc_acc = stats.wilcoxon(accs, cls_accs).pvalue
                    p_wc_auc = stats.wilcoxon(aurocs, cls_aurocs).pvalue
                    p_wc_brier = stats.wilcoxon(briers, cls_briers).pvalue
                except ValueError:
                    p_wc_acc, p_wc_auc, p_wc_brier = 1.0, 1.0, 1.0
            else:
                p_tt_acc, p_tt_auc, p_tt_brier = 1.0, 1.0, 1.0
                p_wc_acc, p_wc_auc, p_wc_brier = 1.0, 1.0, 1.0

            rows.append({
                "Tier": tier_label,
                "Model": m,
                "Accuracy": f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                "AUROC": f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}",
                "Brier": f"{np.mean(briers):.4f} ± {np.std(briers):.4f}",
                "ECE": f"{np.mean(eces):.4f} ± {np.std(eces):.4f}",
                "p_acc (t/wilc)": f"{p_tt_acc:.3e} / {p_wc_acc:.3e}" if m != "classical" else "—",
                "p_auc (t/wilc)": f"{p_tt_auc:.3e} / {p_wc_auc:.3e}" if m != "classical" else "—",
                "p_brier (t/wilc)": f"{p_tt_brier:.3e} / {p_wc_brier:.3e}" if m != "classical" else "—",
            })
        return pd.DataFrame(rows)

    df_t1 = build_summary(runs_t1, "Tier 1 (SimK only)")
    df_t2 = build_summary(runs_t2, "Tier 2 (SimK + Pachy)")
    combined_df = pd.concat([df_t1, df_t2], ignore_index=True)

    print("\n" + "=" * 100)
    print("15-RUN STATISTICAL VALIDATION SUMMARY (3 SEEDS x 5 FOLDS)")
    print("=" * 100)
    print(combined_df.to_string(index=False))

    out_md = config.BASE_DIR / "results" / "multi_seed_significance_summary.md"
    out_json = config.BASE_DIR / "results" / "multi_seed_significance_results.json"
    combined_df.to_markdown(out_md, index=False)
    with open(out_json, "w") as f:
        json.dump({"tier1": runs_t1, "tier2": runs_t2}, f, indent=2, default=str)

    print(f"\nWrote statistical report to: {out_md}")
    print(f"Wrote raw runs data to: {out_json}")


if __name__ == "__main__":
    main()
