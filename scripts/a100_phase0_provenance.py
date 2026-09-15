#!/usr/bin/env python3
"""Phase 0b + 0c: Provenance check and Teacher vs Hybrid comparison.

Phase 0b: Check hybrid_classifier_best.pt mtime vs three MLflow evaluate_model runs
           (Sep 7 12:53, 18:15, 18:16).  If mtime > all runs → file was overwritten;
           report explicitly.  Otherwise re-evaluate with ImageNet norm and report
           whether it exactly reproduces 0.7698/0.8786/[[128,50],[17,96]].

Phase 0c: Side-by-side comparison on the 291-eye held-out test set:
            - CornealEncoder (CNN + supervised_head) from teacher_finetuned_with_pretrain_s42_fold1.pt
            - HybridQuantumClassifier (CNN + VQC head) from hybrid_classifier_best.pt
           Both use identical preprocessing. Report both confusion matrices.

Run:
    python scripts/a100_phase0_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config, get_device
from quantum_kc.data.image_transforms import get_eval_transforms
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.models.encoder import CornealEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# ── MLflow run timestamps for Sep 7 evaluate_model runs ───────────────────────
# The three known evaluate_model run timestamps (local tz → epoch seconds)
# 2023-09-07 12:53, 18:15, 18:16  (user's local timezone — treat as UTC for conservatism)
KNOWN_RUN_TIMESTAMPS = [
    datetime(2023, 9, 7, 12, 53, 0).timestamp(),
    datetime(2023, 9, 7, 18, 15, 0).timestamp(),
    datetime(2023, 9, 7, 18, 16, 0).timestamp(),
]
LATEST_KNOWN_RUN = max(KNOWN_RUN_TIMESTAMPS)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EXPECTED_ACC = 0.7698
EXPECTED_AUC = 0.8786
EXPECTED_CM  = [[128, 50], [17, 96]]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for imgs, _tab, targets in loader:
            imgs = imgs.to(device)
            out = model(imgs, return_logits=True)
            # CornealEncoder returns (latent, logits); HybridQuantumClassifier may differ
            if isinstance(out, tuple):
                logits = out[1]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(np.argmax(probs, axis=1))
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.ndim == 2 else roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "precision": prec, "recall": rec, "auc_roc": auc, "cm": cm}


def phase_0b(device: torch.device) -> dict:
    """Check hybrid_classifier_best.pt provenance."""
    print("\n" + "="*70)
    print("PHASE 0b — hybrid_classifier_best.pt provenance check")
    print("="*70)

    hybrid_ckpt = config.CHECKPOINT_DIR / "hybrid_classifier_best.pt"
    if not hybrid_ckpt.exists():
        print(f"[SKIP] {hybrid_ckpt} does not exist — skipping provenance check.")
        return {"status": "file_missing"}

    mtime = os.path.getmtime(str(hybrid_ckpt))
    mtime_str = datetime.fromtimestamp(mtime).isoformat()
    ckpt_hash = sha256(str(hybrid_ckpt))
    print(f"File:        {hybrid_ckpt}")
    print(f"mtime:       {mtime_str}  (epoch {mtime:.0f})")
    print(f"SHA-256:     {ckpt_hash}")
    print(f"Latest known evaluate_model run: {datetime.fromtimestamp(LATEST_KNOWN_RUN).isoformat()}")

    if mtime > LATEST_KNOWN_RUN + 60:  # 60s grace
        verdict = (
            "OVERWRITTEN — file mtime is LATER than all three evaluate_model runs. "
            "The 0.7698/0.8786/[[128,50],[17,96]] baseline is NOT reproducible from "
            "current weights."
        )
        print(f"\n[VERDICT] {verdict}")
        return {"status": "overwritten", "mtime": mtime_str, "sha256": ckpt_hash, "verdict": verdict}

    # mtime doesn't resolve it → re-evaluate with ImageNet norm
    print(f"\nmtime ({mtime_str}) does NOT clearly post-date the known runs.")
    print("Re-evaluating with ImageNet normalization to check reproducibility...")

    try:
        from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
        enc = CornealEncoder(in_channels=3, latent_dim=config.LATENT_DIM)
        model = HybridQuantumClassifier(encoder=enc)
        state = torch.load(str(hybrid_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
    except Exception as e:
        logger.warning("Could not load HybridQuantumClassifier: %s", e)
        # Fall back to CornealEncoder direct load
        model = CornealEncoder(in_channels=3, latent_dim=config.LATENT_DIM, n_classes=2)
        state = torch.load(str(hybrid_ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model = model.to(device)

    transform = get_eval_transforms(config.IMG_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    test_ds = CornOrbDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        split="test",
        transform=transform,
    )
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    metrics = evaluate_model(model, loader, device)

    print(f"\nImageNet-norm evaluation results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  (expected {EXPECTED_ACC})")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}  (expected {EXPECTED_AUC})")
    print(f"  CM:        {metrics['cm']}  (expected {EXPECTED_CM})")

    reproduces = (
        abs(metrics['accuracy'] - EXPECTED_ACC) < 1e-4
        and abs(metrics['auc_roc'] - EXPECTED_AUC) < 1e-4
        and metrics['cm'] == EXPECTED_CM
    )
    verdict = "REPRODUCED exactly" if reproduces else "NOT reproduced — baseline is non-reproducible from current weights"
    print(f"\n[VERDICT] {verdict}")
    metrics["verdict"] = verdict
    metrics["sha256"] = ckpt_hash
    metrics["mtime"] = mtime_str
    return metrics


def phase_0c(device: torch.device) -> dict:
    """Side-by-side Teacher CNN vs CNN+VQC on 291-eye test split."""
    print("\n" + "="*70)
    print("PHASE 0c — Teacher CNN vs CNN+VQC head (291-eye test set)")
    print("="*70)

    dataset_norm_transform = get_eval_transforms(
        config.IMG_SIZE, mean=config.DATASET_MEAN, std=config.DATASET_STD
    )
    test_ds = CornOrbDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        split="test",
        transform=dataset_norm_transform,
    )
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Test set: {len(test_ds)} eyes")

    results = {}

    # ── Teacher: CNN + supervised_head ────────────────────────────────
    teacher_ckpt = config.CHECKPOINT_DIR / "teacher_finetuned_with_pretrain_s42_fold1.pt"
    teacher_hash = sha256(str(teacher_ckpt))
    print(f"\nTeacher checkpoint: {teacher_ckpt.name}")
    print(f"  SHA-256: {teacher_hash}")

    teacher = CornealEncoder(in_channels=3, latent_dim=config.LATENT_DIM, n_classes=2).to(device)
    state = torch.load(str(teacher_ckpt), map_location=device, weights_only=True)
    teacher.load_state_dict(state)
    teacher_metrics = evaluate_model(teacher, loader, device)
    results["teacher_cnn"] = {**teacher_metrics, "checkpoint": teacher_ckpt.name, "sha256": teacher_hash}

    print(f"  Accuracy:  {teacher_metrics['accuracy']:.4f}")
    print(f"  AUC-ROC:   {teacher_metrics['auc_roc']:.4f}")
    print(f"  CM:        {teacher_metrics['cm']}")

    # ── Hybrid: CNN + VQC head ────────────────────────────────────────
    hybrid_ckpt = config.CHECKPOINT_DIR / "hybrid_classifier_best.pt"
    if hybrid_ckpt.exists():
        hybrid_hash = sha256(str(hybrid_ckpt))
        print(f"\nHybrid checkpoint: {hybrid_ckpt.name}")
        print(f"  SHA-256: {hybrid_hash}")

        try:
            from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
            enc = CornealEncoder(in_channels=3, latent_dim=config.LATENT_DIM)
            hybrid = HybridQuantumClassifier(encoder=enc)
            state = torch.load(str(hybrid_ckpt), map_location=device, weights_only=True)
            hybrid.load_state_dict(state)
            hybrid = hybrid.to(device)
            # Use dataset-specific norm for the hybrid too (same as Teacher)
            hybrid_metrics = evaluate_model(hybrid, loader, device)
            results["hybrid_vqc"] = {**hybrid_metrics, "checkpoint": hybrid_ckpt.name, "sha256": hybrid_hash}
            print(f"  Accuracy:  {hybrid_metrics['accuracy']:.4f}")
            print(f"  AUC-ROC:   {hybrid_metrics['auc_roc']:.4f}")
            print(f"  CM:        {hybrid_metrics['cm']}")
        except Exception as e:
            logger.warning("Could not evaluate HybridQuantumClassifier: %s", e)
            results["hybrid_vqc"] = {"status": f"error: {e}"}
    else:
        print(f"\n[SKIP] {hybrid_ckpt} not found — skipping hybrid evaluation.")
        results["hybrid_vqc"] = {"status": "file_missing"}

    # ── Side-by-side summary ──────────────────────────────────────────
    print("\n" + "-"*70)
    print("PHASE 0c SUMMARY — Teacher CNN vs CNN+VQC (291-eye test set)")
    print("-"*70)
    print(f"{'Metric':<20} {'Teacher CNN':>20} {'CNN+VQC Head':>20}")
    print("-"*70)
    for k in ["accuracy", "precision", "recall", "auc_roc"]:
        t_val = results.get("teacher_cnn", {}).get(k, float("nan"))
        h_val = results.get("hybrid_vqc", {}).get(k, float("nan"))
        t_str = f"{t_val:.4f}" if isinstance(t_val, float) else str(t_val)
        h_str = f"{h_val:.4f}" if isinstance(h_val, float) else str(h_val)
        print(f"  {k:<18} {t_str:>20} {h_str:>20}")
    print(f"  {'CM':<18} {str(results.get('teacher_cnn', {}).get('cm', '?')):>20} "
          f"{str(results.get('hybrid_vqc', {}).get('cm', '?')):>20}")

    return results


def main():
    device = get_device()
    print(f"\nDevice: {device}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    phase0b_results = phase_0b(device)
    phase0c_results = phase_0c(device)

    combined = {
        "phase_0b": phase0b_results,
        "phase_0c": phase0c_results,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = config.RESULTS_DIR / "a100_phase0_results.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n[SAVED] Results → {out_path}")

    # Log to MLflow
    with mlflow.start_run(run_name="a100_phase0_provenance"):
        for k, v in phase0b_results.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"phase0b_{k}", v)
        if "teacher_cnn" in phase0c_results:
            for k, v in phase0c_results["teacher_cnn"].items():
                if isinstance(v, float):
                    mlflow.log_metric(f"teacher_cnn_{k}", v)
        if "hybrid_vqc" in phase0c_results and isinstance(phase0c_results["hybrid_vqc"], dict):
            for k, v in phase0c_results["hybrid_vqc"].items():
                if isinstance(v, float):
                    mlflow.log_metric(f"hybrid_vqc_{k}", v)
        mlflow.log_artifact(str(out_path))
    print("[MLflow] Phase 0 logged.")


if __name__ == "__main__":
    main()
