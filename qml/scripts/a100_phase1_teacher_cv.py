#!/usr/bin/env python3
"""Phase 1: Teacher 5-fold CV on the 1,163 labeled CornOrb training eyes.

Two conditions per fold:
  A) use_pretrained=True   — fine-tune from Stage-1 pretrained_encoder.pt
  B) use_pretrained=False  — train from random initialization

Across all 5 patient-grouped folds, reports mean ± std for Accuracy and AUC-ROC.
At the end, selects the canonical frozen Teacher and states it explicitly.

This ONLY runs 5-fold CV on the 1,163 LABELED eyes (Stage 2).
Stage 1 pretraining on the 2,633 UNLABELED images is a SEPARATE step and
runs ONCE — not per fold.

Usage:
    python scripts/a100_phase1_teacher_cv.py [--epochs N] [--seed S]
    python scripts/a100_phase1_teacher_cv.py --check-pretrain-only  # just verify Stage-1 ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config, get_device
from quantum_kc.training.teacher_finetune import finetune_teacher

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PRETRAINED_CKPT = str(config.CHECKPOINT_DIR / "pretrained_encoder.pt")
CANONICAL_TEACHER_CKPT = str(
    config.CHECKPOINT_DIR / "teacher_finetuned_with_pretrain_s42_fold1.pt"
)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def check_pretrained_encoder() -> dict:
    """Verify Stage-1 pretrained_encoder.pt exists and is non-trivially sized."""
    ckpt = Path(PRETRAINED_CKPT)
    if not ckpt.exists():
        return {
            "exists": False,
            "action": "NEED_PRETRAIN",
            "message": f"pretrained_encoder.pt NOT found at {PRETRAINED_CKPT}. "
                       f"Run Stage 1 pretraining first: python scripts/run_pretrain.py",
        }
    h = sha256(PRETRAINED_CKPT)
    size_mb = ckpt.stat().st_size / 1e6
    return {
        "exists": True,
        "path": str(ckpt),
        "sha256": h,
        "size_mb": round(size_mb, 2),
        "action": "REUSE",
        "message": f"pretrained_encoder.pt exists ({size_mb:.1f} MB). Reusing for Stage-2 init.",
    }


def run_five_fold_cv(epochs: int, seed: int) -> dict:
    """Run 5-fold Teacher CV for both conditions (pretrained vs from scratch)."""
    n_folds = 5
    conditions = [
        {"use_pretrained": True,  "tag": "with_pretrain"},
        {"use_pretrained": False, "tag": "no_pretrain"},
    ]

    all_results = {cond["tag"]: [] for cond in conditions}

    for cond in conditions:
        tag = cond["tag"]
        use_pretrained = cond["use_pretrained"]
        print(f"\n{'='*70}")
        print(f"Condition: {tag} | use_pretrained={use_pretrained} | epochs={epochs} | seed={seed}")
        print("="*70)

        if use_pretrained:
            pretrain_info = check_pretrained_encoder()
            if not pretrain_info["exists"]:
                logger.error(pretrain_info["message"])
                raise FileNotFoundError(pretrain_info["message"])
            logger.info(pretrain_info["message"])

        for fold in range(1, n_folds + 1):
            logger.info("[%s] Fold %d/%d ...", tag, fold, n_folds)
            from quantum_kc.config import Config
            fold_cfg = Config()
            fold_cfg.FINETUNE_EPOCHS = epochs

            result = finetune_teacher(
                cfg=fold_cfg,
                fold=fold,
                use_pretrained=use_pretrained,
                pretrained_ckpt=PRETRAINED_CKPT if use_pretrained else None,
                save_tag=tag,
                seed=seed,
            )
            all_results[tag].append({
                "fold": fold,
                "val_auc": result.get("auc_roc"),
                "val_acc": result.get("accuracy"),
                "test_auc": result.get("test_auc_roc"),
                "test_acc": result.get("test_accuracy", result.get("test_sensitivity")),
                "checkpoint_path": result.get("checkpoint_path"),
            })
            logger.info(
                "  [%s] Fold %d done — val AUC %.4f | test AUC %.4f",
                tag, fold,
                result.get("auc_roc") or 0.0,
                result.get("test_auc_roc") or 0.0,
            )

    return all_results


def compute_summary(results_by_tag: dict) -> dict:
    """Compute mean ± std across folds for each condition."""
    summary = {}
    for tag, fold_results in results_by_tag.items():
        val_aucs = [r["val_auc"] for r in fold_results if r.get("val_auc") is not None]
        test_aucs = [r["test_auc"] for r in fold_results if r.get("test_auc") is not None]
        val_accs = [r["val_acc"] for r in fold_results if r.get("val_acc") is not None]
        summary[tag] = {
            "val_auc_mean": float(np.mean(val_aucs)) if val_aucs else None,
            "val_auc_std":  float(np.std(val_aucs))  if val_aucs else None,
            "test_auc_mean": float(np.mean(test_aucs)) if test_aucs else None,
            "test_auc_std":  float(np.std(test_aucs))  if test_aucs else None,
            "val_acc_mean": float(np.mean(val_accs)) if val_accs else None,
            "val_acc_std":  float(np.std(val_accs))  if val_accs else None,
            "n_folds": len(fold_results),
        }
    return summary


def select_canonical_teacher(results_by_tag: dict, summary: dict) -> dict:
    """
    Select canonical Teacher checkpoint.
    Criterion: highest mean validation AUC across folds.
    Tie-break: prefer the pretrained condition (lower variance expected).
    Then pick the fold checkpoint with highest val_auc within that condition.
    """
    best_tag = max(
        summary.keys(),
        key=lambda t: (summary[t].get("val_auc_mean") or 0.0),
    )
    best_fold_result = max(
        results_by_tag[best_tag],
        key=lambda r: r.get("val_auc") or 0.0,
    )
    return {
        "canonical_tag": best_tag,
        "canonical_fold": best_fold_result.get("fold"),
        "canonical_val_auc": best_fold_result.get("val_auc"),
        "canonical_checkpoint": best_fold_result.get("checkpoint_path"),
        "selection_criterion": "highest mean validation AUC across 5 folds; tie-break: pretrained condition",
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Teacher 5-fold CV")
    parser.add_argument("--epochs", type=int, default=50, help="Finetune epochs per fold (default 50)")
    parser.add_argument("--seed",   type=int, default=42, help="Random seed")
    parser.add_argument(
        "--check-pretrain-only", action="store_true",
        help="Only check Stage-1 pretrained_encoder.pt, then exit.",
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("PHASE 1 — Teacher 5-fold CV (Stage 2, 1,163 labeled eyes)")
    print("="*70)
    print(f"  epochs={args.epochs}, seed={args.seed}")
    print(f"  NOTE: Stage-1 pretraining (2,633 unlabeled) runs ONCE if needed.")
    print(f"        The 5-fold CV runs ONLY on the Stage-2 labeled split.")

    # Always verify Stage-1 encoder first
    pretrain_info = check_pretrained_encoder()
    print(f"\nStage-1 pretrained encoder check:")
    print(f"  {pretrain_info['message']}")
    if pretrain_info["exists"]:
        print(f"  SHA-256: {pretrain_info['sha256']} | Size: {pretrain_info['size_mb']} MB")

    if args.check_pretrain_only:
        print(json.dumps(pretrain_info, indent=2))
        return

    if not pretrain_info["exists"]:
        print(
            "\nStage-1 checkpoint is missing. You must run Stage 1 pretraining first:\n"
            "    python scripts/run_pretrain.py\n"
            "Then re-run this script."
        )
        sys.exit(1)

    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.warning("Could not set MLflow tracking URI: %s", e)

    results_by_tag = run_five_fold_cv(epochs=args.epochs, seed=args.seed)
    summary = compute_summary(results_by_tag)
    canonical = select_canonical_teacher(results_by_tag, summary)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 1 SUMMARY — Teacher 5-Fold CV")
    print("="*70)
    print(f"{'Condition':<20} {'Val AUC (mean±std)':>25} {'Test AUC (mean±std)':>25}")
    print("-"*70)
    for tag, s in summary.items():
        va = f"{s['val_auc_mean']:.4f}±{s['val_auc_std']:.4f}" if s["val_auc_mean"] is not None else "N/A"
        ta = f"{s['test_auc_mean']:.4f}±{s['test_auc_std']:.4f}" if s["test_auc_mean"] is not None else "N/A"
        print(f"  {tag:<18} {va:>25} {ta:>25}")
    print()
    print(f"CANONICAL TEACHER selected:")
    print(f"  Condition : {canonical['canonical_tag']}")
    print(f"  Fold      : {canonical['canonical_fold']}")
    print(f"  Val AUC   : {canonical['canonical_val_auc']:.4f}")
    print(f"  Checkpoint: {canonical['canonical_checkpoint']}")
    print(f"  Criterion : {canonical['selection_criterion']}")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "phase": "1_teacher_5fold_cv",
        "timestamp": datetime.now().isoformat(),
        "epochs": args.epochs,
        "seed": args.seed,
        "pretrained_encoder_sha256": pretrain_info.get("sha256"),
        "results_by_fold": results_by_tag,
        "summary": summary,
        "canonical_teacher": canonical,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "a100_phase1_teacher_cv.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] Results → {out_path}")

    # Log canonical teacher to MLflow
    try:
        with mlflow.start_run(run_name="a100_phase1_teacher_cv_summary"):
            for tag, s in summary.items():
                if s.get("val_auc_mean") is not None:
                    mlflow.log_metric(f"{tag}_val_auc_mean", s["val_auc_mean"])
                    mlflow.log_metric(f"{tag}_val_auc_std",  s["val_auc_std"])
                if s.get("test_auc_mean") is not None:
                    mlflow.log_metric(f"{tag}_test_auc_mean", s["test_auc_mean"])
                    mlflow.log_metric(f"{tag}_test_auc_std",  s["test_auc_std"])
            mlflow.log_param("canonical_tag", canonical["canonical_tag"])
            mlflow.log_param("canonical_fold", canonical["canonical_fold"])
            mlflow.log_param("canonical_checkpoint", canonical["canonical_checkpoint"])
            mlflow.log_artifact(str(out_path))
        print("[MLflow] Phase 1 summary logged.")
    except Exception as e:
        logger.warning("MLflow summary logging failed (%s). Continuing, results already saved to JSON.", e)
        print("[MLflow] Skipped MLflow summary logging — results saved to JSON.")


if __name__ == "__main__":
    main()
