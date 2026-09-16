"""Entry point for the full distillation pipeline.

Usage:
    # Step 1: Compute dataset stats (run once)
    python scripts/compute_dataset_stats.py

    # Step 2: Pre-flight fold balance check
    python scripts/run_distill.py --preflight

    # Step 3: Stage 1 pretrain (if not done)
    python scripts/run_pretrain.py --epochs 100

    # Step 4: Teacher fine-tune
    python scripts/run_distill.py --stage teacher

    # Step 5: Full student ablation
    python scripts/run_distill.py --stage ablation

    # Smoke test (all stages, fast)
    python scripts/run_distill.py --smoke-test

    # Domain-shift probe
    python scripts/run_distill.py --stage domain-shift
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import Config
from quantum_kc.data.split_utils import verify_fold_balance, log_test_set_stats
from quantum_kc.training.teacher_finetune import finetune_teacher
from quantum_kc.training.ablation_runner import run_ablation
from quantum_kc.training.domain_shift import run_stage1_ablation


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="QuantaMED distillation pipeline")
    parser.add_argument(
        "--stage",
        choices=["preflight", "teacher", "ablation", "domain-shift", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test (1 fold, 1 seed)")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (1-5)")
    parser.add_argument("--head-type", choices=["classical", "quantum", "both"], default="both")
    parser.add_argument("--alpha", type=float, default=None, help="Distillation loss weight")
    parser.add_argument("--beta", type=float, default=None, help="Task loss weight")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.DISTILL_EPOCHS = args.epochs
        cfg.FINETUNE_EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.smoke_test:
        cfg.DISTILL_EPOCHS = 3
        cfg.FINETUNE_EPOCHS = 3
        cfg.N_SEEDS = 1

    head_types = (
        ["classical"] if args.head_type == "classical"
        else ["quantum"] if args.head_type == "quantum"
        else ["classical", "quantum"]
    )

    # ── Pre-flight ────────────────────────────────────────────────────
    if args.stage in ("preflight", "all"):
        print("\n[Pre-flight] Verifying fold balance...")
        verify_fold_balance(str(cfg.LABELED_CSV))
        log_test_set_stats(str(cfg.LABELED_CSV))
        if args.stage == "preflight":
            return

    # ── Teacher fine-tune ─────────────────────────────────────────────
    if args.stage in ("teacher", "all"):
        print("\n[Stage] Fine-tuning Teacher encoder...")
        fold = args.fold or 1
        metrics = finetune_teacher(cfg=cfg, fold=fold)
        print(f"Teacher val AUC: {metrics.get('auc_roc', 'N/A'):.4f}")

    # ── Student ablation ──────────────────────────────────────────────
    if args.stage in ("ablation", "all"):
        print("\n[Stage] Running Student ablation matrix...")
        folds = [args.fold] if args.fold else None
        alpha_grid = [args.alpha] if args.alpha is not None else None
        beta_grid = [args.beta] if args.beta is not None else None

        ablation_cfg = Config()
        if alpha_grid:
            ablation_cfg.DISTILL_ALPHA_GRID = alpha_grid
            ablation_cfg.DISTILL_BETA_GRID = beta_grid or [1.0 - a for a in alpha_grid]
        if args.epochs:
            ablation_cfg.DISTILL_EPOCHS = args.epochs
        if args.batch_size:
            ablation_cfg.BATCH_SIZE = args.batch_size
        if args.smoke_test:
            ablation_cfg.DISTILL_EPOCHS = 3
            ablation_cfg.N_SEEDS = 1

        run_ablation(
            cfg=ablation_cfg,
            head_types=head_types,
            folds=folds,
            smoke_test=args.smoke_test,
        )

    # ── Domain-shift probe ────────────────────────────────────────────
    if args.stage in ("domain-shift", "all"):
        print("\n[Stage] Running domain-shift probe...")
        fold = args.fold or 1
        seeds = [cfg.SEED] if args.smoke_test else [42, 43, 44, 45, 46]
        run_stage1_ablation(cfg=cfg, fold=fold, seeds=seeds, epochs=args.epochs)

    print("\n[Done] Pipeline complete.")


if __name__ == "__main__":
    main()
