"""Ablation runner: full (alpha, beta) × head_type × seed × fold matrix.

Runs all combinations of:
    head_type : ['classical', 'quantum']
    (alpha, beta) : [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]   ← from cfg.DISTILL_ALPHA_GRID
    seed : range(cfg.N_SEEDS)
    fold : range(1, 6)

For each combination, calls train_student() and collects best val metrics.
All results are saved to results/ablation_results.json.

The naive logistic regression baseline is also run here for direct comparison.

Usage:
    python scripts/run_distill.py  (calls run_ablation() internally)
    or call run_ablation(cfg) programmatically.
"""

from __future__ import annotations

import json
import logging
import os
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from quantum_kc.config import Config, config as default_config
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.data.image_transforms import get_eval_transforms
from quantum_kc.training.distill import train_student
from quantum_kc.training.utils import (
    compute_metrics,
    get_class_weights,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_logistic_baseline(
    cfg: Config,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a logistic regression directly on the 5 Student tabular features.

    No distillation, no latent bottleneck — the simplest possible baseline.
    Uses all training folds combined and evaluates on the held-out test fold.

    Args:
        cfg: Configuration object.
        seed: Random seed for the LR solver.

    Returns:
        Dict of metrics evaluated on the held-out test set.
    """
    logger.info("Running logistic regression naive baseline (seed=%d)", seed)
    set_seed(seed)
    eval_transform = get_eval_transforms(cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD)

    train_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="train",
        fold=None,
        transform=eval_transform,
    )
    test_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="test",
        transform=eval_transform,
        tabular_scaler=train_ds.tabular_scaler,
        tabular_imputer=train_ds.tabular_imputer,
        student_pipeline=train_ds.student_pipeline,
    )

    X_train = train_ds.student_features
    y_train = train_ds.labels.numpy()
    X_test = test_ds.student_features
    y_test = test_ds.labels.numpy()

    clf = LogisticRegression(
        max_iter=cfg.LOGISTIC_BASELINE_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["model"] = "logistic_regression_baseline"
    metrics["seed"] = seed
    logger.info(
        "LR baseline | AUC=%.4f | Sensitivity=%.4f | Specificity=%.4f",
        metrics.get("auc_roc") or 0.0,
        metrics["sensitivity"],
        metrics["specificity"],
    )
    return metrics


def run_ablation(
    cfg: Optional[Config] = None,
    head_types: Optional[List[str]] = None,
    folds: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run the full ablation matrix and save results to JSON.

    Args:
        cfg: Configuration object.
        head_types: List of head types to run (default ['classical', 'quantum']).
        folds: List of folds to run (default [1,2,3,4,5]).
        smoke_test: If True, run only 1 fold × 1 seed × 1 (alpha,beta) to verify.

    Returns:
        Dict of all collected results.
    """
    if cfg is None:
        cfg = default_config

    head_types = head_types or ["classical", "quantum"]
    folds = folds or list(range(1, 6))
    seeds = list(range(cfg.N_SEEDS))

    alpha_beta_pairs = list(zip(cfg.DISTILL_ALPHA_GRID, cfg.DISTILL_BETA_GRID))

    if smoke_test:
        folds = [folds[0]]
        seeds = [seeds[0]]
        alpha_beta_pairs = [alpha_beta_pairs[1]]  # balanced
        logger.info("SMOKE TEST: running 1 fold × 1 seed × balanced loss only.")

    os.makedirs(str(cfg.RESULTS_DIR), exist_ok=True)
    results_path = str(cfg.RESULTS_DIR / "ablation_results.json")

    # ── Load existing results if resuming ──────────────────────────────
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            logger.info("Resuming ablation from existing %s", results_path)
        except Exception as e:
            logger.warning("Could not load existing %s: %s — starting fresh", results_path, e)
            all_results = {"classical": [], "quantum": [], "logistic_baseline": []}
    else:
        all_results = {"classical": [], "quantum": [], "logistic_baseline": []}

    all_results.setdefault("classical", [])
    all_results.setdefault("quantum", [])
    all_results.setdefault("logistic_baseline", [])

    # ── Logistic regression baseline (once per seed, fold-independent) ──
    existing_lr_seeds = {r.get("seed") for r in all_results["logistic_baseline"] if "seed" in r}
    for seed in seeds:
        if seed in existing_lr_seeds:
            continue
        try:
            m = run_logistic_baseline(cfg, seed=seed)
            all_results["logistic_baseline"].append(m)
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            logger.error("LR baseline failed (seed=%d): %s", seed, e)

    # ── Student distillation matrix ────────────────────────────────────
    total_runs = len(head_types) * len(folds) * len(seeds) * len(alpha_beta_pairs)
    run_idx = 0
    for head_type, fold, seed, (alpha, beta) in product(head_types, folds, seeds, alpha_beta_pairs):
        run_idx += 1

        # Check if this configuration was already completed
        already_completed = any(
            r.get("fold") == fold
            and r.get("seed") == seed
            and abs(float(r.get("alpha", -1.0)) - alpha) < 1e-4
            and abs(float(r.get("beta", -1.0)) - beta) < 1e-4
            for r in all_results.get(head_type, [])
        )
        if already_completed:
            logger.info(
                "[%d/%d] [RESUME - SKIP] head=%s fold=%d seed=%d α=%.1f β=%.1f already done.",
                run_idx, total_runs, head_type, fold, seed, alpha, beta,
            )
            continue

        logger.info(
            "[%d/%d] head=%s fold=%d seed=%d α=%.1f β=%.1f",
            run_idx, total_runs, head_type, fold, seed, alpha, beta,
        )
        try:
            metrics = train_student(
                cfg=cfg,
                fold=fold,
                head_type=head_type,
                alpha=alpha,
                beta=beta,
                seed=seed,
            )
            all_results[head_type].append(metrics)

            # Save immediately after each run so progress is never lost
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info("Saved incremental progress to %s", results_path)

        except Exception as e:
            logger.error(
                "Run failed (head=%s fold=%d seed=%d α=%.1f β=%.1f): %s",
                head_type, fold, seed, alpha, beta, e,
            )

    _print_summary(all_results)
    return all_results


def _print_summary(all_results: Dict[str, List[Dict]]) -> None:
    """Print a brief per-variant summary to stdout."""
    print("\n" + "=" * 70)
    print("Ablation summary (mean ± std over seeds × folds, balanced α=β=0.5)")
    print("=" * 70)
    for variant, results in all_results.items():
        balanced = [r for r in results if r.get("alpha", 0.5) == 0.5]
        if not balanced:
            balanced = results
        if not balanced:
            continue
        aucs = [r["auc_roc"] for r in balanced if r.get("auc_roc") is not None]
        sens = [r["sensitivity"] for r in balanced if "sensitivity" in r]
        spec = [r["specificity"] for r in balanced if "specificity" in r]
        print(
            f"  {variant:<30}  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}"
            f"  Sens={np.mean(sens):.3f}  Spec={np.mean(spec):.3f}"
            f"  (n={len(aucs)})"
        )
    print("=" * 70 + "\n")
