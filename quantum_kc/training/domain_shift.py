"""Domain-shift probe: ablation for Stage 1 pretraining contribution.

Experiment:
    Condition A: Teacher pretrained on Stage 1 (autoencoder) → fine-tuned on Stage 2
    Condition B: Teacher trained from random init → fine-tuned on Stage 2 only

For each condition, a Student (classical tabular MLP) is distilled using the
resulting Teacher's dynamic embeddings.

Rigorous safeguards enforced:
  1. Automated assertions verifying that the Teacher checkpoints for each seed
     have distinct SHA256 hashes and divergent weight tensors.
  2. Multi-seed evaluation (≥5 random seeds) for statistical validity.
  3. Paired significance testing (paired t-test) on Teacher and Student AUC differences.
  4. Scoring on both validation and held-out test splits.
  5. Distinct checkpoint filenames to prevent file overwrite.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as sp_stats
import torch

from quantum_kc.config import Config, config as default_config
from quantum_kc.training.distill import train_student
from quantum_kc.training.teacher_finetune import finetune_teacher

logger = logging.getLogger(__name__)


def compute_checkpoint_hash(filepath: str) -> str:
    """Compute SHA256 hash of a checkpoint file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_stage1_ablation(
    cfg: Optional[Config] = None,
    fold: int = 1,
    seeds: Optional[List[int]] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare Student and Teacher performance with vs. without Stage 1 pretraining.

    Args:
        cfg: Configuration object.
        fold: Fold to use for validation split.
        seeds: List of random seeds (defaults to 5 seeds: [42, 43, 44, 45, 46]).
        epochs: Optional epoch override for faster iteration.

    Returns:
        Dict with full per-seed results, summary statistics, and significance test.
    """
    if cfg is None:
        cfg = default_config
    seeds = seeds or [42, 43, 44, 45, 46]

    if epochs is not None:
        cfg.FINETUNE_EPOCHS = epochs
        cfg.DISTILL_EPOCHS = epochs

    results: Dict[str, Any] = {
        "seeds": seeds,
        "fold": fold,
        "epochs": cfg.FINETUNE_EPOCHS,
        "with_pretraining": [],
        "without_pretraining": [],
        "teacher_comparison": {},
        "student_comparison": {},
        "finding": "",
    }

    print("\n" + "=" * 75)
    print("STAGE 1 PRETRAINING ABLATION (Domain Shift Probe)")
    print(f"Seeds: {seeds} | Fold: {fold} | Epochs: {cfg.FINETUNE_EPOCHS}")
    print("=" * 75)

    for seed in seeds:
        print(f"\n>>> Running Ablation for Seed: {seed} <<<")

        # ── Step 1: Train Teacher WITH Stage 1 pretraining ────────────
        logger.info("[Seed %d] Training Teacher WITH Stage 1 pretraining...", seed)
        t_with = finetune_teacher(
            cfg=cfg,
            fold=fold,
            use_pretrained=True,
            save_tag=f"with_pretrain_s{seed}",
            seed=seed,
        )
        ckpt_with = t_with["checkpoint_path"]
        hash_with = compute_checkpoint_hash(ckpt_with)

        # ── Step 2: Train Teacher WITHOUT Stage 1 pretraining ─────────
        logger.info("[Seed %d] Training Teacher WITHOUT Stage 1 pretraining (random init)...", seed)
        t_without = finetune_teacher(
            cfg=cfg,
            fold=fold,
            use_pretrained=False,
            save_tag=f"without_pretrain_s{seed}",
            seed=seed,
        )
        ckpt_without = t_without["checkpoint_path"]
        hash_without = compute_checkpoint_hash(ckpt_without)

        # ── Step 3: Hard Assertions — Ensure models actually differ ────
        logger.info("[Seed %d] Running divergence assertions...", seed)
        if hash_with == hash_without:
            raise AssertionError(
                f"FATAL: Seed {seed} Teacher checkpoints are byte-for-byte identical (hash: {hash_with})! "
                f"with_pretraining and without_pretraining must produce distinct models."
            )

        # Weight tensor numerical difference
        w_with = torch.load(ckpt_with, map_location="cpu", weights_only=True)
        w_without = torch.load(ckpt_without, map_location="cpu", weights_only=True)
        weight_diff = float(torch.norm(w_with["encoder.0.weight"] - w_without["encoder.0.weight"]).item())
        if weight_diff < 1e-4:
            raise AssertionError(
                f"FATAL: Seed {seed} Teacher conv weights did not diverge (diff norm: {weight_diff:.6f})! "
                f"Pretrained weights were not loaded correctly."
            )
        logger.info(
            "[Seed %d] Divergence check PASSED: hash_with=%s... | hash_without=%s... | weight_diff=%.4f",
            seed, hash_with[:10], hash_without[:10], weight_diff,
        )

        # ── Step 4: Distill Student under Condition A (with pretraining) ─
        logger.info("[Seed %d] Distilling Student with Pretrained Teacher...", seed)
        s_with = train_student(
            cfg=cfg,
            fold=fold,
            head_type="classical",
            alpha=0.5,
            beta=0.5,
            seed=seed,
            teacher_ckpt=ckpt_with,
            save_tag=f"with_pretrain_s{seed}",
        )
        s_with["condition"] = "with_pretraining"
        s_with["seed"] = seed
        s_with["teacher_val_auc"] = t_with.get("auc_roc")
        s_with["teacher_test_auc"] = t_with.get("test_auc_roc")
        s_with["teacher_checkpoint"] = ckpt_with
        s_with["teacher_hash"] = hash_with
        results["with_pretraining"].append(s_with)

        # ── Step 5: Distill Student under Condition B (without pretraining) 
        logger.info("[Seed %d] Distilling Student with Scratch Teacher...", seed)
        s_without = train_student(
            cfg=cfg,
            fold=fold,
            head_type="classical",
            alpha=0.5,
            beta=0.5,
            seed=seed,
            teacher_ckpt=ckpt_without,
            save_tag=f"without_pretrain_s{seed}",
        )
        s_without["condition"] = "without_pretraining"
        s_without["seed"] = seed
        s_without["teacher_val_auc"] = t_without.get("auc_roc")
        s_without["teacher_test_auc"] = t_without.get("test_auc_roc")
        s_without["teacher_checkpoint"] = ckpt_without
        s_without["teacher_hash"] = hash_without
        results["without_pretraining"].append(s_without)

        # Intermediate save after each seed
        os.makedirs(str(cfg.RESULTS_DIR), exist_ok=True)
        out_path = str(cfg.RESULTS_DIR / "domain_shift_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ── Summary & Statistical Testing ─────────────────────────────────
    teacher_with_test = [r["teacher_test_auc"] for r in results["with_pretraining"] if r.get("teacher_test_auc") is not None]
    teacher_without_test = [r["teacher_test_auc"] for r in results["without_pretraining"] if r.get("teacher_test_auc") is not None]

    student_with_test = [r.get("test_auc_roc") or r["auc_roc"] for r in results["with_pretraining"]]
    student_without_test = [r.get("test_auc_roc") or r["auc_roc"] for r in results["without_pretraining"]]

    # Paired t-test on Teacher Test AUC
    t_teacher_stat, p_teacher = (
        sp_stats.ttest_rel(teacher_with_test, teacher_without_test)
        if len(teacher_with_test) > 1 and len(teacher_with_test) == len(teacher_without_test)
        else (float("nan"), float("nan"))
    )

    # Paired t-test on Student Test AUC
    t_student_stat, p_student = (
        sp_stats.ttest_rel(student_with_test, student_without_test)
        if len(student_with_test) > 1 and len(student_with_test) == len(student_without_test)
        else (float("nan"), float("nan"))
    )

    results["teacher_comparison"] = {
        "with_pretraining_mean": float(np.mean(teacher_with_test)),
        "with_pretraining_std": float(np.std(teacher_with_test)),
        "without_pretraining_mean": float(np.mean(teacher_without_test)),
        "without_pretraining_std": float(np.std(teacher_without_test)),
        "delta_mean": float(np.mean(teacher_with_test) - np.mean(teacher_without_test)),
        "t_statistic": float(t_teacher_stat),
        "p_value": float(p_teacher),
        "significant_at_05": bool(p_teacher < 0.05) if not np.isnan(p_teacher) else False,
    }

    results["student_comparison"] = {
        "with_pretraining_mean": float(np.mean(student_with_test)),
        "with_pretraining_std": float(np.std(student_with_test)),
        "without_pretraining_mean": float(np.mean(student_without_test)),
        "without_pretraining_std": float(np.std(student_without_test)),
        "delta_mean": float(np.mean(student_with_test) - np.mean(student_without_test)),
        "t_statistic": float(t_student_stat),
        "p_value": float(p_student),
        "significant_at_05": bool(p_student < 0.05) if not np.isnan(p_student) else False,
    }

    s_delta = results["student_comparison"]["delta_mean"]
    s_sig = results["student_comparison"]["significant_at_05"]
    s_pval = results["student_comparison"]["p_value"]
    t_delta = results["teacher_comparison"]["delta_mean"]
    t_pval = results["teacher_comparison"]["p_value"]

    finding = (
        f"Stage 1 pretraining effect across {len(seeds)} seeds on held-out test data: "
        f"Teacher test AUC was {np.mean(teacher_with_test):.4f} ± {np.std(teacher_with_test):.4f} (with) "
        f"vs {np.mean(teacher_without_test):.4f} ± {np.std(teacher_without_test):.4f} (without) "
        f"(Δ={t_delta:+.4f}, p={t_pval:.4f}). "
        f"Student test AUC was {np.mean(student_with_test):.4f} ± {np.std(student_with_test):.4f} (with) "
        f"vs {np.mean(student_without_test):.4f} ± {np.std(student_without_test):.4f} (without) "
        f"(Δ={s_delta:+.4f}, {'statistically significant p=' + f'{s_pval:.4f}' if s_sig else 'not statistically significant p=' + f'{s_pval:.4f}'}). "
        f"{'Stage 1 pretraining provides measurable positive transfer to the downstream Student.' if s_delta > 0.005 and s_sig else 'Stage 1 unsupervised pretraining provides no statistically significant advantage over training the Teacher from scratch on this clinical cohort.'}"
    )
    results["finding"] = finding

    # Final save
    out_path = str(cfg.RESULTS_DIR / "domain_shift_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 75)
    print("DOMAIN SHIFT PROBE SUMMARY")
    print("=" * 75)
    print(f"Teacher Test AUC (With Pretraining)    : {results['teacher_comparison']['with_pretraining_mean']:.4f} ± {results['teacher_comparison']['with_pretraining_std']:.4f}")
    print(f"Teacher Test AUC (Without Pretraining) : {results['teacher_comparison']['without_pretraining_mean']:.4f} ± {results['teacher_comparison']['without_pretraining_std']:.4f}")
    print(f"Teacher Δ: {t_delta:+.4f} (p={t_pval:.4f})")
    print("-" * 75)
    print(f"Student Test AUC (With Pretraining)    : {results['student_comparison']['with_pretraining_mean']:.4f} ± {results['student_comparison']['with_pretraining_std']:.4f}")
    print(f"Student Test AUC (Without Pretraining) : {results['student_comparison']['without_pretraining_mean']:.4f} ± {results['student_comparison']['without_pretraining_std']:.4f}")
    print(f"Student Δ: {s_delta:+.4f} (p={s_pval:.4f})")
    print("-" * 75)
    print(f"Finding: {finding}")
    print("=" * 75 + "\n")

    _log_metadata_note()
    return results


def _log_metadata_note() -> None:
    """Print the note about missing clinic/device metadata."""
    note = (
        "[Domain Shift Note]\n"
        "clinical_data_and_labels.csv does not contain clinic or acquisition-device "
        "metadata columns. Sub-population holdout by device/clinic is therefore not "
        "feasible. The Stage 1 pretraining ablation above is the sole domain-shift "
        "proxy available in this dataset. This limitation is documented in the "
        "Limitations section of the report.\n"
    )
    print(note)
    logger.info(note.strip())
