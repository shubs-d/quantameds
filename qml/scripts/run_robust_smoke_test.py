#!/usr/bin/env python3
"""Fast, local, single-epoch smoke test for RobustQuantaStudent on CPU.

Verifies the 10 checks required prior to full A100 training:
  1. Data loading: Tabular preprocessing runs, no NaN/Inf, 6 features formatted.
  2. Modality dropout: Genuinely mixed batches (both mask=1 and mask=0 present).
  3. Forward pass: Runs without error/NaN for both mask=1 and mask=0 inputs.
  4. Quantum layer bounds: Latent inputs to VQC are strictly in [0, pi].
  5. Loss calculation: MSE, BCE, and total loss are finite.
  6. Backward pass: VQC weights receive non-None, non-zero gradients.
  7. Optimizer step: Weights update, 1 epoch completes, loss moves.
  8. Checkpoint save/load: Fresh instance produces identical predictions (atol=1e-5).
  9. Evaluation loop: Evaluates Tier 1 and Tier 2 modes, computes full metrics dict.
  10. MLflow logging: Logs to 'smoke-test-robust-student' experiment and verifies query.

Usage:
  .venv/bin/python scripts/run_robust_smoke_test.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.image_transforms import get_eval_transforms
from quantum_kc.data.labeled_dataset import RobustCornOrbDataset
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.robust_student import RobustQuantaStudent
from quantum_kc.training.utils import compute_metrics, set_seed


class SmokeTestRunner:
    """Executes the 10-check smoke test on CPU."""

    def __init__(self, seed: int = 42, device: str = "cpu") -> None:
        self.seed = seed
        self.device = torch.device(device)
        set_seed(seed)

        self.results: Dict[str, Tuple[bool, str]] = {}
        self.teacher_ckpt_path = config.CHECKPOINT_DIR / "teacher_finetuned_with_pretrain_s42_fold1.pt"
        if not self.teacher_ckpt_path.exists():
            self.teacher_ckpt_path = config.CHECKPOINT_DIR / "teacher_finetuned_best.pt"

    def record_check(self, num: int, name: str, passed: bool, details: str) -> None:
        key = f"Check {num}: {name}"
        self.results[key] = (passed, details)
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {key}")
        print(f"       Details: {details}\n")

    def run_check_0_paired_data(self) -> Tuple[bool, str]:
        """Check whether CornOrb training split has paired topography images."""
        print("=" * 70)
        print("PAIRED DATA VERIFICATION (Decide Loss Mode)")
        print("=" * 70)
        df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
        df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
        train_df = df[df["split_80_20"] == "train"]

        modalities = ["Axial", "Anterior", "Posterior", "Pachymetry"]
        missing_count = 0
        total_eyes = len(train_df)

        for _, row in train_df.iterrows():
            p_code = str(row["patient_code"])
            eye = str(row["eye"])
            eye_dir = config.LABELED_ROOT / p_code / eye
            if not eye_dir.exists():
                missing_count += 1
                continue
            for m in modalities:
                if not (eye_dir / f"{p_code}_{eye}_{m}.png").exists():
                    missing_count += 1
                    break

        if missing_count == 0:
            decision = (
                f"YES: 100% paired topography images exist ({total_eyes}/{total_eyes} eyes). "
                "Selected loss mode: Full Distillation (MSE + BCE)."
            )
            print(f"Decision: {decision}\n")
            return True, decision
        else:
            decision = (
                f"NO: {missing_count}/{total_eyes} eyes missing topography files. "
                "Falling back to Task-only loss (BCE only)."
            )
            print(f"Decision: {decision}\n")
            return False, decision

    def run(self) -> bool:
        start_time = time.time()
        print("\n" + "=" * 70)
        print("STARTING ROBUST STUDENT SMOKE TEST (CPU ONLY)")
        print(f"Device: {self.device} | Seed: {self.seed}")
        print("=" * 70 + "\n")

        # 0. Paired data decision
        is_paired, pairing_decision = self.run_check_0_paired_data()
        alpha = 0.5 if is_paired else 0.0
        beta = 0.5 if is_paired else 1.0

        # Load teacher model
        teacher = CornealEncoder(in_channels=3, latent_dim=8).to(self.device)
        if self.teacher_ckpt_path.exists():
            state = torch.load(self.teacher_ckpt_path, map_location=self.device, weights_only=True)
            teacher.load_state_dict(state, strict=False)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            teacher_loaded = True
        else:
            teacher_loaded = False

        # ── Check 1: Data Loading & Preprocessing ──────────────────────
        try:
            raw_df = pd.read_csv(config.LABELED_CSV)
            pipe = RobustStudentTabularPipeline()
            train_df = raw_df[raw_df["split_80_20"] == "train"].reset_index(drop=True)
            pipe.fit(train_df)
            base_train = pipe.transform_base(train_df)

            has_nan = np.isnan(base_train).any()
            has_inf = np.isinf(base_train).any()
            shape_ok = (base_train.shape == (len(train_df), 5))

            # Transform with modality dropout to 6 features
            train_6d = pipe.apply_dropout(base_train, mode="train", p_dropout=0.5, rng=np.random.RandomState(self.seed))
            has_nan_6d = np.isnan(train_6d).any()
            has_inf_6d = np.isinf(train_6d).any()
            shape_6d_ok = (train_6d.shape == (len(train_df), 6))

            check1_pass = (not has_nan) and (not has_inf) and shape_ok and (not has_nan_6d) and (not has_inf_6d) and shape_6d_ok
            check1_details = (
                f"Fitted on {len(train_df)} train samples. Base shape: {base_train.shape}, "
                f"6D shape: {train_6d.shape}. NaN={has_nan_6d}, Inf={has_inf_6d}. "
                f"Pachy mean={pipe._pachy_mean:.1f}um, std={pipe._pachy_std:.1f}um."
            )
        except Exception as e:
            check1_pass = False
            check1_details = f"Exception: {e}"
        self.record_check(1, "Data loading & tabular preprocessing", check1_pass, check1_details)

        # Build full PyTorch Datasets for subsequent checks
        transform = get_eval_transforms(config.IMG_SIZE, mean=config.DATASET_MEAN, std=config.DATASET_STD)
        train_dataset = RobustCornOrbDataset(
            csv_path=str(config.LABELED_CSV),
            data_root=str(config.LABELED_ROOT),
            split="train",
            fold=1,
            transform=transform,
            pipeline=pipe,
            mode="train",
            p_dropout=0.5,
            load_images=True,
        )
        test_dataset = RobustCornOrbDataset(
            csv_path=str(config.LABELED_CSV),
            data_root=str(config.LABELED_ROOT),
            split="test",
            transform=None,
            pipeline=pipe,
            mode="tier2",
            load_images=False,
        )

        # DataLoader with batch size 32
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # ── Check 2: Modality Dropout Mixed Batches ────────────────────
        try:
            imgs, x_batch, y_batch = next(iter(train_loader))
            mask_col = x_batch[:, 5]
            n_ones = int((mask_col == 1.0).sum().item())
            n_zeros = int((mask_col == 0.0).sum().item())
            total = len(mask_col)
            mixed = (0 < n_ones < total)

            zero_pachy_ok = bool((x_batch[mask_col == 0.0, 4] == 0.0).all().item())
            nonzero_pachy_ok = bool((x_batch[mask_col == 1.0, 4] != 0.0).any().item())

            check2_pass = mixed and zero_pachy_ok and nonzero_pachy_ok
            check2_details = (
                f"Batch size {total}: {n_ones} present (mask=1, {n_ones/total*100:.1f}%), "
                f"{n_zeros} dropped (mask=0, {n_zeros/total*100:.1f}%). "
                f"Neutral value 0.0 verified for dropped samples: {zero_pachy_ok}."
            )
        except Exception as e:
            check2_pass = False
            check2_details = f"Exception: {e}"
        self.record_check(2, "Modality dropout creates mixed batches", check2_pass, check2_details)

        # Initialize student model
        student = RobustQuantaStudent(
            input_dim=6,
            hidden_dim=16,
            latent_dim=8,
            n_layers=2,
            dev_name="lightning.qubit",
            diff_method="adjoint",
        ).to(self.device)

        # ── Check 3: Forward Pass for mask=1 and mask=0 ────────────────
        try:
            x_tier1 = x_batch.clone()
            x_tier1[:, 4] = 0.0
            x_tier1[:, 5] = 0.0

            x_tier2 = x_batch.clone()
            x_tier2[:, 5] = 1.0

            z_t1, logits_t1 = student(x_tier1, return_latent=True)
            z_t2, logits_t2 = student(x_tier2, return_latent=True)
            z_mix, logits_mix = student(x_batch, return_latent=True)

            no_nan = (
                torch.isfinite(z_t1).all()
                and torch.isfinite(logits_t1).all()
                and torch.isfinite(z_t2).all()
                and torch.isfinite(logits_t2).all()
                and torch.isfinite(z_mix).all()
                and torch.isfinite(logits_mix).all()
            )
            shape_ok = (logits_mix.shape == (batch_size,)) and (z_mix.shape == (batch_size, 8))

            check3_pass = bool(no_nan and shape_ok)
            check3_details = (
                f"Tier 1 output shape: logits={logits_t1.shape}, z={z_t1.shape}. "
                f"Tier 2 output shape: logits={logits_t2.shape}, z={z_t2.shape}. "
                f"All finite, no NaNs/Infs."
            )
        except Exception as e:
            check3_pass = False
            check3_details = f"Exception: {e}"
        self.record_check(3, "Forward pass for mask=1 and mask=0", check3_pass, check3_details)

        # ── Check 4: Quantum Layer Bounds [0, pi] ──────────────────────
        try:
            x_extreme = torch.tensor([
                [100.0, 100.0, 1.0, 1.0, 100.0, 1.0],
                [-100.0, -100.0, -1.0, -1.0, -100.0, 0.0],
            ], device=self.device)
            z_extreme = student.encoder(x_extreme)

            all_z = torch.cat([z_t1, z_t2, z_mix, z_extreme], dim=0)
            z_min = float(all_z.min().item())
            z_max = float(all_z.max().item())

            in_bounds = (z_min >= 0.0) and (z_max <= math.pi + 1e-6)
            check4_pass = in_bounds
            check4_details = (
                f"Latent z range across normal & extreme inputs: [{z_min:.6f}, {z_max:.6f}]. "
                f"Strictly inside [0, pi={math.pi:.6f}]."
            )
        except Exception as e:
            check4_pass = False
            check4_details = f"Exception: {e}"
        self.record_check(4, "Quantum layer bounds strictly in [0, pi]", check4_pass, check4_details)

        # ── Check 5: Loss Calculation (Finite MSE + BCE) ───────────────
        try:
            imgs = imgs.to(self.device)
            with torch.no_grad():
                z_teacher = teacher(imgs)

            mse_loss_fn = nn.MSELoss()
            bce_loss_fn = nn.BCEWithLogitsLoss()

            loss_distill = mse_loss_fn(z_mix, z_teacher)
            loss_task = bce_loss_fn(logits_mix, y_batch.to(self.device))
            loss_total = alpha * loss_distill + beta * loss_task

            finite_distill = bool(torch.isfinite(loss_distill).item())
            finite_task = bool(torch.isfinite(loss_task).item())
            finite_total = bool(torch.isfinite(loss_total).item())

            check5_pass = finite_distill and finite_task and finite_total
            check5_details = (
                f"Losses on first batch (alpha={alpha}, beta={beta}): "
                f"L_distill(MSE)={loss_distill.item():.4f}, "
                f"L_task(BCE)={loss_task.item():.4f}, "
                f"L_total={loss_total.item():.4f}. All finite."
            )
        except Exception as e:
            check5_pass = False
            check5_details = f"Exception: {e}"
        self.record_check(5, "Loss calculation finite (MSE + BCE)", check5_pass, check5_details)

        # ── Check 6: Backward Pass & Quantum Gradients ──────────────────
        try:
            optimizer = Adam(student.parameters(), lr=1e-3)
            optimizer.zero_grad()
            loss_total.backward()

            vqc_grad = student.vqc_layer.weights.grad
            vqc_grad_not_none = vqc_grad is not None
            vqc_grad_nonzero = bool((vqc_grad != 0).any().item()) if vqc_grad_not_none else False
            vqc_grad_finite = bool(torch.isfinite(vqc_grad).all().item()) if vqc_grad_not_none else False

            enc_grad = student.encoder[0].weight.grad
            cls_grad = student.classifier.weight.grad
            enc_ok = enc_grad is not None and bool((enc_grad != 0).any().item())
            cls_ok = cls_grad is not None and bool((cls_grad != 0).any().item())

            check6_pass = vqc_grad_not_none and vqc_grad_nonzero and vqc_grad_finite and enc_ok and cls_ok
            grad_norm = float(vqc_grad.norm().item()) if vqc_grad_not_none else 0.0
            check6_details = (
                f"VQC weights grad shape: {tuple(vqc_grad.shape)}, norm={grad_norm:.6f}, "
                f"non-zero={vqc_grad_nonzero}, finite={vqc_grad_finite}. "
                f"Encoder & Classifier heads also received valid gradients."
            )
        except Exception as e:
            check6_pass = False
            check6_details = f"Exception: {e}"
        self.record_check(6, "Backward pass & VQC parameter gradients", check6_pass, check6_details)

        # ── Check 7: Optimizer Step & 1-Epoch Completion ───────────────
        try:
            w_before = student.vqc_layer.weights.clone().detach()
            optimizer.step()
            w_after = student.vqc_layer.weights.clone().detach()

            weight_changed = not torch.equal(w_before, w_after)
            weight_diff = float((w_before - w_after).abs().sum().item())

            # Run 1 epoch on a representative subset of 4 batches (64 samples)
            subset_indices = list(range(64))
            train_subset = Subset(train_dataset, subset_indices)
            subset_loader = DataLoader(train_subset, batch_size=16, shuffle=False)

            epoch_losses = []
            student.train()
            for b_idx, (b_imgs, b_x, b_y) in enumerate(subset_loader):
                optimizer.zero_grad()
                with torch.no_grad():
                    b_z_teacher = teacher(b_imgs.to(self.device))
                b_z, b_logits = student(b_x.to(self.device), return_latent=True)
                l_dist = mse_loss_fn(b_z, b_z_teacher)
                l_tsk = bce_loss_fn(b_logits, b_y.to(self.device))
                l_tot = alpha * l_dist + beta * l_tsk
                l_tot.backward()
                optimizer.step()
                epoch_losses.append(l_tot.item())

            loss_first = epoch_losses[0]
            loss_last = epoch_losses[-1]
            loss_moved = abs(loss_first - loss_last) > 1e-5

            check7_pass = weight_changed and loss_moved
            check7_details = (
                f"Optimizer step delta on VQC weights: {weight_diff:.6f} > 0. "
                f"1 epoch (4 batches, 64 samples) completed: "
                f"batch 0 loss={loss_first:.4f} -> batch 3 loss={loss_last:.4f} "
                f"(delta={loss_last - loss_first:+.4f})."
            )
        except Exception as e:
            check7_pass = False
            check7_details = f"Exception: {e}"
        self.record_check(7, "Optimizer step updates weights & 1 epoch completes", check7_pass, check7_details)

        # ── Check 8: Checkpoint Save & Fresh Reload ────────────────────
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                tmp_ckpt_path = tmp_file.name

            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "input_dim": 6,
                    "latent_dim": 8,
                    "n_qubits": 8,
                    "n_layers": 2,
                },
                tmp_ckpt_path,
            )

            # Create fresh model instance
            fresh_student = RobustQuantaStudent(
                input_dim=6,
                hidden_dim=16,
                latent_dim=8,
                n_layers=2,
                dev_name="lightning.qubit",
                diff_method="adjoint",
            ).to(self.device)

            ckpt_loaded = torch.load(tmp_ckpt_path, map_location=self.device)
            fresh_student.load_state_dict(ckpt_loaded["model_state_dict"])
            student.eval()
            fresh_student.eval()

            # Compare outputs on identical test input
            test_x = x_batch[:8].to(self.device)
            with torch.no_grad():
                out_orig = student(test_x)
                out_fresh = fresh_student(test_x)

            allclose = torch.allclose(out_orig, out_fresh, atol=1e-5)
            max_diff = float((out_orig - out_fresh).abs().max().item())

            # Clean up temp file
            if os.path.exists(tmp_ckpt_path):
                os.remove(tmp_ckpt_path)

            check8_pass = bool(allclose)
            check8_details = (
                f"Saved to temp checkpoint, reloaded into fresh instance. "
                f"Output allclose(atol=1e-5): {allclose}, max absolute diff: {max_diff:.2e}."
            )
        except Exception as e:
            check8_pass = False
            check8_details = f"Exception: {e}"
        self.record_check(8, "Checkpoint save and fresh load equality", check8_pass, check8_details)

        # ── Check 9: Post-Epoch Evaluation Loop (Tier 1 vs Tier 2) ─────
        try:
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            student.eval()

            def eval_tier(mode: str) -> Dict[str, Any]:
                test_dataset.set_mode(mode)
                all_probs = []
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for _, t_x, t_y in test_loader:
                        logits = student(t_x.to(self.device))
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds = (probs >= 0.5).astype(int)
                        all_probs.extend(probs)
                        all_preds.extend(preds)
                        all_targets.extend(t_y.numpy().astype(int))

                y_true = np.array(all_targets)
                y_pred = np.array(all_preds)
                y_prob = np.array(all_probs)
                return compute_metrics(y_true, y_pred, y_prob=y_prob)

            tier1_metrics = eval_tier("tier1")
            tier2_metrics = eval_tier("tier2")

            required_metric_keys = [
                "accuracy", "precision", "recall", "sensitivity",
                "specificity", "f1", "auc_roc", "confusion_matrix"
            ]
            t1_has_keys = all(k in tier1_metrics for k in required_metric_keys)
            t2_has_keys = all(k in tier2_metrics for k in required_metric_keys)

            check9_pass = t1_has_keys and t2_has_keys
            check9_details = (
                f"Evaluated on {len(test_dataset)} test eyes. "
                f"Tier 1 (ARK only): AUC={tier1_metrics['auc_roc']:.4f}, Acc={tier1_metrics['accuracy']:.4f}, "
                f"Sens={tier1_metrics['sensitivity']:.4f}, Spec={tier1_metrics['specificity']:.4f}, "
                f"CM={tier1_metrics['confusion_matrix']}. "
                f"Tier 2 (ARK+Pachy): AUC={tier2_metrics['auc_roc']:.4f}, Acc={tier2_metrics['accuracy']:.4f}, "
                f"Sens={tier2_metrics['sensitivity']:.4f}, Spec={tier2_metrics['specificity']:.4f}, "
                f"CM={tier2_metrics['confusion_matrix']}."
            )
        except Exception as e:
            check9_pass = False
            check9_details = f"Exception: {e}"
            tier1_metrics, tier2_metrics = {}, {}
        self.record_check(9, "Evaluation loop for Tier 1 and Tier 2 modes", check9_pass, check9_details)

        # ── Check 10: MLflow Logging & Verification ───────────────────
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            exp_name = "smoke-test-robust-student"
            mlflow.set_experiment(exp_name)

            run_name = f"smoke_test_cpu_{int(time.time())}"
            with mlflow.start_run(run_name=run_name) as run:
                active_run_id = run.info.run_id
                mlflow.log_params({
                    "model_class": "RobustQuantaStudent",
                    "input_dim": 6,
                    "hidden_dim": 16,
                    "latent_dim": 8,
                    "n_qubits": 8,
                    "n_layers": 2,
                    "loss_mode": "MSE+BCE" if is_paired else "BCE-only",
                    "alpha": alpha,
                    "beta": beta,
                    "seed": self.seed,
                    "device": str(self.device),
                })
                mlflow.log_metrics({
                    "batch_0_loss": loss_first,
                    "batch_3_loss": loss_last,
                    "tier1_accuracy": tier1_metrics.get("accuracy", 0.0),
                    "tier1_auc_roc": tier1_metrics.get("auc_roc", 0.0),
                    "tier1_sensitivity": tier1_metrics.get("sensitivity", 0.0),
                    "tier1_specificity": tier1_metrics.get("specificity", 0.0),
                    "tier2_accuracy": tier2_metrics.get("accuracy", 0.0),
                    "tier2_auc_roc": tier2_metrics.get("auc_roc", 0.0),
                    "tier2_sensitivity": tier2_metrics.get("sensitivity", 0.0),
                    "tier2_specificity": tier2_metrics.get("specificity", 0.0),
                })

            # Verify run exists by querying MLflow client
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name(exp_name)
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f'attributes.run_id = "{active_run_id}"'
            )
            found = len(runs) == 1
            if found:
                queried_run = runs[0]
                has_alpha = "alpha" in queried_run.data.params
                has_t1_auc = "tier1_auc_roc" in queried_run.data.metrics
                check10_pass = has_alpha and has_t1_auc
                check10_details = (
                    f"Successfully logged and queried run_id='{active_run_id}' in experiment '{exp_name}'. "
                    f"Logged {len(queried_run.data.params)} params, {len(queried_run.data.metrics)} metrics."
                )
            else:
                check10_pass = False
                check10_details = f"Run ID '{active_run_id}' not found in search_runs."
        except Exception as e:
            check10_pass = False
            check10_details = f"Exception: {e}"
        self.record_check(10, "MLflow logging and queryable run verification", check10_pass, check10_details)

        # ── Final Summary ─────────────────────────────────────────────
        elapsed = time.time() - start_time
        all_passed = all(passed for passed, _ in self.results.values())

        print("=" * 70)
        print("SMOKE TEST SUMMARY")
        print("=" * 70)
        for name, (passed, details) in self.results.items():
            st = "PASS" if passed else "FAIL"
            print(f"[{st:4s}] {name}")

        print("-" * 70)
        print(f"Total time elapsed: {elapsed:.2f} seconds")
        print(f"Paired data mode:   {'Full Distillation (MSE + BCE)' if is_paired else 'Task only (BCE)'}")
        verdict = "READY FOR FULL A100 TRAINING" if all_passed else "NOT READY (Blockers detected)"
        print(f"Final verdict:      {verdict}")
        print("=" * 70 + "\n")

        return all_passed


if __name__ == "__main__":
    runner = SmokeTestRunner()
    success = runner.run()
    sys.exit(0 if success else 1)
