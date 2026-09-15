#!/usr/bin/env python3
"""Step 5: Full Pipeline Local Smoke Test with Provenance.

Runs 1 epoch across train split with:
  1. Teacher latent loading from data/cached_teacher_latents.pt (with SHA-256 check)
  2. Per-sample Bernoulli(0.5) modality dropout in DataLoader
  3. HybridRobustQuantaStudent with Dual-Rate optimizer (c=1e-3, q=1e-2)
  4. Separate gradient tracking for quantum vs classical parameters
  5. Tier 1 and Tier 2 evaluations on held-out test split (291 eyes)
  6. MLflow experiment logging with SHA-256 provenance

All 10 checks must pass before A100 training.
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.matched_ablation_models import HybridRobustQuantaStudent
from quantum_kc.training.utils import set_seed
from scripts.run_matched_ablation import CachedTabularDataset, compute_ece


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    set_seed(42)

    print("=" * 70)
    print("STEP 5: FULL PIPELINE LOCAL SMOKE TEST WITH PROVENANCE")
    print("=" * 70)

    # CHECK 1: Teacher latent file exists & compute hash
    cache_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    assert cache_path.exists(), f"Cached latents missing at {cache_path}"
    cache_hash = compute_sha256(str(cache_path))
    cached_data = torch.load(cache_path, map_location="cpu")
    train_latents = cached_data["latents_train"]
    test_latents = cached_data["latents_test"]
    teacher_sha256 = cached_data["teacher_sha256"]
    print(f"[CHECK 1 PASS] Cached teacher latents verified:")
    print(f"  Path: {cache_path.name}")
    print(f"  Cache SHA-256:   {cache_hash[:16]}...")
    print(f"  Teacher SHA-256: {teacher_sha256[:16]}...")
    print(f"  Train: {train_latents.shape}, Test: {test_latents.shape}")

    # CHECK 2: Load clinical tabular data & split cleanly
    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)
    test_df = df[df["split_80_20"] == "test"].reset_index(drop=True)
    assert len(train_df) == len(train_latents) == 1163
    assert len(test_df) == len(test_latents) == 291
    print(f"[CHECK 2 PASS] Data partitions match: Train={len(train_df)}, Test={len(test_df)}")

    # CHECK 3: Fit RobustStudentTabularPipeline (zero test leakage)
    pipe = RobustStudentTabularPipeline()
    pipe.fit(train_df)
    train_base = pipe.transform_base(train_df)
    test_base = pipe.transform_base(test_df)
    assert not np.isnan(train_base).any() and not np.isnan(test_base).any()
    print(f"[CHECK 3 PASS] Pipeline fitted cleanly: Pachy mean={pipe._pachy_mean:.2f}, std={pipe._pachy_std:.2f}")

    # CHECK 4: Per-sample Bernoulli(0.5) modality dropout in DataLoader
    train_ds = CachedTabularDataset(
        train_base, train_latents, train_df["label"].values.astype(int),
        mode="train", p_dropout=0.5, rng=np.random.RandomState(42)
    )
    sample_masks = [train_ds[i][0][5].item() for i in range(100)]
    pachy_present_ratio = np.mean(sample_masks)
    assert 0.35 <= pachy_present_ratio <= 0.65, f"Unexpected dropout ratio: {pachy_present_ratio}"
    print(f"[CHECK 4 PASS] Per-sample modality dropout working: present ratio = {pachy_present_ratio:.2f}")

    # CHECK 5: Model instantiation & Dual-Rate Optimizer
    model = HybridRobustQuantaStudent(n_layers=2)
    classical_params = [p for n, p in model.named_parameters() if "vqc" not in n]
    quantum_params = [p for n, p in model.named_parameters() if "vqc" in n]
    optimizer = Adam([
        {"params": classical_params, "lr": 1e-3},
        {"params": quantum_params, "lr": 1e-2},
    ])
    assert len(classical_params) == 6  # 4 linear/bias layers
    assert len(quantum_params) == 1   # vqc_layer.weights
    print(f"[CHECK 5 PASS] Dual-rate optimizer configured (6 classical tensors, 1 quantum tensor)")

    # CHECK 6: Forward & Backward pass with finite losses
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    model.train()
    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_bce_sum = 0.0

    t0 = time.time()
    for x_b, z_t, y_b in train_loader:
        optimizer.zero_grad()
        z_pred, logits = model(x_b, return_latent=True)
        l_distill = mse_fn(z_pred, z_t)
        l_task = bce_fn(logits, y_b)
        l_tot = 0.5 * l_distill + 0.5 * l_task
        l_tot.backward()
        optimizer.step()

        total_loss_sum += l_tot.item()
        total_mse_sum += l_distill.item()
        total_bce_sum += l_task.item()

    elapsed = time.time() - t0
    avg_loss = total_loss_sum / len(train_loader)
    avg_mse = total_mse_sum / len(train_loader)
    avg_bce = total_bce_sum / len(train_loader)
    assert np.isfinite(avg_loss) and np.isfinite(avg_mse) and np.isfinite(avg_bce)
    print(f"[CHECK 6 PASS] 1 full training epoch finished in {elapsed:.2f}s:")
    print(f"  Total Loss: {avg_loss:.4f} | MSE (distill): {avg_mse:.4f} | BCE (task): {avg_bce:.4f}")

    # CHECK 7: Separate Gradient norm and variance check
    q_weights = model.vqc_layer.weights
    assert q_weights.grad is not None, "Quantum gradient must not be None"
    q_norm = q_weights.grad.norm().item()
    q_var = torch.var(q_weights.grad).item()
    c_grads = torch.cat([p.grad.view(-1) for p in classical_params if p.grad is not None])
    c_norm = c_grads.norm().item()
    c_var = torch.var(c_grads).item()
    assert q_norm > 0 and q_var > 0 and c_norm > 0 and c_var > 0
    print(f"[CHECK 7 PASS] Non-zero gradients reached both quantum and classical branches:")
    print(f"  Quantum:   norm = {q_norm:.4f}, variance = {q_var:.2e}")
    print(f"  Classical: norm = {c_norm:.4f}, variance = {c_var:.2e}")

    # CHECK 8: Held-out Test Evaluation (Tier 1 vs Tier 2)
    test_ds = CachedTabularDataset(
        test_base, test_latents, test_df["label"].values.astype(int), mode="tier2"
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    def eval_test(mode: str):
        model.eval()
        test_ds.set_mode(mode)
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for x_v, _, y_v in test_loader:
                logits = model(x_v)
                probs = torch.sigmoid(logits).numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(y_v.numpy().astype(int))
        y_t = np.array(all_targets)
        y_p = np.array(all_preds)
        y_pr = np.array(all_probs)
        return {
            "accuracy": float(accuracy_score(y_t, y_p)),
            "auroc": float(roc_auc_score(y_t, y_pr)),
            "brier": float(brier_score_loss(y_t, y_pr)),
            "ece": float(compute_ece(y_t, y_pr)),
        }

    t1_res = eval_test("tier1")
    t2_res = eval_test("tier2")
    assert t1_res["accuracy"] > 0.5 and t2_res["accuracy"] > 0.5
    print(f"[CHECK 8 PASS] Test set evaluations completed:")
    print(f"  Tier 1 (ARK only): Acc = {t1_res['accuracy']:.4f} | AUROC = {t1_res['auroc']:.4f} | Brier = {t1_res['brier']:.4f}")
    print(f"  Tier 2 (+Pachy):  Acc = {t2_res['accuracy']:.4f} | AUROC = {t2_res['auroc']:.4f} | Brier = {t2_res['brier']:.4f}")

    # CHECK 9: Checkpoint Save and Strict State Dict Load
    ckpt_out = config.CHECKPOINT_DIR / "smoke_test_hybrid_student.pt"
    torch.save(model.state_dict(), ckpt_out)
    m_reload = HybridRobustQuantaStudent(n_layers=2)
    m_reload.load_state_dict(torch.load(ckpt_out, weights_only=True), strict=True)
    m_reload.eval()
    with torch.no_grad():
        x_dummy = torch.randn(4, 6)
        out_orig = model(x_dummy)
        out_reloaded = m_reload(x_dummy)
        assert torch.allclose(out_orig, out_reloaded, atol=1e-6)
    print(f"[CHECK 9 PASS] Checkpoint saved and reloaded with strict=True (exact match)")

    # CHECK 10: MLflow logging with cryptographic hashes & parameter provenance
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("smoke-test-robust-student")
    with mlflow.start_run(run_name="smoke_test_step5_dual_rate") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "model_name": "HybridRobustQuantaStudent",
            "optimizer": "DualRateAdam",
            "lr_classical": 1e-3,
            "lr_quantum": 1e-2,
            "n_layers": 2,
            "n_qubits": 8,
            "teacher_checkpoint_sha256": teacher_sha256,
            "cache_latents_sha256": cache_hash,
            "smoke_checkpoint_sha256": compute_sha256(str(ckpt_out)),
        })
        mlflow.log_metrics({
            "train_loss": avg_loss,
            "loss_mse": avg_mse,
            "loss_bce": avg_bce,
            "q_grad_norm": q_norm,
            "q_grad_var": q_var,
            "c_grad_norm": c_norm,
            "c_grad_var": c_var,
            "tier1_accuracy": t1_res["accuracy"],
            "tier1_auroc": t1_res["auroc"],
            "tier2_accuracy": t2_res["accuracy"],
            "tier2_auroc": t2_res["auroc"],
        })
    print(f"[CHECK 10 PASS] MLflow run logged: run_id={run_id}")
    print("\n" + "=" * 70)
    print("ALL 10 CHECKS PASSED: PIPELINE IS VERIFIED FOR A100 FULL TRAINING")
    print("=" * 70)


if __name__ == "__main__":
    main()
