#!/usr/bin/env python3
"""Phase 2: Cache Teacher embeddings over all 1,454 eyes.

One batched GPU forward pass of the canonical Teacher checkpoint.
Saves:
  - data/teacher_embeddings.npz   (keyed by "patient_code+eye")
  - data/cached_teacher_latents.pt (backward-compatible with existing scripts)

Validates every eye has a finite, non-NaN embedding.
Reports shape, range, and SHA-256 of output files.
Logs SHA-256 of input Teacher checkpoint and output files to MLflow.

After this script completes, GPU is no longer needed for Student work.

Usage:
    python scripts/a100_phase2_cache_embeddings.py [--teacher-ckpt PATH]
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
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config, get_device
from quantum_kc.data.image_transforms import get_eval_transforms, load_and_stack_maps
from quantum_kc.models.encoder import CornealEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


class FullCornOrbImagesDataset(Dataset):
    """All 1,454 labeled eye images in exact CSV row order."""

    def __init__(self, csv_path: str, data_root: str, transform=None):
        df = pd.read_csv(csv_path, dtype={"patient_code": str})
        df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_code = str(row["patient_code"])
        eye = str(row["eye"])
        img_dir = os.path.join(self.data_root, patient_code, eye)
        img = load_and_stack_maps(img_dir, patient_code, eye)
        if self.transform:
            img_tensor = self.transform(img)
        else:
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)
        return idx, patient_code, eye, img_tensor, int(row["label"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Cache Teacher Embeddings")
    parser.add_argument(
        "--teacher-ckpt",
        type=str,
        default=str(config.CHECKPOINT_DIR / "teacher_finetuned_with_pretrain_s42_fold1.pt"),
        help="Path to canonical frozen Teacher checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    device = get_device()
    print("\n" + "="*70)
    print("PHASE 2 — Cache Teacher Embeddings (all 1,454 eyes)")
    print("="*70)
    print(f"  Device:       {device}")
    print(f"  Teacher ckpt: {args.teacher_ckpt}")

    ckpt_path = Path(args.teacher_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    teacher_hash = sha256(str(ckpt_path))
    print(f"  Teacher SHA-256: {teacher_hash}")

    # ── Load Teacher (frozen) ──────────────────────────────────────────
    teacher = CornealEncoder(in_channels=3, latent_dim=config.LATENT_DIM).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Dataset ───────────────────────────────────────────────────────
    transform = get_eval_transforms(
        config.IMG_SIZE, mean=config.DATASET_MEAN, std=config.DATASET_STD
    )
    dataset = FullCornOrbImagesDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Dataset size: {len(dataset)} eyes")

    # ── Forward pass ──────────────────────────────────────────────────
    all_latents = torch.zeros((len(dataset), config.LATENT_DIM), dtype=torch.float32)
    all_keys: list[str] = [""] * len(dataset)
    all_splits: list[str] = [""] * len(dataset)
    total = 0

    df = dataset.df
    split_col = df["split_80_20"].values
    patient_codes_col = df["patient_code"].values
    eyes_col = df["eye"].values

    print(f"\nExtracting {config.LATENT_DIM}D latents...")
    with torch.no_grad():
        for idxs, p_codes, eyes, img_tensors, _labels in loader:
            img_tensors = img_tensors.to(device)
            latents = teacher(img_tensors)  # CornealEncoder forward → (B, LATENT_DIM)
            latents_cpu = latents.cpu()
            for i, (idx_t, p, e) in enumerate(zip(idxs, p_codes, eyes)):
                idx = int(idx_t)
                all_latents[idx] = latents_cpu[i]
                all_keys[idx] = f"{p}+{e}"
                all_splits[idx] = str(split_col[idx])
            total += len(img_tensors)
            if total % 200 == 0 or total == len(dataset):
                print(f"  {total}/{len(dataset)} processed")

    # ── Validate ──────────────────────────────────────────────────────
    assert len(all_latents) == len(dataset), "Length mismatch!"
    n_nan = (~torch.isfinite(all_latents)).sum().item()
    if n_nan > 0:
        raise ValueError(f"NaN/Inf detected in {n_nan} latent entries!")

    latent_min = all_latents.min().item()
    latent_max = all_latents.max().item()
    latent_mean = all_latents.mean().item()
    print(f"\nLatent stats: shape={tuple(all_latents.shape)}, "
          f"min={latent_min:.4f}, max={latent_max:.4f}, mean={latent_mean:.4f}")

    # ── Split into train/test ─────────────────────────────────────────
    train_mask = np.array([s == "train" for s in all_splits])
    test_mask  = np.array([s == "test"  for s in all_splits])
    train_latents = all_latents[train_mask]
    test_latents  = all_latents[test_mask]
    print(f"  Train: {train_latents.shape} (expected [1163, {config.LATENT_DIM}])")
    print(f"  Test:  {test_latents.shape}  (expected [291, {config.LATENT_DIM}])")

    assert train_latents.shape[0] == 1163, f"Expected 1163 train eyes, got {train_latents.shape[0]}"
    assert test_latents.shape[0] == 291,   f"Expected 291 test eyes,  got {test_latents.shape[0]}"

    # ── Save NPZ (Phase 2 canonical format, keyed by patient_code+eye) ─
    data_dir = config.BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir / "teacher_embeddings.npz"
    latent_np = all_latents.numpy()  # (1454, 8)
    np.savez(
        str(npz_path),
        latents=latent_np,
        keys=np.array(all_keys),
        splits=np.array(all_splits),
        teacher_ckpt=str(ckpt_path),
        teacher_sha256=teacher_hash,
    )
    npz_hash = sha256(str(npz_path))
    npz_size = os.path.getsize(str(npz_path))
    print(f"\n[SAVED] teacher_embeddings.npz → {npz_path}")
    print(f"  SHA-256: {npz_hash} | Size: {npz_size:,} bytes")

    # ── Save PT (backward-compatible with existing scripts) ────────────
    latents_by_patient_eye = {
        (all_keys[i].split("+")[0], all_keys[i].split("+")[1]): all_latents[i]
        for i in range(len(all_latents))
    }
    pt_path = data_dir / "cached_teacher_latents.pt"
    torch.save(
        {
            "teacher_checkpoint": str(ckpt_path),
            "teacher_sha256": teacher_hash,
            "latents_all": all_latents,
            "latents_train": train_latents,
            "latents_test": test_latents,
            "latents_by_patient_eye": latents_by_patient_eye,
            "latent_dim": config.LATENT_DIM,
            "num_samples": len(all_latents),
        },
        pt_path,
    )
    pt_hash = sha256(str(pt_path))
    pt_size = os.path.getsize(str(pt_path))
    print(f"[SAVED] cached_teacher_latents.pt → {pt_path}")
    print(f"  SHA-256: {pt_hash} | Size: {pt_size:,} bytes")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print(f"  1,454 embeddings cached (1163 train + 291 test)")
    print(f"  NaN check: PASS (0 invalid entries)")
    print(f"  teacher_embeddings.npz SHA-256 : {npz_hash}")
    print(f"  cached_teacher_latents.pt SHA-256: {pt_hash}")
    print(f"\n>>> GPU is no longer needed for Student training/evaluation. <<<")
    print(">>> You may terminate the A100 instance after Phase 3 check (if not gated in). <<<")

    # ── Log to MLflow ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="a100_phase2_cache_embeddings"):
        mlflow.log_params({
            "teacher_checkpoint": str(ckpt_path),
            "teacher_sha256": teacher_hash,
            "npz_sha256": npz_hash,
            "pt_sha256": pt_hash,
            "n_train": train_latents.shape[0],
            "n_test": test_latents.shape[0],
            "latent_dim": config.LATENT_DIM,
        })
        mlflow.log_metrics({
            "latent_min": latent_min,
            "latent_max": latent_max,
            "latent_mean": latent_mean,
            "n_nan": float(n_nan),
        })
        mlflow.log_artifact(str(npz_path))
    print("[MLflow] Phase 2 logged.")

    # ── Save metadata ─────────────────────────────────────────────────
    meta = {
        "phase": "2_cache_embeddings",
        "timestamp": datetime.now().isoformat(),
        "teacher_ckpt": str(ckpt_path),
        "teacher_sha256": teacher_hash,
        "n_total": len(dataset),
        "n_train": int(train_latents.shape[0]),
        "n_test": int(test_latents.shape[0]),
        "latent_dim": config.LATENT_DIM,
        "latent_min": latent_min,
        "latent_max": latent_max,
        "n_nan": n_nan,
        "npz_path": str(npz_path),
        "npz_sha256": npz_hash,
        "pt_path": str(pt_path),
        "pt_sha256": pt_hash,
        "gpu_no_longer_needed": True,
    }
    meta_path = config.RESULTS_DIR / "a100_phase2_cache_embeddings.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVED] Metadata → {meta_path}")


if __name__ == "__main__":
    main()
