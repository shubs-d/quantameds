#!/usr/bin/env python3
"""Cache 8D latent representations from canonical frozen Teacher for all 1,454 eyes.

Uses teacher_finetuned_with_pretrain_s42_fold1.pt and target dataset-specific
normalization (mean=[0.3975, 0.4393, 0.2199], std=[0.2840, 0.3035, 0.2005]).
Saves tensors aligned 1-to-1 with clinical_data_and_labels.csv row indices to
data/cached_teacher_latents.pt.
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.image_transforms import get_eval_transforms, load_and_stack_maps
from quantum_kc.models.encoder import CornealEncoder


class FullCornOrbImagesDataset(Dataset):
    """Dataset of all 1,454 eye images (train + test) in exact CSV row order."""

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


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running latent caching on device: {device}")

    ckpt_path = config.CHECKPOINT_DIR / "teacher_finetuned_with_pretrain_s42_fold1.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found at {ckpt_path}")

    ckpt_hash = compute_sha256(str(ckpt_path))
    print(f"Loaded Teacher checkpoint: {ckpt_path.name}")
    print(f"Teacher SHA-256: {ckpt_hash}")

    teacher = CornealEncoder(in_channels=3, latent_dim=8).to(device)
    state = torch.load(ckpt_path, map_location=device)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    transform = get_eval_transforms(
        config.IMG_SIZE,
        mean=config.DATASET_MEAN,
        std=config.DATASET_STD,
    )
    dataset = FullCornOrbImagesDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_latents = torch.zeros((len(dataset), 8), dtype=torch.float32)
    cached_latents_by_key = {}
    total_processed = 0

    print(f"Extracting 8D latents for {len(dataset)} eyes...")
    with torch.no_grad():
        for idxs, p_codes, eyes, img_tensors, labels in loader:
            img_tensors = img_tensors.to(device)
            latents = teacher(img_tensors).cpu()  # (B, 8)
            for idx, p, e, z in zip(idxs, p_codes, eyes, latents):
                all_latents[idx] = z
                cached_latents_by_key[(p, e)] = z
            total_processed += len(img_tensors)
            if total_processed % 256 == 0 or total_processed == len(dataset):
                print(f"  Processed {total_processed}/{len(dataset)} eyes...")

    # Validate cached tensor
    assert len(all_latents) == len(dataset), f"Mismatch: {len(all_latents)} vs {len(dataset)}"
    assert torch.isfinite(all_latents).all(), "NaN/Inf detected in cached latents!"
    print(f"Cached tensor shape: {all_latents.shape}, min: {all_latents.min():.4f}, max: {all_latents.max():.4f}")

    # Separate train and test splits
    df = dataset.df
    train_mask = (df["split_80_20"] == "train").values
    test_mask = (df["split_80_20"] == "test").values

    train_latents = all_latents[train_mask]
    test_latents = all_latents[test_mask]

    print(f"Train latents shape: {train_latents.shape} (expected 1163, 8)")
    print(f"Test latents shape:  {test_latents.shape} (expected 291, 8)")

    out_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    torch.save(
        {
            "teacher_checkpoint": str(ckpt_path),
            "teacher_sha256": ckpt_hash,
            "latents_all": all_latents,
            "latents_train": train_latents,
            "latents_test": test_latents,
            "latents_by_patient_eye": cached_latents_by_key,
            "latent_dim": 8,
            "num_samples": len(all_latents),
        },
        out_path,
    )
    print(f"\nSuccessfully cached {len(all_latents)} teacher latents to:")
    print(f"  {out_path}")
    print(f"  File size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
