"""One-time script: compute per-channel mean and std for the Stage 1 unlabeled dataset.

These statistics MUST be used instead of ImageNet normalization because the
images are pseudo-colored scalar topography fields, not natural RGB photos.

Run once:
    python scripts/compute_dataset_stats.py

Then paste the printed values into Config.DATASET_MEAN and Config.DATASET_STD.

The script loads images without any normalization (ToTensor only), iterates
through the full dataset in one pass, and computes the Welford online mean/var.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import Config, config as default_config
from quantum_kc.data.unlabeled_dataset import OrbscanUnlabeledDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(cfg: Config = None) -> dict:
    """Compute per-channel mean and std over the unlabeled Stage 1 dataset.

    Uses Welford's online algorithm for numerical stability with large datasets.
    Images are loaded with ToTensor only (no normalization) and resized to
    cfg.IMG_SIZE.

    Returns:
        dict with keys 'mean' and 'std', each a list of 3 floats.
    """
    if cfg is None:
        cfg = default_config

    # Raw transform: resize + ToTensor only (NO normalize)
    raw_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),  # → [0, 1] float32, shape (C, H, W)
    ])

    dataset = OrbscanUnlabeledDataset(
        csv_path=str(cfg.UNLABELED_CSV),
        images_dir=str(cfg.UNLABELED_IMAGES_DIR),
        transform=raw_transform,
        qc_filter=True,
    )
    logger.info("Dataset size (QC-PASS): %d", len(dataset))

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)

    # Welford online mean/variance accumulators per channel
    n_pixels = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)

    for batch in loader:
        imgs = batch[0]  # (B, C, H, W)
        B, C, H, W = imgs.shape
        pixels = B * H * W
        n_pixels += pixels
        channel_sum += imgs.sum(dim=[0, 2, 3])
        channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    result = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_images": len(dataset),
        "n_pixels_per_channel": n_pixels,
    }

    print("\n" + "=" * 60)
    print("Dataset-specific normalization statistics (Stage 1 unlabeled)")
    print("=" * 60)
    print(f"Images processed : {len(dataset)}")
    print(f"DATASET_MEAN     : {[round(v, 4) for v in result['mean']]}")
    print(f"DATASET_STD      : {[round(v, 4) for v in result['std']]}")
    print("\nPaste these values into Config.DATASET_MEAN and Config.DATASET_STD")
    print("=" * 60 + "\n")

    return result


if __name__ == "__main__":
    compute_stats()
