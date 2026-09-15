"""Image transform pipelines for corneal topography maps.

All normalization now uses dataset-specific statistics rather than ImageNet
values.  The ``mean`` and ``std`` parameters default to values in ``Config``
and should be set from the output of ``scripts/compute_dataset_stats.py``.

These are pseudo-colored scalar topography fields — not natural RGB images —
so ImageNet normalization (mean=[0.485,0.456,0.406]) is inappropriate.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default stats — will be replaced by compute_dataset_stats.py output.
# These are intentionally left as placeholders so users notice they need
# to run the stats script.
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD = [0.229, 0.224, 0.225]


def get_pretrain_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Augmented transforms for Stage 1 autoencoder pretraining.

    Augmentations are conservative for topography maps:
      - Horizontal flip: valid (corneal maps are symmetric)
      - Small rotation: valid (simulates head tilt)
      - NO color jitter, random crop, or grayscale — these destroy the
        scalar field semantics of pseudo-colored topography maps.

    Args:
        img_size: Target (H, W).
        mean: Per-channel mean.  Use dataset stats, not ImageNet.
        std: Per-channel std.  Use dataset stats, not ImageNet.
    """
    mean = mean or _DEFAULT_MEAN
    std = std or _DEFAULT_STD
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_finetune_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Augmented transforms for Stage 2 Teacher fine-tuning.

    Adds small affine translation/scale to improve spatial robustness.

    Args:
        img_size: Target (H, W).
        mean: Per-channel mean.
        std: Per-channel std.
    """
    mean = mean or _DEFAULT_MEAN
    std = std or _DEFAULT_STD
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """Deterministic transforms for validation and test evaluation.

    No augmentation — resize and normalize only.

    Args:
        img_size: Target (H, W).
        mean: Per-channel mean.
        std: Per-channel std.
    """
    mean = mean or _DEFAULT_MEAN
    std = std or _DEFAULT_STD
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_and_stack_maps(
    image_dir: str,
    patient_code: str,
    eye: str,
    map_types: Optional[List[str]] = None,
) -> Image.Image:
    """Load 3 grayscale topography maps and stack into a 3-channel image.

    Each map type is loaded as grayscale ('L') and stacked into a synthetic
    RGB image where each channel carries a distinct physical measurement
    (Axial power, Anterior elevation, Posterior elevation).

    Args:
        image_dir: Directory containing the map files.
        patient_code: Patient identifier used in filenames.
        eye: Eye string ('OD' or 'OS').
        map_types: Map types to load.  Defaults to ['Axial','Anterior','Posterior'].

    Returns:
        PIL Image in RGB mode with shape (H, W, 3).

    Raises:
        FileNotFoundError: If any expected map file is missing.
        ValueError: If the number of map types is not 3.
    """
    if map_types is None:
        map_types = ["Axial", "Anterior", "Posterior"]

    channels = []
    for map_type in map_types:
        filename = f"{patient_code}_{eye}_{map_type}.png"
        cand1 = os.path.join(image_dir, filename)
        cand2 = os.path.join(image_dir, patient_code, eye, filename)
        if os.path.exists(cand1):
            filepath = cand1
        elif os.path.exists(cand2):
            filepath = cand2
        else:
            logger.error(
                "Map not found in %s for %s %s %s", image_dir, patient_code, eye, map_type
            )
            raise FileNotFoundError(f"Missing map: {cand1} or {cand2}")

        img = Image.open(filepath)
        if img.mode != "L":
            img = img.convert("L")
        channels.append(img)

    if len(channels) != 3:
        raise ValueError(f"Expected 3 map types, got {len(channels)}")

    # Normalise all channels to the same size
    widths, heights = zip(*(c.size for c in channels))
    target_w, target_h = max(widths), max(heights)
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        logger.debug(
            "Map size mismatch for %s %s — resizing all to (%d, %d)",
            patient_code, eye, target_w, target_h,
        )
        channels = [c.resize((target_w, target_h), Image.LANCZOS) for c in channels]

    return Image.merge("RGB", tuple(channels))
