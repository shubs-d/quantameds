"""Unit tests for CornOrb and Orbscan dataset loaders."""

import pytest
import torch

from quantum_kc.config import config
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.data.image_transforms import get_eval_transforms


def test_cornorb_dataset_initialization():
    """Verify that CornOrbDataset loads from the actual dataset CSV."""
    if not config.LABELED_CSV.exists():
        pytest.skip(f"Labeled CSV not found at {config.LABELED_CSV}")

    transform = get_eval_transforms((224, 224))
    train_ds = CornOrbDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        split="train",
        transform=transform,
    )

    assert len(train_ds) > 0
    # 80% train split has 1,163 records
    assert len(train_ds) == 1163

    # Check class weights
    weights = train_ds.get_class_weights()
    assert weights.shape == (2,)
    assert weights[0] > 0 and weights[1] > 0

    # Check sample item
    img, tab, label = train_ds[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(tab, torch.Tensor)
    assert label.item() in (0, 1)


def test_cornorb_dataset_test_split():
    """Verify test split contains exactly 291 samples."""
    if not config.LABELED_CSV.exists():
        pytest.skip(f"Labeled CSV not found at {config.LABELED_CSV}")

    test_ds = CornOrbDataset(
        csv_path=str(config.LABELED_CSV),
        data_root=str(config.LABELED_ROOT),
        split="test",
    )

    assert len(test_ds) == 291
