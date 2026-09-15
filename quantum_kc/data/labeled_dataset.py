"""PyTorch Dataset for the CornOrb labeled dataset.

Changes from the original:
  - ``build_tabular_pipeline`` now returns a fitted ``SimpleImputer`` too.
    Train datasets fit a new imputer; val/test datasets must receive the
    train-fold imputer to prevent data leakage.
  - ``__getitem__`` returns a 4-tuple: (img_tensor, full_tab_tensor,
    student_tab_tensor, label).  The ``student_tab_tensor`` has exactly 5
    features as defined in ``StudentTabularPipeline``.
  - ``student_pipeline`` is exposed as a property so callers can thread it
    into val/test datasets.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, WeightedRandomSampler

from .image_transforms import load_and_stack_maps
from .preprocessing import (
    RobustStudentTabularPipeline,
    StudentTabularPipeline,
    build_tabular_pipeline,
)

logger = logging.getLogger(__name__)


class CornOrbDataset(Dataset):
    """CornOrb Stage 2 dataset: paired images + tabular + labels.

    Args:
        csv_path: Path to ``clinical_data_and_labels.csv``.
        data_root: Root directory containing patient subdirectories.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        fold: Validation fold index (1-based).  Only used for train/val splits.
        transform: Image transform to apply.
        tabular_scaler: Pre-fitted MinMaxScaler for full tabular features.
            Pass ``None`` for the train fold (a new scaler will be fitted).
            Pass the train-fold scaler for val/test to prevent leakage.
        tabular_imputer: Pre-fitted SimpleImputer for full tabular features.
            Same leakage prevention logic as ``tabular_scaler``.
        student_pipeline: Pre-fitted StudentTabularPipeline.
            Pass ``None`` for the train fold.  Pass the fitted pipeline for
            val/test folds.
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = "train",
        fold: Optional[int] = None,
        transform: Optional[Callable] = None,
        tabular_scaler: Optional[MinMaxScaler] = None,
        tabular_imputer: Optional[SimpleImputer] = None,
        student_pipeline: Optional[StudentTabularPipeline] = None,
        img_mean: Optional[list] = None,
        img_std: Optional[list] = None,
        use_student_features: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform

        # ── Read CSV ──────────────────────────────────────────────────
        df = pd.read_csv(csv_path, dtype={"patient_code": str})
        # Excel artifact: '3E132' was written as scientific notation '3E+132'
        df["patient_code"] = (
            df["patient_code"].astype(str).str.replace("3E+132", "3E132", regex=False)
        )

        # ── Split filtering ───────────────────────────────────────────
        if split == "test":
            if "split_80_20" in df.columns:
                df = df[df["split_80_20"] != "train"]
        elif split in ("train", "val"):
            if "split_80_20" in df.columns:
                df = df[df["split_80_20"] == "train"]
            if fold is not None and "fold" in df.columns:
                if split == "train":
                    df = df[df["fold"] != fold]
                elif split == "val":
                    df = df[df["fold"] == fold]

        self.df = df.reset_index(drop=True)

        # ── Categorical encoding (preserve eye string for image loading) ──
        if "eye" in self.df.columns:
            self.df["eye_str"] = self.df["eye"].copy()
            self.df["eye"] = self.df["eye"].map(
                {"OD": 0, "OS": 1, "od": 0, "os": 1}
            )
        if "gender" in self.df.columns:
            self.df["gender"] = self.df["gender"].map(
                {"m": 0, "f": 1, "M": 0, "F": 1}
            )

        self.use_student_features = use_student_features

        # ── Full tabular features (for Teacher / multimodal models) ───
        numeric_cols = [
            "kmax_value_D", "pachy_central_um", "pachy_thinnest_um",
            "pachy_thinnest_x", "pachy_thinnest_y", "asphericity_anterior",
            "asphericity_posterior", "age_years", "gender", "eye",
            "astig_axis_deg", "kmax_axis_deg", "astig_value_D",
        ]
        existing_num_cols = [c for c in numeric_cols if c in self.df.columns]
        tabular_data = self.df[existing_num_cols]

        angular_cols = [c for c in ["astig_axis_deg", "kmax_axis_deg"] if c in existing_num_cols]
        skewed_cols = [c for c in ["astig_value_D"] if c in existing_num_cols]

        self.tabular_features, self.tabular_scaler, self.tabular_imputer = (
            build_tabular_pipeline(
                tabular_data,
                angular_cols=angular_cols,
                skewed_cols=skewed_cols,
                scaler=tabular_scaler,
                imputer=tabular_imputer,
                return_imputer=True,
            )
        )

        # ── Student tabular features (5-feature contract) ─────────────
        if student_pipeline is None:
            self._student_pipeline = StudentTabularPipeline()
            self.student_features = self._student_pipeline.fit_transform(self.df)
        else:
            self._student_pipeline = student_pipeline
            self.student_features = self._student_pipeline.transform(self.df)

        # ── Labels ────────────────────────────────────────────────────
        if "label" in self.df.columns:
            self.labels = torch.LongTensor(self.df["label"].values.copy())
        else:
            self.labels = torch.zeros(len(self.df), dtype=torch.long)
            logger.warning("No 'label' column found — defaulting to zeros.")

    # ── Properties ────────────────────────────────────────────────────

    @property
    def student_pipeline(self) -> StudentTabularPipeline:
        """Return the fitted StudentTabularPipeline (thread to val/test)."""
        return self._student_pipeline

    # ── Dataset protocol ──────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (img_tensor, tab_tensor, label).

        If use_student_features is True, tab_tensor contains the 5 Student features.
        Otherwise, tab_tensor contains the 13 full tabular features.
        """
        row = self.df.iloc[idx]
        patient_code = str(row["patient_code"])
        eye = (
            str(row["eye_str"])
            if "eye_str" in self.df.columns
            else ("OD" if row["eye"] == 0 else "OS")
        )

        img_dir = os.path.join(self.data_root, patient_code, eye)

        try:
            img = load_and_stack_maps(img_dir, patient_code, eye)
        except Exception as e:
            logger.error("Error loading images for %s %s: %s", patient_code, eye, e)
            img = None

        if img is not None and self.transform:
            img_tensor = self.transform(img)
        elif img is not None:
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)
        else:
            img_tensor = torch.zeros((3, 224, 224))

        if self.use_student_features:
            tab_tensor = torch.FloatTensor(self.student_features[idx])
        else:
            tab_tensor = torch.FloatTensor(self.tabular_features[idx])

        label = self.labels[idx]

        return img_tensor, tab_tensor, label

    def get_both_tabular(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (img_tensor, full_tab_tensor, student_tab_tensor, label)."""
        img_tensor, _, label = self.__getitem__(idx)
        full_tab = torch.FloatTensor(self.tabular_features[idx])
        student_tab = torch.FloatTensor(self.student_features[idx])
        return img_tensor, full_tab, student_tab, label

    # ── Class imbalance helpers ────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss."""
        class_counts = np.bincount(self.labels.numpy())
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler for balanced mini-batches."""
        class_counts = np.bincount(self.labels.numpy())
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels.numpy()]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


class RobustCornOrbDataset(Dataset):
    """CornOrb dataset tailored for RobustQuantaStudent (6 inputs with modality dropout).

    Features:
        0: kmax_norm
        1: log1p_cyl_norm
        2: sin_2axis
        3: cos_2axis
        4: pachy_norm (or 0.0 if dropped)
        5: pachy_mask (1.0 if present, 0.0 if dropped)

    Modes:
        'train': Bernoulli(1 - p_dropout) independently per sample.
        'tier1': mask=0.0, pachy=0.0 (ARK only).
        'tier2': mask=1.0, pachy=pachy_norm (ARK + Pachymetry).
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = "train",
        fold: Optional[int] = None,
        transform: Optional[Callable] = None,
        pipeline: Optional[RobustStudentTabularPipeline] = None,
        mode: str = "train",
        p_dropout: float = 0.5,
        load_images: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.mode = mode
        self.p_dropout = p_dropout
        self.load_images = load_images

        # ── Read CSV ──────────────────────────────────────────────────
        df = pd.read_csv(csv_path, dtype={"patient_code": str})
        df["patient_code"] = (
            df["patient_code"].astype(str).str.replace("3E+132", "3E132", regex=False)
        )

        # ── Split filtering ───────────────────────────────────────────
        if split == "test":
            if "split_80_20" in df.columns:
                df = df[df["split_80_20"] != "train"]
        elif split in ("train", "val"):
            if "split_80_20" in df.columns:
                df = df[df["split_80_20"] == "train"]
            if fold is not None and "fold" in df.columns:
                if split == "train":
                    df = df[df["fold"] != fold]
                elif split == "val":
                    df = df[df["fold"] == fold]

        self.df = df.reset_index(drop=True)

        # ── Fit or apply pipeline ─────────────────────────────────────
        if pipeline is None:
            self._pipeline = RobustStudentTabularPipeline()
            self._pipeline.fit(self.df)
        else:
            self._pipeline = pipeline

        self.base_features = self._pipeline.transform_base(self.df)

        # ── Labels ────────────────────────────────────────────────────
        if "label" in self.df.columns:
            self.labels = torch.tensor(self.df["label"].values.copy(), dtype=torch.float32)
        else:
            self.labels = torch.zeros(len(self.df), dtype=torch.float32)

    @property
    def pipeline(self) -> RobustStudentTabularPipeline:
        return self._pipeline

    def set_mode(self, mode: str) -> None:
        """Switch between 'train', 'tier1', and 'tier2'."""
        if mode not in ("train", "tier1", "tier2"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (img_tensor, x_tabular, label)."""
        row = self.df.iloc[idx]
        patient_code = str(row["patient_code"])
        eye = str(row["eye"])

        # 1. Load image if required
        if self.load_images:
            img_dir = os.path.join(self.data_root, patient_code, eye)
            try:
                img = load_and_stack_maps(img_dir, patient_code, eye)
            except Exception as e:
                logger.error("Error loading images for %s %s: %s", patient_code, eye, e)
                img = None

            if img is not None and self.transform:
                img_tensor = self.transform(img)
            elif img is not None:
                from torchvision.transforms import ToTensor
                img_tensor = ToTensor()(img)
            else:
                img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 2. Modality dropout on 5 base features -> 6 student features
        base = self.base_features[idx]
        if self.mode == "tier1":
            pachy = 0.0
            mask = 0.0
        elif self.mode == "tier2":
            pachy = float(base[4])
            mask = 1.0
        elif self.mode == "train":
            # Per-sample independent Bernoulli trial
            is_present = (np.random.rand() >= self.p_dropout)
            pachy = float(base[4]) if is_present else 0.0
            mask = 1.0 if is_present else 0.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        x_tab = torch.tensor(
            [base[0], base[1], base[2], base[3], pachy, mask],
            dtype=torch.float32,
        )

        label = self.labels[idx]
        return img_tensor, x_tab, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for binary cross-entropy pos_weight."""
        labels_np = self.labels.long().numpy()
        n_neg = (labels_np == 0).sum()
        n_pos = (labels_np == 1).sum()
        pos_weight = float(n_neg / max(n_pos, 1))
        return torch.tensor([pos_weight], dtype=torch.float32)

