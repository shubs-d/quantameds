"""Central configuration for the hybrid quantum-classical keratoconus pipeline.

Provides a single ``Config`` dataclass consumed by every module.  Path
attributes are resolved at import time so downstream code never needs to
hard-code directory strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

import torch


@dataclass
class Config:
    """Immutable-ish bag of hyperparameters, paths, and column lists."""

    # ── Model parameters ──────────────────────────────────────────────
    N_QUBITS: int = 8
    N_LAYERS: int = 2  # StronglyEntanglingLayers depth (shallow → avoids barren plateaus)
    IMG_SIZE: Tuple[int, int] = (224, 224)
    LATENT_DIM: int = 8  # Must equal N_QUBITS for angle embedding

    # ── Training parameters ───────────────────────────────────────────
    BATCH_SIZE: int = 32
    PRETRAIN_EPOCHS: int = 100
    FINETUNE_EPOCHS: int = 50
    DISTILL_EPOCHS: int = 60
    LR_PRETRAIN: float = 1e-3
    LR_FINETUNE: float = 5e-4
    LR_STUDENT: float = 1e-3       # Student MLP encoder learning rate
    LR_CLASSICAL_HEAD: float = 1e-3
    LR_QUANTUM_HEAD: float = 5e-4  # Quantum head may need smaller LR due to gradient scale
    SEED: int = 42
    N_SEEDS: int = 5               # Number of independent seeds for multi-seed evaluation

    # ── Distillation hyperparameters ──────────────────────────────────
    # L_total = DISTILL_ALPHA * L_distill + DISTILL_BETA * L_task
    # Three ablation settings (alpha, beta):
    #   (1.0, 0.0) → distillation-only
    #   (0.5, 0.5) → balanced
    #   (0.0, 1.0) → task-only (sanity check baseline)
    DISTILL_ALPHA: float = 0.5
    DISTILL_BETA: float = 0.5
    DISTILL_ALPHA_GRID: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.0]
    )
    DISTILL_BETA_GRID: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0]
    )

    # ── Quantum backend ───────────────────────────────────────────────
    QML_DEVICE: str = "lightning.qubit"
    DIFF_METHOD: str = "adjoint"  # Efficient for lightning.qubit

    # ── Data paths ────────────────────────────────────────────────────
    BASE_DIR: Path = Path("/home/shubs/Projects/Keratoconus/Dataset")

    # Labeled CornOrb dataset (1,454 labeled eye records, 744 patients)
    LABELED_ROOT: Path = BASE_DIR / "data" / "ORBSCAN_Dataset"
    LABELED_CSV: Path = LABELED_ROOT / "clinical_data_and_labels.csv"

    # Unlabeled Orbscan IIz dataset (2,633 axial power maps)
    UNLABELED_ROOT: Path = BASE_DIR / "data" / "Multimodal Orbscan IIz Dataset 3,000 Axial Power A"
    UNLABELED_CSV: Path = UNLABELED_ROOT / "metadata_numeric.csv"
    UNLABELED_IMAGES_DIR: Path = UNLABELED_ROOT / "Images"

    # ── Artifact paths ────────────────────────────────────────────────
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoints"
    RESULTS_DIR: Path = BASE_DIR / "results"
    MLFLOW_TRACKING_URI: str = f"sqlite:///{BASE_DIR / 'mlruns.db'}"

    # ── Image normalization ───────────────────────────────────────────
    # Dataset-specific channel mean/std for pseudo-colored topography maps.
    # Computed via scripts/compute_dataset_stats.py on 2,633 Stage 1 unlabeled maps.
    DATASET_MEAN: List[float] = field(default_factory=lambda: [0.3975, 0.4393, 0.2199])
    DATASET_STD: List[float] = field(default_factory=lambda: [0.2840, 0.3035, 0.2005])

    # ── Image channels (stacked as pseudo-RGB) ────────────────────────
    IMAGE_CHANNELS: List[str] = field(
        default_factory=lambda: ["Axial", "Anterior", "Posterior"]
    )

    # ── Class weights (inverse-frequency for 61 % Normal / 39 % KC) ──
    NORMAL_WEIGHT: float = 0.82  # 1454 / (2 * 889)
    KC_WEIGHT: float = 1.29      # 1454 / (2 * 565)

    # ── Student tabular feature contract (FIXED — 5 features) ─────────
    # These 5 features represent what a basic clinical screening provides:
    # steep-K proxy, cylinder magnitude + direction, and corneal thinning.
    # DO NOT change this list without updating docs, slides, and all scripts.
    #
    # Raw columns from clinical_data_and_labels.csv → final transformed features:
    #   kmax_value_D              → min-max scaled              (index 0)
    #   astig_value_D             → log1p → min-max scaled      (index 1)
    #   astig_axis_deg            → sin(2θ)                     (index 2)
    #   astig_axis_deg            → cos(2θ)                     (index 3)
    #   pachy_thinnest_um         → min-max scaled              (index 4)
    STUDENT_SCALAR_COLS: List[str] = field(
        default_factory=lambda: ["kmax_value_D", "pachy_thinnest_um"]
    )
    STUDENT_SKEWED_COLS: List[str] = field(
        default_factory=lambda: ["astig_value_D"]
    )
    STUDENT_ANGULAR_COLS: List[str] = field(
        default_factory=lambda: ["astig_axis_deg"]
    )
    # Final feature names after transformation (in order, length == 5)
    STUDENT_FEATURE_NAMES: List[str] = field(
        default_factory=lambda: [
            "kmax_value_D",
            "astig_value_D_log",
            "astig_axis_sin",
            "astig_axis_cos",
            "pachy_thinnest_um",
        ]
    )
    STUDENT_INPUT_DIM: int = 5  # Must equal len(STUDENT_FEATURE_NAMES)

    # ── Full labeled tabular column definitions ────────────────────────
    #   Angular axes → sin(2θ) / cos(2θ) encoding
    LABELED_ANGULAR_COLS: List[str] = field(
        default_factory=lambda: ["astig_axis_deg", "kmax_axis_deg"]
    )
    #   Severely skewed → log(1+|x|)
    LABELED_SKEWED_COLS: List[str] = field(
        default_factory=lambda: ["astig_value_D"]
    )
    #   All numeric features used from the labeled CSV
    LABELED_NUMERIC_COLS: List[str] = field(
        default_factory=lambda: [
            "age_years",
            "astig_value_D",
            "astig_axis_deg",
            "kmax_value_D",
            "kmax_axis_deg",
            "pachy_central_um",
            "pachy_thinnest_um",
            "pachy_thinnest_x",
            "pachy_thinnest_y",
            "asphericity_anterior",
            "asphericity_posterior",
        ]
    )
    #   Categorical columns (binary-encoded)
    LABELED_CATEGORICAL_MAP: dict = field(
        default_factory=lambda: {
            "gender": {"m": 0, "f": 1},
            "eye": {"OD": 0, "OS": 1},
        }
    )

    # ── Logistic regression naive baseline ────────────────────────────
    LOGISTIC_BASELINE_MAX_ITER: int = 1000

    # ── Unlabeled tabular column definitions ──────────────────────────
    UNLABELED_ANGULAR_COLS: List[str] = field(
        default_factory=lambda: [
            "SimK_Astig_Axis_deg",
            "MaxK_Axis_deg",
            "MinK_Axis_deg",
            "Zone3mm_SteepAxis_deg",
            "Zone3mm_FlatAxis_deg",
            "Zone5mm_SteepAxis_deg",
            "Zone5mm_FlatAxis_deg",
            "Kappa_at_deg",
        ]
    )
    UNLABELED_SKEWED_COLS: List[str] = field(
        default_factory=lambda: [
            "SimK_Astig_D",
            "Zone3mm_Irreg_D",
            "Zone5mm_Irreg_D",
        ]
    )
    UNLABELED_NUMERIC_COLS: List[str] = field(
        default_factory=lambda: [
            "SimK_Astig_D",
            "SimK_Astig_Axis_deg",
            "MaxK_D",
            "MaxK_Axis_deg",
            "MinK_D",
            "MinK_Axis_deg",
            "Zone3mm_Irreg_D",
            "Zone3mm_MeanPwr_D",
            "Zone3mm_MeanPwr_SD",
            "Zone3mm_AstigPwr_D",
            "Zone3mm_SteepAxis_deg",
            "Zone3mm_FlatAxis_deg",
            "Zone5mm_Irreg_D",
            "Zone5mm_MeanPwr_D",
            "Zone5mm_MeanPwr_SD",
            "Zone5mm_AstigPwr_D",
            "Zone5mm_SteepAxis_deg",
            "Zone5mm_FlatAxis_deg",
            "WhiteToWhite_mm",
            "PupilDiameter_mm",
            "Thinnest_um",
            "Thinnest_x_mm",
            "Thinnest_y_mm",
            "ACD_Ep_mm",
            "Kappa_deg",
            "Kappa_at_deg",
            "KappaIntercept_x",
            "KappaIntercept_y",
        ]
    )


def get_device() -> torch.device:
    """Return the best available torch device (CUDA → CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Module-level singleton used by all downstream code.
config = Config()
