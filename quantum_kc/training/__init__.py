"""Training loops and optimization strategies."""

from .pretrain import pretrain_autoencoder
from .finetune import finetune_vqc, run_cross_validation
from .teacher_finetune import finetune_teacher
from .distill import train_student
from .ablation_runner import run_ablation, run_logistic_baseline
from .domain_shift import run_stage1_ablation
from .utils import (
    EarlyStopping,
    compute_metrics,
    log_gradient_variance,
    set_seed,
    get_class_weights,
    precision_recall_curve_data,
    find_high_sensitivity_threshold,
)

__all__ = [
    "pretrain_autoencoder",
    "finetune_vqc",
    "run_cross_validation",
    "finetune_teacher",
    "train_student",
    "run_ablation",
    "run_logistic_baseline",
    "run_stage1_ablation",
    "EarlyStopping",
    "compute_metrics",
    "log_gradient_variance",
    "set_seed",
    "get_class_weights",
    "precision_recall_curve_data",
    "find_high_sensitivity_threshold",
]
