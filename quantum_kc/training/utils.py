"""Training utilities: metrics, early stopping, seed, class weights.

Changes from original:
  - compute_metrics now includes specificity and Cohen's kappa.
  - log_gradient_variance fixed to correctly target VQC 'weights' param
    via submodule-level iteration rather than name-string heuristics.
  - precision_recall_curve_data added (returns arrays for plotting).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Returns a tensor of shape (n_classes,) where weight[c] = N / (n_c × C).
    This upweights the minority class for balanced cross-entropy.
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute classification metrics for a binary (Normal vs. KC) task.

    For a triage tool, false negatives (missed KC) are the costlier error, so
    the operating threshold should be chosen to maximise sensitivity.  The
    default threshold=0.5 is used here; callers can pass a clinical threshold.

    Args:
        y_true: Ground truth labels (0=Normal, 1=KC).
        y_pred: Hard predictions (0 or 1).
        y_prob: Probability array of shape (N,) or (N, 2).  Used for AUC.
        threshold: Threshold used to derive y_pred (logged for transparency).

    Returns:
        Dict with keys: accuracy, precision, recall (=sensitivity),
        specificity, f1, kappa, auc_roc, confusion_matrix, threshold.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,  # = recall for positive class
        "recall": sensitivity,       # alias — keep for backward compat
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "n_tp": int(tp),
        "n_fp": int(fp),
        "n_fn": int(fn),
        "n_tn": int(tn),
    }

    if y_prob is not None:
        prob_1d = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, prob_1d))
        except ValueError:
            metrics["auc_roc"] = None

    return metrics


def precision_recall_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return precision, recall, and threshold arrays for a PR curve.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities (1-D).

    Returns:
        (precision, recall, thresholds) arrays as returned by sklearn.
    """
    prob_1d = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    return precision_recall_curve(y_true, prob_1d)


def find_high_sensitivity_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_sensitivity: float = 0.90,
) -> float:
    """Find the lowest threshold that achieves at least ``min_sensitivity``.

    For a keratoconus triage tool, missing KC is the costlier error.  This
    function selects the operating threshold that guarantees >= 90% sensitivity
    (by default) at the cost of lower specificity.

    Args:
        y_true: Binary ground truth.
        y_prob: Positive-class probabilities (1-D or 2-column).
        min_sensitivity: Minimum acceptable sensitivity (default 0.90).

    Returns:
        Selected threshold (float).
    """
    prob_1d = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, prob_1d)
    # thresholds has length N-1 (recall/precision have length N)
    for i, thr in enumerate(thresholds):
        if recall[i] >= min_sensitivity:
            return float(thr)
    return float(thresholds[-1])  # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Gradient variance (quantum barren-plateau diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def log_gradient_variance(model: nn.Module, tag: str = "quantum") -> float:
    """Compute variance of VQC weight gradients for barren-plateau detection.

    This function looks specifically for parameters belonging to the VQC
    TorchLayer submodule, rather than relying on string-matching heuristics
    that can miss TorchLayer's internal 'weights' parameter name.

    Args:
        model: The StudentClassifier or HybridQuantumClassifier model.
        tag: Label used in warning messages.

    Returns:
        Gradient variance (float).  Returns 0.0 if no VQC gradients found.
    """
    grads: List[torch.Tensor] = []

    # Search for TorchLayer submodules (PennyLane's VQC wrapper)
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "TorchLayer" in module_type or "vqc" in name.lower():
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))

    # Fallback: any parameter with 'weights' in a submodule named 'vqc*'
    if not grads:
        for name, param in model.named_parameters():
            if "vqc" in name.lower() and param.grad is not None:
                grads.append(param.grad.view(-1))

    if not grads:
        return 0.0

    all_grads = torch.cat(grads)
    variance = float(torch.var(all_grads).item())

    if variance < 1e-6:
        logger.warning(
            "[%s] Very low gradient variance: %.2e — possible barren plateau.", tag, variance
        )

    return variance


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs to wait for improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: ``'min'`` (loss) or ``'max'`` (metric like AUC).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric: Optional[float] = None

    def __call__(self, metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            return False

        improved = (
            metric < self.best_metric - self.min_delta
            if self.mode == "min"
            else metric > self.best_metric + self.min_delta
        )

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        self.counter = 0
        self.best_metric = None
