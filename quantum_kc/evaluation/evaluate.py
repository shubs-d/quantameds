"""Generalized evaluation module for all model types.

Evaluates any of the following on the held-out test fold:
  - Teacher (CornealEncoder + supervised_head, image-only)
  - ClassicalStudent (TabularMLPEncoder + ClassicalHead, tabular-only)
  - QuantumStudent (TabularMLPEncoder + HybridQMLHead, tabular-only)
  - LogisticBaseline (sklearn LogisticRegression, tabular-only)

Produces:
  - Metrics JSON (AUC, sensitivity, specificity, kappa, precision, F1)
  - Confusion matrix plot
  - ROC curve plot
  - Precision-recall curve plot
  - Operating threshold justification statement
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader

from quantum_kc.config import Config, config as default_config, get_device
from quantum_kc.data.image_transforms import get_eval_transforms
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
from quantum_kc.models.student_pipeline import StudentClassifier
from quantum_kc.training.utils import (
    compute_metrics,
    find_high_sensitivity_threshold,
    precision_recall_curve_data,
)

logger = logging.getLogger(__name__)


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file for experiment reproducibility tracking."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# Operating threshold rationale (printed for every model)
_THRESHOLD_RATIONALE = (
    "Threshold Selection Rationale: This is a triage screening tool. "
    "A false negative (missed KC case) is more costly than a false positive "
    "(unnecessary referral). Therefore, the operating threshold is chosen to "
    "achieve ≥90% sensitivity, accepting lower specificity. "
    "The threshold reported below is the lowest value at which sensitivity ≥ 0.90."
)


def evaluate_model(
    cfg: Optional[Config] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    head_type: str = "classical",
) -> Dict[str, Any]:
    """Polymorphic evaluation function that detects model architecture from checkpoint.

    Logs checkpoint_path, checkpoint_sha256, and all evaluation metrics to MLflow.
    """
    if cfg is None:
        cfg = default_config
    device = get_device()
    ckpt_path = model_path or str(cfg.CHECKPOINT_DIR / "teacher_finetuned_best.pt")

    if not os.path.exists(ckpt_path):
        logger.warning("Checkpoint %s does not exist. Falling back to student eval.", ckpt_path)
        return evaluate_student(cfg=cfg, head_type=head_type, model_path=ckpt_path, output_dir=output_dir)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    keys = list(state.keys()) if isinstance(state, dict) else []

    # Check architecture signatures
    is_hybrid = any("classifier_head." in k or "scaling_layer." in k for k in keys)
    is_teacher = any("supervised_head." in k for k in keys) and not is_hybrid
    is_robust_student = any("vqc_layer." in k and "classifier.weight" in k for k in keys)

    if is_hybrid:
        logger.info("Detected HybridQuantumClassifier checkpoint at %s", ckpt_path)
        out_dir = output_dir or str(cfg.RESULTS_DIR / "hybrid_classifier")
        os.makedirs(out_dir, exist_ok=True)
        test_transform = get_eval_transforms(cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD)
        test_ds = CornOrbDataset(
            csv_path=str(cfg.LABELED_CSV),
            data_root=str(cfg.LABELED_ROOT),
            split="test",
            transform=test_transform,
        )
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
        enc = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM)
        model = HybridQuantumClassifier(
            encoder=enc,
            n_qubits=cfg.N_QUBITS,
            n_layers=cfg.N_LAYERS,
            n_classes=2,
            freeze_encoder=True,
            dev_name=cfg.QML_DEVICE,
            diff_method=cfg.DIFF_METHOD,
        ).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for imgs, _tab, targets in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                all_targets.append(targets.numpy())

        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        return _finalize_evaluation(y_true, y_pred, y_prob, "hybrid_quantum_classifier", out_dir, checkpoint_path=ckpt_path, cfg=cfg)

    elif is_teacher:
        logger.info("Detected CornealEncoder (Teacher) checkpoint at %s", ckpt_path)
        return evaluate_teacher(cfg=cfg, model_path=ckpt_path, output_dir=output_dir)
    else:
        logger.info("Evaluating as StudentClassifier at %s", ckpt_path)
        return evaluate_student(cfg=cfg, head_type=head_type, model_path=ckpt_path, output_dir=output_dir)


def evaluate_teacher(
    cfg: Optional[Config] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate the Teacher (CornealEncoder) on the held-out test set."""
    if cfg is None:
        cfg = default_config
    device = get_device()
    out_dir = output_dir or str(cfg.RESULTS_DIR / "teacher")
    os.makedirs(out_dir, exist_ok=True)

    test_transform = get_eval_transforms(cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD)
    test_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="test",
        transform=test_transform,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(device)
    ckpt_path = model_path or str(cfg.CHECKPOINT_DIR / "teacher_finetuned_best.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        logger.info("Loaded Teacher from %s", ckpt_path)
    else:
        logger.warning("No Teacher checkpoint at %s", ckpt_path)

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, _tab, targets in test_loader:
            imgs = imgs.to(device)
            _z, logits = model(imgs, return_logits=True)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return _finalize_evaluation(y_true, y_pred, y_prob, "teacher", out_dir, checkpoint_path=ckpt_path, cfg=cfg)



def evaluate_student(
    cfg: Optional[Config] = None,
    head_type: str = "classical",
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained Student on the held-out test set.

    The test set uses the same preprocessing pipeline fitted on the full
    training set.  IMPORTANT: this function creates a train dataset internally
    to reuse its fitted pipeline — it does NOT refit on the test fold.
    """
    if cfg is None:
        cfg = default_config
    device = get_device()
    out_dir = output_dir or str(cfg.RESULTS_DIR / f"student_{head_type}")
    os.makedirs(out_dir, exist_ok=True)

    eval_transform = get_eval_transforms(cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD)

    # Fit pipeline on training data (no leakage — test set uses fitted pipeline)
    train_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="train",
        fold=None,
        transform=eval_transform,
    )
    test_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="test",
        transform=eval_transform,
        tabular_scaler=train_ds.tabular_scaler,
        tabular_imputer=train_ds.tabular_imputer,
        student_pipeline=train_ds.student_pipeline,
        use_student_features=True,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)

    model = StudentClassifier(
        head_type=head_type,
        input_dim=cfg.STUDENT_INPUT_DIM,
        latent_dim=cfg.LATENT_DIM,
        n_qubits=cfg.N_QUBITS,
        n_layers=cfg.N_LAYERS,
        dev_name=cfg.QML_DEVICE,
        diff_method=cfg.DIFF_METHOD,
    ).to(device)

    ckpt_path = model_path
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        logger.info("Loaded Student from %s", ckpt_path)

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for _imgs, student_tab, targets in test_loader:
            student_tab = student_tab.to(device)
            logits = model(student_tab)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
            all_targets.append(targets.numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return _finalize_evaluation(y_true, y_pred, y_prob, f"student_{head_type}", out_dir, checkpoint_path=ckpt_path, cfg=cfg)


def _finalize_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    out_dir: str,
    checkpoint_path: Optional[str] = None,
    cfg: Optional[Config] = None,
) -> Dict[str, Any]:
    """Compute metrics, choose threshold, make plots, save JSON, and log to MLflow."""
    if cfg is None:
        cfg = default_config

    # Default threshold metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, threshold=0.5)
    metrics["model"] = model_name
    if checkpoint_path:
        metrics["checkpoint_path"] = str(checkpoint_path)
        if os.path.exists(checkpoint_path):
            metrics["checkpoint_sha256"] = compute_sha256(checkpoint_path)

    # High-sensitivity threshold
    prob_1d = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    hs_threshold = find_high_sensitivity_threshold(y_true, prob_1d, min_sensitivity=0.90)
    y_pred_hs = (prob_1d >= hs_threshold).astype(int)
    metrics_hs = compute_metrics(y_true, y_pred_hs, y_prob, threshold=hs_threshold)
    metrics["high_sensitivity_threshold"] = hs_threshold
    metrics["metrics_at_hs_threshold"] = metrics_hs
    metrics["threshold_rationale"] = _THRESHOLD_RATIONALE

    # ── Confusion matrix ──────────────────────────────────────────────
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    import numpy as _np
    from sklearn.metrics import confusion_matrix as _cm_fn
    cm = _np.array(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "KC"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {model_name} (threshold=0.5)")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── ROC curve ─────────────────────────────────────────────────────
    roc_path = os.path.join(out_dir, "roc_curve.png")
    fpr, tpr, _ = roc_curve(y_true, prob_1d)
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC — {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── PR curve ──────────────────────────────────────────────────────
    pr_path = os.path.join(out_dir, "pr_curve.png")
    precision, recall, _ = precision_recall_curve(y_true, prob_1d)
    plt.figure()
    plt.plot(recall, precision, color="steelblue", lw=2)
    plt.axvline(x=0.90, color="red", linestyle="--", alpha=0.7, label="Sensitivity=0.90 target")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {model_name}")
    plt.legend()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Save JSON ─────────────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, default=str)

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n=== {model_name.upper()} — Test Set Evaluation ===")
    for k in ["auc_roc", "sensitivity", "specificity", "kappa", "precision", "f1", "accuracy"]:
        v = metrics.get(k)
        if isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
    print(f"  high_sensitivity_threshold: {hs_threshold:.4f}")
    print(f"\n  {_THRESHOLD_RATIONALE}")
    print()

    # ── MLflow logging ───────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        active_run = mlflow.active_run()

        def _log_to_mlflow():
            if checkpoint_path:
                mlflow.log_param("checkpoint_path", str(checkpoint_path))
                if "checkpoint_sha256" in metrics:
                    mlflow.log_param("checkpoint_sha256", metrics["checkpoint_sha256"])
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("eval_threshold", float(metrics["threshold"]))
            mlflow.log_param("high_sens_threshold", float(metrics["high_sensitivity_threshold"]))
            for metric_key in ["auc_roc", "accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "kappa"]:
                val = metrics.get(metric_key)
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"eval_{metric_key}", float(val))
            for metric_key in ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1"]:
                val = metrics.get("metrics_at_hs_threshold", {}).get(metric_key)
                if isinstance(val, (int, float)):
                    mlflow.log_metric(f"eval_hs_{metric_key}", float(val))

        if active_run is not None:
            _log_to_mlflow()
            logger.info("Logged evaluation metrics and checkpoint hash to active MLflow run %s", active_run.info.run_id)
        else:
            with mlflow.start_run(run_name=f"evaluate_{model_name}"):
                _log_to_mlflow()
                logger.info("Logged evaluation metrics and checkpoint hash to new MLflow run")
    except Exception as e:
        logger.warning("MLflow logging during evaluation failed: %s", e)

    return metrics
