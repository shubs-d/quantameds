"""Evaluation module for the hybrid quantum-classical keratoconus classifier.

Evaluates trained models on the held-out test split (291 samples), computes
clinical classification metrics (accuracy, precision, recall/sensitivity, F1,
AUC-ROC, specificity), and saves confusion matrices and ROC curves.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import torch
from torch.utils.data import DataLoader

from quantum_kc.config import Config, config as default_config, get_device
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.data.image_transforms import get_eval_transforms
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
from quantum_kc.training.utils import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    cfg: Optional[Config] = None,
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model on the held-out test set.

    Args:
        cfg: Configuration object.
        model_path: Optional path to a trained model state_dict checkpoint.
        output_dir: Output directory for plots and JSON results.

    Returns:
        Dictionary of evaluated metrics.
    """
    if cfg is None:
        cfg = default_config

    device = get_device()
    out_dir = output_dir or str(cfg.RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # ── Test Dataset & Loader ──────────────────────────────────────────
    test_transform = get_eval_transforms(cfg.IMG_SIZE)
    test_dataset = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="test",
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    # ── Model Initialization ───────────────────────────────────────────
    encoder = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM)
    model = HybridQuantumClassifier(
        encoder=encoder,
        n_qubits=cfg.N_QUBITS,
        n_layers=cfg.N_LAYERS,
        n_classes=2,
        freeze_encoder=True,
        dev_name=cfg.QML_DEVICE,
        diff_method=cfg.DIFF_METHOD,
    ).to(device)

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info("Loaded model weights from %s", model_path)
    else:
        logger.warning("No model path provided or file missing (%s). Evaluating untransferred weights.", model_path)

    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, tabular, targets in test_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_targets, all_preds, all_probs)

    # ── Plots & Artifacts ──────────────────────────────────────────────
    cm = np.array(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Keratoconus"])
    disp.plot(cmap="Blues", values_format="d")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.title("Keratoconus Detection Confusion Matrix")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    roc_path = os.path.join(out_dir, "roc_curve.png")
    if all_probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    preds_path = os.path.join(out_dir, "predictions.csv")
    df = pd.DataFrame({
        "True_Label": all_targets,
        "Prediction": all_preds,
    })
    for i in range(all_probs.shape[1]):
        df[f"Prob_Class_{i}"] = all_probs[:, i]
    df.to_csv(preds_path, index=False)

    # ── MLflow Tracking ────────────────────────────────────────────────
    try:
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="evaluate_model"):
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(cm_path)
            if os.path.exists(roc_path):
                mlflow.log_artifact(roc_path)
            mlflow.log_artifact(preds_path)
            mlflow.log_artifact(metrics_path)
    except Exception as e:
        logger.warning("MLflow logging encountered an issue: %s", e)

    print("\n=== Test Set Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>20s}: {v:.4f}")
        else:
            print(f"{k:>20s}: {v}")

    return metrics
