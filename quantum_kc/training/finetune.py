"""Stage 2: Supervised VQC fine-tuning on labeled CornOrb data.

Loads the pretrained CNN encoder, freezes it, and trains a Variational Quantum
Classifier head using PennyLane's StronglyEntanglingLayers.  Supports k-fold
cross-validation using the pre-defined fold assignments in the dataset CSV.
"""

from __future__ import annotations

import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import mlflow
from typing import Dict, Any, Optional

from quantum_kc.config import Config, config as default_config, get_device
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.data.image_transforms import get_finetune_transforms, get_eval_transforms
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier
from quantum_kc.training.utils import (
    set_seed, EarlyStopping, compute_metrics,
    log_gradient_variance, get_class_weights,
)

logger = logging.getLogger(__name__)

def finetune_vqc(cfg: Optional[Config] = None, fold: Optional[int] = None) -> Dict[str, Any]:
    """Train the VQC classifier on labeled CornOrb data.

    Args:
        cfg: Configuration object.  Falls back to module-level singleton.
        fold: If provided, use this fold number for validation (1-5).

    Returns:
        Dictionary with best validation metrics.
    """
    if cfg is None:
        cfg = default_config

    set_seed(cfg.SEED)
    device = get_device()

    # ── Datasets ──────────────────────────────────────────────────────
    train_transform = get_finetune_transforms(cfg.IMG_SIZE)
    val_transform = get_eval_transforms(cfg.IMG_SIZE)

    train_dataset = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="train", fold=fold,
        transform=train_transform,
    )
    val_dataset = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="val", fold=fold,
        transform=val_transform,
        tabular_scaler=train_dataset.tabular_scaler,
    )

    labels = train_dataset.labels.numpy()
    class_weights = get_class_weights(labels).to(device)

    sample_weights = [class_weights[l].item() for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    encoder = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM)
    ckpt_path = str(cfg.CHECKPOINT_DIR / "pretrained_encoder.pt")
    if os.path.exists(ckpt_path):
        encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        logger.info("Loaded pretrained encoder from %s", ckpt_path)
    else:
        logger.warning("No pretrained encoder found at %s — training from scratch", ckpt_path)

    model = HybridQuantumClassifier(
        encoder=encoder,
        n_qubits=cfg.N_QUBITS,
        n_layers=cfg.N_LAYERS,
        n_classes=2,
        freeze_encoder=True,
        dev_name=cfg.QML_DEVICE,
        diff_method=cfg.DIFF_METHOD,
    ).to(device)

    # Only optimize VQC + classifier head (encoder is frozen)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=cfg.LR_FINETUNE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=10, mode="min")
    
    best_val_loss = float('inf')
    best_metrics = {}
    
    # ── MLflow ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_name=f'finetune_vqc_fold_{fold}'):
        mlflow.log_params({"lr": cfg.LR_FINETUNE, "fold": fold, "n_qubits": cfg.N_QUBITS, "n_layers": cfg.N_LAYERS})
        
        for epoch in range(cfg.FINETUNE_EPOCHS):
            model.train()
            train_loss = 0.0
            
            for images, tabular, targets in train_loader:
                images, tabular, targets = images.to(device), tabular.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images, tabular)
                loss = criterion(outputs, targets)
                loss.backward()
                
                grad_var = log_gradient_variance(model, tag='quantum')
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            all_preds, all_targets, all_probs = [], [], []
            
            with torch.no_grad():
                for images, tabular, targets in val_loader:
                    images, tabular, targets = images.to(device), tabular.to(device), targets.to(device)
                    outputs = model(images, tabular)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)
                    
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    
            val_loss /= len(val_loader.dataset)
            
            all_targets = np.concatenate(all_targets)
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            
            metrics = compute_metrics(all_targets, all_preds, all_probs)
            
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': metrics['accuracy'],
                'val_f1': metrics['f1'],
                'val_auc': metrics.get('auc_roc', 0.0),
                'grad_variance': grad_var
            }, step=epoch)
            
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {metrics['accuracy']:.4f}",
                flush=True,
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                os.makedirs(str(cfg.CHECKPOINT_DIR), exist_ok=True)
                ckpt_name = f"hybrid_classifier_fold{fold}.pt" if fold is not None else "hybrid_classifier.pt"
                torch.save(model.state_dict(), str(cfg.CHECKPOINT_DIR / ckpt_name))
                torch.save(model.state_dict(), str(cfg.CHECKPOINT_DIR / "hybrid_classifier_best.pt"))
                
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break
                
    return best_metrics

def run_cross_validation(cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Run cross-validation across 5 folds."""
    if cfg is None:
        cfg = default_config
    all_metrics = {}
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    for k in metric_keys:
        all_metrics[k] = []
        
    for fold in range(1, 6):
        print(f"--- Running Fold {fold} ---")
        metrics = finetune_vqc(cfg, fold=fold)
        for k in metric_keys:
            if k in metrics and metrics[k] is not None:
                all_metrics[k].append(metrics[k])
                
    results = {}
    for k in metric_keys:
        if all_metrics[k]:
            results[f"{k}_mean"] = float(np.mean(all_metrics[k]))
            results[f"{k}_std"] = float(np.std(all_metrics[k]))
            print(f"{k}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")
            
    return results
