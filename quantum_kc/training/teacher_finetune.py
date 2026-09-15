"""Stage 2: Teacher fine-tuning on labeled CornOrb data.

Loads pretrained autoencoder encoder weights, attaches a supervised head,
and jointly fine-tunes the full encoder + head on Stage 2 data.

This produces the Teacher's upper-bound performance reference — the latent
space is now disease-relevant (not just reconstruction-relevant), and the
Teacher AUC on the held-out test set is the ceiling for Student distillation.

Key design choices:
  - Encoder is UN-frozen during fine-tuning (full joint training) for best
    performance upper bound.
  - Stage 1 pretraining contribution is measured via domain_shift.py ablation.
  - Weighted cross-entropy loss with inverse-frequency class weights.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from quantum_kc.config import Config, config as default_config, get_device
from quantum_kc.data.image_transforms import get_finetune_transforms, get_eval_transforms
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.training.utils import (
    EarlyStopping,
    compute_metrics,
    get_class_weights,
    set_seed,
)

logger = logging.getLogger(__name__)


def finetune_teacher(
    cfg: Optional[Config] = None,
    fold: Optional[int] = None,
    use_pretrained: bool = True,
    pretrained_ckpt: Optional[str] = None,
    freeze_encoder: bool = False,
    save_tag: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Fine-tune the CornealEncoder (Teacher) on Stage 2 labeled data.

    Args:
        cfg: Configuration object.
        fold: Validation fold (1-5). None uses the full train split.
        use_pretrained: If True, loads Stage 1 autoencoder weights.
            If False, trains strictly from random initialization.
        pretrained_ckpt: Path to encoder weights from Stage 1 autoencoder.
            Defaults to cfg.CHECKPOINT_DIR / "pretrained_encoder.pt".
        freeze_encoder: If True, freeze the CNN encoder and only train the
            supervised head. Default False (joint training for best perf).
        save_tag: Optional identifier added to the checkpoint filename.
        seed: Random seed for reproducibility.

    Returns:
        Dict of best validation and test metrics.
    """
    if cfg is None:
        cfg = default_config

    run_seed = seed if seed is not None else cfg.SEED
    set_seed(run_seed)
    device = get_device()
    logger.info(
        "Teacher fine-tune | fold=%s | use_pretrained=%s | tag=%s | seed=%d | device=%s",
        fold, use_pretrained, save_tag, run_seed, device,
    )

    # ── Datasets ──────────────────────────────────────────────────────
    train_transform = get_finetune_transforms(
        cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD
    )
    val_transform = get_eval_transforms(
        cfg.IMG_SIZE, mean=cfg.DATASET_MEAN, std=cfg.DATASET_STD
    )

    train_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="train",
        fold=fold,
        transform=train_transform,
    )
    val_ds = CornOrbDataset(
        csv_path=str(cfg.LABELED_CSV),
        data_root=str(cfg.LABELED_ROOT),
        split="val",
        fold=fold,
        transform=val_transform,
        tabular_scaler=train_ds.tabular_scaler,
        tabular_imputer=train_ds.tabular_imputer,
        student_pipeline=train_ds.student_pipeline,
    )

    labels = train_ds.labels.numpy()
    class_weights = get_class_weights(labels).to(device)
    sample_weights = [class_weights[int(l)].item() for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────
    encoder = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM, n_classes=2).to(device)

    # Load pretrained encoder weights (backbone only)
    if use_pretrained:
        ckpt_path = pretrained_ckpt or str(cfg.CHECKPOINT_DIR / "pretrained_encoder.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            # Autoencoder saves model.encoder.state_dict() — keys match CornealEncoder.*
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            logger.info(
                "[Teacher] Loaded Stage 1 pretrained encoder from %s | missing (expected head only): %s | unexpected: %s",
                ckpt_path, missing, unexpected,
            )
        else:
            raise FileNotFoundError(
                f"[Teacher] use_pretrained=True requested, but checkpoint not found at {ckpt_path}"
            )
    else:
        logger.info(
            "[Teacher] Training from RANDOM initialization (Stage 1 pretraining disabled, seed=%d).",
            run_seed,
        )

    if freeze_encoder:
        for p in encoder.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder backbone FROZEN — training supervised_head only.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=cfg.LR_FINETUNE,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=12, mode="min")

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    best_val_loss = float("inf")
    best_metrics: Dict[str, Any] = {}
    tag_part = f"_{save_tag}" if save_tag else ""
    seed_part = f"_s{run_seed}"
    run_name = f"teacher_finetune{tag_part}_fold{fold}{seed_part}" if fold else f"teacher_finetune{tag_part}{seed_part}"

    tag_str = f"_{save_tag}" if save_tag else ""
    ckpt_name = f"teacher_finetuned{tag_str}_fold{fold}.pt" if fold else f"teacher_finetuned{tag_str}.pt"
    best_ckpt_path = str(cfg.CHECKPOINT_DIR / ckpt_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "fold": fold,
            "lr": cfg.LR_FINETUNE,
            "latent_dim": cfg.LATENT_DIM,
            "freeze_encoder": freeze_encoder,
            "use_pretrained": use_pretrained,
            "save_tag": save_tag or "default",
            "seed": run_seed,
        })

        for epoch in range(cfg.FINETUNE_EPOCHS):
            # ── Train ─────────────────────────────────────────────────
            encoder.train()
            train_loss = 0.0
            for imgs, _tab, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                _z, logits = encoder(imgs, return_logits=True)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= len(train_ds)

            # ── Validate ──────────────────────────────────────────────
            encoder.eval()
            val_loss = 0.0
            all_preds, all_targets, all_probs = [], [], []
            with torch.no_grad():
                for imgs, _tab, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    _z, logits = encoder(imgs, return_logits=True)
                    loss = criterion(logits, targets)
                    val_loss += loss.item() * imgs.size(0)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            val_loss /= len(val_ds)
            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
            y_prob = np.concatenate(all_probs)
            metrics = compute_metrics(y_true, y_pred, y_prob)

            scheduler.step(val_loss)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": metrics.get("auc_roc") or 0.0,
                "val_sensitivity": metrics["sensitivity"],
                "val_specificity": metrics["specificity"],
                "val_kappa": metrics["kappa"],
            }, step=epoch)

            logger.info(
                "Epoch %3d | Train %.4f | Val %.4f | AUC %.4f | Sens %.4f | Spec %.4f",
                epoch + 1, train_loss, val_loss,
                metrics.get("auc_roc") or 0.0,
                metrics["sensitivity"], metrics["specificity"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = dict(metrics)
                os.makedirs(str(cfg.CHECKPOINT_DIR), exist_ok=True)
                torch.save(encoder.state_dict(), best_ckpt_path)
                # Always keep a 'best' copy for downstream distillation if default run
                if not save_tag:
                    torch.save(encoder.state_dict(), str(cfg.CHECKPOINT_DIR / "teacher_finetuned_best.pt"))

            if early_stopping(val_loss):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # ── Evaluate on held-out test split ───────────────────────────
        if os.path.exists(best_ckpt_path):
            encoder.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
        encoder.eval()

        test_ds = CornOrbDataset(
            csv_path=str(cfg.LABELED_CSV),
            data_root=str(cfg.LABELED_ROOT),
            split="test",
            transform=val_transform,
            tabular_scaler=train_ds.tabular_scaler,
            tabular_imputer=train_ds.tabular_imputer,
            student_pipeline=train_ds.student_pipeline,
        )
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
        test_preds, test_targets, test_probs = [], [], []
        with torch.no_grad():
            for imgs, _tab, targets in test_loader:
                imgs = imgs.to(device)
                _z, logits = encoder(imgs, return_logits=True)
                probs = torch.softmax(logits, dim=1)
                test_probs.append(probs.cpu().numpy())
                test_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                test_targets.append(targets.numpy())

        y_test_true = np.concatenate(test_targets)
        y_test_pred = np.concatenate(test_preds)
        y_test_prob = np.concatenate(test_probs)
        test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_prob)

        best_metrics["test_auc_roc"] = test_metrics.get("auc_roc")
        best_metrics["test_sensitivity"] = test_metrics.get("sensitivity")
        best_metrics["test_specificity"] = test_metrics.get("specificity")
        best_metrics["test_kappa"] = test_metrics.get("kappa")
        best_metrics["checkpoint_path"] = best_ckpt_path
        best_metrics["use_pretrained"] = use_pretrained
        best_metrics["seed"] = run_seed

        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items() if isinstance(v, float)})
        logger.info(
            "[Teacher] Best Val AUC: %.4f | Test AUC: %.4f | Saved: %s",
            best_metrics.get("auc_roc") or 0.0,
            best_metrics.get("test_auc_roc") or 0.0,
            best_ckpt_path,
        )

    return best_metrics
