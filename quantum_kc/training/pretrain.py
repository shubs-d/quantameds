"""Stage 1: Unsupervised autoencoder pretraining on unlabeled Orbscan IIz data.

Trains a CNN autoencoder on 2,633 QC-PASS anterior axial power maps to learn
the foundational geometries of the human cornea.  The encoder weights are saved
for transfer to the supervised VQC fine-tuning stage.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from quantum_kc.config import Config, config as default_config, get_device
from quantum_kc.data.unlabeled_dataset import OrbscanUnlabeledDataset
from quantum_kc.data.image_transforms import get_pretrain_transforms
from quantum_kc.models.autoencoder import CornealAutoencoder
from quantum_kc.training.utils import set_seed, EarlyStopping

logger = logging.getLogger(__name__)


def pretrain_autoencoder(cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Train the corneal autoencoder on unlabeled Orbscan IIz images.

    Args:
        cfg: Configuration object.  Falls back to the module-level singleton.

    Returns:
        Dictionary with ``best_val_loss`` and training metadata.
    """
    if cfg is None:
        cfg = default_config

    set_seed(cfg.SEED)
    device = get_device()
    logger.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────
    transform = get_pretrain_transforms(cfg.IMG_SIZE)

    dataset = OrbscanUnlabeledDataset(
        csv_path=str(cfg.UNLABELED_CSV),
        images_dir=str(cfg.UNLABELED_IMAGES_DIR),
        transform=transform,
        qc_filter=True,
    )
    logger.info("Unlabeled dataset size: %d", len(dataset))

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = CornealAutoencoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=cfg.LR_PRETRAIN)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, mode="min")

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    best_val_loss = float("inf")
    metrics_dict: Dict[str, Any] = {}

    with mlflow.start_run(run_name="pretrain_autoencoder"):
        mlflow.log_params({
            "lr": cfg.LR_PRETRAIN,
            "batch_size": cfg.BATCH_SIZE,
            "latent_dim": cfg.LATENT_DIM,
            "pretrain_epochs": cfg.PRETRAIN_EPOCHS,
            "seed": cfg.SEED,
        })

        for epoch in range(cfg.PRETRAIN_EPOCHS):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                images = batch[0].to(device)  # (image_tensor, tabular_tensor)

                optimizer.zero_grad()
                x_recon, _z = model(images)  # autoencoder returns (recon, latent)
                loss = criterion(x_recon, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            # ── Validate ──────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    x_recon, _z = model(images)
                    loss = criterion(x_recon, images)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)

            # ── Logging ───────────────────────────────────────────────
            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "lr": lr, "epoch_time": epoch_time},
                step=epoch,
            )

            scheduler.step(val_loss)

            logger.info(
                "Epoch %3d | Train %.5f | Val %.5f | LR %.2e | %.1fs",
                epoch + 1, train_loss, val_loss, lr, epoch_time,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(str(cfg.CHECKPOINT_DIR), exist_ok=True)
                ckpt_path = str(cfg.CHECKPOINT_DIR / "pretrained_encoder.pt")
                torch.save(model.encoder.state_dict(), ckpt_path)
                logger.info("Saved best encoder → %s (val=%.5f)", ckpt_path, val_loss)

            if early_stopping(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        metrics_dict["best_val_loss"] = best_val_loss
        mlflow.log_metric("best_val_loss", best_val_loss)

    return metrics_dict



