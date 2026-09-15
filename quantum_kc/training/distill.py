"""Stage 3: Student distillation training loop.

Trains a StudentClassifier (TabularMLPEncoder + ClassicalHead or HybridQMLHead)
using a compound loss:

    L_total = alpha * L_distill + beta * L_task

where:
    L_distill = MSELoss(z_student, z_teacher.detach())
    L_task    = CrossEntropyLoss(logits, y_true, weight=class_weights)

The Teacher encoder is ALWAYS frozen during distillation.  Its z_teacher
vectors are computed with torch.no_grad() and detached.

Logged separately every epoch:
    - loss_distill
    - loss_task
    - loss_total
    - quantum_grad_variance (for quantum head; 0.0 for classical)
    - val_auc, val_sensitivity, val_specificity, val_kappa
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
from quantum_kc.data.image_transforms import get_eval_transforms, get_finetune_transforms
from quantum_kc.data.labeled_dataset import CornOrbDataset
from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.student_pipeline import StudentClassifier
from quantum_kc.training.utils import (
    EarlyStopping,
    compute_metrics,
    get_class_weights,
    set_seed,
)

logger = logging.getLogger(__name__)


def train_student(
    cfg: Optional[Config] = None,
    fold: int = 1,
    head_type: str = "classical",
    alpha: float = 0.5,
    beta: float = 0.5,
    seed: int = 42,
    teacher_ckpt: Optional[str] = None,
    save_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Train one Student variant for one (fold, head_type, alpha, beta, seed).

    Args:
        cfg: Configuration object.
        fold: Validation fold index (1-5).
        head_type: ``'classical'`` or ``'quantum'``.
        alpha: Weight on distillation loss (L_distill).
        beta: Weight on task loss (L_task).
        seed: Random seed for this run.
        teacher_ckpt: Path to Teacher checkpoint (fine-tuned encoder).
            Defaults to cfg.CHECKPOINT_DIR / 'teacher_finetuned_best.pt'.

    Returns:
        Dict of best validation metrics plus run identifiers.
    """
    if cfg is None:
        cfg = default_config

    set_seed(seed)
    device = get_device()
    logger.info(
        "Student training | fold=%d | head=%s | α=%.2f β=%.2f | seed=%d",
        fold, head_type, alpha, beta, seed,
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
        use_student_features=True,
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
        use_student_features=True,
    )

    labels = train_ds.labels.numpy()
    class_weights = get_class_weights(labels).to(device)
    sample_weights = [class_weights[int(l)].item() for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Teacher (frozen) ──────────────────────────────────────────────
    teacher = CornealEncoder(in_channels=3, latent_dim=cfg.LATENT_DIM).to(device)
    ckpt_path = teacher_ckpt or str(cfg.CHECKPOINT_DIR / "teacher_finetuned_best.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        # Full teacher state_dict includes supervised_head — load with strict=False
        # so the encoder backbone loads even if head key names differ
        teacher.load_state_dict(state, strict=False)
        logger.info("Loaded Teacher from %s", ckpt_path)
    else:
        logger.warning("Teacher checkpoint not found at %s — using random init.", ckpt_path)

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # ── Student ───────────────────────────────────────────────────────
    lr = cfg.LR_QUANTUM_HEAD if head_type == "quantum" else cfg.LR_STUDENT
    student = StudentClassifier(
        head_type=head_type,
        input_dim=cfg.STUDENT_INPUT_DIM,
        latent_dim=cfg.LATENT_DIM,
        n_classes=2,
        n_qubits=cfg.N_QUBITS,
        n_layers=cfg.N_LAYERS,
        dev_name=cfg.QML_DEVICE,
        diff_method=cfg.DIFF_METHOD,
    ).to(device)

    param_counts = student.count_parameters()
    logger.info("Student parameters: %s", param_counts)

    optimizer = Adam(student.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # Losses
    distill_criterion = nn.MSELoss()
    task_criterion = nn.CrossEntropyLoss(weight=class_weights)
    early_stopping = EarlyStopping(patience=12, mode="min")

    # ── MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    run_name = f"student_{head_type}_fold{fold}_a{alpha:.1f}b{beta:.1f}_s{seed}"
    best_val_loss = float("inf")
    best_metrics: Dict[str, Any] = {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "head_type": head_type,
            "fold": fold,
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            "lr": lr,
            "n_qubits": cfg.N_QUBITS,
            "n_layers": cfg.N_LAYERS,
            "student_input_dim": cfg.STUDENT_INPUT_DIM,
            "student_feature_names": ",".join(cfg.STUDENT_FEATURE_NAMES),
            "encoder_params": param_counts["encoder"],
            "head_params": param_counts["head"],
            "total_params": param_counts["total"],
        })

        for epoch in range(cfg.DISTILL_EPOCHS):
            # ── Train ─────────────────────────────────────────────────
            student.train()
            train_loss_total = 0.0
            train_loss_distill = 0.0
            train_loss_task = 0.0

            for imgs, student_tab, targets in train_loader:
                imgs = imgs.to(device)
                student_tab = student_tab.to(device)
                targets = targets.to(device)

                # Teacher embedding (no grad, frozen)
                with torch.no_grad():
                    z_teacher = teacher(imgs, return_logits=False)  # (B, 8)

                optimizer.zero_grad()

                # Student forward
                z_student, logits = student(student_tab, return_embedding=True)

                # Compound loss
                l_distill = distill_criterion(z_student, z_teacher)
                l_task = task_criterion(logits, targets)
                l_total = alpha * l_distill + beta * l_task

                l_total.backward()

                # Log quantum gradient variance BEFORE optimizer.step()
                grad_var = student.quantum_gradient_variance()

                optimizer.step()

                n = imgs.size(0)
                train_loss_total += l_total.item() * n
                train_loss_distill += l_distill.item() * n
                train_loss_task += l_task.item() * n

            n_train = len(train_ds)
            train_loss_total /= n_train
            train_loss_distill /= n_train
            train_loss_task /= n_train

            # ── Validate ──────────────────────────────────────────────
            student.eval()
            val_loss_total = 0.0
            all_preds, all_targets, all_probs = [], [], []

            with torch.no_grad():
                for imgs, student_tab, targets in val_loader:
                    imgs = imgs.to(device)
                    student_tab = student_tab.to(device)
                    targets = targets.to(device)

                    z_teacher = teacher(imgs, return_logits=False)
                    z_student, logits = student(student_tab, return_embedding=True)

                    l_distill = distill_criterion(z_student, z_teacher)
                    l_task = task_criterion(logits, targets)
                    l_total = alpha * l_distill + beta * l_task

                    val_loss_total += l_total.item() * imgs.size(0)
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

            val_loss_total /= len(val_ds)
            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
            y_prob = np.concatenate(all_probs)
            metrics = compute_metrics(y_true, y_pred, y_prob)

            scheduler.step(val_loss_total)
            mlflow.log_metrics({
                "loss_total_train": train_loss_total,
                "loss_distill_train": train_loss_distill,
                "loss_task_train": train_loss_task,
                "loss_total_val": val_loss_total,
                "val_auc": metrics.get("auc_roc") or 0.0,
                "val_sensitivity": metrics["sensitivity"],
                "val_specificity": metrics["specificity"],
                "val_kappa": metrics["kappa"],
                "quantum_grad_variance": grad_var,
            }, step=epoch)

            logger.info(
                "Ep %3d | L_tot=%.4f (L_d=%.4f L_t=%.4f) | AUC=%.4f Sens=%.4f | QGV=%.2e",
                epoch + 1, train_loss_total, train_loss_distill, train_loss_task,
                metrics.get("auc_roc") or 0.0, metrics["sensitivity"], grad_var,
            )

            tag_str = f"_{save_tag}" if save_tag else ""
            ckpt_name = f"student{tag_str}_{head_type}_fold{fold}_a{alpha:.1f}b{beta:.1f}_s{seed}.pt"
            best_ckpt_path = str(cfg.CHECKPOINT_DIR / ckpt_name)

            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_metrics = dict(metrics)
                best_metrics.update({
                    "fold": fold,
                    "head_type": head_type,
                    "alpha": alpha,
                    "beta": beta,
                    "seed": seed,
                    "save_tag": save_tag or "default",
                })
                os.makedirs(str(cfg.CHECKPOINT_DIR), exist_ok=True)
                torch.save(student.state_dict(), best_ckpt_path)

            if early_stopping(val_loss_total):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # ── Evaluate on held-out test split ───────────────────────────
        if os.path.exists(best_ckpt_path):
            student.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
        student.eval()

        test_ds = CornOrbDataset(
            csv_path=str(cfg.LABELED_CSV),
            data_root=str(cfg.LABELED_ROOT),
            split="test",
            transform=val_transform,
            tabular_scaler=train_ds.tabular_scaler,
            tabular_imputer=train_ds.tabular_imputer,
            student_pipeline=train_ds.student_pipeline,
            use_student_features=True,
        )
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2)
        test_preds, test_targets, test_probs = [], [], []
        with torch.no_grad():
            for _imgs, student_tab, targets in test_loader:
                student_tab = student_tab.to(device)
                logits = student(student_tab)
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

        mlflow.log_metrics({
            f"best_{k}": v
            for k, v in best_metrics.items()
            if isinstance(v, (int, float))
        })
        logger.info(
            "[Student] Best Val AUC: %.4f | Test AUC: %.4f | Saved: %s",
            best_metrics.get("auc_roc") or 0.0,
            best_metrics.get("test_auc_roc") or 0.0,
            best_ckpt_path,
        )

    return best_metrics
