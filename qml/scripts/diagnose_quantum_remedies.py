#!/usr/bin/env python3
"""Diagnose and evaluate Step 4 remedies on Fold 1.

Tests in order:
  Baseline: Current Hybrid (L=2, single LR=1e-3, alpha=0.5, beta=0.5)
  Remedy 1: Shrink depth to L=1 entangling layer (24 quantum params)
  Remedy 2: Data re-uploading circuit (2 layers with re-encoded inputs)
  Remedy 3: Dual-rate optimizer (classical lr=1e-3, quantum lr=1e-2)
  Remedy 4: Loss rebalancing (alpha=0.2, beta=0.8, dual-rate)
  Reference: Classical-only reference
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.matched_ablation_models import ClassicalMLPStudent, HybridRobustQuantaStudent
from quantum_kc.training.utils import set_seed
from scripts.run_matched_ablation import CachedTabularDataset, compute_ece


class ReuploadingHybridStudent(nn.Module):
    """Hybrid student with data re-uploading circuit."""

    def __init__(self, input_dim=6, hidden_dim=16, latent_dim=8, n_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Hardtanh(0.0, np.pi),
        )

        dev = qml.device("lightning.qubit", wires=latent_dim)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def qnode(inputs, weights):
            # Data re-uploading: re-encode features between variational layers
            for l in range(n_layers):
                qml.AngleEmbedding(inputs, wires=range(latent_dim), rotation="Y")
                qml.StronglyEntanglingLayers(weights[l].unsqueeze(0), wires=range(latent_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(latent_dim)]

        weight_shapes = {"weights": (n_layers, latent_dim, 3)}
        self.vqc_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        q_out = self.vqc_layer(z)
        logits = self.classifier(q_out).squeeze(-1)
        if return_latent:
            return z, logits
        return logits


def evaluate_candidate(name, model, optimizer, train_loader, val_loader, epochs=15, alpha=0.5, beta=0.5):
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for x_b, z_t, y_b in train_loader:
            optimizer.zero_grad()
            z_pred, logits = model(x_b, return_latent=True)
            loss = alpha * mse_fn(z_pred, z_t) + beta * bce_fn(logits, y_b)
            loss.backward()
            optimizer.step()

    # Eval Tier 2
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for x_v, _, y_v in val_loader:
            logits = model(x_v)
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_v.numpy().astype(int))

    y_t = np.array(all_targets)
    y_p = np.array(all_preds)
    y_pr = np.array(all_probs)

    acc = accuracy_score(y_t, y_p)
    auc_val = roc_auc_score(y_t, y_pr)
    brier = brier_score_loss(y_t, y_pr)
    ece = compute_ece(y_t, y_pr)

    print(f"{name:<42} | Acc: {acc:.4f} | AUROC: {auc_val:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    return {"acc": acc, "auroc": auc_val, "brier": brier, "ece": ece}


def main():
    torch.set_num_threads(4)
    set_seed(42)

    # Load data for Fold 1
    cache_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    cached_data = torch.load(cache_path, map_location="cpu")
    train_latents = cached_data["latents_train"]

    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)

    fold = 1
    train_mask = (train_df["fold"] != fold).values
    val_mask = (train_df["fold"] == fold).values

    pipe = RobustStudentTabularPipeline()
    pipe.fit(train_df[train_mask].reset_index(drop=True))
    train_base = pipe.transform_base(train_df[train_mask].reset_index(drop=True))
    val_base = pipe.transform_base(train_df[val_mask].reset_index(drop=True))

    train_ds = CachedTabularDataset(train_base, train_latents[train_mask].clone(), train_df[train_mask]["label"].values.astype(int), mode="train")
    val_ds = CachedTabularDataset(val_base, train_latents[val_mask].clone(), train_df[val_mask]["label"].values.astype(int), mode="tier2")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    print("=" * 85)
    print("STEP 4 REMEDIES DIAGNOSTIC ON FOLD 1")
    print("=" * 85)

    # Baseline Hybrid (L=2, lr=1e-3, alpha=0.5, beta=0.5)
    set_seed(42)
    m_base = HybridRobustQuantaStudent(n_layers=2)
    opt_base = Adam(m_base.parameters(), lr=1e-3)
    evaluate_candidate("Baseline Hybrid (L=2, lr=1e-3, a=0.5)", m_base, opt_base, train_loader, val_loader, epochs=15)

    # Remedy 1: Depth reduction (L=1, lr=1e-3)
    set_seed(42)
    m_rem1 = HybridRobustQuantaStudent(n_layers=1)
    opt_rem1 = Adam(m_rem1.parameters(), lr=1e-3)
    evaluate_candidate("Remedy 1: Depth L=1 (lr=1e-3)", m_rem1, opt_rem1, train_loader, val_loader, epochs=15)

    # Remedy 2: Data re-uploading circuit (L=2 re-uploading)
    set_seed(42)
    m_rem2 = ReuploadingHybridStudent(n_layers=2)
    opt_rem2 = Adam(m_rem2.parameters(), lr=1e-3)
    evaluate_candidate("Remedy 2: Re-uploading circuit (L=2)", m_rem2, opt_rem2, train_loader, val_loader, epochs=15)

    # Remedy 3: Dual-rate optimizer (classical lr=1e-3, quantum lr=1e-2)
    set_seed(42)
    m_rem3 = HybridRobustQuantaStudent(n_layers=2)
    classical_params = [p for n, p in m_rem3.named_parameters() if "vqc" not in n]
    quantum_params = [p for n, p in m_rem3.named_parameters() if "vqc" in n]
    opt_rem3 = Adam([
        {"params": classical_params, "lr": 1e-3},
        {"params": quantum_params, "lr": 1e-2},
    ])
    evaluate_candidate("Remedy 3: Dual-rate (c=1e-3, q=1e-2)", m_rem3, opt_rem3, train_loader, val_loader, epochs=15)

    # Remedy 4: Loss rebalancing (alpha=0.2, beta=0.8, dual-rate)
    set_seed(42)
    m_rem4 = HybridRobustQuantaStudent(n_layers=2)
    c_p4 = [p for n, p in m_rem4.named_parameters() if "vqc" not in n]
    q_p4 = [p for n, p in m_rem4.named_parameters() if "vqc" in n]
    opt_rem4 = Adam([
        {"params": c_p4, "lr": 1e-3},
        {"params": q_p4, "lr": 1e-2},
    ])
    evaluate_candidate("Remedy 4: Loss rebal (a=0.2, b=0.8, dual-rate)", m_rem4, opt_rem4, train_loader, val_loader, epochs=15, alpha=0.2, beta=0.8)

    # Classical reference
    set_seed(42)
    m_cls = ClassicalMLPStudent()
    opt_cls = Adam(m_cls.parameters(), lr=1e-3)
    evaluate_candidate("Classical Reference (lr=1e-3, a=0.5)", m_cls, opt_cls, train_loader, val_loader, epochs=15, alpha=0.5, beta=0.5)


if __name__ == "__main__":
    main()
