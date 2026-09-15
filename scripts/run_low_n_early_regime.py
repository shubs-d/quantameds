#!/usr/bin/env python3
"""Step 7: Low-N 'Super Early' Regime Testing & Learning Curves.

Carves out the subclinical/early-stage KC cohort (Kmax < 48.0 D, AK Stage 1, N=131)
paired with 131 matched Normal control eyes (total N=262).

Evaluates 5 arms across subsample sizes N in [25, 50, 100, 200, 262]:
  1. Fairly-Tuned Classical MLP (regularization/dropout tuned per N)
  2. Hybrid Dual-Rate Student (VQC + classical encoder)
  3. Quantum-Only Student (VQC alone)
  4. Quantum Kernel / QSVM (fidelity quantum kernel + SVC)
  5. Frozen Teacher Prototype / Probe (direct representation reuse on 8D latent)

Reports fold-averaged Accuracy, AUROC, and Brier score across 5 folds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config
from quantum_kc.data.preprocessing import RobustStudentTabularPipeline
from quantum_kc.models.matched_ablation_models import (
    ClassicalMLPStudent,
    HybridRobustQuantaStudent,
    QuantumOnlyStudent,
)
from quantum_kc.training.utils import set_seed
from scripts.run_matched_ablation import CachedTabularDataset, compute_ece

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Quantum Kernel Circuit (6 qubits) ─────────────────────────────────────────
dev_kernel = qml.device("lightning.qubit", wires=6)

@qml.qnode(dev_kernel)
def _kernel_qnode(x1, x2):
    qml.AngleEmbedding(x1, wires=range(6), rotation="Y")
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(6), rotation="Y")
    return qml.probs(wires=range(6))


def compute_fidelity_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Compute state fidelity kernel matrix K[i, j] = |<phi(x1_i)|phi(x2_j)>|^2."""
    K = np.zeros((len(X1), len(X2)), dtype=np.float64)
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            if X1 is X2 and i == j:
                K[i, j] = 1.0
            elif X1 is X2 and j < i:
                K[i, j] = K[j, i]
            else:
                probs = _kernel_qnode(x1, x2)
                K[i, j] = float(probs[0])
    return K


# ── Fairly-Tuned Classical MLP ────────────────────────────────────────────────
class RegularizedClassicalMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, latent_dim=8, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Hardtanh(0.0, np.pi),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(6, 1),
        )

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        logits = self.head(z).squeeze(-1)
        if return_latent:
            return z, logits
        return logits


def train_tuned_classical(train_ds, val_ds, n_samples: int, epochs: int = 15, seed: int = 42):
    """Tune weight decay and dropout based on sample size N."""
    set_seed(seed)
    # Higher regularization for smaller sample sizes
    if n_samples <= 50:
        dropout = 0.4
        weight_decay = 1e-2
    elif n_samples <= 100:
        dropout = 0.3
        weight_decay = 1e-3
    else:
        dropout = 0.2
        weight_decay = 1e-4

    model = RegularizedClassicalMLP(dropout=dropout)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        train_ds.set_mode("train")
        for x_b, z_t, y_b in loader:
            optimizer.zero_grad()
            z_pred, logits = model(x_b, return_latent=True)
            loss = 0.5 * mse_fn(z_pred, z_t) + 0.5 * bce_fn(logits, y_b)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    val_ds.set_mode("tier2")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xv, _, yv in val_loader:
            l = model(xv)
            probs = torch.sigmoid(l).numpy()
            all_probs.extend(probs)
            all_targets.extend(yv.numpy().astype(int))

    y_t = np.array(all_targets)
    y_p = np.array(all_probs)
    return {
        "accuracy": float(accuracy_score(y_t, (y_p >= 0.5).astype(int))),
        "auroc": float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_t, y_p)),
    }


def train_gradient_model(model_type, train_ds, val_ds, epochs=15, seed=42):
    set_seed(seed)
    if model_type == "hybrid":
        model = HybridRobustQuantaStudent(n_layers=2)
        cp = [p for n, p in model.named_parameters() if "vqc" not in n]
        qp = [p for n, p in model.named_parameters() if "vqc" in n]
        optimizer = Adam([{"params": cp, "lr": 1e-3}, {"params": qp, "lr": 1e-2}])
    elif model_type == "quantum_only":
        model = QuantumOnlyStudent(n_layers=4)
        optimizer = Adam(model.parameters(), lr=1e-2)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        train_ds.set_mode("train")
        for x_b, z_t, y_b in loader:
            optimizer.zero_grad()
            z_pred, logits = model(x_b, return_latent=True)
            loss = 0.5 * mse_fn(z_pred, z_t) + 0.5 * bce_fn(logits, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    val_ds.set_mode("tier2")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    all_probs, all_targets = [], []
    with torch.no_grad():
        for xv, _, yv in val_loader:
            l = model(xv)
            probs = torch.sigmoid(l).numpy()
            all_probs.extend(probs)
            all_targets.extend(yv.numpy().astype(int))

    y_t = np.array(all_targets)
    y_p = np.array(all_probs)
    return {
        "accuracy": float(accuracy_score(y_t, (y_p >= 0.5).astype(int))),
        "auroc": float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_t, y_p)),
    }


def eval_qsvm(train_x, train_y, val_x, val_y):
    """Train and evaluate Quantum Support Vector Classifier with precomputed fidelity kernel."""
    # Scale features into [0, pi]
    K_train = compute_fidelity_kernel(train_x, train_x)
    K_val = compute_fidelity_kernel(val_x, train_x)

    clf = SVC(kernel="precomputed", probability=True, C=1.0)
    clf.fit(K_train, train_y)
    probs = clf.predict_proba(K_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(val_y, preds)),
        "auroc": float(roc_auc_score(val_y, probs)) if len(np.unique(val_y)) > 1 else 0.5,
        "brier": float(brier_score_loss(val_y, probs)),
    }


def eval_teacher_prototype(train_z, train_y, val_z, val_y):
    """Evaluate few-shot representation reuse on frozen 8D Teacher latents."""
    clf = LogisticRegression(C=1.0, max_iter=200)
    clf.fit(train_z, train_y)
    probs = clf.predict_proba(val_z)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(val_y, preds)),
        "auroc": float(roc_auc_score(val_y, probs)) if len(np.unique(val_y)) > 1 else 0.5,
        "brier": float(brier_score_loss(val_y, probs)),
    }


def main():
    torch.set_num_threads(4)
    set_seed(42)

    # 1. Carve out early-stage/subclinical KC cohort
    df = pd.read_csv(config.LABELED_CSV, dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)

    kc_early = df[(df["label"] == 1) & (df["kmax_value_D"] < 48.0)].copy()
    normal_matched = df[df["label"] == 0].sample(n=len(kc_early), random_state=42).copy()
    cohort_df = pd.concat([kc_early, normal_matched]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print("=" * 80)
    print(f"STEP 7: LOW-N 'SUPER EARLY' REGIME (Kmax < 48.0 D, AK Stage 1)")
    print(f"Cohort size: {len(cohort_df)} eyes ({len(kc_early)} KC, {len(normal_matched)} Normal)")
    print("=" * 80)

    # Load cached teacher latents
    cached_data = torch.load(config.BASE_DIR / "data" / "cached_teacher_latents.pt", map_location="cpu")
    latents_by_key = cached_data["latents_by_patient_eye"]

    cohort_latents = torch.stack([
        latents_by_key[(str(r["patient_code"]), str(r["eye"]))] for _, r in cohort_df.iterrows()
    ])

    # Fit tabular pipeline
    pipe = RobustStudentTabularPipeline()
    pipe.fit(cohort_df)
    cohort_base = pipe.transform_base(cohort_df)
    cohort_y = cohort_df["label"].values.astype(int)

    # Subsample sizes to test
    subsample_sizes = [25, 50, 100, 150, 200]
    arms = ["classical_tuned", "hybrid_dual_rate", "quantum_only", "qsvm_kernel", "teacher_prototype"]
    learning_curves: Dict[str, Dict[int, Dict[str, float]]] = {a: {} for a in arms}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for n_sub in subsample_sizes:
        print(f"\n--- Testing Subsample Size N = {n_sub} across 5 Folds ---")
        fold_results: Dict[str, List[Dict[str, float]]] = {a: [] for a in arms}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cohort_base, cohort_y), start=1):
            # Subsample train indices
            rng = np.random.RandomState(42 + fold_idx * 100 + n_sub)
            if len(train_idx) > n_sub:
                sub_train_idx = rng.choice(train_idx, size=n_sub, replace=False)
            else:
                sub_train_idx = train_idx

            sub_base_train = cohort_base[sub_train_idx]
            sub_latents_train = cohort_latents[sub_train_idx]
            sub_y_train = cohort_y[sub_train_idx]

            base_val = cohort_base[val_idx]
            latents_val = cohort_latents[val_idx]
            y_val = cohort_y[val_idx]

            train_ds = CachedTabularDataset(sub_base_train, sub_latents_train, sub_y_train, mode="train", rng=rng)
            val_ds = CachedTabularDataset(base_val, latents_val, y_val, mode="tier2", rng=rng)

            # 1. Classical Tuned
            r_cls = train_tuned_classical(train_ds, val_ds, n_samples=n_sub, epochs=15, seed=42 + fold_idx)
            fold_results["classical_tuned"].append(r_cls)

            # 2. Hybrid Dual-Rate
            r_hyb = train_gradient_model("hybrid", train_ds, val_ds, epochs=15, seed=42 + fold_idx)
            fold_results["hybrid_dual_rate"].append(r_hyb)

            # 3. Quantum Only
            r_qonly = train_gradient_model("quantum_only", train_ds, val_ds, epochs=15, seed=42 + fold_idx)
            fold_results["quantum_only"].append(r_qonly)

            # 4. QSVM Kernel
            # Use 6D Tier 2 features for kernel
            val_x_6d = np.array([val_ds[i][0].numpy() for i in range(len(val_ds))])
            train_x_6d = np.array([train_ds[i][0].numpy() for i in range(len(train_ds))])
            r_qsvm = eval_qsvm(train_x_6d, sub_y_train, val_x_6d, y_val)
            fold_results["qsvm_kernel"].append(r_qsvm)

            # 5. Teacher Prototype
            r_proto = eval_teacher_prototype(sub_latents_train.numpy(), sub_y_train, latents_val.numpy(), y_val)
            fold_results["teacher_prototype"].append(r_proto)

        # Average across folds for this N
        for a in arms:
            avg_acc = float(np.mean([r["accuracy"] for r in fold_results[a]]))
            avg_auc = float(np.mean([r["auroc"] for r in fold_results[a]]))
            avg_brier = float(np.mean([r["brier"] for r in fold_results[a]]))
            learning_curves[a][n_sub] = {"accuracy": avg_acc, "auroc": avg_auc, "brier": avg_brier}
            print(f"  N={n_sub:<3} | {a:<20} -> Acc: {avg_acc:.4f} | AUROC: {avg_auc:.4f} | Brier: {avg_brier:.4f}")

    # Output table
    print("\n" + "=" * 80)
    print("LOW-N LEARNING CURVE SUMMARY (AUROC vs. N)")
    print("=" * 80)
    header = f"{'Model Arm':<22} | " + " | ".join([f"N={n}" for n in subsample_sizes])
    print(header)
    print("-" * len(header))

    table_rows = []
    for a in arms:
        row_str = f"{a:<22} | " + " | ".join([f"{learning_curves[a][n]['auroc']:.4f}" for n in subsample_sizes])
        print(row_str)
        table_rows.append({
            "Arm": a,
            **{f"AUROC_N{n}": learning_curves[a][n]["auroc"] for n in subsample_sizes},
            **{f"Acc_N{n}": learning_curves[a][n]["accuracy"] for n in subsample_sizes},
        })

    out_md = config.BASE_DIR / "results" / "low_n_learning_curves.md"
    out_json = config.BASE_DIR / "results" / "low_n_learning_curves.json"
    pd.DataFrame(table_rows).to_markdown(out_md, index=False)
    with open(out_json, "w") as f:
        json.dump(learning_curves, f, indent=2)

    print(f"\nWrote learning curves to: {out_md}")
    print(f"Wrote raw JSON to: {out_json}")


if __name__ == "__main__":
    main()
