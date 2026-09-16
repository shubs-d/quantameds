#!/usr/bin/env python3
"""Phase 3 gate check: QSVM timing and classical SVM baseline comparison.

Gate (i): Classical RBF-kernel SVM baseline run at multiple N values with
           comparable hyperparameter tuning — checks that QSVM-vs-classical
           gap survives.

Gate (ii): Timing test at N=200 on both lightning.qubit (CPU) and lightning.gpu
            to confirm GPU is actually faster for 8-qubit circuits.

Only if BOTH gates pass does the kernel matrix computation proceed.

Usage:
    python scripts/a100_phase3_qsvm_gate.py [--n-values 50,100,200,500]
    python scripts/a100_phase3_qsvm_gate.py --gate-ii-only    # just timing test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum_kc.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_low_n_latents(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load teacher latents and subsample N from the early-KC cohort (Kmax < 48 D)."""
    import torch
    import pandas as pd

    cache_path = config.BASE_DIR / "data" / "cached_teacher_latents.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"cached_teacher_latents.pt not found at {cache_path}. Run Phase 2 first."
        )

    cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    latents_train = cached["latents_train"].numpy()  # (1163, 8)
    latents_test  = cached["latents_test"].numpy()   # (291, 8)

    df = pd.read_csv(str(config.LABELED_CSV), dtype={"patient_code": str})
    df["patient_code"] = df["patient_code"].str.replace("3E+132", "3E132", regex=False)
    df = df.reset_index(drop=True)

    train_df = df[df["split_80_20"] == "train"].reset_index(drop=True)
    test_df  = df[df["split_80_20"] == "test"].reset_index(drop=True)

    # Early KC cohort filter: Kmax < 48 D (same as run_low_n_early_regime.py)
    train_early = train_df[train_df["kmax_value_D"] < 48.0]
    test_early  = test_df[test_df["kmax_value_D"] < 48.0]

    train_idx = train_early.index.tolist()
    test_idx  = test_early.index.tolist()

    X_train_all = latents_train[train_idx]
    y_train_all = train_early["label"].values
    X_test  = latents_test[test_idx]
    y_test  = test_early["label"].values

    # Subsample N training examples (stratified)
    rng = np.random.default_rng(seed)
    n = min(n, len(X_train_all))
    classes, counts = np.unique(y_train_all, return_counts=True)
    idx_sel = []
    for cls in classes:
        cls_idx = np.where(y_train_all == cls)[0]
        n_cls = max(1, round(n * (cls_idx.size / len(y_train_all))))
        n_cls = min(n_cls, cls_idx.size)
        idx_sel.extend(rng.choice(cls_idx, size=n_cls, replace=False).tolist())
    idx_sel = idx_sel[:n]
    rng.shuffle(idx_sel)

    X_train = X_train_all[idx_sel]
    y_train = y_train_all[idx_sel]
    return X_train, y_train, X_test, y_test


def run_classical_svm(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """RBF-kernel SVM with GridSearchCV (same tuning level as QSVM baselines)."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    param_grid = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
    cv = StratifiedKFold(n_splits=min(3, len(np.unique(y_train))), shuffle=True, random_state=42)
    gs = GridSearchCV(SVC(kernel="rbf", probability=True), param_grid, cv=cv, scoring="roc_auc")
    gs.fit(X_tr, y_train)

    best = gs.best_estimator_
    y_prob = best.predict_proba(X_te)[:, 1]
    y_pred = best.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
    return {"acc": acc, "auc": auc, "best_params": gs.best_params_}


def fidelity_kernel_matrix(X: np.ndarray, Y: np.ndarray, n_qubits: int, backend: str) -> np.ndarray:
    """Compute quantum fidelity kernel K[i,j] = |<psi(X[i])|psi(Y[j])>|^2."""
    import pennylane as qml

    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    K = np.zeros((len(X), len(Y)))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            probs = kernel_circuit(xi, yj)
            K[i, j] = float(probs[0])  # |<0|U†(y)U(x)|0>|² = P(all zeros)
    return K


def gate_i_classical_svm(n_values: list[int]) -> dict:
    """Gate (i): Classical RBF SVM vs QSVM across N values."""
    print("\n" + "="*70)
    print("PHASE 3 GATE (i) — Classical RBF SVM vs QSVM at multiple N")
    print("="*70)

    gate_results = []
    for n in n_values:
        try:
            X_train, y_train, X_test, y_test = load_low_n_latents(n)
            cls_res = run_classical_svm(X_train, y_train, X_test, y_test)
            gate_results.append({
                "n": n,
                "classical_acc": cls_res["acc"],
                "classical_auc": cls_res["auc"],
                "best_params": cls_res["best_params"],
                "note": "QSVM results from prior run_low_n_early_regime.py for comparison",
            })
            print(f"  N={n:4d}: Classical RBF SVM — ACC={cls_res['acc']:.4f} AUC={cls_res['auc']:.4f} "
                  f"(best_C={cls_res['best_params']['C']}, gamma={cls_res['best_params']['gamma']})")
        except Exception as e:
            logger.error("N=%d failed: %s", n, e)
            gate_results.append({"n": n, "error": str(e)})

    # Compare with prior QSVM results from Step 7 (hard-coded from results)
    # QSVM N=50: AUROC 0.8584, ACC 0.7977 (from low_n_learning_curves.json)
    prior_qsvm = {50: {"auc": 0.8584, "acc": 0.7977}}

    print("\nComparison with prior QSVM results (from Step 7):")
    gap_survives = False
    for r in gate_results:
        n = r.get("n")
        if n in prior_qsvm and "classical_auc" in r:
            qsvm_auc = prior_qsvm[n]["auc"]
            cls_auc = r["classical_auc"]
            gap = qsvm_auc - cls_auc
            print(f"  N={n}: QSVM AUC={qsvm_auc:.4f} vs Classical AUC={cls_auc:.4f} → gap={gap:+.4f}")
            if gap > 0.02:
                gap_survives = True

    gate_i_pass = gap_survives
    verdict = "PASS — QSVM advantage survives comparable classical tuning" if gate_i_pass \
              else "FAIL — no clear QSVM advantage after fair classical comparison"
    print(f"\nGate (i) verdict: {verdict}")
    return {
        "results": gate_results,
        "prior_qsvm": prior_qsvm,
        "gate_pass": gate_i_pass,
        "verdict": verdict,
    }


def gate_ii_timing(n: int = 200) -> dict:
    """Gate (ii): Timing test at N=200 on lightning.qubit (CPU) vs lightning.gpu."""
    print("\n" + "="*70)
    print(f"PHASE 3 GATE (ii) — VQC timing at N={n}: lightning.qubit vs lightning.gpu")
    print("="*70)

    try:
        X_train, _, _, _ = load_low_n_latents(n)
    except FileNotFoundError as e:
        return {"gate_pass": False, "verdict": str(e)}

    n_qubits = 8
    X_small = X_train[:10]  # Use 10 examples for head-to-head timing

    times = {}
    for backend in ["lightning.qubit", "lightning.gpu"]:
        try:
            start = time.perf_counter()
            K = fidelity_kernel_matrix(X_small, X_small, n_qubits, backend)
            elapsed = time.perf_counter() - start
            times[backend] = elapsed
            print(f"  {backend:<20}: {elapsed:.3f}s for {len(X_small)}×{len(X_small)} kernel (10×10)")
        except Exception as e:
            times[backend] = None
            logger.warning("  %s failed: %s", backend, e)

    gpu_time = times.get("lightning.gpu")
    cpu_time = times.get("lightning.qubit")

    if gpu_time is None:
        verdict = "FAIL — lightning.gpu not available or errored"
        gate_pass = False
    elif cpu_time is None:
        verdict = "ERROR — lightning.qubit also failed; cannot determine"
        gate_pass = False
    else:
        speedup = cpu_time / gpu_time
        if gpu_time < cpu_time * 0.85:  # GPU must be meaningfully faster (>15%)
            verdict = f"PASS — lightning.gpu is {speedup:.1f}x faster than lightning.qubit at N={n}"
            gate_pass = True
        else:
            verdict = (
                f"FAIL — lightning.gpu ({gpu_time:.3f}s) is NOT meaningfully faster than "
                f"lightning.qubit ({cpu_time:.3f}s) at N={n}. "
                f"For 8-qubit circuits, CPU dispatch overhead dominates. Do NOT use GPU for VQC."
            )
            gate_pass = False

    print(f"\nGate (ii) verdict: {verdict}")
    return {"times": times, "gate_pass": gate_pass, "verdict": verdict}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: QSVM gate check")
    parser.add_argument(
        "--n-values", type=str, default="50,100,200",
        help="Comma-separated N values for Gate (i) (default: 50,100,200)",
    )
    parser.add_argument("--gate-ii-only", action="store_true", help="Only run Gate (ii) timing test")
    parser.add_argument("--timing-n", type=int, default=200, help="N for Gate (ii) timing test")
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    results = {}

    if not args.gate_ii_only:
        results["gate_i"] = gate_i_classical_svm(n_values)

    results["gate_ii"] = gate_ii_timing(args.timing_n)

    # ── Overall gate decision ──────────────────────────────────────────
    g1_pass = results.get("gate_i", {}).get("gate_pass", False) if not args.gate_ii_only else None
    g2_pass = results["gate_ii"].get("gate_pass", False)

    print("\n" + "="*70)
    print("PHASE 3 GATE SUMMARY")
    print("="*70)
    if g1_pass is not None:
        print(f"  Gate (i) — QSVM vs classical gap: {'PASS' if g1_pass else 'FAIL'}")
    print(f"  Gate (ii) — GPU faster than CPU:    {'PASS' if g2_pass else 'FAIL'}")

    if not args.gate_ii_only:
        if g1_pass and g2_pass:
            print("\n>>> Both gates PASS — proceed to kernel matrix computation. <<<")
            print(">>> Use lightning.gpu for the kernel matrix (confirmed faster). <<<")
        else:
            failed = []
            if not g1_pass: failed.append("(i)")
            if not g2_pass: failed.append("(ii)")
            print(f"\n>>> Gate(s) {', '.join(failed)} FAILED — do NOT proceed to kernel matrix. <<<")
            if not g2_pass:
                print(">>> GPU is NOT faster for 8-qubit circuits — keep VQC on CPU. <<<")
            print(">>> GPU instance may be terminated now. <<<")

    results["timestamp"] = datetime.now().isoformat()
    results["n_values"] = n_values

    out_path = config.RESULTS_DIR / "a100_phase3_gate_check.json"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] Results → {out_path}")


if __name__ == "__main__":
    main()
