"""Statistical comparison framework for Classical vs. Hybrid-QML ablation.

Provides:
  - DeLong's test for comparing two AUC-ROC values
  - Paired bootstrap test for AUC differences
  - build_comparison_table: formats the headline comparison table
  - plot_roc_curves / plot_precision_recall_curves: multi-model overlays
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DeLong's AUC significance test
# ─────────────────────────────────────────────────────────────────────────────

def _delong_roc_variance(ground_truth: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
    """Compute AUC and variance via DeLong's method (single classifier)."""
    n_pos = int(np.sum(ground_truth == 1))
    n_neg = int(np.sum(ground_truth == 0))

    pos_scores = predictions[ground_truth == 1]
    neg_scores = predictions[ground_truth == 0]

    # Placement values
    def placement(a, b):
        # P(a > b) + 0.5*P(a == b)
        n = len(b)
        val = np.array([np.sum(a[i] > b) + 0.5 * np.sum(a[i] == b) for i in range(len(a))])
        return val / n

    theta_pos = placement(pos_scores, neg_scores)
    theta_neg = placement(neg_scores, pos_scores)

    auc = float(np.mean(theta_pos))

    # Variance components (DeLong et al. 1988)
    v_pos = np.var(theta_pos, ddof=1) if n_pos > 1 else 0.0
    v_neg = np.var(theta_neg, ddof=1) if n_neg > 1 else 0.0
    variance = v_pos / n_pos + v_neg / n_neg

    return auc, variance


def delong_test(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
) -> Dict[str, float]:
    """DeLong's test for comparing two ROC-AUC values.

    Implements the method from DeLong, DeLong & Clarke-Pearson (1988).

    Args:
        y_true: Binary ground truth labels.
        y_prob_1: Positive-class probabilities for model 1.
        y_prob_2: Positive-class probabilities for model 2.

    Returns:
        Dict with auc1, auc2, delta_auc, z_stat, p_value, significant_at_05.
    """
    p1 = y_prob_1 if y_prob_1.ndim == 1 else y_prob_1[:, 1]
    p2 = y_prob_2 if y_prob_2.ndim == 1 else y_prob_2[:, 1]

    auc1, var1 = _delong_roc_variance(y_true, p1)
    auc2, var2 = _delong_roc_variance(y_true, p2)

    delta = auc1 - auc2
    se = np.sqrt(var1 + var2)
    z = delta / se if se > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    return {
        "auc1": float(auc1),
        "auc2": float(auc2),
        "delta_auc": float(delta),
        "z_stat": float(z),
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auc_difference(
    y_true: np.ndarray,
    y_prob_1: np.ndarray,
    y_prob_2: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap test for AUC difference (model1 - model2).

    Args:
        y_true: Binary ground truth.
        y_prob_1: Probabilities for model 1.
        y_prob_2: Probabilities for model 2.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with observed_delta, ci_lower, ci_upper, p_value, significant_at_05.
    """
    from sklearn.metrics import roc_auc_score

    p1 = y_prob_1 if y_prob_1.ndim == 1 else y_prob_1[:, 1]
    p2 = y_prob_2 if y_prob_2.ndim == 1 else y_prob_2[:, 1]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    obs_auc1 = roc_auc_score(y_true, p1)
    obs_auc2 = roc_auc_score(y_true, p2)
    observed_delta = obs_auc1 - obs_auc2

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            a1 = roc_auc_score(yt, p1[idx])
            a2 = roc_auc_score(yt, p2[idx])
            deltas.append(a1 - a2)
        except ValueError:
            continue

    deltas = np.array(deltas)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    # Two-sided p-value: fraction of bootstrap deltas with same sign as null (0)
    p_value = float(np.mean(np.abs(deltas) >= np.abs(observed_delta)))

    return {
        "auc1": float(obs_auc1),
        "auc2": float(obs_auc2),
        "observed_delta": float(observed_delta),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "n_bootstrap": n_bootstrap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results_path: str) -> pd.DataFrame:
    """Load ablation_results.json and build the headline comparison table.

    The table has one row per variant × (alpha,beta) condition, with columns:
        Variant, alpha, beta, AUC (mean±std), Sensitivity (mean±std),
        Specificity (mean±std), Kappa (mean±std), N_runs

    Args:
        results_path: Path to ablation_results.json.

    Returns:
        pd.DataFrame with formatted mean±std columns.
    """
    with open(results_path) as f:
        all_results = json.load(f)

    rows = []

    for variant, results in all_results.items():
        if not results:
            continue

        # Group by (alpha, beta)
        grouped: Dict[Tuple, List] = {}
        for r in results:
            key = (r.get("alpha", "N/A"), r.get("beta", "N/A"))
            grouped.setdefault(key, []).append(r)

        for (alpha, beta), group in grouped.items():
            def _stat(key: str) -> str:
                vals = [r[key] for r in group if r.get(key) is not None]
                if not vals:
                    return "N/A"
                return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

            rows.append({
                "Variant": variant,
                "alpha": alpha,
                "beta": beta,
                "AUC-ROC": _stat("auc_roc"),
                "Sensitivity": _stat("sensitivity"),
                "Specificity": _stat("specificity"),
                "Kappa": _stat("kappa"),
                "N_runs": len(group),
            })

    df = pd.DataFrame(rows)
    return df


def print_comparison_table(results_path: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """Build, print, and optionally save the comparison table.

    Args:
        results_path: Path to ablation_results.json.
        output_dir: If provided, saves comparison_table.csv and .md here.

    Returns:
        The comparison DataFrame.
    """
    df = build_comparison_table(results_path)

    print("\n" + "=" * 90)
    print("COMPARISON TABLE — Classical vs. Hybrid-QML Student Distillation")
    print("(mean ± std across seeds × folds)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
        df.to_markdown(os.path.join(output_dir, "comparison_table.md"), index=False)
        logger.info("Saved comparison table to %s", output_dir)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: str,
    title: str = "ROC Curves — Classical vs. Hybrid-QML Student",
) -> None:
    """Plot overlaid ROC curves for multiple models.

    Args:
        curves: Dict mapping model name → (fpr, tpr, auc).
        output_path: Where to save the figure.
        title: Plot title.
    """
    plt.figure(figsize=(7, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate (1 − Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved ROC curves to %s", output_path)


def plot_precision_recall_curves(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "Precision-Recall Curves",
) -> None:
    """Plot overlaid PR curves for multiple models.

    Args:
        curves: Dict mapping model name → (precision, recall).
        output_path: Where to save the figure.
        title: Plot title.
    """
    plt.figure(figsize=(7, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for i, (name, (precision, recall)) in enumerate(curves.items()):
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2, label=name)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved PR curves to %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Plain-language finding generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_finding(
    classical_aucs: List[float],
    quantum_aucs: List[float],
    significance: Dict[str, Any],
) -> str:
    """Generate a 2–4 sentence plain-language finding.

    This is the required honest statement regardless of direction of the result.

    Args:
        classical_aucs: AUC values for the classical head (across seeds/folds).
        quantum_aucs: AUC values for the quantum head (across seeds/folds).
        significance: Output of delong_test or bootstrap_auc_difference.

    Returns:
        A 2-4 sentence plain-language finding string.
    """
    c_mean = np.mean(classical_aucs)
    c_std = np.std(classical_aucs)
    q_mean = np.mean(quantum_aucs)
    q_std = np.std(quantum_aucs)
    delta = q_mean - c_mean
    sig = significance.get("significant_at_05", False)
    p_val = significance.get("p_value", float("nan"))

    direction = "outperforms" if delta > 0 else "underperforms"
    if np.isnan(p_val):
        sig_phrase = "p-value N/A (single run; multi-seed required)"
    elif sig:
        sig_phrase = f"statistically significant (p={p_val:.3f})"
    else:
        sig_phrase = f"not statistically significant (p={p_val:.3f})"

    if abs(delta) < 0.005:
        finding = (
            f"The hybrid quantum bottleneck matches the classical linear head on this task: "
            f"AUC {q_mean:.3f} ± {q_std:.3f} vs. {c_mean:.3f} ± {c_std:.3f} "
            f"(difference {delta:+.3f}, {sig_phrase}). "
            f"At equal depth (2 StronglyEntanglingLayers) the quantum circuit provides "
            f"no measurable advantage over a simple linear classifier for this dataset size "
            f"and feature dimensionality. "
            f"This is a legitimate and interpretable finding: the quantum bottleneck explored "
            f"here matches classical performance, positioning it as a forward-looking direction "
            f"rather than a current-generation improvement."
        )
    else:
        finding = (
            f"The hybrid quantum bottleneck {direction} the classical linear head: "
            f"AUC {q_mean:.3f} ± {q_std:.3f} vs. {c_mean:.3f} ± {c_std:.3f} "
            f"(Δ={delta:+.3f}, {sig_phrase}). "
            f"{'This advantage should be interpreted cautiously given the small dataset size (N≈1,454 eyes) and the parameter asymmetry between the two heads.' if sig else 'The difference is not statistically significant and should not be reported as a definitive quantum advantage.'} "
            f"Both variants demonstrate that tabular-only distillation can recover meaningful "
            f"diagnostic signal from the Teacher image encoder."
        )
    return finding
