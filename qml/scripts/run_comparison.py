"""Generate the comparison table, significance tests, and plots.

Run after run_distill.py has completed the ablation:
    python scripts/run_comparison.py

Reads results/ablation_results.json and produces:
  - results/comparison_table.csv
  - results/comparison_table.md
  - results/roc_comparison.png
  - results/pr_comparison.png
  - results/significance_tests.json
  - A printed plain-language finding
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from quantum_kc.config import Config, config as default_config
from quantum_kc.evaluation.comparison import (
    build_comparison_table,
    print_comparison_table,
    delong_test,
    bootstrap_auc_difference,
    generate_finding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = default_config
    results_path = str(cfg.RESULTS_DIR / "ablation_results.json")
    output_dir = str(cfg.RESULTS_DIR)

    if not Path(results_path).exists():
        print(f"ERROR: {results_path} not found.  Run run_distill.py first.")
        sys.exit(1)

    with open(results_path) as f:
        all_results = json.load(f)

    # ── Comparison table ──────────────────────────────────────────────
    df = print_comparison_table(results_path, output_dir=output_dir)

    # ── Significance testing on balanced (alpha=beta=0.5) variant ─────
    classical = [
        r for r in all_results.get("classical", [])
        if r.get("alpha") == 0.5
    ]
    quantum = [
        r for r in all_results.get("quantum", [])
        if r.get("alpha") == 0.5
    ]

    classical_aucs = [r["auc_roc"] for r in classical if r.get("auc_roc") is not None]
    quantum_aucs = [r["auc_roc"] for r in quantum if r.get("auc_roc") is not None]

    sig_results = {}

    if classical_aucs and quantum_aucs:
        # Two-sample t-test on seed-level AUC differences (if n > 1)
        from scipy import stats as sp_stats
        if len(classical_aucs) > 1 and len(quantum_aucs) > 1:
            t_stat, p_val = sp_stats.ttest_ind(quantum_aucs, classical_aucs, equal_var=False)
            t_stat = float(t_stat)
            p_val = float(p_val)
        else:
            t_stat = float("nan")
            p_val = float("nan")

        sig_results["welch_t_test"] = {
            "t_stat": t_stat,
            "p_value": p_val,
            "significant_at_05": (not np.isnan(p_val)) and (p_val < 0.05),
            "classical_auc_mean": float(np.mean(classical_aucs)),
            "classical_auc_std": float(np.std(classical_aucs)),
            "quantum_auc_mean": float(np.mean(quantum_aucs)),
            "quantum_auc_std": float(np.std(quantum_aucs)),
            "delta": float(np.mean(quantum_aucs) - np.mean(classical_aucs)),
            "n_runs": len(classical_aucs),
        }

        print("\n=== Significance Test (Welch t-test, quantum vs classical AUC) ===")
        for k, v in sig_results["welch_t_test"].items():
            print(f"  {k}: {v}")

        # ── Plain-language finding ─────────────────────────────────────
        finding = generate_finding(classical_aucs, quantum_aucs, sig_results["welch_t_test"])
        print("\n=== FINDING ===")
        print(finding)
        sig_results["plain_language_finding"] = finding

    # ── LR baseline summary ───────────────────────────────────────────
    lr_results = all_results.get("logistic_baseline", [])
    if lr_results:
        lr_aucs = [r["auc_roc"] for r in lr_results if r.get("auc_roc") is not None]
        print(f"\nNaive LR baseline AUC: {np.mean(lr_aucs):.3f} ± {np.std(lr_aucs):.3f}")
        sig_results["lr_baseline_auc"] = {
            "mean": float(np.mean(lr_aucs)),
            "std": float(np.std(lr_aucs)),
        }

    # ── Save significance results ──────────────────────────────────────
    sig_path = str(Path(output_dir) / "significance_tests.json")
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2, default=str)
    print(f"\nSignificance results saved to {sig_path}")


if __name__ == "__main__":
    main()
