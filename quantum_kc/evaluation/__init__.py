"""Evaluation, comparison, and inference logic."""

from .evaluate import evaluate_model, evaluate_teacher, evaluate_student
from .comparison import (
    delong_test,
    bootstrap_auc_difference,
    build_comparison_table,
    print_comparison_table,
    plot_roc_curves,
    plot_precision_recall_curves,
    generate_finding,
)

__all__ = [
    "evaluate_model",
    "evaluate_teacher",
    "evaluate_student",
    "delong_test",
    "bootstrap_auc_difference",
    "build_comparison_table",
    "print_comparison_table",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "generate_finding",
]
