"""Fold balance verification and patient grouping utilities.

Run verify_fold_balance as a pre-flight check before any training to confirm
that the StratifiedGroupKFold splits maintain class balance across folds.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BALANCE_TOLERANCE = 0.08  # Accept ±8 pp from overall prevalence


def verify_fold_balance(csv_path: str, label_col: str = "label", fold_col: str = "fold") -> None:
    """Load the labeled CSV and assert class balance across folds.

    Prints a per-fold balance table and raises ``AssertionError`` if any fold
    deviates more than ``_BALANCE_TOLERANCE`` from the overall KC prevalence.

    This is a pre-flight gate — run it before any training.

    Args:
        csv_path: Path to ``clinical_data_and_labels.csv``.
        label_col: Column name for the binary diagnosis label (0=Normal, 1=KC).
        fold_col: Column name for the fold assignment.
    """
    df = pd.read_csv(csv_path, dtype={"patient_code": str})
    df["patient_code"] = (
        df["patient_code"].astype(str).str.replace("3E+132", "3E132", regex=False)
    )

    if fold_col not in df.columns:
        raise ValueError(f"Column '{fold_col}' not found.  Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found.  Available: {list(df.columns)}")

    # Only look at train rows (test rows have no fold assignment)
    train_df = df[df["split_80_20"] == "train"].copy() if "split_80_20" in df.columns else df.copy()
    train_df = train_df.dropna(subset=[fold_col])

    overall_prev = train_df[label_col].mean()
    folds = sorted(train_df[fold_col].unique())

    print("\n" + "=" * 60)
    print("Fold class balance verification")
    print(f"Overall KC prevalence (train): {overall_prev:.3f} ({overall_prev*100:.1f}%)")
    print("-" * 60)
    print(f"{'Fold':>6}  {'N':>6}  {'KC':>6}  {'Prev':>8}  {'Δ from overall':>16}  {'Status':>8}")
    print("-" * 60)

    violations: List[str] = []
    for fold in folds:
        fold_df = train_df[train_df[fold_col] == fold]
        n = len(fold_df)
        n_kc = fold_df[label_col].sum()
        prev = fold_df[label_col].mean()
        delta = prev - overall_prev
        status = "OK" if abs(delta) <= _BALANCE_TOLERANCE else "FAIL"
        if status == "FAIL":
            violations.append(f"Fold {fold}: prevalence={prev:.3f} (Δ={delta:+.3f})")
        print(f"{fold:>6}  {n:>6}  {n_kc:>6}  {prev:>8.3f}  {delta:>+16.3f}  {status:>8}")

    print("=" * 60)

    # Also log patient-level grouping sanity
    if "patient_code" in df.columns:
        n_patients = train_df["patient_code"].nunique()
        n_eyes = len(train_df)
        print(f"\nPatient groups: {n_patients} patients, {n_eyes} eyes")
        # Check for patient leakage across folds
        patient_folds = (
            train_df.groupby("patient_code")[fold_col].nunique()
        )
        leaked = patient_folds[patient_folds > 1]
        if len(leaked) > 0:
            print(f"WARNING: {len(leaked)} patients appear in multiple folds — patient leakage!")
            logger.warning("%d patients appear in multiple folds.", len(leaked))
        else:
            print("Patient leakage check: PASS (each patient is in exactly one fold)")

    if violations:
        msg = "Fold balance check FAILED:\n" + "\n".join(violations)
        raise AssertionError(msg)

    print("\nFold balance check: PASS\n")


def get_patient_groups(df: pd.DataFrame, patient_col: str = "patient_code") -> np.ndarray:
    """Return an array of patient group codes for use with StratifiedGroupKFold.

    Args:
        df: DataFrame containing the patient column.
        patient_col: Column name for patient identifiers.

    Returns:
        np.ndarray of shape (n_eyes,) with patient codes as group labels.
    """
    if patient_col not in df.columns:
        raise ValueError(f"Column '{patient_col}' not found in DataFrame.")
    return df[patient_col].values


def log_test_set_stats(csv_path: str, label_col: str = "label") -> None:
    """Print the test set size and class balance for record-keeping."""
    df = pd.read_csv(csv_path, dtype={"patient_code": str})
    if "split_80_20" not in df.columns:
        return
    test_df = df[df["split_80_20"] != "train"]
    n = len(test_df)
    if n == 0:
        print("No test rows found.")
        return
    prev = test_df[label_col].mean() if label_col in test_df.columns else float("nan")
    print(f"\nHeld-out test set: {n} eyes, KC prevalence={prev:.3f} ({prev*100:.1f}%)")
    if "patient_code" in test_df.columns:
        print(f"  Unique patients in test: {test_df['patient_code'].nunique()}")
