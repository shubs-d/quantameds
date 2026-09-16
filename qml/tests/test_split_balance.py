"""Pre-flight test: verify that StratifiedGroupKFold splits are balanced.

Asserts:
  - Each patient appears in exactly one fold (no patient leakage across folds)
  - KC prevalence in each fold is within ±8pp of overall prevalence
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import config
from quantum_kc.data.split_utils import verify_fold_balance


def test_fold_balance_passes() -> None:
    """verify_fold_balance must not raise AssertionError on valid splits."""
    csv_path = str(config.LABELED_CSV)
    if not Path(csv_path).exists():
        pytest.skip(f"Labeled CSV not found at {csv_path}")
    # Should not raise
    verify_fold_balance(csv_path)


def test_csv_has_required_columns() -> None:
    """clinical_data_and_labels.csv must contain all Student feature columns."""
    import pandas as pd

    csv_path = str(config.LABELED_CSV)
    if not Path(csv_path).exists():
        pytest.skip(f"Labeled CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required = ["kmax_value_D", "astig_value_D", "astig_axis_deg", "pachy_thinnest_um",
                "label", "patient_code", "fold", "split_80_20"]
    for col in required:
        assert col in df.columns, f"Missing required column: {col}"
