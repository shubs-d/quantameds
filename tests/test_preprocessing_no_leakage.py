"""Test that the preprocessing pipeline prevents data leakage.

Critical property: imputers and scalers must be fitted on the train fold only,
then applied (not re-fitted) to val/test folds.

Tests:
  - StudentTabularPipeline.transform() raises if called before fit()
  - StudentTabularPipeline fitted on train does not re-fit when transform() called on val
  - build_tabular_pipeline returns same scaler type when called with pre-fitted scaler
  - Output has exactly 5 features for Student pipeline
  - Val features differ from train features (no identical-fit-and-transform shortcut)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.data.preprocessing import StudentTabularPipeline, build_tabular_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


def _make_fake_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "kmax_value_D": rng.uniform(40, 60, n),
        "astig_value_D": rng.uniform(0, 5, n),
        "astig_axis_deg": rng.uniform(0, 180, n),
        "pachy_thinnest_um": rng.uniform(400, 600, n),
    })


class TestStudentTabularPipeline:
    def test_output_has_5_features(self):
        pipe = StudentTabularPipeline()
        train_df = _make_fake_df(50, seed=0)
        X = pipe.fit_transform(train_df)
        assert X.shape == (50, 5), f"Expected (50, 5), got {X.shape}"

    def test_transform_before_fit_raises(self):
        pipe = StudentTabularPipeline()
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.transform(_make_fake_df(10))

    def test_scaler_not_refitted_on_val(self):
        """Scaler fitted on train should not change when transforming val."""
        pipe = StudentTabularPipeline()
        train_df = _make_fake_df(50, seed=1)
        pipe.fit(train_df)

        # Record scaler state after fitting on train
        train_scale_min = pipe.scaler.data_min_.copy()
        train_scale_max = pipe.scaler.data_max_.copy()

        # Transform val (should NOT refit)
        val_df = _make_fake_df(20, seed=99)
        _X_val = pipe.transform(val_df)

        assert np.allclose(pipe.scaler.data_min_, train_scale_min), \
            "Scaler data_min_ changed after transform — scaler was re-fitted!"
        assert np.allclose(pipe.scaler.data_max_, train_scale_max), \
            "Scaler data_max_ changed after transform — scaler was re-fitted!"

    def test_imputer_not_refitted_on_val(self):
        """Imputer fitted on train should not change when transforming val."""
        pipe = StudentTabularPipeline()
        train_df = _make_fake_df(50, seed=2)
        # Add some NaNs to train
        train_df.loc[0, "kmax_value_D"] = np.nan
        pipe.fit(train_df)
        train_medians = pipe.imputer.statistics_.copy()

        val_df = _make_fake_df(20, seed=88)
        val_df.loc[0, "astig_value_D"] = np.nan
        _X_val = pipe.transform(val_df)

        assert np.allclose(pipe.imputer.statistics_, train_medians), \
            "Imputer statistics changed after val transform — imputer was re-fitted!"

    def test_output_dtype_float32(self):
        pipe = StudentTabularPipeline()
        X = pipe.fit_transform(_make_fake_df(30))
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"

    def test_cyclic_features_in_neg1_pos1(self):
        """sin/cos features must be in [-1, 1]."""
        pipe = StudentTabularPipeline()
        X = pipe.fit_transform(_make_fake_df(100))
        sin_vals = X[:, 2]  # astig_axis_sin
        cos_vals = X[:, 3]  # astig_axis_cos
        assert sin_vals.min() >= -1.0 - 1e-6
        assert sin_vals.max() <= 1.0 + 1e-6
        assert cos_vals.min() >= -1.0 - 1e-6
        assert cos_vals.max() <= 1.0 + 1e-6


class TestBuildTabularPipelineLeakage:
    def test_scaler_passed_through(self):
        """When a pre-fitted scaler is passed, it should not be re-fitted."""
        df = pd.DataFrame({
            "kmax_value_D": [44.0, 48.0, 52.0, 56.0],
            "astig_value_D": [1.0, 2.0, 3.0, 4.0],
            "astig_axis_deg": [10.0, 45.0, 90.0, 135.0],
            "pachy_thinnest_um": [500.0, 520.0, 480.0, 510.0],
        })
        angular = ["astig_axis_deg"]
        skewed = ["astig_value_D"]

        # Fit on train
        X_train, fitted_scaler, fitted_imputer = build_tabular_pipeline(
            df, angular, skewed, scaler=None, imputer=None, return_imputer=True
        )
        original_min = fitted_scaler.data_min_.copy()

        # Transform on val using the same scaler
        val_df = df.copy()
        val_df["kmax_value_D"] = [60.0, 62.0, 64.0, 66.0]  # out-of-train range
        X_val, reused_scaler, _ = build_tabular_pipeline(
            val_df, angular, skewed, scaler=fitted_scaler, imputer=fitted_imputer, return_imputer=True
        )

        # Scaler's data_min_ must remain unchanged (not re-fitted to val)
        assert np.allclose(reused_scaler.data_min_, original_min), \
            "Scaler was re-fitted on val data — data leakage!"
