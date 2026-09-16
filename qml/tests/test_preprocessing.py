"""Unit tests for tabular preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from quantum_kc.data.preprocessing import (
    encode_periodic_axes,
    log_transform_skewed,
    scale_to_quantum_range,
    impute_missing,
    build_tabular_pipeline,
)


def test_encode_periodic_axes():
    """Verify that angular columns are mapped to sin(2θ) and cos(2θ)."""
    # 0 deg: sin(0)=0, cos(0)=1
    # 45 deg: sin(90°)=1, cos(90°)=0
    # 90 deg: sin(180°)=0, cos(180°)=-1
    # 180 deg: sin(360°)=0, cos(360°)=1
    df = pd.DataFrame({
        "axis": [0.0, 45.0, 90.0, 180.0],
        "other": [1, 2, 3, 4],
    })
    encoded = encode_periodic_axes(df, ["axis"])

    assert "axis" not in encoded.columns
    assert "axis_sin" in encoded.columns
    assert "axis_cos" in encoded.columns

    np.testing.assert_allclose(encoded["axis_sin"].values, [0.0, 1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(encoded["axis_cos"].values, [1.0, 0.0, -1.0, 1.0], atol=1e-6)


def test_log_transform_skewed():
    """Verify log1p transform applied on skewed clinical markers."""
    df = pd.DataFrame({
        "irreg": [0.0, 1.0, 9.0],
        "non_skewed": [10.0, 20.0, 30.0],
    })
    transformed = log_transform_skewed(df, ["irreg"])

    assert "irreg" in transformed.columns
    np.testing.assert_allclose(
        transformed["irreg"].values,
        [np.log1p(0.0), np.log1p(1.0), np.log1p(9.0)],
    )


def test_scale_to_quantum_range():
    """Verify that all features are scaled strictly to [0, π]."""
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [5.0, 50.0],
    ])
    scaled, scaler = scale_to_quantum_range(X)

    assert scaled.shape == X.shape
    assert np.all(scaled >= 0.0)
    assert np.all(scaled <= np.pi)

    # Test transform on new data with existing scaler
    X_new = np.array([[3.0, 30.0]])
    scaled_new, _ = scale_to_quantum_range(X_new, scaler=scaler)
    assert np.all(scaled_new >= 0.0)
    assert np.all(scaled_new <= np.pi)


def test_impute_missing():
    """Verify that missing NaNs are imputed using median values."""
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],  # median is 3.0
        "B": [10.0, np.nan, 30.0, 40.0, 50.0],  # median is 35.0
    })
    imputed = impute_missing(df, strategy="median")

    assert not imputed.isna().any().any()
    assert imputed.loc[2, "A"] == 3.0
    assert imputed.loc[1, "B"] == 35.0


def test_build_tabular_pipeline():
    """Test full tabular pipeline orchestration."""
    df = pd.DataFrame({
        "axis_deg": [0.0, 90.0, 45.0, np.nan],
        "skewed_val": [0.0, 5.0, np.nan, 10.0],
        "regular_val": [100.0, 200.0, 300.0, 400.0],
    })
    X_scaled, scaler = build_tabular_pipeline(
        df,
        angular_cols=["axis_deg"],
        skewed_cols=["skewed_val"],
    )

    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape[0] == 4
    # axis_deg replaced by axis_deg_sin and axis_deg_cos -> 4 features total
    assert X_scaled.shape[1] == 4
    assert np.all(X_scaled >= 0.0)
    assert np.all(X_scaled <= np.pi)
