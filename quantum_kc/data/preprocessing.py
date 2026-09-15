"""Tabular feature preprocessing pipeline.

Two pipelines are provided:

1. ``build_tabular_pipeline`` – original full-feature pipeline used by the
   Teacher CNN and the MultimodalHybridClassifier.  *Backward-compatible.*

2. ``StudentTabularPipeline`` – strict 5-feature pipeline used exclusively by
   the Student ``TabularMLPEncoder``.  The feature set is locked by
   ``Config.STUDENT_FEATURE_NAMES`` and must not be changed without
   updating documentation and all downstream scripts.

Leakage prevention (critical):
   Both pipelines expose ``fit`` / ``transform`` semantics.  Callers MUST:
     - Call ``fit`` (or ``fit_transform``) on the *training fold only*.
     - Call ``transform`` on val/test folds using the *same fitted instance*.
   The old ``build_tabular_pipeline`` function returns a fitted
   ``SimpleImputer`` alongside the fitted ``MinMaxScaler`` so callers can
   thread both through to val/test sets without re-fitting.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode_periodic_axes(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Encode angular features using sine and cosine of twice the angle.

    Replaces each angular column ``col`` with ``col_sin = sin(2θ)`` and
    ``col_cos = cos(2θ)`` (θ in radians), then drops the original column.
    This produces a continuous, cyclic encoding with period π (appropriate for
    astigmatism axes, which are undefined modulo 180°).
    """
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns:
            theta_rad = np.deg2rad(df_encoded[col])
            df_encoded[f"{col}_sin"] = np.sin(2 * theta_rad)
            df_encoded[f"{col}_cos"] = np.cos(2 * theta_rad)
            df_encoded.drop(columns=[col], inplace=True)
            logger.debug("Encoded and dropped periodic column: %s", col)
        else:
            logger.warning("Column %s not found for periodic encoding.", col)
    return df_encoded


def log_transform_skewed(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log1p(|x|) transformation to specified skewed columns."""
    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(np.abs(df_transformed[col]))
            logger.debug("Log-transformed skewed column: %s", col)
        else:
            logger.warning("Column %s not found for log transform.", col)
    return df_transformed


def impute_missing(
    df: pd.DataFrame,
    imputer: Optional[SimpleImputer] = None,
    strategy: str = "median",
    return_imputer: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, SimpleImputer]]:
    """Impute missing values.

    Args:
        df: Input dataframe.
        imputer: Pre-fitted SimpleImputer. Pass ``None`` to fit a new one
            (training fold). Pass a fitted imputer for val/test folds to
            prevent data leakage.
        strategy: Imputation strategy when creating a new imputer.
        return_imputer: If True, return (df_imputed, imputer). If False,
            return df_imputed (for backward compatibility).

    Returns:
        df_imputed if return_imputer=False, else (df_imputed, imputer).
    """
    if imputer is None:
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = imputer.fit_transform(df)
        logger.debug("Fitted new SimpleImputer (strategy=%s) and transformed.", strategy)
    else:
        X_imputed = imputer.transform(df)
        logger.debug("Transformed using pre-fitted SimpleImputer.")
    df_imputed = pd.DataFrame(X_imputed, columns=df.columns, index=df.index)
    if return_imputer:
        return df_imputed, imputer
    return df_imputed


def scale_to_quantum_range(
    X: Union[pd.DataFrame, np.ndarray],
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale tabular features to range [0, π] for quantum angle embedding.

    Args:
        X: Data to scale.
        scaler: Pre-fitted MinMaxScaler. Pass ``None`` to fit a new one.

    Returns:
        Tuple of (scaled_array, fitted_scaler).
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_scaled = scaler.fit_transform(X)
        logger.debug("Fitted new MinMaxScaler([0, π]) and transformed.")
    else:
        X_scaled = scaler.transform(X)
        logger.debug("Transformed using pre-fitted MinMaxScaler.")
    return X_scaled, scaler


def scale_standard(
    X: Union[pd.DataFrame, np.ndarray],
    scaler: Optional[MinMaxScaler] = None,
) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale tabular features to [0, 1]. Used for Student features (not QML).

    The Student MLP learns its own internal representation; we feed raw [0,1]
    scaled features rather than [0, π], which is reserved for quantum angle
    embedding.
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        logger.debug("Fitted new MinMaxScaler([0, 1]) and transformed.")
    else:
        X_scaled = scaler.transform(X)
        logger.debug("Transformed using pre-fitted MinMaxScaler([0, 1]).")
    return X_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Full-feature pipeline (backward-compatible, used by Teacher / multimodal)
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular_pipeline(
    df: pd.DataFrame,
    angular_cols: List[str],
    skewed_cols: List[str],
    scaler: Optional[MinMaxScaler] = None,
    imputer: Optional[SimpleImputer] = None,
    return_imputer: bool = False,
) -> Union[Tuple[np.ndarray, MinMaxScaler], Tuple[np.ndarray, MinMaxScaler, SimpleImputer]]:
    """Orchestrate the full preprocessing pipeline.

    Args:
        df: Raw tabular dataframe (numeric columns only).
        angular_cols: Column names to encode as (sin(2θ), cos(2θ)).
        skewed_cols: Column names to apply log1p transform.
        scaler: Pre-fitted MinMaxScaler for val/test folds (``None`` = fit new).
        imputer: Pre-fitted SimpleImputer for val/test folds (``None`` = fit new).
        return_imputer: If True, return (X_scaled, fitted_scaler, fitted_imputer).
            If False, return (X_scaled, fitted_scaler) with fitted_scaler.imputer_ set.

    Returns:
        (X_scaled, fitted_scaler) or (X_scaled, fitted_scaler, fitted_imputer).
    """
    logger.info("Starting full tabular preprocessing pipeline.")
    df_processed, fitted_imputer = impute_missing(df, imputer=imputer, return_imputer=True)
    df_processed = encode_periodic_axes(df_processed, angular_cols)
    df_processed = log_transform_skewed(df_processed, skewed_cols)
    X_scaled, fitted_scaler = scale_to_quantum_range(df_processed, scaler)
    fitted_scaler.imputer_ = fitted_imputer
    logger.info("Finished full tabular preprocessing pipeline. Shape: %s", X_scaled.shape)
    if return_imputer:
        return X_scaled, fitted_scaler, fitted_imputer
    return X_scaled, fitted_scaler


# ─────────────────────────────────────────────────────────────────────────────
# Student tabular pipeline (strict 5-feature contract)
# ─────────────────────────────────────────────────────────────────────────────

class StudentTabularPipeline:
    """Preprocessing pipeline strictly for the 5 Student input features.

    Feature set (FIXED — do not change without updating docs and all scripts):
        Index 0: kmax_value_D          (min-max scaled to [0, 1])
        Index 1: astig_value_D_log     (log1p applied, then min-max scaled)
        Index 2: astig_axis_sin        (sin(2 * astig_axis_deg in radians))
        Index 3: astig_axis_cos        (cos(2 * astig_axis_deg in radians))
        Index 4: pachy_thinnest_um     (min-max scaled to [0, 1])

    The cyclic (sin, cos) features are already in [-1, 1] and do NOT need
    further scaling.  The scaler is fit only on the scalar/skewed features.

    Usage (leakage-safe):
        pipe = StudentTabularPipeline()
        X_train = pipe.fit_transform(train_df)
        X_val   = pipe.transform(val_df)
        X_test  = pipe.transform(test_df)
    """

    SCALAR_COLS = ["kmax_value_D", "pachy_thinnest_um"]
    SKEWED_COLS = ["astig_value_D"]
    ANGULAR_COL = "astig_axis_deg"
    REQUIRED_COLS = SCALAR_COLS + SKEWED_COLS + [ANGULAR_COL]
    OUTPUT_DIM = 5  # Must always equal 5

    def __init__(self) -> None:
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[MinMaxScaler] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "StudentTabularPipeline":
        """Fit imputer and scaler on a training dataframe.

        Args:
            df: DataFrame that must contain all ``REQUIRED_COLS``.

        Returns:
            self (for method chaining).
        """
        self._validate_columns(df)
        subset = df[self.REQUIRED_COLS].copy()

        # 1. Impute (fit on train)
        self._imputer = SimpleImputer(strategy="median")
        self._imputer.fit(subset)

        # 2. Derive features on imputed data (no scaler fitted yet)
        processed = self._apply_transforms(subset, fitting=True)

        # 3. Fit scaler on scalar + skewed columns only (cyclic cols excluded)
        scalar_skewed_cols = self.SCALAR_COLS + [f"{c}_log" for c in self.SKEWED_COLS]
        # Only the columns that are present after transforms
        scale_targets = [c for c in scalar_skewed_cols if c in processed.columns]
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(processed[scale_targets])

        self._fitted = True
        logger.info(
            "StudentTabularPipeline fitted. Output columns: %s",
            self._get_output_col_order(processed),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe using the fitted pipeline.

        Args:
            df: DataFrame containing all ``REQUIRED_COLS``.

        Returns:
            np.ndarray of shape (n_samples, 5).
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() before transform().")
        self._validate_columns(df)
        subset = df[self.REQUIRED_COLS].copy()
        processed = self._apply_transforms(subset, fitting=False)
        return self._assemble_output(processed)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one pass (for training fold only).

        Args:
            df: Training DataFrame.

        Returns:
            np.ndarray of shape (n_samples, 5).
        """
        self.fit(df)
        return self.transform(df)

    # ── Private helpers ───────────────────────────────────────────────

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"StudentTabularPipeline: missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    def _apply_transforms(self, subset: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Apply imputation, log-transform, and cyclic encoding."""
        # Impute
        if fitting:
            X_imp = self._imputer.transform(subset)
        else:
            X_imp = self._imputer.transform(subset)
        df_imp = pd.DataFrame(X_imp, columns=self.REQUIRED_COLS, index=subset.index)

        # Log-transform skewed column
        for col in self.SKEWED_COLS:
            df_imp[f"{col}_log"] = np.log1p(np.abs(df_imp[col]))
            df_imp.drop(columns=[col], inplace=True)

        # Cyclic encoding for axis
        theta_rad = np.deg2rad(df_imp[self.ANGULAR_COL])
        df_imp["astig_axis_sin"] = np.sin(2 * theta_rad)
        df_imp["astig_axis_cos"] = np.cos(2 * theta_rad)
        df_imp.drop(columns=[self.ANGULAR_COL], inplace=True)

        return df_imp

    def _assemble_output(self, processed: pd.DataFrame) -> np.ndarray:
        """Scale scalar/skewed cols, leave cyclic cols as-is, return ordered array."""
        scalar_skewed_cols = self.SCALAR_COLS + [f"{c}_log" for c in self.SKEWED_COLS]
        scale_targets = [c for c in scalar_skewed_cols if c in processed.columns]
        cyclic_cols = ["astig_axis_sin", "astig_axis_cos"]

        scaled = self._scaler.transform(processed[scale_targets])
        scaled_df = pd.DataFrame(scaled, columns=scale_targets, index=processed.index)

        # Reconstruct in the canonical order: kmax, astig_log, sin, cos, pachy
        output_cols = [
            "kmax_value_D",
            "astig_value_D_log",
            "astig_axis_sin",
            "astig_axis_cos",
            "pachy_thinnest_um",
        ]
        result = pd.DataFrame(index=processed.index)
        for col in output_cols:
            if col in scaled_df.columns:
                result[col] = scaled_df[col].values
            elif col in processed.columns:
                result[col] = processed[col].values
            else:
                raise ValueError(f"Output column {col!r} not found after transforms.")

        assert result.shape[1] == self.OUTPUT_DIM, (
            f"StudentTabularPipeline produced {result.shape[1]} features, expected {self.OUTPUT_DIM}"
        )
        return result.values.astype(np.float32)

    @staticmethod
    def _get_output_col_order(processed: pd.DataFrame) -> List[str]:
        return [
            "kmax_value_D",
            "astig_value_D_log",
            "astig_axis_sin",
            "astig_axis_cos",
            "pachy_thinnest_um",
        ]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def imputer(self) -> Optional[SimpleImputer]:
        return self._imputer

    @property
    def scaler(self) -> Optional[MinMaxScaler]:
        return self._scaler


# ─────────────────────────────────────────────────────────────────────────────
# Robust Student tabular pipeline (6-feature contract with modality dropout)
# ─────────────────────────────────────────────────────────────────────────────

class RobustStudentTabularPipeline:
    """Preprocessing pipeline for the 6-feature Robust Student with modality dropout.

    Base features extracted:
        Index 0: kmax_norm          (min-max scaled to [0, 1])
        Index 1: log1p_cyl_norm     (log1p(|astig_value_D|), min-max scaled to [0, 1])
        Index 2: sin_2axis          (sin(2 * astig_axis_deg in radians))
        Index 3: cos_2axis          (cos(2 * astig_axis_deg in radians))
        Index 4: pachy_norm         (z-score standardized central corneal thickness)

    Final 6-dimensional input to student:
        Index 0: kmax_norm
        Index 1: log1p_cyl_norm
        Index 2: sin_2axis
        Index 3: cos_2axis
        Index 4: pachy_val          (pachy_norm if mask=1, else 0.0)
        Index 5: pachy_mask         (1.0 if present, else 0.0)
    """

    def __init__(self, pachy_col: str = "pachy_central_um") -> None:
        self.pachy_col = pachy_col
        self.required_cols = ["kmax_value_D", "astig_value_D", "astig_axis_deg", pachy_col]
        self._imputer: Optional[SimpleImputer] = None
        self._scaler_kmax: Optional[MinMaxScaler] = None
        self._scaler_cyl: Optional[MinMaxScaler] = None
        self._pachy_mean: float = 0.0
        self._pachy_std: float = 1.0
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "RobustStudentTabularPipeline":
        """Fit imputer, scalers, and standardization statistics on training fold."""
        for col in self.required_cols:
            if col not in df.columns:
                raise ValueError(f"RobustStudentTabularPipeline missing required column: {col}")

        subset = df[self.required_cols].copy()
        self._imputer = SimpleImputer(strategy="median")
        imputed = self._imputer.fit_transform(subset)
        df_imp = pd.DataFrame(imputed, columns=self.required_cols, index=subset.index)

        # 1. Kmax min-max scaler
        self._scaler_kmax = MinMaxScaler(feature_range=(0, 1))
        self._scaler_kmax.fit(df_imp[["kmax_value_D"]])

        # 2. Cylinder log1p min-max scaler
        cyl_log = np.log1p(np.abs(df_imp[["astig_value_D"]].values))
        self._scaler_cyl = MinMaxScaler(feature_range=(0, 1))
        self._scaler_cyl.fit(cyl_log)

        # 3. Pachymetry z-score stats
        self._pachy_mean = float(df_imp[self.pachy_col].mean())
        self._pachy_std = float(df_imp[self.pachy_col].std())
        if self._pachy_std < 1e-6:
            self._pachy_std = 1.0

        self._fitted = True
        logger.info(
            "RobustStudentTabularPipeline fitted. Pachy mean=%.2f, std=%.2f",
            self._pachy_mean, self._pachy_std
        )
        return self

    def transform_base(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe to base 5 continuous features (no mask applied yet).

        Returns:
            np.ndarray of shape (N, 5):
                [kmax_norm, log1p_cyl_norm, sin_2axis, cos_2axis, pachy_norm]
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() before transform().")

        for col in self.required_cols:
            if col not in df.columns:
                raise ValueError(f"RobustStudentTabularPipeline missing required column: {col}")

        subset = df[self.required_cols].copy()
        imputed = self._imputer.transform(subset)
        df_imp = pd.DataFrame(imputed, columns=self.required_cols, index=subset.index)

        # Kmax scaled
        kmax_norm = self._scaler_kmax.transform(df_imp[["kmax_value_D"]]).squeeze(1)

        # Cyl log scaled
        cyl_log = np.log1p(np.abs(df_imp[["astig_value_D"]].values))
        cyl_norm = self._scaler_cyl.transform(cyl_log).squeeze(1)

        # Cyclic axis encoding
        theta_rad = np.deg2rad(df_imp["astig_axis_deg"].values)
        sin_2axis = np.sin(2 * theta_rad)
        cos_2axis = np.cos(2 * theta_rad)

        # Pachymetry z-scored
        pachy_norm = (df_imp[self.pachy_col].values - self._pachy_mean) / self._pachy_std

        base = np.column_stack([kmax_norm, cyl_norm, sin_2axis, cos_2axis, pachy_norm])
        return base.astype(np.float32)

    def apply_dropout(
        self,
        base_features: np.ndarray,
        mode: str = "train",
        p_dropout: float = 0.5,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Apply modality dropout or tier mask to base 5 features.

        Args:
            base_features: Array of shape (5,) or (N, 5).
            mode: 'train' (Bernoulli dropout with prob p_dropout),
                  'tier1' (mask forced to 0.0, pachy forced to 0.0),
                  'tier2' (mask forced to 1.0, pachy intact).
            p_dropout: Probability of dropping pachymetry during training.
            rng: Optional numpy RandomState for reproducibility.

        Returns:
            Array of shape (6,) or (N, 6).
        """
        is_1d = (base_features.ndim == 1)
        X = np.atleast_2d(base_features).copy()
        N = len(X)

        kmax = X[:, 0:1]
        cyl = X[:, 1:2]
        sin_ax = X[:, 2:3]
        cos_ax = X[:, 3:4]
        pachy = X[:, 4:5]

        if mode == "tier1":
            pachy_val = np.zeros_like(pachy)
            pachy_mask = np.zeros_like(pachy)
        elif mode == "tier2":
            pachy_val = pachy
            pachy_mask = np.ones_like(pachy)
        elif mode == "train":
            if rng is None:
                mask = (np.random.rand(N, 1) >= p_dropout).astype(np.float32)
            else:
                mask = (rng.rand(N, 1) >= p_dropout).astype(np.float32)
            pachy_val = pachy * mask
            pachy_mask = mask
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'train', 'tier1', or 'tier2'.")

        out = np.hstack([kmax, cyl, sin_ax, cos_ax, pachy_val, pachy_mask]).astype(np.float32)
        return out[0] if is_1d else out

    def fit_transform(
        self,
        df: pd.DataFrame,
        mode: str = "train",
        p_dropout: float = 0.5,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        base = self.transform_base(df)
        return self.apply_dropout(base, mode=mode, p_dropout=p_dropout, rng=rng)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

