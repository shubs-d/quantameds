"""Tabular feature preprocessing pipeline."""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def encode_periodic_axes(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Encode angular features using sine and cosine of twice the angle."""
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns:
            theta_rad = np.deg2rad(df_encoded[col])
            df_encoded[f"{col}_sin"] = np.sin(2 * theta_rad)
            df_encoded[f"{col}_cos"] = np.cos(2 * theta_rad)
            df_encoded.drop(columns=[col], inplace=True)
            logger.debug(f"Encoded and dropped periodic column: {col}")
        else:
            logger.warning(f"Column {col} not found in dataframe for periodic encoding.")
    return df_encoded

def log_transform_skewed(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Apply log1p transformation to specified skewed columns."""
    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns:
            df_transformed[col] = np.log1p(np.abs(df_transformed[col]))
            logger.debug(f"Log transformed skewed column: {col}")
        else:
            logger.warning(f"Column {col} not found in dataframe for log transform.")
    return df_transformed

def scale_to_quantum_range(X: Union[pd.DataFrame, np.ndarray], scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale tabular features to range (0, np.pi) for quantum embedding."""
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_scaled = scaler.fit_transform(X)
        logger.debug("Fit and transformed data using new MinMaxScaler.")
    else:
        X_scaled = scaler.transform(X)
        logger.debug("Transformed data using provided MinMaxScaler.")
    return X_scaled, scaler

def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Impute missing values in the dataframe."""
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    logger.debug(f"Imputed missing values using strategy: {strategy}")
    return df_imputed

def build_tabular_pipeline(df: pd.DataFrame, angular_cols: List[str], skewed_cols: List[str], scaler: Optional[MinMaxScaler] = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Orchestrate the full preprocessing pipeline."""
    logger.info("Starting tabular preprocessing pipeline.")
    df_processed = impute_missing(df)
    df_processed = encode_periodic_axes(df_processed, angular_cols)
    df_processed = log_transform_skewed(df_processed, skewed_cols)
    X_scaled, fitted_scaler = scale_to_quantum_range(df_processed, scaler)
    logger.info("Finished tabular preprocessing pipeline.")
    return X_scaled, fitted_scaler
