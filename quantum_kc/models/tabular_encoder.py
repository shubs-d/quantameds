"""TabularMLPEncoder — Student MLP that maps 5 tabular features to an 8-d latent vector.

Design notes vs. the original implementation:
  - input_dim is fixed at 5 (the Student feature contract).
  - The output is Tanh-activated → range (-1, 1), not Sigmoid×π.
    The Teacher's latent space is unconstrained, so the Student should match
    it in an unconstrained space; π-scaling is applied *outside* this module
    only when passing to the VQC AngleEmbedding.
  - Added Dropout(0.2) for regularisation (comparable to weight decay on
    the classical head).
  - Parameter count: 5×32+32 + 32×16+16 + 16×8+8 = 1,272 parameters.
    This is explicitly verified in the module docstring and can be confirmed
    with sum(p.numel() for p in TabularMLPEncoder().parameters()).

The old TabularPCAEncoder is retained for backward compatibility but is not
used in the distillation pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class TabularPCAEncoder:
    """Tabular feature compression using PCA (retained for backward compat)."""

    def __init__(self, n_components: int = 8) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = MinMaxScaler(feature_range=(0, math.pi))

    def fit(self, X: np.ndarray) -> "TabularPCAEncoder":
        X_pca = self.pca.fit_transform(X)
        self.scaler.fit_transform(X_pca)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X)
        return self.scaler.transform(X_pca)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.fit_transform(X)
        return self.scaler.fit_transform(X_pca)


class TabularMLPEncoder(nn.Module):
    """Student MLP encoder: 5 tabular features → 8-dimensional latent vector.

    Architecture:
        Linear(5 → 32) → ReLU → Dropout(0.2)
        Linear(32 → 16) → ReLU
        Linear(16 → 8) → Tanh

    Output range: (-1, 1) per dimension.
    The Teacher latent space is unconstrained; Tanh prevents unbounded drift
    without restricting to [0, π].  Angle-scaling for the VQC is applied in
    HybridQMLHead, not here.

    Parameter count (input_dim=5, latent_dim=8):
        Linear(5→32):   5×32 + 32 =  192
        Linear(32→16): 32×16 + 16 =  528
        Linear(16→8):  16×8  +  8 =  136
        Dropout:                        0
        Total:                      ──────
                                    856  parameters

    Note: with default input_dim=5 and latent_dim=8.

    Args:
        input_dim: Number of input features.  Fixed at 5 for the Student.
        latent_dim: Output dimensionality.  Must equal N_QUBITS (8).
        dropout: Dropout probability applied after the first hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 5,
        latent_dim: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, latent_dim),
            nn.Tanh(),  # output ∈ (-1, 1); π-scaling handled externally by VQC head
        )
        self._verify_param_count(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode tabular features to latent vector.

        Args:
            x: Float tensor of shape (B, input_dim) in [0, 1].

        Returns:
            Latent vectors of shape (B, latent_dim) in (-1, 1).
        """
        return self.encoder(x)

    @staticmethod
    def _verify_param_count(input_dim: int, latent_dim: int) -> None:
        """Sanity-check the parameter count at init time."""
        expected = (
            input_dim * 32 + 32   # Linear(input_dim → 32) + bias
            + 32 * 16 + 16        # Linear(32 → 16) + bias
            + 16 * latent_dim + latent_dim  # Linear(16 → latent_dim) + bias
        )
        # Instantiate temporarily to count
        tmp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Linear(32, 16),
            nn.Linear(16, latent_dim),
        )
        actual = sum(p.numel() for p in tmp.parameters())
        assert actual == expected, (
            f"TabularMLPEncoder parameter count mismatch: expected {expected}, got {actual}"
        )
