import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class TabularPCAEncoder:
    """
    Tabular feature compression using PCA to match qubit budget.
    """
    def __init__(self, n_components: int = 8):
        """
        Initializes the TabularPCAEncoder.

        Args:
            n_components (int): Number of components to keep.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.scaler = MinMaxScaler(feature_range=(0, math.pi))

    def fit(self, X: np.ndarray) -> 'TabularPCAEncoder':
        """
        Fits the PCA and scaler to the data.

        Args:
            X (np.ndarray): Tabular feature data.
            
        Returns:
            TabularPCAEncoder: self
        """
        X_pca = self.pca.fit_transform(X)
        self.scaler.fit_transform(X_pca)   # fit_transform keeps scaler fitted
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data using PCA and scaling.

        Args:
            X (np.ndarray): Tabular feature data.
            
        Returns:
            np.ndarray: Compressed features scaled to [0, pi].
        """
        X_pca = self.pca.transform(X)
        return self.scaler.transform(X_pca)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits and transforms the data in one pass.

        Args:
            X (np.ndarray): Tabular feature data.

        Returns:
            np.ndarray: Compressed features scaled to [0, pi].
        """
        # Reuse the PCA output from fit to avoid floating-point discrepancies
        # between pca.fit_transform() and pca.transform() that can push
        # boundary values fractionally outside [0, π].
        X_pca = self.pca.fit_transform(X)
        return self.scaler.fit_transform(X_pca)


class TabularMLPEncoder(nn.Module):
    """
    MLP encoder for tabular features to match qubit budget.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8):
        """
        Initializes the TabularMLPEncoder.

        Args:
            input_dim (int): Number of input tabular features.
            latent_dim (int): Dimension of the output latent vector.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP encoder.

        Args:
            x (torch.Tensor): Tabular feature data of shape (batch, input_dim).
            
        Returns:
            torch.Tensor: Latent vectors of shape (batch, latent_dim) in range [0, pi].
        """
        # Multiply Sigmoid output by pi to ensure [0, pi] range
        return self.encoder(x) * math.pi
