import torch
import torch.nn as nn
from .encoder import CornealEncoder

class CornealDecoder(nn.Module):
    """
    CNN decoder for reconstructing corneal topography images from latent vectors.
    """
    def __init__(self, latent_dim: int = 8, out_channels: int = 3):
        """
        Initializes the CornealDecoder.

        Args:
            latent_dim (int): Dimension of the input latent vector.
            out_channels (int): Number of output channels (e.g., 3 for RGB).
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 14 * 14),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            # Input: (batch, 128, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (batch, 64, 28, 28)
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (batch, 32, 56, 56)
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output: (batch, 16, 112, 112)
            
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            # Output: (batch, out_channels, 224, 224)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch, latent_dim).
            
        Returns:
            torch.Tensor: Reconstructed images of shape (batch, out_channels, 224, 224).
        """
        x = self.fc(z)
        x = x.view(-1, 128, 14, 14)
        return self.decoder(x)


class CornealAutoencoder(nn.Module):
    """
    Autoencoder for unsupervised pretraining on corneal topography images.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 8):
        """
        Initializes the autoencoder.

        Args:
            in_channels (int): Number of input/output channels.
            latent_dim (int): Dimension of the bottleneck latent vector.
        """
        super().__init__()
        self.encoder = CornealEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = CornealDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input images of shape (batch, in_channels, H, W).
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the reconstructed image and the latent vector.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to encode images.

        Args:
            x (torch.Tensor): Input images.
            
        Returns:
            torch.Tensor: Latent vectors.
        """
        return self.encoder(x)
