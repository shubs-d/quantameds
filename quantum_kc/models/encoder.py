import torch
import torch.nn as nn

class CornealEncoder(nn.Module):
    """
    CNN encoder for compressing corneal topography images to latent vectors.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 8):
        """
        Initializes the CornealEncoder.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            latent_dim (int): Dimension of the output latent vector.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: (batch, in_channels, 224, 224)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Output: (batch, 16, 112, 112)

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (batch, 32, 56, 56)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (batch, 64, 28, 28)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (batch, 128, 14, 14)

            nn.AdaptiveAvgPool2d(1),
            # Output: (batch, 128, 1, 1)

            nn.Flatten(),
            # Output: (batch, 128)

            nn.Linear(128, latent_dim)
            # Output: (batch, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input images of shape (batch, in_channels, H, W).
            
        Returns:
            torch.Tensor: Latent vectors of shape (batch, latent_dim).
        """
        return self.encoder(x)
