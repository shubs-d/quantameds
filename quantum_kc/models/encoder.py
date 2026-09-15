"""CornealEncoder — CNN image encoder for corneal topography maps.

Changes from original:
  - Added ``supervised_head`` (linear) so the Teacher can be fine-tuned on
    Stage 2 with a disease classification objective.
  - ``forward`` accepts a ``return_logits`` flag.  When True, returns
    ``(z, logits)``; when False (default), returns only ``z``.
  - The encoder backbone is unchanged so pretrained weights can be loaded
    without key mismatches.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class CornealEncoder(nn.Module):
    """CNN encoder compressing corneal topography stacks to a latent vector.

    Architecture:
        4× Conv2d(stride=2) + BN + ReLU → AdaptiveAvgPool → Flatten → Linear

    Parameter count (in_channels=3, latent_dim=8):
        ~131 K parameters in the backbone + 8×2+2 = 18 in the head.

    Args:
        in_channels: Number of input channels (3 for Axial/Anterior/Posterior stack).
        latent_dim: Dimension of the output latent vector z.  Must equal N_QUBITS.
        n_classes: Number of classes for the supervised head (default 2).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 8,
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # Input:  (B, in_channels, 224, 224)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # →  (B, 16, 112, 112)

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # →  (B, 32, 56, 56)

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # →  (B, 64, 28, 28)

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # →  (B, 128, 14, 14)

            nn.AdaptiveAvgPool2d(1),  # →  (B, 128, 1, 1)
            nn.Flatten(),             # →  (B, 128)
            nn.Linear(128, latent_dim),  # →  (B, latent_dim)
        )

        # Supervised head: used during Teacher fine-tuning on Stage 2.
        # Produces logits for Normal/KC classification from the latent vector.
        self.supervised_head = nn.Linear(latent_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input images of shape (B, in_channels, H, W).
            return_logits: If True, also return classification logits.

        Returns:
            z of shape (B, latent_dim) when return_logits=False.
            (z, logits) when return_logits=True, where logits has shape (B, n_classes).
        """
        z = self.encoder(x)
        if return_logits:
            logits = self.supervised_head(z)
            return z, logits
        return z
