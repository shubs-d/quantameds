"""RobustQuantaStudent: Two-tier Keratoconus detector architecture.

Inputs (6 dims):
  0: kmax_norm (steep curvature, scaled)
  1: log1p_cyl_norm (log1p astigmatism magnitude, scaled)
  2: sin_2axis (sin(2*theta) cyclic encoding)
  3: cos_2axis (cos(2*theta) cyclic encoding)
  4: pachy_norm (z-scored central corneal thickness, 0.0 if missing/dropped)
  5: pachy_mask (1.0 if present, 0.0 if missing/dropped)

Architecture:
  Linear(6, 16) -> ReLU -> Linear(16, 8) -> Hardtanh(0, pi)
  -> 8-qubit VQC (AngleEmbedding + StronglyEntanglingLayers on lightning.qubit)
  -> Linear(8, 1) classifier head.

Supports per-sample modality dropout (Tier 1 vs Tier 2).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from quantum_kc.models.quantum_circuit import create_vqc_torch_layer


class RobustQuantaStudent(nn.Module):
    """Two-tier Keratoconus Student with hybrid quantum bottleneck."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        n_layers: int = 2,
        dev_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Classical encoder front-end
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Hardtanh(min_val=0.0, max_val=math.pi),
        )

        # 8-qubit VQC layer (lightning.qubit, AngleEmbedding, StronglyEntanglingLayers)
        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=latent_dim,
            n_layers=n_layers,
            dev_name=dev_name,
            diff_method=diff_method,
        )

        # Single logit classifier head
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 6).
            return_latent: If True, also return student_latent of shape (B, 8).

        Returns:
            logits of shape (B,) or (student_latent, logits).
        """
        # 1. Classical encoding
        z = self.encoder(x)  # (B, 8), strictly in [0, pi]

        # 2. Quantum expectation values
        q_out = self.vqc_layer(z)  # (B, 8) in [-1, 1]

        # 3. Linear classification head
        logits = self.classifier(q_out).squeeze(-1)  # (B,)

        if return_latent:
            return z, logits
        return logits
