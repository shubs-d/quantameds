"""Matched-capacity student architectures for Step 2 ablation.

All models take 6 tabular features:
  0: kmax_norm
  1: log1p_cyl_norm
  2: sin_2axis
  3: cos_2axis
  4: pachy_norm (or 0.0 if masked)
  5: pachy_mask (1.0 if present, 0.0 if masked)

Architectures:
  1. ClassicalMLPStudent: 309 parameters (pure classical)
  2. QuantumOnlyStudent:  257 parameters (VQC alone with direct projection)
  3. HybridRobustQuantaStudent: 305 parameters (current hybrid)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from quantum_kc.models.quantum_circuit import create_vqc_torch_layer


class ClassicalMLPStudent(nn.Module):
    """Classical-only student model matching hybrid parameter capacity (~309 params)."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 6 -> 16 -> 8 (248 params)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Hardtanh(min_val=0.0, max_val=math.pi),
        )

        # Classical bottleneck replacing VQC: 8 -> 6 -> 1 (61 params)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encoder(x)  # (B, 8)
        logits = self.head(z).squeeze(-1)  # (B,)
        if return_latent:
            return z, logits
        return logits

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        enc = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"encoder": enc, "head": head, "quantum": 0, "total": total}


class QuantumOnlyStudent(nn.Module):
    """Quantum-only student: direct projection into an 8-qubit VQC (~257 params)."""

    def __init__(
        self,
        input_dim: int = 6,
        n_qubits: int = 8,
        n_layers: int = 8,
        dev_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Direct linear angle scaler: 6 -> 8 (56 params)
        self.angle_proj = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Hardtanh(min_val=0.0, max_val=math.pi),
        )

        # 8-qubit VQC with 8 entangling layers (192 quantum params)
        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            dev_name=dev_name,
            diff_method=diff_method,
        )

        # Linear readout: 8 -> 1 (9 params)
        self.classifier = nn.Linear(n_qubits, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        angles = self.angle_proj(x)  # (B, 8) in [0, pi]
        q_out = self.vqc_layer(angles)  # (B, 8) in [-1, 1]
        logits = self.classifier(q_out).squeeze(-1)  # (B,)
        if return_latent:
            # Scale q_out to [0, pi] for distillation matching
            z = (q_out + 1.0) * (math.pi / 2.0)
            return z, logits
        return logits

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        vqc = sum(p.numel() for p in self.vqc_layer.parameters())
        proj = sum(p.numel() for p in self.angle_proj.parameters())
        cls = sum(p.numel() for p in self.classifier.parameters())
        return {"proj": proj, "quantum": vqc, "classifier": cls, "total": total}


class HybridRobustQuantaStudent(nn.Module):
    """Hybrid student: classical front-end + 8-qubit VQC + linear readout (~305 params)."""

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
        self.n_layers = n_layers

        # Classical encoder: 6 -> 16 -> 8 (248 params)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Hardtanh(min_val=0.0, max_val=math.pi),
        )

        # 8-qubit VQC with 2 layers (48 quantum params)
        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=latent_dim,
            n_layers=n_layers,
            dev_name=dev_name,
            diff_method=diff_method,
        )

        # Readout: 8 -> 1 (9 params)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z = self.encoder(x)  # (B, 8) in [0, pi]
        q_out = self.vqc_layer(z)  # (B, 8) in [-1, 1]
        logits = self.classifier(q_out).squeeze(-1)  # (B,)
        if return_latent:
            return z, logits
        return logits

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        enc = sum(p.numel() for p in self.encoder.parameters())
        vqc = sum(p.numel() for p in self.vqc_layer.parameters())
        cls = sum(p.numel() for p in self.classifier.parameters())
        return {"encoder": enc, "quantum": vqc, "classifier": cls, "total": total}
