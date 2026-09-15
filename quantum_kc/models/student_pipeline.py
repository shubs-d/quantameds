"""Student classifier pipeline: TabularMLPEncoder + downstream head.

Two head variants, evaluated under identical conditions:

ClassicalHead  (9 parameters)
    Linear(8 → 2)

HybridQMLHead  (66 parameters = 48 VQC + 18 linear)
    AngleScale (learned Linear(8→8)+Sigmoid×π) → VQC → Linear(8→2)
    VQC: 8-qubit AngleEmbedding + 2×StronglyEntanglingLayers (8×3×2=48 params)
    AngleScale params: 8×8+8 = 72 (these are NOT quantum params)
    Total head params: 72 + 48 + 18 = 138

NOTE on parameter asymmetry:
    The classical head (9 params) is not parameter-matched to the quantum head
    (138 params including angle-scale).  This asymmetry is inherent to the
    comparison and is documented in the report.  We use the simplest possible
    classical head (linear) because the quantum circuit's expressive power at
    this parameter count is exactly what we are testing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantum_circuit import create_vqc_torch_layer
from .tabular_encoder import TabularMLPEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Downstream heads
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalHead(nn.Module):
    """Simple linear classifier head.

    Parameter count: latent_dim × n_classes + n_classes = 8×2 + 2 = 18.
    (We use a 2-layer MLP to be slightly more expressive while staying lean.)
    Actually we use nn.Linear(8→2) only: 8×2+2 = 18 parameters.
    """

    def __init__(self, latent_dim: int = 8, n_classes: int = 2) -> None:
        super().__init__()
        self.head = nn.Linear(latent_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (B, latent_dim) in (-1, 1) from TabularMLPEncoder.
        Returns:
            Logits (B, n_classes).
        """
        return self.head(z)


class HybridQMLHead(nn.Module):
    """Quantum head: angle-scale → VQC → linear classifier.

    The angle-scale layer maps the latent vector (in (-1, 1)) to [0, π]
    so it can be used with AngleEmbedding.  It is trainable, giving the
    circuit flexibility in how it encodes the latent state.

    Parameter breakdown:
        AngleScale (Linear 8→8 + Sigmoid×π):  8×8 + 8 =  72
        VQC (StronglyEntanglingLayers 2×8×3):            48
        Linear head (8→2):                    8×2 + 2 =  18
        Total:                                           138

    The 48 VQC parameters and the gradient variance of those parameters
    are tracked separately for barren-plateau diagnostics.

    Args:
        n_qubits: Number of qubits (must equal latent_dim = 8).
        n_layers: Number of StronglyEntanglingLayers (default 2).
        n_classes: Output classes (default 2).
        dev_name: PennyLane device (default 'lightning.qubit').
        diff_method: Differentiation method (default 'adjoint').
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        n_classes: int = 2,
        dev_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # Learnable angle-scale: maps (-1, 1) → (0, π)
        self.angle_scale = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.Sigmoid(),
        )

        self.vqc_layer = create_vqc_torch_layer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            dev_name=dev_name,
            diff_method=diff_method,
        )

        self.classifier = nn.Linear(n_qubits, n_classes)

        # Verify VQC parameter count: n_layers × n_qubits × 3
        expected_vqc_params = n_layers * n_qubits * 3
        actual_vqc_params = sum(p.numel() for p in self.vqc_layer.parameters())
        assert actual_vqc_params == expected_vqc_params, (
            f"VQC parameter count mismatch: expected {expected_vqc_params}, "
            f"got {actual_vqc_params}.  "
            f"StronglyEntanglingLayers({n_layers}, {n_qubits}) should have "
            f"{n_layers}×{n_qubits}×3 = {expected_vqc_params} params."
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector (B, n_qubits) from TabularMLPEncoder (Tanh output).
        Returns:
            Logits (B, n_classes).
        """
        # Scale (-1,1) → (0, π)
        z_scaled = self.angle_scale(z) * math.pi  # (B, n_qubits)
        q_out = self.vqc_layer(z_scaled)           # (B, n_qubits) — Pauli-Z expvals
        return self.classifier(q_out)

    def quantum_gradient_variance(self) -> float:
        """Return variance of VQC weight gradients for barren-plateau detection.

        Returns:
            Gradient variance (float).  Returns 0.0 if gradients are not yet
            computed (before the first backward pass).
        """
        grads = []
        for name, param in self.vqc_layer.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        if not grads:
            return 0.0
        all_grads = torch.cat(grads)
        return float(torch.var(all_grads).item())


# ─────────────────────────────────────────────────────────────────────────────
# Student classifier (encoder + head)
# ─────────────────────────────────────────────────────────────────────────────

class StudentClassifier(nn.Module):
    """Full Student pipeline: 5 tabular features → 8-d latent → classification.

    Combines a ``TabularMLPEncoder`` (shared backbone) with either a
    ``ClassicalHead`` or a ``HybridQMLHead`` for the downstream classification.

    Args:
        head_type: ``'classical'`` or ``'quantum'``.
        input_dim: Number of Student input features (fixed at 5).
        latent_dim: Latent vector size (must equal N_QUBITS = 8).
        n_classes: Output classes (default 2).
        dropout: Dropout rate in the MLP encoder.
        n_qubits: Qubits for the quantum head (ignored for classical).
        n_layers: VQC layers (ignored for classical).
        dev_name: PennyLane device (ignored for classical).
        diff_method: Differentiation method (ignored for classical).
    """

    def __init__(
        self,
        head_type: str = "classical",
        input_dim: int = 5,
        latent_dim: int = 8,
        n_classes: int = 2,
        dropout: float = 0.2,
        n_qubits: int = 8,
        n_layers: int = 2,
        dev_name: str = "lightning.qubit",
        diff_method: str = "adjoint",
    ) -> None:
        super().__init__()
        if head_type not in ("classical", "quantum"):
            raise ValueError(f"head_type must be 'classical' or 'quantum', got {head_type!r}")

        self.head_type = head_type
        self.encoder = TabularMLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        if head_type == "classical":
            self.head = ClassicalHead(latent_dim=latent_dim, n_classes=n_classes)
        else:
            self.head = HybridQMLHead(
                n_qubits=n_qubits,
                n_layers=n_layers,
                n_classes=n_classes,
                dev_name=dev_name,
                diff_method=diff_method,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ):
        """Forward pass.

        Args:
            x: Input tabular features (B, input_dim).
            return_embedding: If True, also return the latent vector z.

        Returns:
            logits (B, n_classes), or (z, logits) if return_embedding=True.
        """
        z = self.encoder(x)
        logits = self.head(z)
        if return_embedding:
            return z, logits
        return logits

    def quantum_gradient_variance(self) -> float:
        """Proxy to HybridQMLHead.quantum_gradient_variance (0.0 for classical)."""
        if self.head_type == "quantum" and isinstance(self.head, HybridQMLHead):
            return self.head.quantum_gradient_variance()
        return 0.0

    def count_parameters(self) -> dict:
        """Return dict with trainable parameter counts per component."""
        enc_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {
            "encoder": enc_params,
            "head": head_params,
            "total": enc_params + head_params,
        }
