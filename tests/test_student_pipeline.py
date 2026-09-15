"""Test the StudentClassifier forward pass and parameter counts.

Covers:
  - Classical head forward pass: correct output shape
  - Quantum head VQC parameter count assertion (48 params)
  - TabularMLPEncoder output in (-1, 1)
  - StudentClassifier.count_parameters() returns expected dict
  - return_embedding=True returns (z, logits) tuple
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.models.student_pipeline import StudentClassifier, ClassicalHead, HybridQMLHead
from quantum_kc.models.tabular_encoder import TabularMLPEncoder


BATCH = 8
INPUT_DIM = 5
LATENT_DIM = 8


def _random_input(batch: int = BATCH, dim: int = INPUT_DIM) -> torch.Tensor:
    return torch.rand(batch, dim)  # [0, 1]


class TestTabularMLPEncoder:
    def test_output_shape(self):
        model = TabularMLPEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        z = model(_random_input())
        assert z.shape == (BATCH, LATENT_DIM), f"Expected ({BATCH}, {LATENT_DIM}), got {z.shape}"

    def test_output_range(self):
        """Tanh output must be strictly in (-1, 1)."""
        model = TabularMLPEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        z = model(_random_input())
        assert z.min().item() > -1.0 - 1e-6
        assert z.max().item() < 1.0 + 1e-6

    def test_no_pi_scaling(self):
        """Output must NOT be scaled to [0, π]."""
        model = TabularMLPEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
        z = model(_random_input())
        # If output were Sigmoid × π, max would be close to π ≈ 3.14
        # Tanh output is in (-1, 1), so max < 1.5 for typical init
        assert z.max().item() < 1.5, "Output looks like it's scaled to [0,π] — check Tanh vs Sigmoid"


class TestClassicalStudentPipeline:
    def test_forward_shape(self):
        model = StudentClassifier(head_type="classical", input_dim=INPUT_DIM)
        logits = model(_random_input())
        assert logits.shape == (BATCH, 2)

    def test_embedding_return(self):
        model = StudentClassifier(head_type="classical", input_dim=INPUT_DIM)
        z, logits = model(_random_input(), return_embedding=True)
        assert z.shape == (BATCH, LATENT_DIM)
        assert logits.shape == (BATCH, 2)

    def test_parameter_count(self):
        model = StudentClassifier(head_type="classical", input_dim=INPUT_DIM)
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert "encoder" in counts
        assert "head" in counts

    def test_gradient_variance_zero_classical(self):
        model = StudentClassifier(head_type="classical", input_dim=INPUT_DIM)
        # No backward pass yet — should return 0.0
        assert model.quantum_gradient_variance() == 0.0


class TestQuantumStudentPipeline:
    def test_vqc_param_count(self):
        """StronglyEntanglingLayers(n_layers=2, n_qubits=8) must have 48 params."""
        head = HybridQMLHead(n_qubits=8, n_layers=2)
        vqc_params = sum(p.numel() for p in head.vqc_layer.parameters())
        assert vqc_params == 48, (
            f"Expected 48 VQC params (2×8×3), got {vqc_params}"
        )

    @pytest.mark.slow
    def test_quantum_forward_shape(self):
        """Full quantum forward pass (slow due to simulation)."""
        model = StudentClassifier(head_type="quantum", input_dim=INPUT_DIM)
        logits = model(_random_input())
        assert logits.shape == (BATCH, 2)
