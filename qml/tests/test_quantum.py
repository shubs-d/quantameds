"""Unit tests for PennyLane quantum circuits and hybrid quantum-classical classifier."""

import math
import pytest
import torch
import torch.nn as nn

from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.quantum_circuit import create_vqc_qnode, create_vqc_torch_layer
from quantum_kc.models.hybrid_classifier import HybridQuantumClassifier, MultimodalHybridClassifier


def test_vqc_qnode():
    """Verify PennyLane QNode forward execution and Pauli-Z expectation values."""
    n_qubits = 8
    n_layers = 2
    # Use default.qubit for fast local unit testing
    qnode = create_vqc_qnode(n_qubits=n_qubits, n_layers=n_layers, dev_name="default.qubit")

    inputs = torch.rand(n_qubits) * math.pi
    weights = torch.randn(n_layers, n_qubits, 3)

    expvals = qnode(inputs, weights)

    assert len(expvals) == n_qubits
    for val in expvals:
        # Pauli-Z expectation values must lie in [-1, 1]
        assert -1.0 - 1e-5 <= float(val) <= 1.0 + 1e-5


def test_vqc_torch_layer():
    """Verify TorchLayer integration, weight shapes, and batch forward execution."""
    n_qubits = 8
    n_layers = 2
    layer = create_vqc_torch_layer(n_qubits=n_qubits, n_layers=n_layers, dev_name="default.qubit")

    assert "weights" in layer.qnode_weights
    assert layer.weights.shape == (n_layers, n_qubits, 3)

    batch_inputs = torch.rand(3, n_qubits) * math.pi
    outputs = layer(batch_inputs)

    assert outputs.shape == (3, n_qubits)
    assert not torch.isnan(outputs).any()


def test_hybrid_quantum_classifier_forward_backward():
    """Verify end-to-end forward pass and backpropagation through the quantum circuit."""
    n_qubits = 8
    encoder = CornealEncoder(in_channels=3, latent_dim=n_qubits)
    model = HybridQuantumClassifier(
        encoder=encoder,
        n_qubits=n_qubits,
        n_layers=2,
        n_classes=2,
        freeze_encoder=True,
        dev_name="default.qubit",
    )

    # Ensure encoder is frozen
    for p in model.encoder.parameters():
        assert not p.requires_grad

    dummy_image = torch.rand(2, 3, 224, 224)
    logits = model(dummy_image)

    assert logits.shape == (2, 2)

    # Test backward pass
    targets = torch.tensor([0, 1], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    # Quantum layer weights must have gradients
    assert model.vqc_layer.weights.grad is not None
    assert not torch.isnan(model.vqc_layer.weights.grad).any()


def test_multimodal_hybrid_classifier():
    """Verify forward execution of the multimodal (image + tabular) hybrid model."""
    n_qubits = 8
    encoder = CornealEncoder(in_channels=3, latent_dim=n_qubits)
    model = MultimodalHybridClassifier(
        encoder=encoder,
        tabular_input_dim=12,
        n_qubits=n_qubits,
        n_layers=2,
        n_classes=2,
        freeze_encoder=True,
        dev_name="default.qubit",
    )

    dummy_image = torch.rand(2, 3, 224, 224)
    dummy_tabular = torch.rand(2, 12)

    logits = model(dummy_image, dummy_tabular)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()
