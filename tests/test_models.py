"""Unit tests for classical encoder, decoder, and autoencoder architectures."""

import numpy as np
import pytest
import torch

from quantum_kc.models.encoder import CornealEncoder
from quantum_kc.models.autoencoder import CornealDecoder, CornealAutoencoder
from quantum_kc.models.tabular_encoder import TabularPCAEncoder, TabularMLPEncoder


def test_corneal_encoder():
    """Verify that CornealEncoder maps 3x224x224 images to an 8-dim latent vector."""
    latent_dim = 8
    model = CornealEncoder(in_channels=3, latent_dim=latent_dim)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, latent_dim)
    assert not torch.isnan(out).any()


def test_corneal_decoder():
    """Verify that CornealDecoder reconstructs images of shape 3x224x224 from 8-dim latent."""
    latent_dim = 8
    model = CornealDecoder(latent_dim=latent_dim, out_channels=3)
    model.eval()

    dummy_latent = torch.randn(2, latent_dim)
    with torch.no_grad():
        recon = model(dummy_latent)

    assert recon.shape == (2, 3, 224, 224)
    # Output has sigmoid activation so should be in [0, 1]
    assert (recon >= 0.0).all() and (recon <= 1.0).all()


def test_corneal_autoencoder():
    """Verify the full autoencoder forward pass and encode method."""
    model = CornealAutoencoder(in_channels=3, latent_dim=8)
    model.eval()

    dummy_input = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        recon, latent = model(dummy_input)
        latent_direct = model.encode(dummy_input)

    assert recon.shape == (2, 3, 224, 224)
    assert latent.shape == (2, 8)
    torch.testing.assert_close(latent, latent_direct)


def test_tabular_pca_encoder():
    """Verify TabularPCAEncoder compresses tabular features to 8 dims in [0, π]."""
    X = np.random.randn(50, 20)
    encoder = TabularPCAEncoder(n_components=8)
    X_trans = encoder.fit_transform(X)

    assert X_trans.shape == (50, 8)
    assert np.all(X_trans >= 0.0)
    assert np.all(X_trans <= np.pi + 1e-6)


def test_tabular_mlp_encoder():
    """Verify TabularMLPEncoder compresses tabular features to 8 dims in (-1, 1)."""
    model = TabularMLPEncoder(input_dim=15, latent_dim=8)
    model.eval()

    dummy_tab = torch.randn(4, 15)
    with torch.no_grad():
        out = model(dummy_tab)

    assert out.shape == (4, 8)
    assert (out >= -1.0).all()
    assert (out <= 1.0).all()
