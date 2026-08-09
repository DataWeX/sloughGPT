"""Tests for Variational Autoencoder (VAE) — image compression and generation."""

import numpy as np
import pytest
from domains.training.slonet import Tensor, tensor as _tensor
from domains.multimodal.vae import (
    _group_norm,
    SloVAEEncoder,
    SloVAEDecoder,
    SloVAE,
)


class TestGroupNorm:
    def test_output_shape(self):
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        out = _group_norm(x, num_groups=4)
        assert out.data.shape == (1, 16, 7, 7)

    def test_normalizes_per_group(self):
        x = _tensor(np.random.randn(2, 8, 4, 4).astype(np.float32) * 10)
        out = _group_norm(x, num_groups=4)
        B, C, H, W = out.data.shape
        reshaped = out.data.reshape(B, 4, C // 4, H, W)
        means = reshaped.mean(axis=(2, 3, 4))
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_raises_on_indivisible_channels(self):
        x = _tensor(np.random.randn(1, 5, 4, 4).astype(np.float32))
        with pytest.raises(AssertionError, match="not divisible"):
            _group_norm(x, num_groups=3)


class TestSloVAEEncoder:
    def test_forward_shapes(self):
        encoder = SloVAEEncoder(latent_dim=16)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32)
        mean, log_var = encoder.forward(images)
        # Encoder reduces spatial dims; for 32x32 input with 5 stride-2 convs:
        # 32 -> 16 -> 8 -> 4 -> 2 -> 1, then final conv keeps spatial
        assert mean.data.shape[0] == 1
        assert mean.data.shape[1] == 16  # latent_dim
        assert log_var.data.shape == mean.data.shape

    def test_sample_shape(self):
        encoder = SloVAEEncoder(latent_dim=16)
        mean = _tensor(np.random.randn(1, 16, 2, 2).astype(np.float32))
        log_var = _tensor(np.zeros((1, 16, 2, 2), dtype=np.float32))
        z = encoder.sample(mean, log_var)
        assert z.data.shape == mean.data.shape

    def test_sample_stochastic(self):
        mean = _tensor(np.zeros((1, 8, 4, 4), dtype=np.float32))
        log_var = _tensor(np.zeros((1, 8, 4, 4), dtype=np.float32))
        encoder = SloVAEEncoder(latent_dim=8)
        z1 = encoder.sample(mean, log_var)
        z2 = encoder.sample(mean, log_var)
        assert not np.allclose(z1.data, z2.data)

    def test_parameters(self):
        encoder = SloVAEEncoder(latent_dim=8)
        params = encoder.parameters()
        assert len(params) > 0


class TestSloVAEDecoder:
    def test_forward_shape(self):
        decoder = SloVAEDecoder(latent_dim=16)
        latents = _tensor(np.random.randn(1, 16, 2, 2).astype(np.float32))
        out = decoder.forward(latents)
        # Decoder upsamples spatial dims
        assert out.data.shape[0] == 1
        assert out.data.shape[1] == 3  # RGB output

    def test_output_in_0_1(self):
        decoder = SloVAEDecoder(latent_dim=8)
        latents = _tensor(np.random.randn(1, 8, 2, 2).astype(np.float32) * 0.1)
        out = decoder.forward(latents)
        assert out.data.min() >= 0.0
        assert out.data.max() <= 1.0

    def test_upsample_doubles_spatial(self):
        decoder = SloVAEDecoder(latent_dim=8)
        x = _tensor(np.random.randn(1, 8, 4, 4).astype(np.float32))
        up = decoder._upsample(x)
        assert up.data.shape == (1, 8, 8, 8)

    def test_parameters(self):
        decoder = SloVAEDecoder(latent_dim=8)
        params = decoder.parameters()
        assert len(params) > 0


class TestSloVAE:
    def test_forward_returns_three(self):
        vae = SloVAE(latent_dim=8)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
        reconstructed, mean, log_var = vae.forward(images)
        assert reconstructed.data.shape[0] == 1
        assert reconstructed.data.shape[1] == 3
        assert mean.data.shape[1] == 8
        assert log_var.data.shape[1] == 8

    def test_loss_is_positive(self):
        vae = SloVAE(latent_dim=8)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
        loss = vae.loss(images)
        assert float(loss.data) > 0

    def test_train_step_returns_float(self):
        vae = SloVAE(latent_dim=8)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
        loss_val = vae.train_step(images)
        assert isinstance(loss_val, float)
        assert loss_val > 0

    def test_encode_decode_shapes(self):
        vae = SloVAE(latent_dim=8)
        images = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
        latents = vae.encode(images)
        assert latents.shape[0] == 1
        assert latents.shape[1] == 8
        decoded = vae.decode(latents)
        assert decoded.shape[0] == 1
        assert decoded.shape[1] == 3

    def test_parameters(self):
        vae = SloVAE(latent_dim=8)
        params = vae.parameters()
        assert len(params) > 0
        # VAE has both encoder and decoder params
        assert len(params) > 20
