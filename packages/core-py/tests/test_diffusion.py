"""Tests for Latent Diffusion Model — text-to-image generation."""

import numpy as np
import pytest
from domains.training.slonet import Tensor, tensor as _tensor
from domains.multimodal.diffusion import (
    _group_norm,
    _timestep_embedding,
    TimestepEmbedder,
    ResBlock,
    UNetBlock,
    LatentUNet,
    LatentDiffusionModel,
)


class TestGroupNorm:
    def test_output_shape(self):
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        out = _group_norm(x, num_groups=8)
        assert out.data.shape == (1, 16, 7, 7)

    def test_normalizes_per_group(self):
        x = _tensor(np.random.randn(2, 8, 4, 4).astype(np.float32) * 10)
        out = _group_norm(x, num_groups=4)
        B, C, H, W = out.data.shape
        reshaped = out.data.reshape(B, 4, C // 4, H, W)
        means = reshaped.mean(axis=(2, 3, 4))
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_fewer_channels_than_groups(self):
        x = _tensor(np.random.randn(1, 3, 4, 4).astype(np.float32))
        out = _group_norm(x, num_groups=32)
        assert out.data.shape == (1, 3, 4, 4)

    def test_channels_not_divisible_by_groups(self):
        x = _tensor(np.random.randn(1, 5, 4, 4).astype(np.float32))
        out = _group_norm(x, num_groups=3)
        assert out.data.shape == (1, 5, 4, 4)


class TestTimestepEmbedding:
    def test_output_shape(self):
        t = np.array([0, 100, 999])
        emb = _timestep_embedding(t, dim=64)
        assert emb.shape == (3, 64)

    def test_odd_dim(self):
        t = np.array([0, 50])
        emb = _timestep_embedding(t, dim=65)
        assert emb.shape == (2, 65)

    def test_deterministic(self):
        t = np.array([42])
        emb1 = _timestep_embedding(t, dim=32)
        emb2 = _timestep_embedding(t, dim=32)
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_timesteps_different_embeddings(self):
        t = np.array([0, 500])
        emb = _timestep_embedding(t, dim=32)
        assert not np.allclose(emb[0], emb[1])


class TestTimestepEmbedder:
    def test_forward_shape(self):
        embedder = TimestepEmbedder(dim=64)
        t = np.array([0, 100, 500])
        out = embedder.forward(t)
        assert out.data.shape == (3, 64)

    def test_parameters_count(self):
        embedder = TimestepEmbedder(dim=64)
        params = embedder.parameters()
        assert len(params) > 0
        total = sum(p.data.size for p in params)
        assert total == 64 * 256 + 256 + 256 * 64 + 64


class TestResBlock:
    def test_forward_shape(self):
        block = ResBlock(in_channels=16, out_channels=32, temb_dim=64)
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        temb = _tensor(np.random.randn(1, 64).astype(np.float32))
        out = block.forward(x, temb)
        assert out.data.shape == (1, 32, 7, 7)

    def test_skip_connection_channels(self):
        block = ResBlock(in_channels=16, out_channels=32, temb_dim=64)
        assert block.skip is not None

    def test_no_skip_same_channels(self):
        block = ResBlock(in_channels=32, out_channels=32, temb_dim=64)
        assert block.skip is None

    def test_parameters_count(self):
        block = ResBlock(in_channels=16, out_channels=32, temb_dim=64)
        params = block.parameters()
        assert len(params) > 0


class TestUNetBlock:
    def test_without_cross_attention(self):
        block = UNetBlock(in_channels=16, out_channels=32, temb_dim=64, has_cross_attn=False)
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        temb = _tensor(np.random.randn(1, 64).astype(np.float32))
        out = block.forward(x, temb)
        assert out.data.shape == (1, 32, 7, 7)

    def test_with_cross_attention(self):
        block = UNetBlock(in_channels=16, out_channels=32, temb_dim=64,
                          context_dim=64, n_heads=4, has_cross_attn=True)
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        temb = _tensor(np.random.randn(1, 64).astype(np.float32))
        context = _tensor(np.random.randn(1, 4, 64).astype(np.float32))
        out = block.forward(x, temb, context)
        assert out.data.shape == (1, 32, 7, 7)


class TestLatentUNet:
    def test_forward_shape(self):
        unet = LatentUNet(in_channels=16, model_channels=32, out_channels=16,
                          temb_dim=64, context_dim=64, n_heads=4)
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        t = np.array([100])
        out = unet.forward(x, t)
        assert out.data.shape == (1, 16, 7, 7)

    def test_with_context(self):
        unet = LatentUNet(in_channels=16, model_channels=32, out_channels=16,
                          temb_dim=64, context_dim=64, n_heads=4)
        x = _tensor(np.random.randn(1, 16, 7, 7).astype(np.float32))
        t = np.array([100])
        context = _tensor(np.random.randn(1, 4, 64).astype(np.float32))
        out = unet.forward(x, t, context)
        assert out.data.shape == (1, 16, 7, 7)

    def test_batch_size_2(self):
        unet = LatentUNet(in_channels=16, model_channels=32, out_channels=16,
                          temb_dim=64, context_dim=64, n_heads=4)
        x = _tensor(np.random.randn(2, 16, 7, 7).astype(np.float32))
        t = np.array([100, 500])
        out = unet.forward(x, t)
        assert out.data.shape == (2, 16, 7, 7)

    def test_parameters_count(self):
        unet = LatentUNet(in_channels=16, model_channels=32, out_channels=16,
                          temb_dim=64, context_dim=64, n_heads=4)
        params = unet.parameters()
        assert len(params) > 100


class TestLatentDiffusionModel:
    def test_init(self):
        model = LatentDiffusionModel(latent_dim=16, model_channels=32,
                                     temb_dim=64, context_dim=64, n_heads=4)
        assert model.num_timesteps == 1000
        assert model.betas.shape == (1000,)
        assert model.alphas_cumprod.shape == (1000,)

    def test_add_noise(self):
        model = LatentDiffusionModel(latent_dim=16, num_timesteps=100)
        latents = np.random.randn(1, 16, 7, 7).astype(np.float32)
        t = np.array([50])
        noisy, noise = model.add_noise(latents, t)
        assert noisy.shape == latents.shape
        assert noise.shape == latents.shape

    def test_noise_schedule_monotonic(self):
        model = LatentDiffusionModel(num_timesteps=1000)
        assert np.all(np.diff(model.alphas_cumprod) <= 0)

    def test_train_step_returns_float(self):
        model = LatentDiffusionModel(latent_dim=16, model_channels=16,
                                     temb_dim=32, context_dim=32, n_heads=2,
                                     num_timesteps=100)
        latents = np.random.randn(1, 16, 7, 7).astype(np.float32)
        text_emb = np.random.randn(1, 4, 32).astype(np.float32)
        loss = model.train_step(latents, text_emb)
        assert isinstance(loss, float)
        assert loss > 0

    def test_sample_output_shape(self):
        model = LatentDiffusionModel(latent_dim=16, model_channels=16,
                                     temb_dim=32, context_dim=32, n_heads=2,
                                     num_timesteps=100)
        text_emb = np.random.randn(1, 4, 32).astype(np.float32)
        latents = model.sample(text_emb, num_steps=3)
        assert latents.shape == (1, 16, 7, 7)

    def test_sample_deterministic_with_seed(self):
        model = LatentDiffusionModel(latent_dim=16, model_channels=16,
                                     temb_dim=32, context_dim=32, n_heads=2,
                                     num_timesteps=100)
        text_emb = np.random.randn(1, 4, 32).astype(np.float32)
        np.random.seed(42)
        latents1 = model.sample(text_emb, num_steps=3)
        np.random.seed(42)
        latents2 = model.sample(text_emb, num_steps=3)
        np.testing.assert_array_equal(latents1, latents2)

    def test_parameters(self):
        model = LatentDiffusionModel(latent_dim=16, model_channels=16,
                                     temb_dim=32, context_dim=32, n_heads=2)
        params = model.parameters()
        assert len(params) > 0
