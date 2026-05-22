"""Tests for multimodal generation components (VAE, diffusion, TTS, video)."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add core-py to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.multimodal.vae import SloVAE, SloVAEEncoder, SloVAEDecoder
from domains.multimodal.diffusion import LatentDiffusionModel, LatentUNet
from domains.multimodal.text_encoder import TextEncoder
from domains.multimodal.video import VideoProcessor, TemporalEncoder
from domains.multimodal.tts import TTSEngine, GriffinLimVocoder


class TestSloVAE:
    """Test VAE encoder/decoder."""

    def test_vae_encode_decode_shape(self):
        """VAE should encode and decode images with correct shapes."""
        vae = SloVAE(latent_dim=32)
        
        # Create test image (1, 3, 224, 224)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        # Encode
        mean, log_var = vae.encoder.forward(img)
        assert mean.data.shape[1] == 32  # latent_dim
        assert mean.data.shape[2] == 7   # 224 / 32
        assert mean.data.shape[3] == 7
        
        # Sample and decode
        latent = vae.encoder.sample(mean, log_var)
        reconstructed = vae.decoder.forward(latent)
        assert reconstructed.data.shape == img.shape

    def test_vae_loss_computation(self):
        """VAE loss should be computable."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        loss = vae.loss(img)
        assert loss.data > 0


class TestLatentDiffusion:
    """Test latent diffusion model."""

    def test_unet_forward_shape(self):
        """UNet should predict noise with correct shape."""
        unet = LatentUNet(in_channels=32, model_channels=64, out_channels=32)
        
        # Create noisy latent
        B = 1
        x = np.random.randn(B, 32, 7, 7).astype(np.float32)
        timesteps = np.array([500])
        context = np.random.randn(B, 10, 64).astype(np.float32)
        
        from domains.training.slonet import Tensor
        x_tensor = Tensor(x, requires_grad=False)
        context_tensor = Tensor(context, requires_grad=False)
        
        noise_pred = unet.forward(x_tensor, timesteps, context_tensor)
        assert noise_pred.data.shape == (B, 32, 7, 7)

    def test_diffusion_add_noise(self):
        """Diffusion should add noise correctly."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        
        latents = np.random.randn(1, 32, 7, 7).astype(np.float32)
        t = np.array([50])
        
        noisy, noise = diffusion.add_noise(latents, t)
        assert noisy.shape == latents.shape
        assert noise.shape == latents.shape


class TestTextEncoder:
    """Test text encoder."""

    def test_text_encoder_train_and_encode(self):
        """Text encoder should train tokenizer and encode text."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        
        texts = ["a red circle", "a blue square", "a green triangle"]
        encoder.train_tokenizer(texts)
        
        assert encoder.tokenizer._built
        
        embeddings = encoder.encode_text(["a red circle"])
        assert embeddings.ndim == 3  # (B, seq_len, embed_dim)
        assert embeddings.shape[2] == 64  # embed_dim


class TestVideoProcessor:
    """Test video processing."""

    def test_temporal_encoder_forward(self):
        """Temporal encoder should process frame sequences."""
        encoder = TemporalEncoder(embed_dim=64, n_heads=2, n_layers=2, max_frames=8)
        
        # Create frame embeddings
        frames = np.random.randn(1, 5, 64).astype(np.float32)
        
        output = encoder.forward(frames)
        assert output.data.shape == (1, 5, 64)

    def test_video_processor_extract_frames(self):
        """Video processor should extract frames from video."""
        processor = VideoProcessor(max_frames=8)
        
        # Test with non-existent file (should fallback to random frames)
        frames = processor.extract_frames("/nonexistent/video.mp4", num_frames=4)
        assert len(frames) == 4
        assert frames[0].shape == (224, 224, 3)


class TestTTSEngine:
    """Test text-to-speech."""

    def test_griffin_lim_vocoder(self):
        """Griffin-Lim should convert spectrogram to waveform."""
        vocoder = GriffinLimVocoder(n_mels=80, sample_rate=22050)
        
        # Create random mel spectrogram
        mel_spec = np.random.randn(80, 50).astype(np.float32)
        mel_spec = np.abs(mel_spec)
        
        waveform = vocoder.generate_waveform(mel_spec, num_iterations=8)
        assert waveform.ndim == 1
        assert len(waveform) > 0

    def test_tts_engine_generate(self):
        """TTS engine should generate waveform from text."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        
        waveform = tts.text_to_waveform("hello", max_frames=20)
        assert waveform.ndim == 1
        assert len(waveform) > 0
