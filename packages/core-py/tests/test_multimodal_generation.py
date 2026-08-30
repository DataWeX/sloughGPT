"""Tests for multimodal generation components (VAE, diffusion, TTS, video)."""

import numpy as np
import pytest
pytestmark = pytest.mark.slow
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

    def test_vae_different_latent_dims(self):
        """VAE should work with different latent dimensions."""
        for ld in [16, 32, 64]:
            vae = SloVAE(latent_dim=ld)
            img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            mean, log_var = vae.encoder.forward(img)
            assert mean.data.shape[1] == ld

    def test_vae_encode_returns_mean(self):
        """VAE encoder should return mean tensor."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        mean, log_var = vae.encoder.forward(img)
        assert mean.data is not None
        assert mean.data.dtype == np.float32

    def test_vae_encode_log_var(self):
        """VAE encoder should return log_var tensor."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        mean, log_var = vae.encoder.forward(img)
        assert log_var.data is not None

    def test_vae_sample_shape(self):
        """VAE sampling should produce correct latent shape."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        mean, log_var = vae.encoder.forward(img)
        latent = vae.encoder.sample(mean, log_var)
        assert latent.data.shape == mean.data.shape

    def test_vae_encode_method(self):
        """VAE encode() should return numpy array."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        latent = vae.encode(img)
        assert isinstance(latent, np.ndarray)
        assert latent.shape[1] == 32

    def test_vae_decode_method(self):
        """VAE decode() should return numpy array."""
        vae = SloVAE(latent_dim=32)
        latents = np.random.randn(1, 32, 7, 7).astype(np.float32)
        decoded = vae.decode(latents)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (1, 3, 224, 224)

    def test_vae_encode_decode_roundtrip_shape(self):
        """VAE encode->decode should produce correct output shape."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        reconstructed, mean, log_var = vae.forward(img)
        assert reconstructed.data.shape == img.shape

    def test_vae_loss_positive(self):
        """VAE loss should always be positive."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        loss = vae.loss(img)
        assert loss.data > 0

    def test_vae_parameters(self):
        """VAE should have parameters."""
        vae = SloVAE(latent_dim=32)
        params = vae.parameters()
        assert len(params) > 0

    def test_vae_encoder_parameters(self):
        """VAE encoder should have parameters."""
        enc = SloVAEEncoder(latent_dim=32)
        params = enc.parameters()
        assert len(params) > 0

    def test_vae_decoder_parameters(self):
        """VAE decoder should have parameters."""
        dec = SloVAEDecoder(latent_dim=32)
        params = dec.parameters()
        assert len(params) > 0

    def test_vae_decoder_upsample(self):
        """VAE decoder upsample should double spatial dims."""
        dec = SloVAEDecoder(latent_dim=32)
        x = np.random.randn(1, 32, 7, 7).astype(np.float32)
        from domains.training.slonet import tensor as _tensor
        x_tensor = _tensor(x, requires_grad=False)
        up = dec._upsample(x_tensor)
        assert up.data.shape == (1, 32, 14, 14)

    def test_vae_reconstruction_range(self):
        """VAE reconstruction should be in valid range (sigmoid output)."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        reconstructed, _, _ = vae.forward(img)
        assert reconstructed.data.min() >= 0.0
        assert reconstructed.data.max() <= 1.0

    def test_vae_loss_reduces_after_train_step(self):
        """VAE loss should reduce after a training step."""
        vae = SloVAE(latent_dim=32)
        img = np.random.randn(1, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        loss1 = vae.loss(img).data
        vae.train_step(img)
        loss2 = vae.loss(img).data
        assert loss2 <= loss1 + 0.1

    def test_vae_batch_encode(self):
        """VAE should handle batch encoding."""
        vae = SloVAE(latent_dim=32)
        batch = np.random.randn(2, 3, 224, 224).astype(np.float32) * 0.5 + 0.5
        batch = np.clip(batch, 0, 1)
        mean, log_var = vae.encoder.forward(batch)
        assert mean.data.shape[0] == 2


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

    def test_diffusion_noise_schedule(self):
        """Diffusion should have correct noise schedule."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        assert len(diffusion.betas) == 100
        assert len(diffusion.alphas) == 100
        assert len(diffusion.alphas_cumprod) == 100

    def test_diffusion_betas_range(self):
        """Diffusion betas should be in [1e-4, 0.02]."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        assert diffusion.betas[0] >= 1e-4
        assert diffusion.betas[-1] <= 0.02

    def test_diffusion_add_noise_different_timesteps(self):
        """Diffusion should produce different noise at different timesteps."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        latents = np.random.randn(1, 32, 7, 7).astype(np.float32)
        noisy1, _ = diffusion.add_noise(latents, np.array([10]))
        noisy2, _ = diffusion.add_noise(latents, np.array([90]))
        assert not np.allclose(noisy1, noisy2)

    def test_diffusion_parameters(self):
        """Diffusion model should have parameters."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        params = diffusion.parameters()
        assert len(params) > 0

    def test_unet_parameters(self):
        """UNet should have parameters."""
        unet = LatentUNet(in_channels=32, model_channels=64, out_channels=32)
        params = unet.parameters()
        assert len(params) > 0

    def test_diffusion_train_step(self):
        """Diffusion train step should return a loss."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        latents = np.random.randn(1, 32, 7, 7).astype(np.float32)
        text_emb = np.random.randn(1, 10, 64).astype(np.float32)
        loss = diffusion.train_step(latents, text_emb)
        assert isinstance(loss, float)
        assert loss > 0

    def test_diffusion_sqrt_alpha_bar(self):
        """Diffusion _get_sqrt_alpha_bar should return correct shape."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        t = np.array([0, 50, 99])
        result = diffusion._get_sqrt_alpha_bar(t)
        assert result.shape == (3, 1, 1, 1)

    def test_diffusion_sqrt_one_minus_alpha_bar(self):
        """Diffusion _get_sqrt_one_minus_alpha_bar should return correct shape."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        t = np.array([0, 50, 99])
        result = diffusion._get_sqrt_one_minus_alpha_bar(t)
        assert result.shape == (3, 1, 1, 1)

    def test_diffusion_sample_output_shape(self):
        """Diffusion sample should produce latent with correct shape."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        text_emb = np.random.randn(1, 10, 64).astype(np.float32)
        latents = diffusion.sample(text_emb, num_steps=5)
        assert latents.shape == (1, 32, 7, 7)

    def test_unet_no_context(self):
        """UNet should work without text context."""
        unet = LatentUNet(in_channels=32, model_channels=64, out_channels=32)
        B = 1
        x = np.random.randn(B, 32, 7, 7).astype(np.float32)
        timesteps = np.array([500])
        from domains.training.slonet import Tensor
        x_tensor = Tensor(x, requires_grad=False)
        noise_pred = unet.forward(x_tensor, timesteps)
        assert noise_pred.data.shape == (B, 32, 7, 7)

    def test_diffusion_num_timesteps(self):
        """Diffusion model should use configured num_timesteps."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=200)
        assert diffusion.num_timesteps == 200

    def test_diffusion_latent_dim(self):
        """Diffusion model should store latent_dim."""
        diffusion = LatentDiffusionModel(latent_dim=32, model_channels=64, num_timesteps=100)
        assert diffusion.latent_dim == 32


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

    def test_text_encoder_auto_trains_tokenizer(self):
        """Text encoder should auto-train tokenizer on first encode."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        embeddings = encoder.encode_text(["hello world"])
        assert encoder.tokenizer._built
        assert embeddings.ndim == 3

    def test_text_encoder_encode_tokens(self):
        """Text encoder should encode token IDs."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        embeddings = encoder.encode_tokens(token_ids)
        assert embeddings.data.shape == (1, 5, 64)

    def test_text_encoder_vocab_size(self):
        """Text encoder should store vocab size."""
        encoder = TextEncoder(vocab_size=512, embed_dim=64, n_heads=2, n_layers=2)
        assert encoder.vocab_size == 512

    def test_text_encoder_embed_dim(self):
        """Text encoder should store embed dim."""
        encoder = TextEncoder(vocab_size=256, embed_dim=128, n_heads=2, n_layers=2)
        assert encoder.embed_dim == 128

    def test_text_encoder_max_seq_len(self):
        """Text encoder should store max seq len."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2, max_seq_len=32)
        assert encoder.max_seq_len == 32

    def test_text_encoder_parameters(self):
        """Text encoder should have parameters."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        params = encoder.parameters()
        assert len(params) > 0

    def test_text_encoder_multiple_texts(self):
        """Text encoder should encode multiple texts."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        embeddings = encoder.encode_text(["hello", "world", "test"])
        assert embeddings.shape[0] == 3

    def test_text_encoder_embedding_dim_match(self):
        """Text encoder embedding dim should match output dim."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        embeddings = encoder.encode_text(["test"])
        assert embeddings.shape[2] == 64

    def test_text_encoder_train_tokenizer(self):
        """Text encoder train_tokenizer should build tokenizer."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, n_heads=2, n_layers=2)
        encoder.train_tokenizer(["hello world", "foo bar"])
        assert encoder.tokenizer._built


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
        cv2 = pytest.importorskip("cv2")
        processor = VideoProcessor(max_frames=8)

        # Test with non-existent file (should fallback to random frames)
        frames = processor.extract_frames("/nonexistent/video.mp4", num_frames=4)
        assert len(frames) == 4
        assert frames[0].shape == (224, 224, 3)

    def test_temporal_encoder_different_lengths(self):
        """Temporal encoder should handle different frame counts."""
        encoder = TemporalEncoder(embed_dim=64, n_heads=2, n_layers=2, max_frames=8)
        for n_frames in [1, 3, 5, 8]:
            frames = np.random.randn(1, n_frames, 64).astype(np.float32)
            output = encoder.forward(frames)
            assert output.data.shape == (1, n_frames, 64)

    def test_temporal_encoder_parameters(self):
        """Temporal encoder should have parameters."""
        encoder = TemporalEncoder(embed_dim=64, n_heads=2, n_layers=2, max_frames=8)
        params = encoder.parameters()
        assert len(params) > 0

    def test_temporal_encoder_embed_dim(self):
        """Temporal encoder should store embed_dim."""
        encoder = TemporalEncoder(embed_dim=128, n_heads=2, n_layers=2, max_frames=16)
        assert encoder.embed_dim == 128

    def test_temporal_encoder_max_frames(self):
        """Temporal encoder should store max_frames."""
        encoder = TemporalEncoder(embed_dim=64, n_heads=2, n_layers=2, max_frames=16)
        assert encoder.max_frames == 16

    def test_video_processor_init(self):
        """Video processor should initialize correctly."""
        processor = VideoProcessor(embed_dim=64, n_heads=2, n_temporal_layers=2, max_frames=8)
        assert processor.embed_dim == 64
        assert processor.max_frames == 8

    def test_video_processor_temporal_encoder(self):
        """Video processor should have temporal encoder."""
        processor = VideoProcessor(embed_dim=64, n_heads=2, n_temporal_layers=2, max_frames=8)
        assert processor.temporal_encoder is not None

    def test_video_processor_parameters(self):
        """Video processor should have parameters."""
        processor = VideoProcessor(embed_dim=64, n_heads=2, n_temporal_layers=2, max_frames=8)
        params = processor.parameters()
        assert len(params) > 0

    def test_temporal_encoder_single_frame(self):
        """Temporal encoder should handle single frame."""
        encoder = TemporalEncoder(embed_dim=64, n_heads=2, n_layers=2, max_frames=8)
        frames = np.random.randn(1, 1, 64).astype(np.float32)
        output = encoder.forward(frames)
        assert output.data.shape == (1, 1, 64)

    def test_extract_frames_num_frames_default(self):
        """Video processor extract_frames should use max_frames default."""
        cv2 = pytest.importorskip("cv2")
        processor = VideoProcessor(max_frames=4)
        frames = processor.extract_frames("/nonexistent/video.mp4")
        assert len(frames) == 4

    def test_extract_frames_output_shape(self):
        """Video processor extract_frames should return 224x224x3 frames."""
        cv2 = pytest.importorskip("cv2")
        processor = VideoProcessor(max_frames=4)
        frames = processor.extract_frames("/nonexistent/video.mp4", num_frames=4)
        for frame in frames:
            assert frame.shape == (224, 224, 3)


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

    def test_griffin_lim_empty_mel(self):
        """Griffin-Lim should handle empty mel spectrogram."""
        vocoder = GriffinLimVocoder(n_mels=80, sample_rate=22050)
        mel_spec = np.zeros((80, 1), dtype=np.float32)
        waveform = vocoder.generate_waveform(mel_spec, num_iterations=4)
        assert waveform.ndim == 1

    def test_griffin_lim_custom_params(self):
        """Griffin-Lim should accept custom parameters."""
        vocoder = GriffinLimVocoder(n_fft=512, hop_length=128, n_mels=40, sample_rate=16000)
        assert vocoder.n_fft == 512
        assert vocoder.hop_length == 128
        assert vocoder.n_mels == 40
        assert vocoder.sample_rate == 16000

    def test_griffin_lim_hz_to_mel(self):
        """Griffin-Lim should convert Hz to mel."""
        vocoder = GriffinLimVocoder()
        mel = vocoder._hz_to_mel(700.0)
        assert mel > 0

    def test_griffin_lim_mel_to_hz(self):
        """Griffin-Lim should convert mel to Hz."""
        vocoder = GriffinLimVocoder()
        hz = vocoder._mel_to_hz(1000.0)
        assert hz > 0

    def test_griffin_lim_stft_istft_roundtrip(self):
        """STFT -> ISTFT should approximately reconstruct signal."""
        vocoder = GriffinLimVocoder(n_fft=256, hop_length=64)
        signal = np.random.randn(1024).astype(np.float32)
        spec = vocoder._stft(signal)
        reconstructed = vocoder._istft(spec)
        assert reconstructed.shape == signal.shape

    def test_griffin_lim_mel_basis_shape(self):
        """Griffin-Lim mel basis should have correct shape."""
        vocoder = GriffinLimVocoder(n_fft=256, n_mels=40)
        assert vocoder.mel_basis.shape == (40, 129)

    def test_tts_engine_sample_rate(self):
        """TTS engine should store sample rate."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40, sample_rate=16000)
        assert tts.sample_rate == 16000

    def test_tts_engine_decoder(self):
        """TTS engine should have decoder."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        assert tts.decoder is not None

    def test_tts_engine_vocoder(self):
        """TTS engine should have vocoder."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        assert tts.vocoder is not None

    def test_tts_engine_parameters(self):
        """TTS engine should have parameters."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        params = tts.parameters()
        assert len(params) > 0

    def test_tts_text_to_waveform_empty(self):
        """TTS should handle empty text."""
        tts = TTSEngine(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        waveform = tts.text_to_waveform("", max_frames=10)
        assert waveform.ndim == 1

    def test_spectrogram_decoder_encode_text(self):
        """Spectrogram decoder should encode text to hidden states."""
        from domains.multimodal.tts import SpectrogramDecoder
        decoder = SpectrogramDecoder(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        phoneme_ids = np.array([[65, 66, 67]], dtype=np.int32)
        enc_out, h, c = decoder.encode_text(phoneme_ids)
        assert enc_out.data.ndim == 3

    def test_spectrogram_decoder_generate(self):
        """Spectrogram decoder should generate mel spectrogram."""
        from domains.multimodal.tts import SpectrogramDecoder
        decoder = SpectrogramDecoder(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        phoneme_ids = np.array([[65, 66, 67]], dtype=np.int32)
        mel = decoder.generate(phoneme_ids, max_frames=10)
        assert mel.ndim == 2
        assert mel.shape[0] == 40

    def test_spectrogram_decoder_parameters(self):
        """Spectrogram decoder should have parameters."""
        from domains.multimodal.tts import SpectrogramDecoder
        decoder = SpectrogramDecoder(vocab_size=256, embed_dim=64, hidden_dim=128, n_mels=40)
        params = decoder.parameters()
        assert len(params) > 0
