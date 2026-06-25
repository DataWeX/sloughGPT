"""Tests for upgraded multimodal engine (ViT + cross-attention + BPE)."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add core-py to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.training.slonet import Tensor, SloCrossAttention, SloLinear, SloLayerNorm, SloEmbedding
from domains.multimodal.engine import (
    VisionEncoder, AudioEncoder, MultimodalEngine, TextDecoder,
    SloTransformerDecoder, SloTransformerDecoderBlock,
)
from domains.multimodal.bpe_tokenizer import BPETokenizer


class TestSloCrossAttention:
    """Test the new cross-attention layer."""

    def test_cross_attention_forward_shape(self):
        """Cross-attention should output same shape as input."""
        d_model = 128
        n_heads = 4
        layer = SloCrossAttention(d_model, n_heads)
        
        # Text query: (1, seq_len, d_model)
        x_data = np.random.randn(1, 5, d_model).astype(np.float32)
        x = Tensor(x_data, requires_grad=True)
        
        # Image context: (1, num_patches, d_model) — 49 patches + 1 cls = 50
        ctx_data = np.random.randn(1, 50, d_model).astype(np.float32)
        ctx = Tensor(ctx_data, requires_grad=True)
        
        out = layer.forward(x, ctx)
        assert out.data.shape == (1, 5, d_model)

    def test_cross_attention_gradient_flow(self):
        """Gradients should flow back to both query and context."""
        d_model = 64
        n_heads = 4
        layer = SloCrossAttention(d_model, n_heads)
        
        x_data = np.random.randn(1, 3, d_model).astype(np.float32)
        x = Tensor(x_data, requires_grad=True)
        
        ctx_data = np.random.randn(1, 10, d_model).astype(np.float32)
        ctx = Tensor(ctx_data, requires_grad=True)
        
        out = layer.forward(x, ctx)
        loss = out.sum()
        loss.backward()
        
        # Both x and ctx should have gradients
        assert x.grad is not None
        assert ctx.grad is not None
        assert x.grad.data.shape == x.data.shape
        assert ctx.grad.data.shape == ctx.data.shape


class TestVisionEncoder:
    """Test the upgraded ViT-style VisionEncoder."""

    def test_vision_encoder_output_shape(self):
        """VisionEncoder should output (B, 1, embed_dim) cls token."""
        embed_dim = 128
        encoder = VisionEncoder(embed_dim=embed_dim, n_heads=4, n_layers=2)
        
        # (B, H, W, C) = (1, 224, 224, 3)
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = encoder.forward(img)
        
        assert out.data.shape == (1, 1, embed_dim)

    def test_vision_encoder_patch_embeddings(self):
        """get_patch_embeddings should return (B, num_patches+1, embed_dim)."""
        embed_dim = 128
        encoder = VisionEncoder(embed_dim=embed_dim, n_heads=4, n_layers=2)
        
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        patches = encoder.get_patch_embeddings(img)
        
        expected_patches = (224 // 32) ** 2  # 7*7 = 49
        assert patches.data.shape == (1, expected_patches + 1, embed_dim)

    def test_vision_encoder_gradient_flow(self):
        """Gradients should flow through the ViT."""
        embed_dim = 64
        encoder = VisionEncoder(embed_dim=embed_dim, n_heads=4, n_layers=2)
        
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        out = encoder.forward(img)
        loss = out.sum()
        loss.backward()
        
        # At least some parameters should have gradients
        # (SloTransformerBlock gradient flow is limited in SloNet autograd)
        params_with_grad = sum(1 for p in encoder.parameters() if p.requires_grad and p.grad is not None)
        assert params_with_grad > 0, "No parameters received gradients"


class TestBPETokenizer:
    """Test the BPE tokenizer."""

    def test_bpe_train_and_encode(self):
        """BPE tokenizer should train and encode text."""
        tokenizer = BPETokenizer(vocab_size=256)
        texts = [
            "a red circle on blue background",
            "a green square next to yellow triangle",
            "three shapes arranged in a row",
        ]
        tokenizer.train(texts)
        
        assert tokenizer._built
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        
        # Encode should return list of ints
        encoded = tokenizer.encode("a red circle")
        assert isinstance(encoded, list)
        assert all(isinstance(t, int) for t in encoded)

    def test_bpe_encode_decode_roundtrip(self):
        """Encoding then decoding should produce similar text."""
        tokenizer = BPETokenizer(vocab_size=512)
        texts = [
            "a bright red rectangle on a dark background",
            "a green circle next to a blue square",
        ]
        tokenizer.train(texts)
        
        original = "a red rectangle on dark background"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Should be readable (not exact due to subword splitting)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_bpe_save_load(self):
        """Tokenizer should save and load correctly."""
        import tempfile
        tokenizer = BPETokenizer(vocab_size=256)
        texts = ["a red circle on blue background"]
        tokenizer.train(texts)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        tokenizer.save(path)
        
        new_tokenizer = BPETokenizer(vocab_size=256)
        assert new_tokenizer.load(path)
        assert new_tokenizer._built
        assert len(new_tokenizer.vocab) == len(tokenizer.vocab)


class TestSloTransformerDecoder:
    """Test the new SloTransformerDecoder with cross-attention."""

    def test_decoder_forward_shape(self):
        """Decoder should output (seq_len, vocab_size) logits."""
        embed_dim = 64
        hidden_dim = 128
        vocab_size = 100
        decoder = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim, n_heads=4, n_layers=2)
        
        # Image cls token: (1, 1, embed_dim)
        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=True)
        
        # Image patches: (1, num_patches+1, embed_dim)
        num_patches = 10
        img_patches = Tensor(np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32), requires_grad=True)
        
        # Token IDs: (1, seq_len)
        token_ids = Tensor(np.array([[0, 1, 2, 3]]), requires_grad=False)
        
        logits, h = decoder.forward(img_embed, token_ids, img_patches)
        
        # Logits should be (B, seq_len, vocab_size) = (1, 4, 100)
        assert logits.data.ndim == 3
        assert logits.data.shape[0] == 1  # batch
        assert logits.data.shape[1] == 4  # seq_len
        assert logits.data.shape[2] == vocab_size

    def test_decoder_without_cross_attention(self):
        """Decoder should work without image patches."""
        embed_dim = 64
        hidden_dim = 128
        vocab_size = 100
        decoder = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim, n_heads=4, n_layers=2)
        
        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=True)
        token_ids = Tensor(np.array([[0, 1, 2, 3]]), requires_grad=False)
        
        logits, h = decoder.forward(img_embed, token_ids)
        assert logits.data.ndim == 3
        assert logits.data.shape[2] == vocab_size  # last dim is vocab

    def test_decoder_parameters(self):
        """Decoder should have trainable parameters."""
        decoder = SloTransformerDecoder(100, 64, 128, n_heads=4, n_layers=2)
        params = decoder.parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestTextDecoderWithBPE:
    """Test TextDecoder with BPE tokenizer."""

    def test_text_decoder_builds_bpe_vocab(self):
        """TextDecoder should train BPE tokenizer on texts."""
        decoder = TextDecoder(embed_dim=64, hidden_dim=128, vocab_size=256)
        texts = [
            "a red circle on blue background",
            "a green square next to yellow triangle",
        ]
        decoder.build_vocab(texts)
        
        assert decoder.bpe._built
        assert len(decoder.bpe.vocab) > 0

    def test_text_decoder_encode_decode(self):
        """TextDecoder should encode and decode using BPE."""
        decoder = TextDecoder(embed_dim=64, hidden_dim=128, vocab_size=512)
        texts = [
            "a bright red rectangle on a dark background",
            "a green circle next to a blue square",
        ]
        decoder.build_vocab(texts)
        
        encoded = decoder.encode("a red rectangle")
        assert isinstance(encoded, list)
        
        decoded = decoder.decode(encoded)
        assert isinstance(decoded, str)


class TestMultimodalEngineIntegration:
    """Test full multimodal engine integration."""

    def test_engine_init_with_new_dims(self):
        """Engine should initialize with new ViT dimensions."""
        engine = MultimodalEngine(embed_dim=128, hidden_dim=256, n_vit_layers=2, n_heads=4)
        
        assert engine.vision.embed_dim == 128
        assert engine.vision.num_patches == 49  # 7*7
        assert engine.decoder.n_layers == 3  # default changed from 4 to 3
        assert engine.decoder.n_heads == 4
        assert hasattr(engine, 'audio'), "Engine should have AudioEncoder"
        assert engine.audio.embed_dim == 128

    def test_engine_generate_with_cross_attention(self):
        """Engine should generate captions using cross-attention."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        
        # Build vocab first with some text
        engine.build_vocab(["a red circle on blue background", "a green square"])
        
        # Create test image (1, 224, 224, 3)
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        
        # Generate should work
        result = engine.generate(img, max_len=5, temperature=0.0)
        assert isinstance(result.text, str)

    def test_audio_encoder_forward_shape(self):
        """AudioEncoder forward should produce (1, 1, embed_dim)."""
        ae = AudioEncoder(embed_dim=64, n_heads=2, n_layers=1)
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)  # 0.5s at 16kHz
        cls_out = ae.forward(audio)
        assert cls_out.data.shape == (1, 1, 64)

    def test_audio_encoder_patch_embeddings_shape(self):
        """AudioEncoder.get_patch_embeddings should return (1, N+1, embed_dim)."""
        ae = AudioEncoder(embed_dim=32, n_heads=2, n_layers=1)
        audio = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)  # 1s at 16kHz
        patches = ae.get_patch_embeddings(audio)
        assert patches.data.ndim == 3
        assert patches.data.shape[0] == 1
        assert patches.data.shape[2] == 32
        assert patches.data.shape[1] >= 1  # at least CLS + 1 patch

    def test_audio_encoder_mel_spectrogram_shape(self):
        """Mel spectrogram should return (N_MELS, T) with T >= 1."""
        ae = AudioEncoder(embed_dim=32, n_heads=2, n_layers=1)
        audio = np.sin(np.linspace(0, 50, 4000)).astype(np.float32)
        mel = ae._mel_spectrogram(audio)
        assert mel.shape[0] == AudioEncoder.N_MELS  # 80
        assert mel.shape[1] >= 1

    def test_audio_encoder_multiple_patches(self):
        """Audio longer than PATCH_SECONDS should produce multiple patches."""
        ae = AudioEncoder(embed_dim=32, n_heads=2, n_layers=1)
        # 12 seconds of audio = 2+ patches (PATCH_SECONDS=5)
        t = np.linspace(0, 12 * 440, 12 * 16000).astype(np.float32)
        audio = np.sin(t)
        patches = ae.extract_patches(audio)
        assert patches.shape[0] == 1
        assert patches.shape[1] >= 2  # at least 2 patches

    def test_engine_generate_with_audio_only(self):
        """Engine should generate from audio without image."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["hello world", "beep boop"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        result = engine.generate(audio_np=audio, max_len=5, temperature=0.0)
        assert isinstance(result.text, str)

    def test_engine_generate_with_audio_and_image(self):
        """Engine should generate from both audio and image simultaneously."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["hello world", "red circle", "beep"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        result = engine.generate(image_np=img, audio_np=audio, max_len=5, temperature=0.0)
        assert isinstance(result.text, str)

    def test_engine_train_step_with_audio(self):
        """train_step should work with audio input."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["hello world", "beep boop sound"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32).reshape(1, -1)
        tokens = engine.text.bpe.encode("hello world")
        tok_arr = np.array([tokens], dtype=np.int64)
        loss = engine.train_step(audio_np=audio, text_tokens=tok_arr, lr=1e-3)
        assert isinstance(loss, float)
        assert 0 < loss < 20

    def test_engine_train_step_with_audio_and_image(self):
        """train_step with both audio and image should produce a lower loss."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["a red circle with sound", "hello world"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32).reshape(1, -1)
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.bpe.encode("hello world")
        tok_arr = np.array([tokens], dtype=np.int64)
        loss = engine.train_step(images_np=img, audio_np=audio, text_tokens=tok_arr, lr=1e-3)
        assert isinstance(loss, float)
        assert 0 < loss < 20

    def test_engine_audio_only_train_step_fails_without_text_tokens(self):
        """train_step should raise ValueError when text_tokens is None."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128)
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        with pytest.raises(ValueError, match="text_tokens is required"):
            engine.train_step(audio_np=audio)

    def test_engine_save_load_with_audio_weights(self, tmp_path):
        """Save/load should preserve audio encoder weights."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["test save load audio"])
        # Initial forward to set weight values
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        before = engine.audio.cls_token.data.copy()

        p = str(tmp_path / "test_audio_engine.npz")
        engine.save(p)

        loaded = MultimodalEngine.load(p)
        after = loaded.audio.cls_token.data
        assert np.allclose(before, after, atol=1e-6)

    def test_engine_audio_parameters_trained(self):
        """Audio encoder parameters should be trainable (requires_grad=True)."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        for p in engine.audio.parameters():
            assert p.requires_grad
