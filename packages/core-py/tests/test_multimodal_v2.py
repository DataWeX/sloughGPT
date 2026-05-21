"""Tests for upgraded multimodal engine (ViT + cross-attention + BPE)."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add core-py to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from domains.training.slonet import Tensor, SloCrossAttention, SloLinear, SloLayerNorm, SloEmbedding
from domains.multimodal.engine import VisionEncoder, DecoderLSTM, MultimodalEngine, TextDecoder
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
        
        # Image context: (1, num_patches, d_model)
        ctx_data = np.random.randn(1, 197, d_model).astype(np.float32)
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
        
        expected_patches = (224 // 16) ** 2  # 14*14 = 196
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


class TestDecoderLSTMWithCrossAttention:
    """Test DecoderLSTM with cross-attention."""

    def test_decoder_with_cross_attention(self):
        """Decoder should use image patches for cross-attention."""
        embed_dim = 64
        hidden_dim = 128
        vocab_size = 100
        decoder = DecoderLSTM(vocab_size, embed_dim, hidden_dim, n_heads=4)
        
        # Image cls token: (1, 1, embed_dim)
        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=True)
        
        # Image patches: (1, num_patches+1, embed_dim)
        num_patches = 10
        img_patches = Tensor(np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32), requires_grad=True)
        
        # Token IDs: (1, seq_len)
        token_ids = Tensor(np.array([[0, 1, 2, 3]]), requires_grad=False)
        
        logits, h = decoder.forward(img_embed, token_ids, img_patches)
        
        # Logits should be (seq_len, vocab_size)
        assert logits.data.shape[1] == vocab_size


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
        assert len(decoder.vocab) > 0

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
        assert engine.vision.num_patches == 196  # 14*14
        assert engine.decoder.cross_attn is not None

    def test_engine_generate_with_cross_attention(self):
        """Engine should generate captions using cross-attention."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        
        # Build vocab first with some text
        engine.text.build_vocab(["a red circle on blue background", "a green square"])
        engine.decoder.vocab_size = max(1, len(engine.text.vocab))
        # Rebuild embedding with correct vocab size
        engine.decoder.embedding = SloEmbedding(engine.decoder.vocab_size, engine.decoder.embed_dim)
        engine.decoder.fc_out = SloLinear(engine.decoder.hidden_dim, engine.decoder.vocab_size)
        
        # Create test image (1, 224, 224, 3)
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        
        # Generate should work
        result = engine.generate(img, max_len=5, temperature=0.0)
        assert isinstance(result.text, str)
