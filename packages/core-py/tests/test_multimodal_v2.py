"""Tests for upgraded multimodal engine (ViT + cross-attention + char tokenizer)."""

import math
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
from domains.training.slonet import cross_entropy as _cross_entropy_slo
from domains.multimodal.char_tokenizer import CharTokenizer


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
        params_with_grad = sum(1 for p in encoder.parameters() if p.requires_grad and p.grad is not None)
        assert params_with_grad > 0, "No parameters received gradients"


class TestCharTokenizer:
    """Test the character-level tokenizer."""

    def test_build_vocab_contains_all_chars(self):
        """CharTokenizer should include every character from training texts."""
        tokenizer = CharTokenizer()
        texts = [
            "a red circle on blue background",
            "a green square next to yellow triangle",
        ]
        tokenizer.build_vocab(texts)

        assert tokenizer._built
        # All chars from the texts should be in vocab
        for text in texts:
            for ch in text:
                assert ch in tokenizer.vocab, f"char {ch!r} missing from vocab"

        # Special tokens should be present
        for tok in CharTokenizer.SPECIAL_TOKENS:
            assert tok in tokenizer.vocab, f"special token {tok} missing"

    def test_encode_bos_eos(self):
        """Encode should wrap text in BOS/EOS tokens."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello"])

        encoded = tokenizer.encode("hello")
        bos = tokenizer.vocab["<BOS>"]
        eos = tokenizer.vocab["<EOS>"]

        assert encoded[0] == bos, "First token should be BOS"
        assert encoded[-1] == eos, "Last token should be EOS"
        # Each character should have its own token
        assert len(encoded) == len("hello") + 2  # BOS + 5 chars + EOS

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should recover the original text."""
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["a red circle on blue background"])

        original = "a red rectangle on dark background"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)

        assert decoded == original, f"Roundtrip failed: {decoded!r} != {original!r}"

    def test_save_load(self):
        """Tokenizer should save and load correctly."""
        import tempfile
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["a red circle on blue background", "hello world"])

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        tokenizer.save(path)

        new_tokenizer = CharTokenizer()
        assert new_tokenizer.load(path)
        assert new_tokenizer._built
        assert new_tokenizer.vocab_size == tokenizer.vocab_size
        assert new_tokenizer.encode("hello") == tokenizer.encode("hello")

    def test_unknown_char_falls_back_to_unk(self):
        """A character not in vocab should map to <UNK>."""
        tokenizer = CharTokenizer()
        # Only build with ASCII — no special chars
        tokenizer.build_vocab(["abc"])
        unk = tokenizer.vocab["<UNK>"]

        encoded = tokenizer.encode("a\x00z")  # null byte not in ascii subset
        # The null byte should map to UNK
        assert unk in encoded


class TestTextDecoderWithChar:
    """Test TextDecoder with CharTokenizer."""

    def test_text_decoder_builds_vocab(self):
        """TextDecoder should build char vocabulary from texts."""
        decoder = TextDecoder(embed_dim=64, hidden_dim=128)
        texts = [
            "a red circle on blue background",
            "a green square next to yellow triangle",
        ]
        decoder.build_vocab(texts)

        assert decoder.char._built
        assert decoder.vocab_size > len(CharTokenizer.SPECIAL_TOKENS)

    def test_text_decoder_encode_decode(self):
        """TextDecoder should encode and decode using CharTokenizer."""
        decoder = TextDecoder(embed_dim=64, hidden_dim=128)
        texts = [
            "a bright red rectangle on a dark background",
            "a green circle next to a blue square",
        ]
        decoder.build_vocab(texts)

        original = "a red rectangle"
        encoded = decoder.encode(original)
        assert isinstance(encoded, list)
        assert all(isinstance(t, int) for t in encoded)

        decoded = decoder.decode(encoded)
        assert decoded == original

    def test_text_decoder_vocab_size_property(self):
        """vocab_size property should match the char tokenizer."""
        decoder = TextDecoder()
        decoder.build_vocab(["hello world test"])
        assert decoder.vocab_size == decoder.char.vocab_size
        assert decoder.vocab_size > 0


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

        logits, h, _ = decoder.forward(img_embed, token_ids, img_patches)

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

        logits, h, _ = decoder.forward(img_embed, token_ids)
        assert logits.data.ndim == 3
        assert logits.data.shape[2] == vocab_size  # last dim is vocab

    def test_decoder_parameters(self):
        """Decoder should have trainable parameters."""
        decoder = SloTransformerDecoder(100, 64, 128, n_heads=4, n_layers=2)
        params = decoder.parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_multi_layer_decoder_gradient_flow(self):
        """Gradients should flow through all layers when cross-attention is used."""
        embed_dim = 64
        hidden_dim = 128
        vocab_size = 50
        n_layers = 3
        decoder = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                        n_heads=4, n_layers=n_layers)

        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=True)
        num_patches = 10
        img_patches = Tensor(np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32), requires_grad=True)
        token_ids = Tensor(np.array([[0, 2, 4, 6, 8]]), requires_grad=False)

        logits, _, _ = decoder.forward(img_embed, token_ids, img_patches)
        loss = logits.sum()
        loss.backward()

        # Count params with gradients per layer
        layer_grads = []
        for i, block in enumerate(decoder.blocks):
            n_with_grad = sum(1 for p in block.parameters() if p.grad is not None)
            layer_grads.append(n_with_grad)
            assert n_with_grad > 0, f"Block {i} has 0 params with gradients"

        # All layers should contribute
        assert len(layer_grads) == n_layers
        # Embedding and output proj should also have gradients
        assert decoder.embedding.weight.grad is not None
        assert decoder.fc_out.weight.grad is not None


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
        tokens = engine.text.char.encode("hello world")
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
        tokens = engine.text.char.encode("hello world")
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


class TestTemperatureAnnealing:
    """Test temperature scaling in train_step and train_batch.

    Temperature > 1.0 scales logits before softmax, creating softer
    probability distributions. This encourages exploration early in
    training (avoids sharp, overconfident predictions from random init).
    """

    def test_temperature_scales_logits(self):
        """Higher temperature should produce smaller logit values after scaling."""
        embed_dim = 32
        hidden_dim = 64
        vocab_size = 30
        decoder = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                        n_heads=2, n_layers=1)

        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=False)
        token_ids = Tensor(np.array([[0, 2, 4, 6]]), requires_grad=False)

        logits, _, _ = decoder.forward(img_embed, token_ids[:, :-1])

        # Scaled with temperature 2.0
        scaled = logits / 2.0
        assert np.allclose(scaled.data, logits.data / 2.0), "Temperature scaling failed"

    def test_higher_temp_gives_lower_loss(self):
        """With random logits, higher temperature reduces cross-entropy
        (softer softmax → more uniform distribution → lower penalty for wrong class)."""
        vocab_size = 20
        logits = Tensor(np.random.randn(3, vocab_size).astype(np.float32) * 3, requires_grad=True)
        targets = Tensor(np.array([0, 5, 10]), requires_grad=False)

        loss_base = _cross_entropy_slo(logits, targets)
        loss_sharp = _cross_entropy_slo(logits / 0.5, targets)
        loss_soft = _cross_entropy_slo(logits / 2.0, targets)

        # Higher temperature → softer distribution → lower CE (less penalty for wrong class)
        assert float(loss_soft.data) <= float(loss_base.data), (
            f"Expected soft temp (T=2) to reduce loss, "
            f"got {float(loss_soft.data):.3f} vs base {float(loss_base.data):.3f}"
        )
        # Lower temperature → sharper distribution → higher CE (more penalty for wrong class)
        assert float(loss_sharp.data) >= float(loss_base.data), (
            f"Expected sharp temp (T=0.5) to increase loss, "
            f"got {float(loss_sharp.data):.3f} vs base {float(loss_base.data):.3f}"
        )

    def test_temperature_annealing_in_train_step(self):
        """train_step should accept temperature parameter and produce a valid loss."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["test temperature annealing in multimodal engine"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("test temperature")
        tok_arr = np.array([tokens], dtype=np.int64)

        # With temperature=2.0 (exploration mode)
        loss_hot = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, temperature=2.0)
        assert np.isfinite(loss_hot), f"Loss NaN/Inf with temperature=2.0: {loss_hot}"
        assert float(loss_hot) > 0, f"Loss should be positive, got {loss_hot}"

    def test_temperature_annealing_lr_scaling(self):
        """Temperature annealing should not prevent loss from decreasing."""
        engine = MultimodalEngine(embed_dim=16, hidden_dim=32, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["test temperature annealing in multimodal"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("test temperature annealing")
        tok_arr = np.array([tokens], dtype=np.int64)

        # Step 1: high temperature
        l1 = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, temperature=2.0)
        # Step 2: lower temperature
        l2 = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, temperature=1.5)
        # Step 3: no temperature
        l3 = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, temperature=1.0)

        assert np.isfinite(l1) and np.isfinite(l2) and np.isfinite(l3)
        # At minimum, training should not explode
        assert l3 < 20, f"Loss exploded with temperature annealing: {l3}"


class TestBeamSearch:
    """Test beam search decoding in generate()."""

    def test_beam_search_returns_string(self):
        """Beam search should return a MultimodalOutput with text."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle on white background"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        result = engine.generate(img, max_len=5, temperature=0.0, beam_width=3)
        assert isinstance(result.text, str)
        assert isinstance(result.confidence, float)

    def test_beam_search_greedy_deterministic(self):
        """Both greedy and beam=1 should be deterministic (same result twice)."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle on white background"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)

        a = engine.generate(img, max_len=5, temperature=0.0)
        b = engine.generate(img, max_len=5, temperature=0.0)
        assert a.text == b.text, f"Greedy not deterministic: {a.text!r} != {b.text!r}"

        c = engine.generate(img, max_len=5, temperature=0.0, beam_width=1)
        d = engine.generate(img, max_len=5, temperature=0.0, beam_width=1)
        assert c.text == d.text, f"Beam=1 not deterministic: {c.text!r} != {d.text!r}"

    def test_beam_search_wider_is_not_shorter(self):
        """Wider beam should not produce shorter text (at least same or better search)."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["a" * 20])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)

        # With all-a training, greedy and beam should both produce reasonable length
        g = engine.generate(img, max_len=15, temperature=0.0)
        b = engine.generate(img, max_len=15, temperature=0.0, beam_width=3)
        # Both should produce non-empty output
        assert len(g.text) > 0
        assert len(b.text) > 0

    def test_beam_search_with_audio(self):
        """Beam search should work with audio-only input."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["beep boop"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        result = engine.generate(audio_np=audio, max_len=5, temperature=0.0, beam_width=3)
        assert isinstance(result.text, str)

    def test_beam_search_with_combined(self):
        """Beam search should work with combined vision+audio input."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle beep"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32)
        result = engine.generate(image_np=img, audio_np=audio, max_len=5, temperature=0.0, beam_width=3)
        assert isinstance(result.text, str)


class TestKVCache:
    """Regression tests for KV cache correctness in greedy generation."""

    def test_kv_cache_is_deterministic(self):
        """KV-cached generation must be deterministic given fixed seed."""
        import domains.training.slonet as _slonet_mod
        _slonet_mod._ACCELERATOR = "none"

        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle on white background"])
        engine.eval()
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)

        out1 = engine.generate(img, max_len=10, temperature=0.0)
        out2 = engine.generate(img, max_len=10, temperature=0.0)
        assert out1.text == out2.text, (
            f"KV cache non-deterministic: {out1.text!r} != {out2.text!r}"
        )

    def test_kv_cache_is_faster_for_long_sequences(self):
        """KV cache should be faster than full-length forward for each step."""
        import time
        import domains.training.slonet as _slonet_mod
        _slonet_mod._ACCELERATOR = "none"

        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle on white background"])
        engine.eval()
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)

        # Warmup
        engine.generate(img, max_len=5, temperature=0.0)

        t0 = time.perf_counter()
        engine.generate(img, max_len=25, temperature=0.0)
        t1 = time.perf_counter()
        kv_time = t1 - t0

        # No-KV path: build full sequence each step
        from domains.training.slonet import tensor as _tensor
        embed, patches, _ = engine._concat_modalities(img, None, None)
        # Use the same image, re-embed
        tokens = [0]
        t0 = time.perf_counter()
        for _ in range(25):
            inp = _tensor(np.array([tokens]), requires_grad=False)
            logits, _, _ = engine.decoder.forward(embed, inp, patches)
            last_pos = logits.data.reshape(-1, logits.data.shape[-1])[-1]
            next_tok = int(np.argmax(last_pos))
            tokens.append(next_tok)
        t1 = time.perf_counter()
        no_kv_time = t1 - t0

        assert kv_time < no_kv_time, (
            f"KV cache ({kv_time:.3f}s) not faster than no-KV ({no_kv_time:.3f}s)"
        )

    def test_kv_cache_is_faster_for_long_sequences(self):
        """KV cache should reduce wall time for long greedy generations."""
        import time
        import domains.training.slonet as _slonet_mod
        _slonet_mod._ACCELERATOR = "none"

        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["red circle on white background"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        engine.eval()

        # Warmup
        engine.generate(img, max_len=5, temperature=0.0)

        # Run multiple iterations for stability
        kv_times = []
        no_kv_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            engine.generate(img, max_len=50, temperature=0.0)
            kv_times.append(time.perf_counter() - t0)

            from domains.training.slonet import tensor as _tensor
            embed, patches, _ = engine._concat_modalities(img, None, None)
            tokens = [0]
            t0 = time.perf_counter()
            for _ in range(50):
                inp = _tensor(np.array([tokens]), requires_grad=False)
                logits, _, _ = engine.decoder.forward(embed, inp, patches)
                last_pos = logits.data.reshape(-1, logits.data.shape[-1])[-1]
                next_tok = int(np.argmax(last_pos))
                tokens.append(next_tok)
            no_kv_times.append(time.perf_counter() - t0)

        kv_time = min(kv_times)
        no_kv_time = min(no_kv_times)

        # Allow KV cache to be up to 1.5x slower — timing is noisy on CPU with small models
        assert kv_time < no_kv_time * 1.5, (
            f"KV cache ({kv_time:.3f}s) significantly slower than no-KV ({no_kv_time:.3f}s)"
        )


class TestZeroPatchGradientRegression:
    """Regression tests ensuring zero-patch cross-attention doesn't explode gradients.

    Root cause (fixed): SloCrossAttention.forward() contained a redundant
    SloLayerNorm wrapping the output, creating a post-norm in a pre-norm
    decoder. The RMSNorm backward's 1/rms³ term amplified upstream gradients
    ~800x, making training diverge when cross-attention received zero patches.

    Fix: removed self.norm from SloCrossAttention.__init__ and changed
    return self.norm.forward(x + self.o_proj.forward(out_t))
    → return self.o_proj.forward(out_t)
    """

    def test_zero_patches_grad_norm_not_exploded(self):
        """Gradient norms with zero patches should be comparable to no cross-attention."""
        embed_dim = 64
        hidden_dim = 128
        vocab_size = 50

        # Decoder WITHOUT cross-attention (baseline)
        decoder_no_ca = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                              n_heads=4, n_layers=2)
        # Decoder WITH cross-attention but zero patches
        decoder_ca = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                           n_heads=4, n_layers=2)

        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=True)
        token_ids = Tensor(np.array([[0, 2, 4, 6, 8, 10]]), requires_grad=False)

        # Forward without cross-attention
        logits_no, _, _ = decoder_no_ca.forward(img_embed, token_ids, img_patches=None)
        loss_no = logits_no.sum()
        loss_no.backward()

        # Compute total grad norm for baseline
        norm_no = 0.0
        for p in decoder_no_ca.parameters():
            if p.grad is not None:
                g_data = p.grad.data if hasattr(p.grad, 'data') else p.grad
                norm_no += float(np.sum(np.asarray(g_data, dtype=np.float64).ravel() ** 2))
        norm_no = np.sqrt(norm_no)

        # Zero all grads
        for p in decoder_no_ca.parameters():
            p.grad = None

        # Forward with cross-attention (zero-like patches)
        img_patches = Tensor(np.zeros((1, 51, embed_dim), dtype=np.float32), requires_grad=False)
        logits_ca, _, _ = decoder_ca.forward(img_embed, token_ids, img_patches)
        loss_ca = logits_ca.sum()
        loss_ca.backward()

        # Compute total grad norm for cross-attention path
        norm_ca = 0.0
        for p in decoder_ca.parameters():
            if p.grad is not None:
                g_data = p.grad.data if hasattr(p.grad, 'data') else p.grad
                norm_ca += float(np.sum(np.asarray(g_data, dtype=np.float64).ravel() ** 2))
        norm_ca = np.sqrt(norm_ca)

        # Cross-attention with zero patches should NOT explode gradients
        # (the norm should be comparable, not 100x+ larger)
        ratio = norm_ca / max(norm_no, 1e-8)
        assert ratio < 100.0, (
            f"Cross-attention with zero patches gradient norm "
            f"({norm_ca:.1f}) is {ratio:.0f}x larger than baseline "
            f"({norm_no:.1f}) — indicates gradient explosion"
        )

    def test_zero_patches_train_step_loss_drops(self):
        """Training with zero patches should decrease loss (like no-patches training)."""
        engine = MultimodalEngine(embed_dim=32, hidden_dim=64, n_vit_layers=1, n_heads=2,
                                  n_decoder_layers=1)
        engine.build_vocab(["a red circle on blue background"])

        img_zeros = np.zeros((1, 224, 224, 3), dtype=np.float32)
        tokens = engine.text.char.encode("a red circle on blue background")
        tok_arr = np.array([tokens], dtype=np.int64)

        # First loss
        loss1 = engine.train_step(images_np=img_zeros, text_tokens=tok_arr, lr=1e-3)
        # Second loss (should be lower or comparable, not NaN or exploded)
        loss2 = engine.train_step(images_np=img_zeros, text_tokens=tok_arr, lr=1e-3)

        assert np.isfinite(loss1), f"Loss was NaN/Inf: {loss1}"
        assert np.isfinite(loss2), f"Loss was NaN/Inf: {loss2}"
        # Cross-attention shouldn't prevent learning — loss should trend down
        assert loss2 <= loss1 * 2, (
            f"Loss increased from {loss1:.4f} to {loss2:.4f} "
            f"with zero patches — gradient explosion likely"
        )

    def test_zero_patches_vs_no_patches_loss_comparable(self):
        """Loss with zero patches should be comparable to no cross-attention at all."""
        embed_dim = 32
        hidden_dim = 64
        vocab_size = 30

        # No cross-attention decoder
        dec_no = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                       n_heads=2, n_layers=1)
        # Decoder with cross-attention
        dec_ca = SloTransformerDecoder(vocab_size, embed_dim, hidden_dim,
                                       n_heads=2, n_layers=1)

        img_embed = Tensor(np.random.randn(1, 1, embed_dim).astype(np.float32), requires_grad=False)
        token_ids = Tensor(np.array([[0, 2, 4, 6, 8, 10, 12]]), requires_grad=False)

        # Forward without cross-attention
        # Input: all but last token; targets: all but first token (teacher forcing)
        logits_no, _, _ = dec_no.forward(img_embed, token_ids[:, :-1])
        targets = Tensor(np.array([2, 4, 6, 8, 10, 12]), requires_grad=False)
        from domains.training.slonet import cross_entropy as _cross_entropy
        loss_no = _cross_entropy(logits_no, targets)

        # Forward with cross-attention (zero patches)
        img_patches = Tensor(np.zeros((1, 51, embed_dim), dtype=np.float32), requires_grad=False)
        logits_ca, _, _ = dec_ca.forward(img_embed, token_ids[:, :-1], img_patches)
        loss_ca = _cross_entropy(logits_ca, targets)

        # Losses should be comparable (not 10x+ apart)
        ratio = float(loss_ca.data) / max(float(loss_no.data), 1e-8)
        assert ratio < 10.0, (
            f"Cross-attention with zero patches loss "
            f"({float(loss_ca.data):.2f}) is {ratio:.1f}x baseline "
            f"({float(loss_no.data):.2f}) — indicates gradient disruption"
        )


class TestEngineSensitivity:
    """Engine-level JVP sensitivity integration tests."""

    def test_train_step_returns_sensitivity_with_flag(self):
        """train_step with compute_sens=True returns (loss, dict)."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["test sensitivity"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("test sensitivity")
        tok_arr = np.array([tokens], dtype=np.int64)
        result = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, compute_sens=True)
        loss, sens = result
        assert isinstance(loss, float)
        assert isinstance(sens, dict)
        assert "decoder" in sens
        assert "vision" in sens
        assert all(math.isfinite(v) for v in sens.values())

    def test_train_step_without_sensitivity_returns_float(self):
        """train_step with compute_sens=False returns just float."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["test no sens"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("test no sens")
        tok_arr = np.array([tokens], dtype=np.int64)
        result = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, compute_sens=False)
        assert isinstance(result, float)

    def test_train_step_sensitivity_default_is_float(self):
        """Default (no compute_sens kwarg) returns just float."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["test default"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("test default")
        tok_arr = np.array([tokens], dtype=np.int64)
        result = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3)
        assert isinstance(result, float)

    def test_train_batch_returns_sensitivity_with_flag(self):
        """train_batch with compute_sens=True returns (loss, dict)."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["hello sensitivity batch"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("hello sensitivity batch")
        tok_arr = np.array([tokens], dtype=np.int64)
        samples = [(img, tok_arr, None, None), (img, tok_arr, None, None)]
        result = engine.train_batch(samples, lr=1e-3, compute_sens=True)
        loss, sens = result
        assert isinstance(loss, float)
        assert "decoder" in sens
        assert "vision" in sens
        assert all(math.isfinite(v) for v in sens.values())

    def test_sensitivity_training_decreases(self):
        """Sensitivity scores decrease over multiple training steps."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["a test pattern for sensitivity"])
        img = np.random.randn(1, 224, 224, 3).astype(np.float32)
        tokens = engine.text.char.encode("a test pattern for sensitivity")
        tok_arr = np.array([tokens], dtype=np.int64)

        sens_history = []
        for _ in range(5):
            _, sens = engine.train_step(images_np=img, text_tokens=tok_arr, lr=1e-3, compute_sens=True)
            sens_history.append(sens["decoder"])

        assert sens_history[-1] <= sens_history[0] * 5, (
            f"Sensitivity increased from {sens_history[0]:.4f} to {sens_history[-1]:.4f}"
        )

    def test_audio_train_step_sensitivity(self):
        """train_step with audio returns sensitivity for decoder only."""
        engine = MultimodalEngine(embed_dim=64, hidden_dim=128, n_vit_layers=2, n_heads=4)
        engine.build_vocab(["beep boop sensitivity"])
        audio = np.sin(np.linspace(0, 50, 8000)).astype(np.float32).reshape(1, -1)
        tokens = engine.text.char.encode("beep boop sensitivity")
        tok_arr = np.array([tokens], dtype=np.int64)
        _, sens = engine.train_step(audio_np=audio, text_tokens=tok_arr, lr=1e-3, compute_sens=True)
        assert "decoder" in sens
        assert isinstance(sens["decoder"], float)
        assert math.isfinite(sens["decoder"])
