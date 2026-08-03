"""Coverage-focused tests for the multimodal package.

Raises statement coverage on previously untested public APIs across
``domains.multimodal``: tokenizers, text encoder, vision CNN, video
temporal encoder, VAE, latent diffusion, the multimodal engine, and the
multimodal manager. Uses small model sizes to keep the suite fast.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.slow

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from domains.multimodal.bpe_tokenizer import BPETokenizer
from domains.multimodal.char_tokenizer import CharTokenizer
from domains.multimodal.diffusion import (
    TimestepEmbedder, LatentDiffusionModel, LatentUNet,
    _group_norm as diffusion_group_norm, _timestep_embedding,
)
from domains.multimodal.engine import (
    AudioEncoder, MultimodalEngine, ReplayBuffer, TextDecoder,
    VisionEncoder, augment_image, contrastive_loss, contrastive_step,
    get_multimodal_engine, replay_train_step,
)
from domains.multimodal.manager import (
    MultimodalManager, get_multimodal_manager, initialize_multimodal,
)
from domains.multimodal.text_encoder import TextEncoder
from domains.multimodal.vae import (
    SloVAE, SloVAEDecoder, SloVAEEncoder, _group_norm as vae_group_norm,
)
from domains.multimodal.video import TemporalEncoder, VideoProcessor
from domains.multimodal.vision import VisionCNN, get_vision_model
from domains.training.slonet import Tensor, tensor as _tensor


def _sample_image() -> np.ndarray:
    """Small random (1, 224, 224, 3) image in [0, 1]."""
    return np.random.rand(1, 224, 224, 3).astype(np.float32)


def _make_engine() -> MultimodalEngine:
    """Small multimodal engine with a character vocabulary."""
    engine = MultimodalEngine(
        embed_dim=64, hidden_dim=128, n_vit_layers=1, n_heads=2,
        n_decoder_layers=1, n_audio_layers=1,
    )
    engine.build_vocab(["a red circle on a dark background"])
    return engine


def _make_manager(engine: MultimodalEngine) -> MultimodalManager:
    """Manager with a small engine wired in (skips lazy default-size engine)."""
    mgr = MultimodalManager()
    mgr._multimodal_engine = engine
    return mgr


class TestCharTokenizer:
    def test_build_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        tok.build_vocab(["hello world"])
        ids = tok.encode("hello world")
        assert ids[0] == tok.vocab["<BOS>"]
        assert ids[-1] == tok.vocab["<EOS>"]
        assert tok.decode(ids) == "hello world"
        assert tok.vocab_size > 4
        assert tok._built

    def test_unknown_chars_map_to_unk(self):
        tok = CharTokenizer()
        tok.build_vocab(["abc"])
        ids = tok.encode("a\x01\x02z")
        unk = tok.vocab["<UNK>"]
        assert unk in ids
        assert tok.decode([tok.vocab["a"], unk]) == "a"

    def test_encode_before_build_raises(self):
        tok = CharTokenizer()
        with pytest.raises(RuntimeError):
            tok.encode("hi")

    def test_save_load_roundtrip(self, tmp_path):
        tok = CharTokenizer(pad_to=16)
        tok.build_vocab(["cat dog bird"])
        path = str(tmp_path / "char.json")
        tok.save(path)
        loaded = CharTokenizer()
        assert loaded.load(path) is True
        assert loaded._built
        assert loaded.vocab_size == tok.vocab_size
        assert loaded.pad_to == 16
        assert loaded.decode(loaded.encode("cat dog")) == "cat dog"

    def test_load_missing_file_returns_false(self, tmp_path):
        tok = CharTokenizer()
        assert tok.load(str(tmp_path / "nope.json")) is False


class TestBPETokenizer:
    def test_train_encode_decode(self):
        tok = BPETokenizer(vocab_size=64)
        texts = [
            "a red circle on a dark background",
            "a red square on a white background",
            "a blue circle over a gray background",
        ]
        tok.train(texts)
        assert tok._built
        assert len(tok.merges) > 0
        ids = tok.encode("a red circle")
        assert isinstance(ids, list) and len(ids) > 0
        assert tok.decode(ids).strip() != ""

    def test_encode_before_train_raises(self):
        tok = BPETokenizer(vocab_size=32)
        with pytest.raises(RuntimeError):
            tok.encode("hello")

    def test_merge_helpers(self):
        from collections import Counter
        tok = BPETokenizer(vocab_size=64)
        vocab = Counter({"l o w </w>": 5, "l o w e r </w>": 2})
        pairs = tok._get_stats(vocab)
        assert ("l", "o") in pairs
        merged = tok._merge_vocab(("l", "o"), vocab)
        assert "lo w </w>" in merged

    def test_train_empty_texts_breaks(self):
        tok = BPETokenizer(vocab_size=64)
        tok.train(["a"])
        assert tok._built
        assert tok.merges == []

    def test_decode_skips_special_tokens(self):
        tok = BPETokenizer(vocab_size=64)
        tok.train(["a red circle"])
        ids = tok.encode("a red circle")
        ids = [tok.vocab["<BOS>"]] + ids + [tok.vocab["<EOS>"], tok.vocab["<PAD>"], tok.vocab["<UNK>"]]
        decoded = tok.decode(ids)
        assert "<" not in decoded

    def test_save_load_roundtrip(self, tmp_path):
        tok = BPETokenizer(vocab_size=32)
        tok.train(["a red circle", "a blue square", "the cat sat"])
        path = str(tmp_path / "bpe.json")
        tok.save(path)
        loaded = BPETokenizer()
        assert loaded.load(path) is True
        assert loaded._built
        assert loaded.vocab == tok.vocab
        assert loaded.merges == tok.merges
        assert loaded.encode("the cat") == tok.encode("the cat")

    def test_load_missing_file_returns_false(self, tmp_path):
        tok = BPETokenizer()
        assert tok.load(str(tmp_path / "nope.json")) is False

    def test_preprocess_splits_words(self):
        tok = BPETokenizer()
        tokens = tok._preprocess("Hello, world!")
        assert any(t.endswith("</w>") for t in tokens)


class TestTextEncoder:
    def test_encode_tokens_shape(self):
        enc = TextEncoder(vocab_size=128, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=16)
        ids = np.zeros((2, 5), dtype=np.int32)
        out = enc.encode_tokens(ids)
        assert out.data.shape == (2, 5, 16)

    def test_encode_text_auto_trains(self):
        enc = TextEncoder(vocab_size=256, embed_dim=16, n_heads=2, n_layers=1, max_seq_len=16)
        emb = enc.encode_text(["the cat sat on the mat"])
        assert emb.shape[0] == 1
        assert emb.shape[-1] == 16
        assert enc.tokenizer._built

    def test_train_tokenizer_explicit(self):
        enc = TextEncoder(vocab_size=64, embed_dim=8, n_heads=2, n_layers=1)
        enc.train_tokenizer(["a red circle"])
        assert enc.tokenizer._built

    def test_parameters_nonempty(self):
        enc = TextEncoder(vocab_size=64, embed_dim=8, n_heads=2, n_layers=1)
        assert len(enc.parameters()) > 0


class TestVisionCNN:
    def _pil(self, size=32):
        return Image.new("RGB", (size, size), (200, 40, 40))

    def test_build_and_forward(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=16)
        x = _tensor(np.random.rand(1, 3, 32, 32).astype(np.float32), requires_grad=False)
        out = cnn.forward(x)
        assert out.data.shape == (1, 16)

    def test_get_embedding_and_untrained_caption(self):
        cnn = VisionCNN()
        emb = cnn.get_embedding(self._pil())
        assert emb.shape == (128,)
        cap = cnn.caption(self._pil())
        assert "untrained" in cap.text
        assert cap.confidence == 0.0

    def test_train_on_batch_then_caption(self):
        cnn = VisionCNN()
        cnn.build_model(embed_dim=16)
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        y = np.random.rand(1, 16).astype(np.float32)
        loss = cnn.train_on_batch(x, y)
        assert np.isfinite(loss)
        assert cnn._learned
        cap = cnn.caption(self._pil())
        assert cap.text.startswith("learned_feat_")
        assert len(cap.tags) > 0

    def test_detect_returns_object(self):
        cnn = VisionCNN()
        objs = cnn.detect(self._pil())
        assert len(objs) == 1
        assert objs[0].bbox == [0, 0, 0, 0]

    def test_get_vision_model(self):
        model = get_vision_model("slonet")
        assert isinstance(model, VisionCNN)

    def test_preprocess_string_path(self, tmp_path):
        cnn = VisionCNN()
        path = str(tmp_path / "img.png")
        Image.new("RGB", (32, 32), (10, 200, 10)).save(path)
        x = cnn._preprocess(path)
        assert x.shape == (1, 3, 32, 32)
        assert x.max() <= 1.0

    def test_caption_error_path(self):
        cnn = VisionCNN()
        cap = cnn.caption(None)
        assert cap.text == "[vision model error]"
        assert cap.confidence == 0.0

    def test_train_on_batch_auto_build(self):
        cnn = VisionCNN()
        x = np.random.rand(1, 3, 32, 32).astype(np.float32)
        y = np.random.rand(1, 128).astype(np.float32)
        loss = cnn.train_on_batch(x, y)
        assert np.isfinite(loss)
        assert cnn._learned


class TestTemporalEncoder:
    def test_forward_shape(self):
        enc = TemporalEncoder(embed_dim=16, n_heads=2, n_layers=1, max_frames=8)
        x = np.random.rand(2, 4, 16).astype(np.float32)
        out = enc.forward(x)
        assert out.data.shape == (2, 4, 16)

    def test_parameters_nonempty(self):
        enc = TemporalEncoder(embed_dim=8, n_heads=2, n_layers=1, max_frames=4)
        assert len(enc.parameters()) > 0


class TestVideoProcessor:
    def test_extract_frames_placeholder(self):
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        frames = proc.extract_frames("/nonexistent.mp4", num_frames=3)
        assert len(frames) == 3
        assert frames[0].shape == (224, 224, 3)

    def test_extract_frames_cv2_path(self, monkeypatch):
        import types
        import sys

        class FakeCapture:
            def __init__(self, path):
                self.path = path
                self.pos = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return 10

            def set(self, prop, value):
                self.pos = value
                return True

            def read(self):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, frame

            def release(self):
                return None

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = FakeCapture
        fake_cv2.CAP_PROP_FRAME_COUNT = 1
        fake_cv2.CAP_PROP_POS_FRAMES = 2
        fake_cv2.COLOR_BGR2RGB = 3
        fake_cv2.cvtColor = lambda img, code: img
        fake_cv2.resize = lambda img, size: img[:224, :224]
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        frames = proc.extract_frames("/fake.mp4", num_frames=3)
        assert len(frames) == 3
        assert frames[0].shape == (224, 224, 3)
        assert 0.0 <= frames[0].min() <= frames[0].max() <= 1.0

    def test_extract_frames_cv2_unopenable(self, monkeypatch):
        import types
        import sys

        class ClosedCapture:
            def isOpened(self):
                return False

            def release(self):
                return None

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = lambda path: ClosedCapture()
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        frames = proc.extract_frames("/fake.mp4", num_frames=2)
        assert len(frames) == 2

    def test_extract_frames_cv2_read_failure(self, monkeypatch):
        import types
        import sys

        class FailingCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return 10

            def set(self, prop, value):
                return True

            def read(self):
                return False, None

            def release(self):
                return None

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = lambda path: FailingCapture()
        fake_cv2.CAP_PROP_FRAME_COUNT = 1
        fake_cv2.CAP_PROP_POS_FRAMES = 2
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        frames = proc.extract_frames("/fake.mp4", num_frames=3)
        assert len(frames) == 3
        assert frames[0].shape == (224, 224, 3)

    def test_extract_frames_cv2_zero_frames(self, monkeypatch):
        import types
        import sys

        class ZeroCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return 0

            def release(self):
                return None

        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = lambda path: ZeroCapture()
        fake_cv2.CAP_PROP_FRAME_COUNT = 1
        fake_cv2.CAP_PROP_POS_FRAMES = 2
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        frames = proc.extract_frames("/fake.mp4", num_frames=2)
        assert len(frames) == 2

    def test_encode_video(self):
        proc = VideoProcessor(embed_dim=16, n_heads=2, n_temporal_layers=1, max_frames=4)
        vision = VisionEncoder(embed_dim=16, n_heads=2, n_layers=1)
        frames = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(2)]
        out = proc.encode_video(frames, vision)
        assert out.data.shape == (1, 2, 16)

    def test_parameters(self):
        proc = VideoProcessor(embed_dim=8, n_heads=2, n_temporal_layers=1, max_frames=4)
        assert len(proc.parameters()) > 0


class TestVAE:
    def test_group_norm_valid(self):
        x = _tensor(np.random.rand(1, 32, 8, 8).astype(np.float32), requires_grad=True)
        out = vae_group_norm(x, num_groups=4)
        assert out.data.shape == (1, 32, 8, 8)

    def test_group_norm_invalid_raises(self):
        x = _tensor(np.random.rand(1, 32, 8, 8).astype(np.float32), requires_grad=True)
        with pytest.raises(AssertionError):
            vae_group_norm(x, num_groups=7)

    def test_encoder_forward_and_sample(self):
        enc = SloVAEEncoder(latent_dim=4)
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        mean, log_var = enc.forward(img)
        assert mean.data.shape == (1, 4, 7, 7)
        assert log_var.data.shape == (1, 4, 7, 7)
        z = enc.sample(mean, log_var)
        assert z.data.shape == (1, 4, 7, 7)
        assert len(enc.parameters()) > 0

    def test_decoder_forward(self):
        dec = SloVAEDecoder(latent_dim=4)
        z = _tensor(np.random.rand(1, 4, 7, 7).astype(np.float32), requires_grad=False)
        out = dec.forward(z)
        assert out.data.shape == (1, 3, 224, 224)

    def test_vae_train_step_and_codec(self):
        vae = SloVAE(latent_dim=4)
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        loss = vae.train_step(img)
        assert np.isfinite(loss)
        latent = vae.encode(img)
        assert latent.shape == (1, 4, 7, 7)
        recon = vae.decode(latent)
        assert recon.shape == (1, 3, 224, 224)
        assert recon.min() >= 0.0 and recon.max() <= 1.0

    def test_vae_forward_tuple(self):
        vae = SloVAE(latent_dim=4)
        img = np.random.rand(1, 3, 224, 224).astype(np.float32)
        recon, mean, log_var = vae.forward(img)
        assert recon.data.shape == (1, 3, 224, 224)
        assert mean.data.shape == (1, 4, 7, 7)
        assert log_var.data.shape == (1, 4, 7, 7)

    def test_vae_parameters(self):
        vae = SloVAE(latent_dim=4)
        assert len(vae.parameters()) > 0

    def test_vae_encoder_parameters_method(self):
        enc = SloVAEEncoder(latent_dim=4)
        assert len(enc.parameters()) > 0

    def test_vae_decoder_parameters_method(self):
        dec = SloVAEDecoder(latent_dim=4)
        assert len(dec.parameters()) > 0


class TestDiffusion:
    def test_group_norm_fewer_channels_than_groups(self):
        x = _tensor(np.random.rand(1, 3, 8, 8).astype(np.float32), requires_grad=True)
        out = diffusion_group_norm(x, num_groups=32)
        assert out.data.shape == (1, 3, 8, 8)

    def test_group_norm_non_divisible(self):
        x = _tensor(np.random.rand(1, 6, 8, 8).astype(np.float32), requires_grad=True)
        out = diffusion_group_norm(x, num_groups=4)
        assert out.data.shape == (1, 6, 8, 8)

    def test_timestep_embedding_odd_dim(self):
        emb = _timestep_embedding(np.array([0, 5, 10]), dim=5)
        assert emb.shape == (3, 5)
        emb_even = _timestep_embedding(np.array([1, 2]), dim=4)
        assert emb_even.shape == (2, 4)

    def test_timestep_embedder_forward(self):
        emb = TimestepEmbedder(dim=8)
        out = emb.forward(np.array([0, 3]))
        assert out.data.shape == (2, 8)
        assert len(emb.parameters()) > 0

    def test_latent_unet_forward(self):
        unet = LatentUNet(in_channels=4, model_channels=8, out_channels=4,
                          temb_dim=8, context_dim=8, n_heads=2)
        x = _tensor(np.random.rand(1, 4, 7, 7).astype(np.float32), requires_grad=False)
        ctx = _tensor(np.random.rand(1, 3, 8).astype(np.float32), requires_grad=False)
        out = unet.forward(x, np.array([3]), ctx)
        assert out.data.shape == (1, 4, 7, 7)

    def test_diffusion_noise_and_train_step(self):
        model = LatentDiffusionModel(latent_dim=4, model_channels=8, temb_dim=8,
                                     context_dim=8, n_heads=2, num_timesteps=100)
        latents = np.random.rand(1, 4, 7, 7).astype(np.float32)
        noisy, noise = model.add_noise(latents, np.array([5]))
        assert noisy.shape == (1, 4, 7, 7)
        assert noise.shape == (1, 4, 7, 7)
        text_emb = np.random.rand(1, 3, 8).astype(np.float32)
        loss = model.train_step(latents, text_emb)
        assert np.isfinite(loss)

    def test_diffusion_sample_shape(self):
        model = LatentDiffusionModel(latent_dim=4, model_channels=8, temb_dim=8,
                                     context_dim=8, n_heads=2, num_timesteps=50)
        text_emb = np.random.rand(1, 3, 8).astype(np.float32)
        latents = model.sample(text_emb, num_steps=2)
        assert latents.shape == (1, 4, 7, 7)

    def test_diffusion_parameters(self):
        model = LatentDiffusionModel(latent_dim=4, model_channels=8, temb_dim=8,
                                     context_dim=8, n_heads=2, num_timesteps=50)
        assert len(model.parameters()) > 0

    def test_latent_unet_parameters(self):
        unet = LatentUNet(in_channels=4, model_channels=8, out_channels=4,
                          temb_dim=8, context_dim=8, n_heads=2)
        assert len(unet.parameters()) > 0


class TestReplayBuffer:
    def test_add_sample_size_and_eviction(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(np.full((1, 1, 1, 3), float(i)), f"cap {i}")
        assert buf.size == 3
        assert "cap 0" not in buf.captions
        assert "cap 4" in buf.captions
        imgs, caps = buf.sample(2)
        assert len(imgs) == 2 and len(caps) == 2

    def test_sample_when_short_returns_copies(self):
        buf = ReplayBuffer(capacity=10)
        buf.add(np.zeros((1, 2, 2, 3)), "a")
        imgs, caps = buf.sample(8)
        assert imgs == buf.images
        assert caps == buf.captions

    def test_sample_diverse_weighting(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(6):
            buf.add(np.zeros((1, 2, 2, 3)), f"unique{i}")
        for _ in range(3):
            buf.add(np.zeros((1, 2, 2, 3)), "common")
        imgs, caps = buf.sample(2)
        assert len(caps) == 2
        assert len(imgs) == 2

    def test_counts_bookkeeping_on_eviction(self):
        buf = ReplayBuffer(capacity=2)
        buf.add(np.zeros((1, 1, 1, 1)), "x")
        buf.add(np.zeros((1, 1, 1, 1)), "y")
        buf.add(np.zeros((1, 1, 1, 1)), "x")
        assert buf._counts.get("x", 0) == 1
        assert buf._counts.get("y", 0) == 1


class TestAugmentAndContrastive:
    def test_augment_image_preserves_shape(self):
        img = _sample_image()
        out = augment_image(img)
        assert out.shape == (1, 224, 224, 3)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_contrastive_loss_scalar_and_backward(self):
        z1 = Tensor(np.random.rand(1, 8).astype(np.float32), requires_grad=True)
        z2 = Tensor(np.random.rand(1, 8).astype(np.float32), requires_grad=True)
        negs = [Tensor(np.random.rand(1, 8).astype(np.float32), requires_grad=True)]
        loss = contrastive_loss(z1, z2, negs, temperature=0.5)
        assert np.isfinite(float(loss.data))
        loss.backward()
        assert np.isfinite(float(loss.data))

    def test_contrastive_step_empty_buffer(self):
        engine = _make_engine()
        assert contrastive_step(engine, _sample_image(), ReplayBuffer()) == 0.0

    def test_contrastive_step_with_buffer(self):
        engine = _make_engine()
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add(_sample_image(), f"caption {i}")
        loss = contrastive_step(engine, _sample_image(), buf)
        assert np.isfinite(loss)

    def test_replay_train_step_empty_buffer(self):
        engine = _make_engine()
        assert replay_train_step(engine, ReplayBuffer()) == 0.0

    def test_replay_train_step_with_buffer(self):
        engine = _make_engine()
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add(_sample_image(), f"a red circle {i}")
        loss = replay_train_step(engine, buf, batch_size=2)
        assert np.isfinite(loss)


class TestMultimodalEngine:
    def test_model_id_embed_dim_metadata(self):
        engine = _make_engine()
        assert engine.model_id == "multimodal-v1"
        assert engine.embed_dim == 64
        meta = engine.metadata
        assert meta["vocab_size"] == engine.text.vocab_size
        assert meta["trained"] is False

    def test_capabilities(self):
        engine = _make_engine()
        caps = engine.capabilities
        assert caps.vision is True
        assert caps.chat is True

    def test_text_decoder_roundtrip(self):
        td = TextDecoder(embed_dim=16, hidden_dim=32)
        td.build_vocab(["a red circle"])
        ids = td.encode("a red circle")
        assert td.decode(ids) == "a red circle"
        assert td.vocab_size > 4

    def test_vision_encoder_shapes(self):
        vision = VisionEncoder(embed_dim=16, n_heads=2, n_layers=1)
        img = _sample_image()
        embed = vision.forward(img)
        assert embed.data.shape == (1, 1, 16)
        patches = vision.get_patch_embeddings(img)
        assert patches.data.shape == (1, 50, 16)
        assert vision.extract_patches(img).shape == (1, 49, 3072)

    def test_audio_encoder_short_waveform_padding(self):
        audio = AudioEncoder(embed_dim=16, n_heads=2, n_layers=1)
        t = np.arange(16000, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        mel = audio._mel_spectrogram(wave)
        assert mel.shape[0] == audio.N_MELS
        patches = audio.extract_patches(wave)
        assert patches.shape[0] == 1
        embed = audio.get_patch_embeddings(wave)
        assert embed.data.shape[1] == audio.max_patches + 1
        cls = audio.forward(wave)
        assert cls.data.shape == (1, 1, 16)
        params = audio.parameters()
        assert len(params) > 0

    def test_audio_encoder_long_waveform_multiple_patches(self):
        audio = AudioEncoder(embed_dim=16, n_heads=2, n_layers=1)
        t = np.arange(16000 * 20, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        patches = audio.extract_patches(wave)
        assert patches.shape[1] > 1
        embed = audio._embed_patches(patches)
        assert embed.data.shape == (1, audio.max_patches + 1, 16)

    def test_train_step_and_sensitivity(self):
        engine = _make_engine()
        img = _sample_image()
        tokens = np.array([engine.text.encode("a red circle")], dtype=np.int64)
        loss, sens = engine.train_step(img, tokens, lr=1e-3, compute_sens=True)
        assert np.isfinite(loss)
        assert set(sens.keys()) == {"decoder", "vision"}
        assert engine._trained

    def test_param_groups(self):
        engine = _make_engine()
        groups = engine.param_groups()
        assert set(groups.keys()) == {"decoder", "vision", "audio"}
        assert len(groups["decoder"]) > 0

    def test_generate_greedy_and_sampling(self):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        img = _sample_image()
        out_greedy = engine.generate(img, max_len=8, temperature=0.0)
        assert isinstance(out_greedy.text, str)
        assert out_greedy.confidence >= 0.0
        out_samp = engine.generate(img, max_len=8, temperature=0.8, top_k=10)
        assert isinstance(out_samp.text, str)
        out_beam = engine.generate(img, max_len=8, temperature=0.5, beam_width=3)
        assert isinstance(out_beam.text, str)

    def test_generate_untrained_argmax_path(self):
        engine = _make_engine()
        out = engine.generate(_sample_image(), max_len=6, temperature=0.8)
        assert isinstance(out.text, str)

    def test_forward_and_precompute_audio(self):
        engine = _make_engine()
        img = _sample_image()
        tokens = np.array([engine.text.encode("a red")], dtype=np.int64)
        logits, embed = engine.forward(img, tokens)
        assert logits.data.ndim == 3
        assert embed.data.shape == (1, 1, 64)
        t = np.arange(16000, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        raw = engine.precompute_audio_patches(wave)
        assert raw.shape[1] >= 1

    def test_concat_modalities_variants(self):
        engine = _make_engine()
        img = _sample_image()
        embed, patches, opts = engine._concat_modalities(images_np=img)
        assert embed.data.shape == (1, 1, 64)
        assert patches.data.shape == (1, 50, 64)
        assert len(opts) == 2
        t = np.arange(16000, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio_patches = engine.precompute_audio_patches(wave)
        embed_a, patches_a, opts_a = engine._concat_modalities(audio_patches=audio_patches)
        assert embed_a.data.shape == (1, 1, 64)
        assert patches_a.data.shape[1] == engine.audio.max_patches + 1
        embed_b, patches_b, _ = engine._concat_modalities(images_np=img, audio_patches=audio_patches)
        assert patches_b.data.shape[1] == 50 + engine.audio.max_patches + 1
        with pytest.raises(ValueError):
            engine._concat_modalities()

    def test_embed_property_untrained_then_trained(self):
        engine = _make_engine()
        assert engine.embed("a red circle") == [0.0] * 128
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        vec = engine.embed("a red circle")
        assert len(vec) == 128

    def test_extract_images_from_messages(self):
        engine = _make_engine()
        msgs = [
            {"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]},
            {"content": "look: data:image/png;base64,BBB and done"},
        ]
        imgs = engine._extract_images(msgs)
        assert len(imgs) == 2
        assert imgs[0].startswith("data:image/")
        assert engine._extract_images([{"content": "no image here"}]) == []

    def test_causal_mask_shape(self):
        from domains.multimodal.engine import _causal_mask
        mask = _causal_mask(6)
        assert mask.data.shape == (1, 1, 6, 6)

    def test_pil_to_np_and_mode_transitions(self):
        engine = _make_engine()
        engine.train()
        engine.eval()
        vision = engine.vision
        vision.train()
        vision.eval()
        audio = engine.audio
        audio.train()
        audio.eval()
        decoder = engine.decoder
        decoder.train()
        decoder.eval()
        img = Image.new("RGB", (448, 448), (255, 0, 0))
        arr = engine._pil_to_np(img)
        assert arr.shape == (1, 224, 224, 3)
        assert arr.max() <= 1.0

    def test_forward_with_audio_np(self):
        engine = _make_engine()
        img = _sample_image()
        tokens = np.array([engine.text.encode("a red")], dtype=np.int64)
        t = np.arange(16000, dtype=np.float32) / 16000.0
        wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        logits, embed = engine.forward(img, tokens, audio_np=wave)
        assert logits.data.ndim == 3
        assert embed.data.shape == (1, 1, 64)

    def test_sum_grads_and_empty_train_batch(self):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        params = list(engine.vision.parameters())
        p = params[0]
        p.grad = Tensor(np.ones_like(p.data), requires_grad=False)
        engine._sum_grads([p], scale=0.5)
        g_data = p.grad.data if hasattr(p.grad, "data") else p.grad
        assert float(np.asarray(g_data).ravel()[0]) == 0.5
        assert engine.train_batch([]) == 0.0

    def test_greedy_topk_untrained(self):
        engine = _make_engine()
        out = engine.generate(_sample_image(), max_len=6, temperature=0.0, top_k=5)
        assert isinstance(out.text, str)

    def test_beam_search_all_eos(self):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        out = engine.generate(_sample_image(), max_len=2, temperature=0.5, beam_width=2)
        assert isinstance(out.text, str)

    def test_mel_spectrogram_short_waveform(self):
        audio = AudioEncoder(embed_dim=16, n_heads=2, n_layers=1)
        mel = audio._mel_spectrogram(np.zeros(100, dtype=np.float32))
        assert mel.shape == (audio.N_MELS, 1)

    def test_decoder_forward_1d_tokens(self):
        engine = _make_engine()
        img = _sample_image()
        embed = engine.vision.forward(img)
        patches = engine.vision.get_patch_embeddings(img)
        inp = _tensor(np.array([0, 1]), requires_grad=False)
        logits, last_out, kv = engine.decoder.forward(embed, inp, patches)
        assert logits.data.ndim == 3
        assert kv is not None

    def test_train_batch_with_temperature_and_none_tokens(self):
        engine = _make_engine()
        img = _sample_image()
        tokens = np.array([engine.text.encode("a red circle")], dtype=np.int64)
        samples = [(img, tokens, None, None), (img, None, None, None)]
        loss = engine.train_batch(samples, lr=0.01, temperature=0.7)
        assert loss > 0.0
        assert engine.train_batch([(img, None, None, None)]) == 0.0

    def test_params_for_optimizer_unknown(self):
        engine = _make_engine()
        class _Dummy:
            pass
        dummy = _Dummy()
        assert engine._params_for_optimizer(dummy, None, None) == []

    def test_replay_buffer_eviction_duplicate_counts(self):
        buf = ReplayBuffer(capacity=3)
        img = np.zeros((8, 8), dtype=np.float32)
        buf.add(img, "same caption")
        buf.add(img + 1, "same caption")
        buf.add(img + 2, "same caption")
        buf.add(img + 3, "same caption")
        assert len(buf.images) == 3
        assert buf._counts["same caption"] == 3

    def test_replay_train_step_edge_cases(self):
        engine = _make_engine()
        buf = ReplayBuffer(capacity=4)
        img = np.zeros((8, 8), dtype=np.float32)
        buf.add(img, "")
        buf.add(img, "")
        assert replay_train_step(engine, buf, batch_size=2) == 0.0

    def test_replay_train_step_exception_path(self):
        engine = _make_engine()
        buf = ReplayBuffer(capacity=4)
        img = np.zeros((8, 8), dtype=np.float32)
        buf.add(img, "a red circle")
        buf.add(img, "a blue square")
        import domains.multimodal.engine as engine_mod
        with patch.object(engine_mod.ReplayBuffer, "sample", return_value=([img], ["bad caption"])):
            with patch.object(engine_mod.TextDecoder, "encode", side_effect=RuntimeError("boom")):
                assert replay_train_step(engine, buf, batch_size=2) == 0.0

    def test_beam_search_no_completed(self):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        out = engine.generate(_sample_image(), max_len=6, temperature=0.99, beam_width=2, top_k=3)
        assert isinstance(out.text, str)

    def test_beam_search_all_eos_break(self):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        out = engine.generate(_sample_image(), max_len=30, temperature=0.5, beam_width=1, top_k=1)
        assert isinstance(out.text, str)

    def test_chat_stream_valid_image(self):
        engine = _make_engine()
        import io
        import base64
        import asyncio
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (255, 0, 0)).save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        chunks = []
        class _Cap:
            text = "a red circle"
        class _Mgr:
            def caption_image(self, img):
                return _Cap()
        with patch("domains.multimodal.manager.get_multimodal_manager", return_value=_Mgr()):
            async def _run():
                async for chunk in engine.chat_stream(
                    [{"role": "user", "content": "data:image/png;base64," + b64}]
                ):
                    chunks.append(chunk)
            asyncio.run(_run())
        assert chunks[0] == "a red circle"

    def test_chat_stream_no_image(self):
        engine = _make_engine()
        import asyncio
        chunks = []
        async def _run():
            async for chunk in engine.chat_stream([{"role": "user", "content": "hello"}]):
                chunks.append(chunk)
        asyncio.run(_run())
        assert "no image" in chunks[0]

    def test_chat_stream_bad_image(self):
        engine = _make_engine()
        import asyncio
        chunks = []
        async def _run():
            async for chunk in engine.chat_stream([{"role": "user", "content": "data:image/png;base64,notvalid!!!"}]):
                chunks.append(chunk)
        asyncio.run(_run())
        assert "[error:" in chunks[0]

    def test_chat_no_image(self):
        engine = _make_engine()
        import asyncio
        out = asyncio.run(engine.chat([{"role": "user", "content": "hi"}]))
        assert "no image" in out

    def test_decoder_kv_cache_incremental(self):
        engine = _make_engine()
        img = _sample_image()
        embed = engine.vision.forward(img)
        patches = engine.vision.get_patch_embeddings(img)
        inp = _tensor(np.array([[0, 1]]), requires_grad=False)
        logits1, _, kv1 = engine.decoder.forward(embed, inp, patches)
        assert logits1.data.ndim == 3
        assert kv1 is not None
        inp2 = _tensor(np.array([[5]]), requires_grad=False)
        logits2, _, kv2 = engine.decoder.forward(embed, inp2, patches, kv_cache=kv1, start_pos=2)
        assert logits2.data.shape[-1] == logits1.data.shape[-1]
        assert kv2 is not None

    def test_save_load_roundtrip(self, tmp_path):
        engine = _make_engine()
        engine.train_step(_sample_image(), np.array([engine.text.encode("a red circle")], dtype=np.int64))
        path = str(tmp_path / "mm" / "engine.npz")
        saved = engine.save(path, extra_meta={"images_learned": 3})
        assert saved == path
        loaded = MultimodalEngine.load(path)
        assert loaded._trained
        assert loaded.embed_dim == engine.embed_dim
        assert loaded.text.vocab_size == engine.text.vocab_size
        assert loaded.model_id == "multimodal-v1"

    def test_get_multimodal_engine(self):
        engine = get_multimodal_engine(embed_dim=16, hidden_dim=32, n_vit_layers=1,
                                       n_heads=2, n_decoder_layers=1, n_audio_layers=1)
        assert isinstance(engine, MultimodalEngine)


class TestMultimodalManager:
    def test_init_attributes(self):
        mgr = MultimodalManager()
        assert mgr._initialized is False
        assert mgr._learning_count == 0
        assert mgr._replay_buffer.capacity == 200
        assert mgr._caption_history == []
        assert mgr._accuracy_history == []
        assert mgr._multimodal_engine is None

    def test_capabilities_fresh_then_with_engine(self):
        mgr = MultimodalManager()
        caps = mgr.capabilities
        assert caps.image_caption is False
        assert caps.speech_to_text is False
        mgr._multimodal_engine = _make_engine()
        caps2 = mgr.capabilities
        assert caps2.image_caption is True
        assert caps2.speech_model == "browser"

    def test_pil_to_np(self):
        mgr = MultimodalManager()
        img = Image.new("RGB", (448, 448), (255, 0, 0))
        arr = mgr._pil_to_np(img)
        assert arr.shape == (1, 224, 224, 3)
        assert arr.min() >= 0.0 and arr.max() <= 1.0

    def test_count_trained_images(self, tmp_path, monkeypatch):
        from domains.multimodal import engine as engine_mod
        monkeypatch.setattr(engine_mod.MultimodalEngine, "SAVE_PATH", str(tmp_path / "mm.npz"))
        mgr = MultimodalManager()
        assert mgr._count_trained_images() == 0
        import json
        with open(str(tmp_path / "mm.npz.json"), "w") as f:
            json.dump({"images_learned": 7}, f)
        assert mgr._count_trained_images() == 7

    def test_recognize_speech_graceful_degradation(self):
        mgr = MultimodalManager()
        mgr._speech_server_mode = True
        result = mgr.recognize_speech(b"\x00" * 160)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_recognize_speech_browser(self):
        mgr = MultimodalManager()
        cfg = mgr.get_browser_speech_config()
        assert cfg["language"] == "en-US"

    def test_get_browser_speech_config(self):
        mgr = MultimodalManager()
        cfg = mgr.get_browser_speech_config()
        assert cfg["language"] == "en-US"

    def test_caption_image_supervised_path(self):
        engine = _make_engine()
        mgr = _make_manager(engine)
        img = Image.new("RGB", (224, 224), (50, 150, 50))
        cap = mgr.caption_image(img, ground_truth="a green circle")
        assert cap.text == "a green circle"
        assert cap.confidence >= 0.0
        assert cap.tags == ["vision", "supervised"]
        assert mgr._learning_count == 1
        assert len(mgr._accuracy_history) == 1
        assert mgr._replay_buffer.size == 1

    def test_caption_image_self_supervised_seed(self):
        engine = _make_engine()
        mgr = _make_manager(engine)
        img = Image.new("RGB", (224, 224), (150, 50, 150))
        cap = mgr.caption_image(img)
        assert cap.text in mgr._SEED_CAPTIONS
        assert cap.tags == ["vision", "learned"]
        assert cap.accuracy == 0.0
        assert mgr._learning_count == 1

    def test_caption_image_error_path(self):
        mgr = MultimodalManager()
        cap = mgr.caption_image(None)
        assert cap.text == "[caption failed]"
        assert cap.confidence == 0.0

    def test_caption_image_auto_save_every_five(self, tmp_path, monkeypatch):
        from domains.multimodal import engine as engine_mod
        monkeypatch.setattr(engine_mod.MultimodalEngine, "SAVE_PATH", str(tmp_path / "mm.npz"))
        engine = _make_engine()
        mgr = _make_manager(engine)
        mgr._learning_count = 4
        img = Image.new("RGB", (224, 224), (10, 10, 200))
        cap = mgr.caption_image(img)
        assert cap.text in mgr._SEED_CAPTIONS
        assert mgr._learning_count == 5
        assert (tmp_path / "mm.npz").exists()

    def test_train_on_path(self, tmp_path):
        engine = _make_engine()
        mgr = _make_manager(engine)
        img = Image.new("RGB", (224, 224), (200, 10, 10))
        path = str(tmp_path / "img.png")
        img.save(path)
        cap = mgr.train_on_path(path)
        assert cap.text in mgr._SEED_CAPTIONS
        assert mgr._learning_count == 1

    def test_detect_objects(self):
        engine = _make_engine()
        mgr = _make_manager(engine)
        img = Image.new("RGB", (224, 224), (30, 30, 30))
        objs = mgr.detect_objects(img)
        assert len(objs) == 1
        assert objs[0].bbox == [0, 0, 0, 0]

    def test_get_multimodal_manager_singleton(self):
        first = get_multimodal_manager()
        second = get_multimodal_manager()
        assert first is second
        assert isinstance(first, MultimodalManager)

    def test_initialize_multimodal(self, monkeypatch):
        from domains.multimodal import manager as manager_mod
        fresh = MultimodalManager()
        monkeypatch.setattr(manager_mod, "_multimodal_manager", fresh)
        monkeypatch.setattr(MultimodalManager, "_pretrain_engine", lambda self, **kw: 0.0)
        initialize_multimodal(speech_server=False, vision_model="slonet")
        assert fresh._initialized is True
        assert fresh._multimodal_engine is not None
        assert fresh._speech_recognizer is not None

    def test_initialize_speech_server_mode(self, monkeypatch):
        from domains.multimodal import manager as manager_mod
        fresh = MultimodalManager()
        monkeypatch.setattr(manager_mod, "_multimodal_manager", fresh)
        monkeypatch.setattr(MultimodalManager, "_pretrain_engine", lambda self, **kw: 0.0)
        fresh.initialize(speech_server=True, vision_model="slonet")
        assert fresh._speech_server_mode is True
        assert fresh._speech_recognizer is not None
        caps = fresh.capabilities
        assert caps.speech_model == "whisper" or caps.speech_model == "browser"

    def test_initialize_loads_saved_engine(self, tmp_path, monkeypatch):
        from domains.multimodal import engine as engine_mod
        saved = _make_engine()
        saved._trained = True
        monkeypatch.setattr(engine_mod.MultimodalEngine, "SAVE_PATH", str(tmp_path / "mm.npz"))
        monkeypatch.setattr(engine_mod.MultimodalEngine, "load", classmethod(lambda cls: saved))
        import json
        with open(str(tmp_path / "mm.npz.json"), "w") as f:
            json.dump({"images_learned": 3}, f)
        mgr = MultimodalManager()
        mgr.initialize(speech_server=False, vision_model="slonet")
        assert mgr._initialized is True
        assert mgr._multimodal_engine is saved
        assert mgr._learning_count == 3

    def test_initialize_saved_engine_fails(self, tmp_path, monkeypatch):
        from domains.multimodal import engine as engine_mod
        from domains.multimodal import manager as manager_mod
        monkeypatch.setattr(engine_mod.MultimodalEngine, "SAVE_PATH", str(tmp_path / "mm.npz"))
        def boom(cls):
            raise RuntimeError("corrupt")
        monkeypatch.setattr(engine_mod.MultimodalEngine, "load", classmethod(boom))
        import json
        with open(str(tmp_path / "mm.npz.json"), "w") as f:
            json.dump({"images_learned": 3}, f)
        mgr = MultimodalManager()
        mgr.initialize(speech_server=False, vision_model="slonet")
        assert mgr._multimodal_engine is not None

    def test_gen_synthetic_data(self):
        mgr = MultimodalManager()
        images, captions = mgr._gen_synthetic_data(8)
        assert images.shape == (8, 224, 224, 3)
        assert len(captions) == 8
        assert all(isinstance(c, str) and "background" in c for c in captions)

    def test_pick_seed_caption_stable_and_edge_cases(self):
        mgr = MultimodalManager()
        first = mgr._pick_seed_caption(np.ones((1, 8)))
        second = mgr._pick_seed_caption(np.ones((1, 8)))
        assert first == second
        assert first in mgr._SEED_CAPTIONS
        assert mgr._pick_seed_caption(np.array([])) == mgr._SEED_CAPTIONS[0]
        empty = MultimodalManager()
        empty._SEED_CAPTIONS = []
        assert empty._pick_seed_caption(np.ones((1, 8))) == "an image"

    def test_pretrain_engine(self):
        mgr = _make_manager(_make_engine())
        loss = mgr._pretrain_engine(epochs=1, samples=4, batch_size=2)
        assert np.isfinite(loss)
        assert mgr._replay_buffer.size == 4
        assert mgr._multimodal_engine._trained is True

    def test_pretrain_engine_no_engine(self):
        mgr = MultimodalManager()
        assert mgr._pretrain_engine() == float("inf")

    def test_caption_image_generated_text_path(self, monkeypatch):
        engine = _make_engine()
        engine._trained = True
        mgr = _make_manager(engine)
        mgr._learning_count = 12
        orig = engine.generate

        def fake_generate(img, max_len=16, temperature=0.8):
            return type("R", (), {"text": "a red circle", "confidence": 0.5})()
        monkeypatch.setattr(engine, "generate", fake_generate)
        cap = mgr.caption_image(Image.new("RGB", (224, 224), (10, 200, 10)))
        assert cap.text == "a red circle"
        assert mgr._learning_count == 13

    def test_caption_image_generate_falls_back_to_seed(self, monkeypatch):
        engine = _make_engine()
        engine._trained = True
        mgr = _make_manager(engine)
        mgr._learning_count = 12
        monkeypatch.setattr(engine, "generate",
                            lambda img, max_len=16, temperature=0.8: type("R", (), {"text": ""})())
        cap = mgr.caption_image(Image.new("RGB", (224, 224), (10, 200, 10)))
        assert cap.text in mgr._SEED_CAPTIONS

    def test_caption_image_decoder_train_skipped(self, monkeypatch):
        engine = _make_engine()
        mgr = _make_manager(engine)
        orig_encode = engine.text.encode
        def boom(s):
            raise RuntimeError("encode fail")
        monkeypatch.setattr(engine.text, "encode", boom)
        cap = mgr.caption_image(Image.new("RGB", (224, 224), (10, 200, 10)))
        assert cap.text in mgr._SEED_CAPTIONS
        assert mgr._learning_count == 1

    def test_caption_image_replay_and_save_failure(self, tmp_path, monkeypatch):
        from domains.multimodal import engine as engine_mod
        monkeypatch.setattr(engine_mod.MultimodalEngine, "SAVE_PATH", str(tmp_path / "mm.npz"))
        engine = _make_engine()
        mgr = _make_manager(engine)
        mgr._learning_count = 4
        def boom_save(self, **kwargs):
            raise IOError("disk full")
        monkeypatch.setattr(engine_mod.MultimodalEngine, "save", boom_save)
        cap = mgr.caption_image(Image.new("RGB", (224, 224), (10, 200, 10)))
        assert cap.text in mgr._SEED_CAPTIONS
        assert mgr._learning_count == 5

    def test_caption_image_replay_step(self):
        engine = _make_engine()
        mgr = _make_manager(engine)
        mgr._learning_count = 5
        for i in range(3):
            mgr._replay_buffer.add(np.random.rand(1, 224, 224, 3).astype(np.float32), f"a red circle {i}")
        cap = mgr.caption_image(Image.new("RGB", (224, 224), (10, 200, 10)))
        assert cap.text in mgr._SEED_CAPTIONS
        assert mgr._learning_count == 6

    def test_initialize_register_provider_failure(self, monkeypatch):
        from domains.multimodal import manager as manager_mod
        fresh = MultimodalManager()
        monkeypatch.setattr(manager_mod, "_multimodal_manager", fresh)
        monkeypatch.setattr(MultimodalManager, "_pretrain_engine", lambda self, **kw: 0.0)
        import domains.models.provider as provider_mod
        def boom(*a, **kw):
            raise RuntimeError("no provider")
        monkeypatch.setattr(provider_mod, "register_provider", boom)
        fresh.initialize(speech_server=False, vision_model="slonet")
        assert fresh._initialized is True
        assert fresh._multimodal_engine is not None

    def test_get_browser_speech_config_no_config(self):
        mgr = MultimodalManager()
        mgr._speech_recognizer = object()
        assert mgr.get_browser_speech_config() == {"language": "en-US"}
