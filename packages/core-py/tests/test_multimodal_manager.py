"""Tests for multimodal.manager — MultimodalManager."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from domains.multimodal.manager import (
    MultimodalCapabilities,
    MultimodalManager,
    get_multimodal_manager,
    initialize_multimodal,
)


# ── MultimodalCapabilities ─────────────────────────────────────────────────


class TestMultimodalCapabilities:

    def test_defaults(self):
        c = MultimodalCapabilities()
        assert c.speech_to_text is False
        assert c.image_caption is False
        assert c.object_detection is False
        assert c.vqa is False
        assert c.speech_model is None
        assert c.vision_model is None

    def test_custom(self):
        c = MultimodalCapabilities(speech_to_text=True, vision_model="slonet")
        assert c.speech_to_text is True
        assert c.vision_model == "slonet"


# ── MultimodalManager ──────────────────────────────────────────────────────


class TestMultimodalManager:

    def test_init(self):
        m = MultimodalManager()
        assert m._initialized is False
        assert m._learning_count == 0
        assert m._caption_history == []

    @patch("domains.multimodal.manager.get_speech_recognizer")
    @patch("domains.multimodal.manager.get_multimodal_engine")
    @patch("os.path.exists", return_value=False)
    def test_initialize(self, mock_exists, mock_engine, mock_speech):
        m = MultimodalManager()
        m.initialize(speech_server=False, vision_model="slonet")
        assert m._initialized is True
        mock_speech.assert_called_once_with(use_server=False)

    @patch("domains.multimodal.manager.get_speech_recognizer")
    @patch("domains.multimodal.manager.get_multimodal_engine")
    @patch("os.path.exists", return_value=False)
    def test_initialize_with_server(self, mock_exists, mock_engine, mock_speech):
        m = MultimodalManager()
        m.initialize(speech_server=True, vision_model="slonet")
        mock_speech.assert_called_once_with(use_server=True)

    def test_capabilities_not_initialized(self):
        m = MultimodalManager()
        c = m.capabilities
        assert c.speech_to_text is False
        assert c.image_caption is False

    def test_capabilities_initialized(self):
        m = MultimodalManager()
        m._speech_server_mode = True
        m._speech_recognizer = MagicMock()
        m._multimodal_engine = MagicMock()
        c = m.capabilities
        assert c.speech_to_text is True
        assert c.image_caption is True
        assert c.object_detection is True

    def test_capabilities_server_no_recognizer(self):
        m = MultimodalManager()
        m._speech_server_mode = True
        m._speech_recognizer = None
        c = m.capabilities
        assert c.speech_to_text is False

    def test_image_hash(self):
        arr = np.zeros((1, 4, 4, 3), dtype=np.float32)
        h = MultimodalManager._image_hash(arr)
        assert isinstance(h, int)

    def test_image_hash_deterministic(self):
        arr = np.ones((1, 4, 4, 3), dtype=np.float32)
        h1 = MultimodalManager._image_hash(arr)
        h2 = MultimodalManager._image_hash(arr)
        assert h1 == h2

    def test_pick_seed_caption(self):
        m = MultimodalManager()
        embed = np.array([0.5, 0.3, 0.2])
        cap = m._pick_seed_caption(embed)
        assert isinstance(cap, str)
        assert len(cap) > 0

    def test_pick_seed_caption_empty(self):
        m = MultimodalManager()
        cap = m._pick_seed_caption(np.array([]))
        assert cap == MultimodalManager._SEED_CAPTIONS[0]

    def test_pick_seed_caption_no_captions(self):
        m = MultimodalManager()
        m._SEED_CAPTIONS = []
        cap = m._pick_seed_caption(np.array([1.0]))
        assert cap == "an image"

    def test_pil_to_np(self):
        from PIL import Image
        m = MultimodalManager()
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        arr = m._pil_to_np(img)
        assert arr.shape == (1, 224, 224, 3)
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0

    def test_gen_synthetic_data(self):
        m = MultimodalManager()
        images, captions = m._gen_synthetic_data(10)
        assert images.shape == (10, 224, 224, 3)
        assert len(captions) == 10
        assert all(isinstance(c, str) for c in captions)

    def test_count_trained_images(self):
        m = MultimodalManager()
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={"images_learned": 42}):
                count = m._count_trained_images()
                assert count == 42

    def test_count_trained_images_error(self):
        m = MultimodalManager()
        with patch("builtins.open", side_effect=FileNotFoundError):
            count = m._count_trained_images()
            assert count == 0


# ── recognize_speech ────────────────────────────────────────────────────────


class TestRecognizeSpeech:

    def test_recognize_speech(self):
        m = MultimodalManager()
        mock_recognizer = MagicMock()
        m._speech_recognizer = mock_recognizer
        # Non-silent audio (sine wave) so VAD passes — need >= 250ms at 16kHz
        audio = np.sin(np.linspace(0, 10, 4800)) * 10000
        audio = audio.astype(np.int16).tobytes()
        m.recognize_speech(audio, "en")
        mock_recognizer.recognize.assert_called_once()

    @patch("domains.multimodal.manager.get_speech_recognizer")
    def test_recognize_speech_no_recognizer(self, mock_get):
        m = MultimodalManager()
        mock_rec = MagicMock()
        mock_get.return_value = mock_rec
        audio = np.sin(np.linspace(0, 10, 4800)) * 10000
        audio = audio.astype(np.int16).tobytes()
        m.recognize_speech(audio)
        mock_get.assert_called_once()

    def test_recognize_speech_empty_audio(self):
        m = MultimodalManager()
        mock_recognizer = MagicMock()
        m._speech_recognizer = mock_recognizer
        result = m.recognize_speech(b"", "en")
        assert result.text == ""
        assert result.is_valid == False
        mock_recognizer.recognize.assert_not_called()

    def test_recognize_speech_vad_silence(self):
        m = MultimodalManager()
        mock_recognizer = MagicMock()
        m._speech_recognizer = mock_recognizer
        # Very quiet audio — VAD should detect silence
        audio = np.full(16000, 10, dtype=np.int16).tobytes()
        result = m.recognize_speech(audio, "en")
        assert result.text == ""
        assert result.is_valid == False
        mock_recognizer.recognize.assert_not_called()

    def test_recognize_speech_with_audio_filter_config(self):
        m = MultimodalManager()
        mock_recognizer = MagicMock()
        m._speech_recognizer = mock_recognizer
        from domains.multimodal.audio_filter import AudioFilterConfig, FilterMode
        m._audio_filter_config = AudioFilterConfig(mode=FilterMode.NONE)
        audio = np.zeros(160, dtype=np.int16).tobytes()
        m.recognize_speech(audio, "en")
        mock_recognizer.recognize.assert_called_once()


# ── caption_image ──────────────────────────────────────────────────────────


class TestCaptionImage:

    def test_caption_no_engine(self):
        m = MultimodalManager()
        engine_mock = MagicMock()
        engine_mock.generate.return_value = MagicMock(text="generated caption")
        engine_mock.vision.forward.return_value = MagicMock(data=np.array([0.5]))
        engine_mock.text.encode.return_value = [1, 2, 3]
        engine_mock.train_step.return_value = 0.5

        from PIL import Image
        img = Image.new("RGB", (64, 64), (128, 128, 128))

        with patch("domains.multimodal.manager.get_multimodal_engine", return_value=engine_mock):
            with patch("domains.multimodal.manager.contrastive_step", return_value=0.1):
                result = m.caption_image(img, generate_only=True)
                assert result.text == "generated caption"
                assert result.tags == ["vision", "generated"]

    def test_caption_generate_only_cached(self):
        m = MultimodalManager()
        engine_mock = MagicMock()
        engine_mock.vision.forward.return_value = MagicMock(data=np.array([0.5]))
        m._multimodal_engine = engine_mock
        m._embed_cache = {hash(b"cached"): "cached caption"}

        from PIL import Image
        img = Image.new("RGB", (64, 64), (128, 128, 128))

        with patch.object(m, "_image_hash", return_value=hash(b"cached")):
            result = m.caption_image(img, generate_only=True)
            assert result.text == "cached caption"

    def test_caption_error_handling(self):
        m = MultimodalManager()
        m._multimodal_engine = MagicMock()
        m._multimodal_engine.generate.side_effect = RuntimeError("fail")

        from PIL import Image
        img = Image.new("RGB", (64, 64), (128, 128, 128))

        result = m.caption_image(img, generate_only=True)
        assert result.text == "[caption failed]"
        assert result.tags == ["error"]


# ── detect_objects ──────────────────────────────────────────────────────────


class TestDetectObjects:

    def test_detect_objects(self):
        m = MultimodalManager()
        m._learning_count = 10  # >= 10 to skip seed caption
        m._multimodal_engine = MagicMock()
        m._multimodal_engine.generate.return_value = MagicMock(text="a red circle")
        m._multimodal_engine.vision.forward.return_value = MagicMock(data=np.array([0.5]))
        m._multimodal_engine.text.encode.return_value = [1, 2]

        from PIL import Image
        img = Image.new("RGB", (64, 64))
        with patch("domains.multimodal.manager.contrastive_step", return_value=0.1):
            objs = m.detect_objects(img)
            assert len(objs) == 1
            assert objs[0].label == "a red circle"
            assert objs[0].confidence >= 0.0


# ── get_browser_speech_config ──────────────────────────────────────────────


class TestBrowserSpeechConfig:

    def test_get_config_no_recognizer(self):
        m = MultimodalManager()
        with patch("domains.multimodal.manager.get_speech_recognizer") as mock_get:
            mock_rec = MagicMock()
            mock_rec.get_config.return_value = {"language": "en-US"}
            mock_get.return_value = mock_rec
            config = m.get_browser_speech_config()
            assert config["language"] == "en-US"

    def test_get_config_no_get_config_method(self):
        m = MultimodalManager()
        m._speech_recognizer = MagicMock(spec=[])  # no get_config
        config = m.get_browser_speech_config()
        assert config == {"language": "en-US"}


# ── Singleton ───────────────────────────────────────────────────────────────


class TestSingleton:

    def test_get_multimodal_manager(self):
        import domains.multimodal.manager as mod
        mod._multimodal_manager = None
        m = get_multimodal_manager()
        assert isinstance(m, MultimodalManager)
        assert get_multimodal_manager() is m

    def test_initialize_multimodal(self):
        import domains.multimodal.manager as mod
        mod._multimodal_manager = None
        with patch.object(MultimodalManager, "initialize"):
            initialize_multimodal(speech_server=False, vision_model="slonet")
            m = get_multimodal_manager()
            m.initialize.assert_called_once()


# ── _pretrain_engine ────────────────────────────────────────────────────────


class TestPretrain:

    def test_pretrain_no_engine(self):
        m = MultimodalManager()
        m._multimodal_engine = None
        result = m._pretrain_engine(epochs=1, samples=4)
        assert result == float("inf")

    def test_pretrain_with_engine(self):
        m = MultimodalManager()
        engine = MagicMock()
        engine.text.build_vocab = MagicMock()
        engine.train_step.return_value = 0.5
        engine.generate.return_value = MagicMock(text="sample caption")
        engine.vision.forward.return_value = MagicMock(data=np.array([0.1]))
        engine.save = MagicMock()
        m._multimodal_engine = engine

        result = m._pretrain_engine(epochs=2, samples=8, batch_size=4)
        assert isinstance(result, float)
        assert engine.save.called
