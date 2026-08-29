"""Tests for domains.multimodal.manager — MultimodalCapabilities."""

from domains.multimodal.manager import MultimodalCapabilities


class TestMultimodalCapabilities:
    def test_defaults(self):
        mc = MultimodalCapabilities()
        assert mc.speech_to_text is False
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is False
        assert mc.speech_model is None
        assert mc.vision_model is None

    def test_custom(self):
        mc = MultimodalCapabilities(
            speech_to_text=True,
            image_caption=True,
            speech_model="whisper",
            vision_model="clip",
        )
        assert mc.speech_to_text is True
        assert mc.image_caption is True
        assert mc.speech_model == "whisper"
        assert mc.vision_model == "clip"
