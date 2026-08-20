"""Tests for domains.multimodal — MultiModalConfig."""

from domains.multimodal import MultiModalConfig


class TestMultiModalConfig:
    def test_defaults(self):
        cfg = MultiModalConfig()
        assert cfg.image_size == 224
        assert cfg.patch_size == 16
        assert cfg.fusion_type == "cross_attention"
        assert cfg.max_seq_length == 512

    def test_custom(self):
        cfg = MultiModalConfig(image_size=512, patch_size=32, fusion_type="concat")
        assert cfg.image_size == 512
        assert cfg.patch_size == 32
        assert cfg.fusion_type == "concat"
