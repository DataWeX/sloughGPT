"""Tests for domains.training.huggingface.local_loader — HFLocalConfig."""

from domains.training.huggingface.local_loader import HFLocalConfig


class TestHFLocalConfig:
    def test_fields(self):
        hc = HFLocalConfig(model="gpt2")
        assert hc.model == "gpt2"
        assert hc.device == "auto"
        assert hc.load_in_8bit is False
        assert hc.load_in_4bit is False
        assert hc.local_files_only is True
        assert hc.max_new_tokens == 256

    def test_custom(self):
        hc = HFLocalConfig(model="llama", device="cuda", load_in_4bit=True)
        assert hc.device == "cuda"
        assert hc.load_in_4bit is True
