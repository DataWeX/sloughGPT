"""Tests for domains.training.huggingface.api_loader — HFAPIConfig."""

from domains.training.huggingface.api_loader import HFAPIConfig


class TestHFAPIConfig:
    def test_fields(self):
        hc = HFAPIConfig(model="gpt2")
        assert hc.model == "gpt2"
        assert hc.api_key is None
        assert hc.timeout == 60
        assert hc.max_new_tokens == 256
        assert hc.temperature == 0.7

    def test_custom(self):
        hc = HFAPIConfig(model="llama", api_key="key", temperature=0.5)
        assert hc.model == "llama"
        assert hc.api_key == "key"
        assert hc.temperature == 0.5
