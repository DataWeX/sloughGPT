"""Tests for domains.infrastructure.model_server — IdleManager."""

from domains.infrastructure.model_server import IdleManager


class TestIdleManager:
    def test_init(self):
        im = IdleManager()
        assert im._idle_timeout_s == 300.0
        assert im._check_interval_s == 30.0

    def test_register(self):
        im = IdleManager()
        im.register("model1")
        assert "model1" in im._models

    def test_unregister(self):
        im = IdleManager()
        im.register("model1")
        im.unregister("model1")
        assert "model1" not in im._models

    def test_touch(self):
        im = IdleManager()
        im.register("model1")
        result = im.touch("model1")
        assert isinstance(result, bool)
