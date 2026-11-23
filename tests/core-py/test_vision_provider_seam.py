"""Vision provider seam — config-selected vision model (card 20261116_144837).

``config.vision.model`` / env ``SLO_VISION__MODEL`` names the vision provider.
Consumers (``analyze_image``, ``VisionProcessor``, router ``describe_image``)
resolve it through ``domain.models._internal.provider.vision`` instead of
hardcoding ``VisionCNN`` / ``"multimodal"``.

Honesty contract: a configured provider that is missing or incapable reports
unavailable — never a silent fallback to a different model. The built-in
``"multimodal"`` backend keeps its existing VisionCNN analysis path.

The seam module is imported lazily inside the tests: it is the feature under
test, so pre-implementation RED is an ImportError, not a collection error.
"""

from __future__ import annotations

import base64

import pytest

from domain.infrastructure._internal.config import (
    AppConfig,
    ConfigManager,
    _apply_env_overrides,
    get_config_manager,
    set_config_manager,
)
from domain.models._internal.provider.registry import (
    clear_providers,
    register_provider,
)


@pytest.fixture(autouse=True)
def _isolated():
    """Fresh provider registry + config manager per test, restored after."""
    clear_providers()
    saved_manager = get_config_manager()
    yield
    clear_providers()
    set_config_manager(saved_manager)


@pytest.fixture
def vision_config(tmp_path, monkeypatch):
    """Install a config manager whose ``vision.model`` comes from the env."""

    def _set(name: str | None) -> None:
        if name is None:
            monkeypatch.delenv("SLO_VISION__MODEL", raising=False)
        else:
            monkeypatch.setenv("SLO_VISION__MODEL", name)
        set_config_manager(ConfigManager(config_dir=str(tmp_path)))

    return _set


def _boom(*_args, **_kwargs):
    raise AssertionError("multimodal init must not be attempted")


def _write_image(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG fake")
    return str(img)


# ── config ──────────────────────────────────────────────────────────────────


class TestVisionConfig:
    def test_default_vision_model(self):
        assert AppConfig().vision.model == "multimodal"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SLO_VISION__MODEL", "vlm")
        cfg = _apply_env_overrides(AppConfig())
        assert cfg.vision.model == "vlm"


# ── resolver ────────────────────────────────────────────────────────────────


class TestGetVisionModelName:
    def test_reads_config(self, vision_config):
        vision_config("vlm")
        from domain.models._internal.provider.vision import get_vision_model_name

        assert get_vision_model_name() == "vlm"

    def test_default(self, vision_config):
        vision_config(None)
        from domain.models._internal.provider.vision import get_vision_model_name

        assert get_vision_model_name() == "multimodal"


class TestGetVisionProvider:
    def test_registered_provider_returned(self, vision_config):
        vision_config("vlm")
        from domain.models._internal.provider.vision import get_vision_provider

        sentinel = object()
        register_provider("vlm", sentinel)
        assert get_vision_provider() is sentinel

    def test_explicit_name_overrides_config(self, vision_config):
        vision_config("vlm")
        from domain.models._internal.provider.vision import get_vision_provider

        sentinel = object()
        register_provider("mm2", sentinel)
        assert get_vision_provider(name="mm2") is sentinel

    def test_unknown_configured_name_is_none_without_init(self, vision_config, monkeypatch):
        vision_config("vlm")
        from domain.models._internal.provider.vision import get_vision_provider
        from domain.multimodal._internal import manager as mm_manager

        monkeypatch.setattr(mm_manager, "initialize_multimodal", _boom)
        assert get_vision_provider() is None

    def test_default_name_no_side_effect_without_init(self, vision_config, monkeypatch):
        vision_config(None)  # "multimodal", registry empty
        from domain.models._internal.provider.vision import get_vision_provider
        from domain.multimodal._internal import manager as mm_manager

        monkeypatch.setattr(mm_manager, "initialize_multimodal", _boom)
        assert get_vision_provider() is None

    def test_init_true_lazily_registers_multimodal(self, vision_config, monkeypatch):
        vision_config(None)
        from domain.models._internal.provider.vision import get_vision_provider
        from domain.multimodal._internal import manager as mm_manager

        sentinel = object()

        def _fake_init(*_args, **_kwargs):
            register_provider("multimodal", sentinel)

        monkeypatch.setattr(mm_manager, "initialize_multimodal", _fake_init)
        assert get_vision_provider(init=True) is sentinel


# ── analyze_image ───────────────────────────────────────────────────────────


class _TrioVision:
    """Provider exposing the analysis trio analyze_image contracts on."""

    def caption(self, _path):
        from domain.multimodal._internal.vision import ImageCaption

        return ImageCaption(text="from-provider", confidence=0.9, tags=["x"])

    def detect(self, _path):
        return []

    def get_embedding(self, _path):
        return None


class TestAnalyzeImageSeam:
    def test_uses_configured_trio_provider(self, vision_config, tmp_path):
        vision_config("vlm")
        register_provider("vlm", _TrioVision())
        from domain.inference import analyze_image

        res = analyze_image(_write_image(tmp_path))
        assert res.success is True
        assert res.caption == "from-provider"
        assert res.error_kind == ""

    def test_missing_configured_provider_reports_unavailable(self, vision_config, tmp_path):
        vision_config("vlm")  # nothing registered under it
        from domain.inference import analyze_image

        res = analyze_image(_write_image(tmp_path))
        assert res.success is False
        assert res.error_kind == "unavailable"
        assert "Vision model not available" in res.error


# ── VisionProcessor ─────────────────────────────────────────────────────────


class TestVisionProcessorConfig:
    def test_default_name_from_config(self, vision_config):
        vision_config(None)
        from domain.models._internal.provider.processors.vision import VisionProcessor

        assert VisionProcessor()._provider_name == "multimodal"

    def test_configured_name_used(self, vision_config):
        vision_config("vlm")
        from domain.models._internal.provider.processors.vision import VisionProcessor

        assert VisionProcessor()._provider_name == "vlm"

    def test_explicit_name_wins_over_config(self, vision_config):
        vision_config("vlm")
        from domain.models._internal.provider.processors.vision import VisionProcessor

        assert VisionProcessor("explicit")._provider_name == "explicit"

    async def test_ensure_provider_resolves_configured_name(self, vision_config, monkeypatch):
        vision_config("vlm")
        from domain.models._internal.provider.processors.vision import VisionProcessor
        from domain.multimodal._internal import manager as mm_manager

        register_provider("vlm", object())
        monkeypatch.setattr(mm_manager, "initialize_multimodal", _boom)
        assert VisionProcessor()._ensure_provider() is True


# ── router describe_image tool ──────────────────────────────────────────────


class _ChatVision:
    async def chat_stream(self, _messages, **_kwargs):
        yield "from-vlm"


class TestDescribeImageSeam:
    async def test_router_tool_uses_configured_provider(self, vision_config, monkeypatch):
        vision_config("vlm")
        from domain.models._internal.provider.router import ProviderRouter
        from domain.multimodal._internal import manager as mm_manager

        register_provider("vlm", _ChatVision())
        # A non-multimodal config must never fall back to the multimodal backend.
        monkeypatch.setattr(mm_manager, "initialize_multimodal", _boom)
        monkeypatch.setattr(mm_manager, "get_multimodal_manager", _boom)

        router = ProviderRouter()
        data_url = "data:image/png;base64," + base64.b64encode(b"fake").decode()
        result = await router._execute_tool("describe_image", data_url)
        assert result == "from-vlm"

    async def test_router_tool_reports_unavailable_for_missing_provider(
        self, vision_config, monkeypatch
    ):
        vision_config("vlm")  # nothing registered
        from domain.models._internal.provider.router import ProviderRouter
        from domain.multimodal._internal import manager as mm_manager

        monkeypatch.setattr(mm_manager, "initialize_multimodal", _boom)
        monkeypatch.setattr(mm_manager, "get_multimodal_manager", _boom)

        router = ProviderRouter()
        data_url = "data:image/png;base64," + base64.b64encode(b"fake").decode()
        result = await router._execute_tool("describe_image", data_url)
        assert "not available" in result
