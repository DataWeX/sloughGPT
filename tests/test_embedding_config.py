"""Generic embedding provider config — api_key + base_url (card 90839119).

Back-compat: ``openai_api_key`` stays readable/writable; new generic fields
``api_key`` + ``base_url`` let embeddings target any OpenAI-compatible
endpoint (ollama, LM Studio, vLLM) — user ask 2026-08-28, repeated.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


class TestEmbeddingConfigFields:
    def test_openai_api_key_back_compat_fills_generic_field(self):
        from domain.infrastructure._internal.config import EmbeddingConfig

        cfg = EmbeddingConfig(openai_api_key="sk-old")
        assert cfg.api_key == "sk-old"

    def test_generic_api_key_fills_legacy_alias(self):
        from domain.infrastructure._internal.config import EmbeddingConfig

        cfg = EmbeddingConfig(api_key="sk-new")
        assert cfg.openai_api_key == "sk-new"

    def test_base_url_field(self):
        from domain.infrastructure._internal.config import EmbeddingConfig

        cfg = EmbeddingConfig(base_url="http://localhost:11434/v1")
        assert cfg.base_url == "http://localhost:11434/v1"

    def test_env_overrides_map_generic_keys(self, monkeypatch):
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.setenv("SLO_EMBEDDING__API_KEY", "sk-env")
        monkeypatch.setenv("SLO_EMBEDDING__BASE_URL", "http://llm.local:8080/v1")
        cfg = _apply_env_overrides(AppConfig())
        assert cfg.embedding.api_key == "sk-env"
        assert cfg.embedding.base_url == "http://llm.local:8080/v1"
        assert cfg.embedding.openai_api_key == "sk-env"

    def test_env_legacy_openai_key_still_fills_generic_field(self, monkeypatch):
        # The env name .env.example documented BEFORE this change must keep
        # working end-to-end (old deployments): SLO_EMBEDDING__OPENAI_API_KEY
        # -> env walk -> post-env model_validate -> alias mirror -> api_key.
        from domain.infrastructure._internal.config import AppConfig, _apply_env_overrides

        monkeypatch.delenv("SLO_EMBEDDING__API_KEY", raising=False)
        monkeypatch.setenv("SLO_EMBEDDING__OPENAI_API_KEY", "sk-legacy-env")
        cfg = _apply_env_overrides(AppConfig())
        assert cfg.embedding.api_key == "sk-legacy-env"
        assert cfg.embedding.openai_api_key == "sk-legacy-env"

    def test_new_field_wins_when_both_set(self):
        from domain.infrastructure._internal.config import EmbeddingConfig

        cfg = EmbeddingConfig(api_key="sk-new", openai_api_key="sk-old")
        assert cfg.api_key == "sk-new"
        # mirror keeps old-field readers in agreement with the winning value
        assert cfg.openai_api_key == "sk-new"


@pytest.fixture
def fake_openai(monkeypatch):
    """Stub the openai package (not installed in the shared venv)."""
    mod = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            self.api_key = api_key
            self.base_url = base_url

    mod.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


class TestOpenAIEmbedderProviderConfig:
    def _config(self, **embedding):
        base = {"provider": "n_gram", "api_key": "", "openai_api_key": "", "base_url": ""}
        base.update(embedding)
        return SimpleNamespace(embedding=SimpleNamespace(**base))

    def test_base_url_taken_from_config(self, fake_openai, monkeypatch):
        from domain.inference._internal import embeddings as emb_mod

        monkeypatch.setattr(
            emb_mod,
            "get_config",
            lambda: self._config(api_key="sk-cfg", base_url="http://localhost:11434/v1"),
        )
        embedder = emb_mod.OpenAIEmbedder()
        assert embedder.base_url == "http://localhost:11434/v1"
        assert embedder.client.base_url == "http://localhost:11434/v1"

    def test_legacy_openai_api_key_still_used(self, fake_openai, monkeypatch):
        from domain.inference._internal import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "get_config", lambda: self._config(openai_api_key="sk-legacy"))
        embedder = emb_mod.OpenAIEmbedder()
        assert embedder.api_key == "sk-legacy"
        assert embedder.client.api_key == "sk-legacy"

    def test_missing_key_still_raises(self, fake_openai, monkeypatch):
        from domain.inference._internal import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "get_config", lambda: self._config())
        with pytest.raises(ValueError):
            emb_mod.OpenAIEmbedder()

    def test_embedder_factory_passes_base_url(self, fake_openai):
        from domain.inference._internal import embeddings as emb_mod

        embedder = emb_mod.Embedder(provider="openai", api_key="sk-x", base_url="http://host/v1")
        impl = embedder._impl
        assert impl.base_url == "http://host/v1"
        assert impl.client.api_key == "sk-x"

    def test_no_base_url_config_means_none(self, fake_openai, monkeypatch):
        # Empty base_url must reach the OpenAI client as None (provider
        # default / OPENAI_BASE_URL env), never as "" or a fabricated URL.
        from domain.inference._internal import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "get_config", lambda: self._config(api_key="sk-cfg"))
        embedder = emb_mod.OpenAIEmbedder()
        assert embedder.base_url is None
        assert embedder.client.base_url is None
