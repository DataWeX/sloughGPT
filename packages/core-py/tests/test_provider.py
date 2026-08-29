"""Tests for domains.models.provider — ProviderRouter, ModelCapabilities,
provider registry, processors (VisionProcessor, KnowledgeProcessor,
ToolUseProcessor, PersonalityProcessor, StyleProcessor).

Covers: dataclass creation, router setup, processor detection, registry CRUD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.models.provider import (
    ModelCapabilities,
    ProviderRouter,
    VisionProcessor,
    KnowledgeProcessor,
    ToolUseProcessor,
    PersonalityProcessor,
    StyleProcessor,
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
)


class TestModelCapabilities:
    def test_defaults(self):
        c = ModelCapabilities()
        assert c.chat is False
        assert c.streaming is False
        assert c.embedding is False
        assert c.vision is False

    def test_custom(self):
        c = ModelCapabilities(chat=True, streaming=True, embedding=True, vision=True)
        assert c.chat is True
        assert c.vision is True


class TestProviderRouter:
    def test_creation(self):
        r = ProviderRouter()
        assert r.model_id == "router-v1"
        assert r._processors == []

    def test_add_processor(self):
        r = ProviderRouter()
        p = KnowledgeProcessor()
        result = r.add_processor(p)
        assert result is r  # fluent API
        assert len(r._processors) == 1

    def test_set_text_provider(self):
        r = ProviderRouter()
        r.set_text_provider("slonet")
        assert r._text_name == "slonet"

    def test_capabilities(self):
        r = ProviderRouter()
        caps = r.capabilities
        assert caps.chat is True
        assert caps.vision is True

    def test_metadata(self):
        r = ProviderRouter()
        r.add_processor(KnowledgeProcessor())
        r.set_text_provider("test")
        meta = r.metadata
        assert "KnowledgeProcessor" in meta["processors"]
        assert meta["text_provider"] == "test"

    def test_find_tool_processor(self):
        r = ProviderRouter()
        assert r._find_tool_processor() is None
        r.add_processor(ToolUseProcessor())
        assert r._find_tool_processor() is not None


class TestProcessors:
    def test_vision_processor(self):
        p = VisionProcessor("multimodal")
        assert p is not None

    def test_knowledge_processor(self):
        p = KnowledgeProcessor()
        assert p is not None

    def test_tool_use_processor(self):
        p = ToolUseProcessor()
        assert p is not None

    def test_personality_processor(self):
        p = PersonalityProcessor()
        assert p is not None

    def test_style_processor(self):
        p = StyleProcessor()
        assert p is not None


class TestProviderRegistry:
    def test_register_and_get(self):
        register_provider("test_provider", "mock")
        assert get_provider("test_provider") == "mock"

    def test_get_missing(self):
        assert get_provider("nonexistent_xyz") is None

    def test_list_providers(self):
        register_provider("test_list_1", "mock1")
        register_provider("test_list_2", "mock2")
        providers = list_providers()
        assert "test_list_1" in providers
        assert "test_list_2" in providers

    def test_clear(self):
        register_provider("clear_test", "mock")
        clear_providers()
        assert get_provider("clear_test") is None
