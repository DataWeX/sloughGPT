"""Tests for domains.models.provider — ProviderRouter, ModelCapabilities,
provider registry, processors (VisionProcessor, KnowledgeProcessor,
ToolUseProcessor, PersonalityProcessor, StyleProcessor).

Covers: dataclass creation, router setup, processor detection, registry CRUD,
attach_process_guard_to_provider, update_personality_traits, _server_from_provider,
ProviderRouter.chat/chat_stream.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    ToolDef,
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
    register_processor,
    get_processor,
    list_processors,
    apply_processors,
    attach_process_guard_to_provider,
    update_personality_traits,
    _server_from_provider,
)


# =============================================================================
# ModelCapabilities
# =============================================================================

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

    def test_all_true(self):
        c = ModelCapabilities(chat=True, streaming=True, embedding=True, vision=True, functions=True)
        assert c.functions is True
        assert all([c.chat, c.streaming, c.embedding, c.vision, c.functions])

    def test_all_false_explicit(self):
        c = ModelCapabilities(chat=False, streaming=False, embedding=False, vision=False, functions=False)
        assert c.chat is False
        assert c.streaming is False
        assert c.embedding is False
        assert c.vision is False
        assert c.functions is False

    def test_functions_default_false(self):
        c = ModelCapabilities()
        assert c.functions is False

    def test_partial_capabilities(self):
        c = ModelCapabilities(chat=True, vision=True)
        assert c.chat is True
        assert c.streaming is False
        assert c.embedding is False
        assert c.vision is True
        assert c.functions is False

    def test_equality(self):
        c1 = ModelCapabilities(chat=True, streaming=False)
        c2 = ModelCapabilities(chat=True, streaming=False)
        assert c1 == c2

    def test_inequality(self):
        c1 = ModelCapabilities(chat=True)
        c2 = ModelCapabilities(chat=False)
        assert c1 != c2

    def test_hashability_not_required(self):
        """ModelCapabilities is a dataclass — equality works without hash."""
        c1 = ModelCapabilities(chat=True)
        c2 = ModelCapabilities(chat=True)
        assert c1 == c2

    def test_repr(self):
        c = ModelCapabilities(chat=True)
        assert "chat=True" in repr(c)


# =============================================================================
# ProviderRouter
# =============================================================================

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
        assert caps.streaming is True
        assert caps.embedding is False

    def test_metadata(self):
        r = ProviderRouter()
        r.add_processor(KnowledgeProcessor())
        r.set_text_provider("test")
        meta = r.metadata
        assert "KnowledgeProcessor" in meta["processors"]
        assert meta["text_provider"] == "test"
        assert meta["max_tool_rounds"] == 3

    def test_find_tool_processor(self):
        r = ProviderRouter()
        assert r._find_tool_processor() is None
        r.add_processor(ToolUseProcessor())
        assert r._find_tool_processor() is not None

    def test_find_tool_processor_among_many(self):
        r = ProviderRouter()
        r.add_processor(KnowledgeProcessor())
        r.add_processor(StyleProcessor())
        assert r._find_tool_processor() is None
        r.add_processor(ToolUseProcessor())
        tp = r._find_tool_processor()
        assert tp is not None

    def test_find_tool_processor_first_is_tool(self):
        r = ProviderRouter()
        r.add_processor(ToolUseProcessor())
        r.add_processor(KnowledgeProcessor())
        assert isinstance(r._find_tool_processor(), ToolUseProcessor)

    def test_metadata_empty(self):
        r = ProviderRouter()
        meta = r.metadata
        assert meta["processors"] == []
        assert meta["text_provider"] is None

    def test_add_multiple_processors(self):
        r = ProviderRouter()
        r.add_processor(VisionProcessor("mm"))
        r.add_processor(KnowledgeProcessor())
        r.add_processor(ToolUseProcessor())
        r.add_processor(PersonalityProcessor())
        r.add_processor(StyleProcessor())
        assert len(r._processors) == 5

    def test_model_id_always_router_v1(self):
        r1 = ProviderRouter()
        r2 = ProviderRouter()
        assert r1.model_id == r2.model_id == "router-v1"

    def test_embed_returns_empty(self):
        r = ProviderRouter()
        assert r.embed("hello") == []

    def test_find_tool_processor_no_tool(self):
        r = ProviderRouter()
        r.add_processor(KnowledgeProcessor())
        r.add_processor(VisionProcessor("mm"))
        assert r._find_tool_processor() is None

    def test_add_processor_returns_self(self):
        r = ProviderRouter()
        result = r.add_processor(StyleProcessor())
        assert result is r

    def test_capabilities_streaming(self):
        r = ProviderRouter()
        assert r.capabilities.streaming is True

    def test_capabilities_embedding_false(self):
        r = ProviderRouter()
        assert r.capabilities.embedding is False


# =============================================================================
# ProcessorRegistry
# =============================================================================

class TestProcessorRegistry:
    def test_register_and_get(self):
        register_processor("test_proc", "mock")
        assert get_processor("test_proc") == "mock"

    def test_get_missing(self):
        assert get_processor("nonexistent_xyz") is None

    def test_list_processors(self):
        register_processor("test_list_1", "mock1")
        register_processor("test_list_2", "mock2")
        procs = list_processors()
        assert "test_list_1" in procs
        assert "test_list_2" in procs

    def test_clear_does_not_affect_providers(self):
        register_provider("proc_test_p", "mock")
        register_processor("proc_test_proc", "mock")
        clear_providers()
        assert get_provider("proc_test_p") is None
        assert get_processor("proc_test_proc") == "mock"

    def test_register_overwrites(self):
        register_processor("ow_test", "old")
        register_processor("ow_test", "new")
        assert get_processor("ow_test") == "new"


# =============================================================================
# apply_processors
# =============================================================================

class TestApplyProcessors:
    @pytest.mark.asyncio
    async def test_empty_processors(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [])
        assert result == msgs

    @pytest.mark.asyncio
    async def test_processor_transforms(self):
        class Adder:
            async def process(self, messages):
                return messages + [{"role": "system", "content": "added"}]

        result = await apply_processors([{"role": "user", "content": "hi"}], [Adder()])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_processor_exception_is_swallowed(self):
        class Bad:
            async def process(self, messages):
                raise RuntimeError("boom")

        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [Bad()])
        assert result == msgs

    @pytest.mark.asyncio
    async def test_multiple_processors_chain(self):
        class Double:
            async def process(self, messages):
                return messages + messages

        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [Double(), Double()])
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_first_processor_fails_second_runs(self):
        class Bad:
            async def process(self, messages):
                raise RuntimeError("fail")

        class Good:
            async def process(self, messages):
                return messages + [{"role": "system", "content": "ok"}]

        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [Bad(), Good()])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_preserves_original_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [])
        assert result is msgs


# =============================================================================
# VisionProcessor
# =============================================================================

class TestVisionProcessor:
    def test_creation(self):
        p = VisionProcessor("multimodal")
        assert p._provider_name == "multimodal"

    def test_default_provider_name(self):
        p = VisionProcessor()
        assert p._provider_name == "multimodal"

    def test_extract_images_none(self):
        p = VisionProcessor("mm")
        assert p._extract_images([{"role": "user", "content": "hello"}]) == []

    def test_extract_images_list_content(self):
        p = VisionProcessor("mm")
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "describe"},
        ]}]
        imgs = p._extract_images(msgs)
        assert len(imgs) == 1
        assert "abc" in imgs[0]

    def test_extract_images_string_base64(self):
        p = VisionProcessor("mm")
        msgs = [{"role": "user", "content": "data:image/jpeg;base64,XYZDATA"}]
        imgs = p._extract_images(msgs)
        assert len(imgs) == 1

    def test_extract_images_no_images(self):
        p = VisionProcessor("mm")
        msgs = [{"role": "user", "content": "plain text"}]
        assert p._extract_images(msgs) == []

    def test_extract_images_multiple_messages(self):
        p = VisionProcessor("mm")
        msgs = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img1"}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img2"}}]},
        ]
        assert len(p._extract_images(msgs)) == 2

    @pytest.mark.asyncio
    async def test_process_no_images_returns_unchanged(self):
        p = VisionProcessor("mm")
        msgs = [{"role": "user", "content": "hello"}]
        result = await p.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_process_provider_unavailable_strips_images(self):
        p = VisionProcessor("nonexistent_provider_xyz")
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = await p.process(msgs)
        assert len(result) == 1
        assert result[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_process_provider_unavailable_plain_text(self):
        p = VisionProcessor("nonexistent_provider_xyz")
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = await p.process(msgs)
        assert "describe this" in result[0]["content"]
        assert "[Image attached" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_process_provider_unavailable_preserves_role(self):
        p = VisionProcessor("nonexistent_provider_xyz")
        msgs = [{"role": "assistant", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = await p.process(msgs)
        assert result[0]["role"] == "assistant"
        assert "Image attached" not in result[0].get("content", "")

    def test_extract_images_empty_messages(self):
        p = VisionProcessor("mm")
        assert p._extract_images([]) == []

    def test_extract_images_string_without_base64(self):
        p = VisionProcessor("mm")
        msgs = [{"role": "user", "content": "just a plain string"}]
        assert p._extract_images(msgs) == []


# =============================================================================
# KnowledgeProcessor
# =============================================================================

class TestKnowledgeProcessor:
    @pytest.mark.asyncio
    async def test_process_empty_knowledge(self):
        p = KnowledgeProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        assert await p.process(msgs) == msgs

    @pytest.mark.asyncio
    async def test_process_injects_knowledge(self):
        p = KnowledgeProcessor(knowledge=["fact1", "fact2"])
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert len(result) == 2
        assert "Knowledge context:" in result[0]["content"]
        assert "fact1" in result[0]["content"]
        assert "fact2" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_set_knowledge(self):
        p = KnowledgeProcessor()
        p.set_knowledge(["new_fact"])
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "new_fact" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_knowledge_prepends_system_message(self):
        p = KnowledgeProcessor(knowledge=["k1"])
        msgs = [{"role": "user", "content": "q"}]
        result = await p.process(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_set_knowledge_overwrites(self):
        p = KnowledgeProcessor(knowledge=["old"])
        p.set_knowledge(["new"])
        msgs = [{"role": "user", "content": "q"}]
        result = await p.process(msgs)
        assert "new" in result[0]["content"]
        assert "old" not in result[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_knowledge_list(self):
        p = KnowledgeProcessor(knowledge=[])
        msgs = [{"role": "user", "content": "q"}]
        assert await p.process(msgs) == msgs

    @pytest.mark.asyncio
    async def test_single_knowledge_fact(self):
        p = KnowledgeProcessor(knowledge=["only fact"])
        msgs = [{"role": "user", "content": "q"}]
        result = await p.process(msgs)
        assert "only fact" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_preserves_original_messages(self):
        p = KnowledgeProcessor(knowledge=["k"])
        msgs = [{"role": "user", "content": "q"}]
        original = msgs.copy()
        result = await p.process(msgs)
        assert len(result) == len(original) + 1


# =============================================================================
# ToolUseProcessor
# =============================================================================

class TestToolUseProcessor:
    def test_creation_default_tools(self):
        p = ToolUseProcessor()
        assert len(p._tools) > 0

    def test_creation_custom_tools(self):
        tools = [ToolDef(name="my_tool", provider_name="prov")]
        p = ToolUseProcessor(tools=tools)
        assert len(p._tools) == 1
        assert p._tools[0].name == "my_tool"

    def test_has_image_string(self):
        assert ToolUseProcessor._has_image([{"role": "user", "content": "data:image/png;base64,abc"}]) is True

    def test_has_image_list(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        assert ToolUseProcessor._has_image(msgs) is True

    def test_has_image_no_image(self):
        assert ToolUseProcessor._has_image([{"role": "user", "content": "hello"}]) is False

    def test_has_image_empty(self):
        assert ToolUseProcessor._has_image([]) is False

    def test_has_image_non_dict_part(self):
        msgs = [{"role": "user", "content": ["just a string"]}]
        assert ToolUseProcessor._has_image(msgs) is False

    @pytest.mark.asyncio
    async def test_process_no_image_returns_unchanged(self):
        p = ToolUseProcessor()
        msgs = [{"role": "user", "content": "hello"}]
        result = await p.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_process_injects_tool_prompt(self):
        p = ToolUseProcessor()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        result = await p.process(msgs)
        assert any("tools" in m.get("content", "").lower() for m in result)

    @pytest.mark.asyncio
    async def test_process_appends_to_existing_system(self):
        p = ToolUseProcessor()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]
        result = await p.process(msgs)
        sys_msgs = [m for m in result if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert "You are helpful." in sys_msgs[0]["content"]
        assert "tools" in sys_msgs[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_process_inserts_system_when_missing(self):
        p = ToolUseProcessor()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        result = await p.process(msgs)
        assert result[0]["role"] == "system"

    def test_match_tool_found(self):
        p = ToolUseProcessor()
        result = p.match_tool("I need to call [[TOOL: describe_image]] abc123")
        assert result is not None
        assert result[0] == "describe_image"
        assert result[1] == "abc123"

    def test_match_tool_placeholder_rejected(self):
        p = ToolUseProcessor()
        result = p.match_tool("[[TOOL: describe_image]] <base64_image_data>")
        assert result is None

    def test_match_tool_no_match(self):
        p = ToolUseProcessor()
        assert p.match_tool("no tool call here") is None

    def test_match_tool_unknown_tool_name(self):
        p = ToolUseProcessor()
        result = p.match_tool("[[TOOL: nonexistent_tool]] arg")
        assert result is None

    def test_tool_def_dataclass(self):
        td = ToolDef(name="test", provider_name="prov", description="desc")
        assert td.name == "test"
        assert td.description == "desc"

    def test_tool_def_default_description(self):
        td = ToolDef(name="x", provider_name="y")
        assert td.description == ""

    def test_has_image_string_with_png(self):
        msgs = [{"role": "user", "content": "data:image/png;base64,abc"}]
        assert ToolUseProcessor._has_image(msgs) is True

    def test_has_image_string_with_jpeg(self):
        msgs = [{"role": "user", "content": "data:image/jpeg;base64,abc"}]
        assert ToolUseProcessor._has_image(msgs) is True


# =============================================================================
# PersonalityProcessor
# =============================================================================

class TestPersonalityProcessor:
    @pytest.mark.asyncio
    async def test_process_empty_traits(self):
        p = PersonalityProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        assert await p.process(msgs) == msgs

    @pytest.mark.asyncio
    async def test_process_injects_traits(self):
        p = PersonalityProcessor(traits={"warmth": 0.8, "humor": 0.5})
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "Personality:" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_set_traits(self):
        p = PersonalityProcessor()
        p.set_traits({"confidence": 0.9})
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "Personality:" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_process_appends_to_system(self):
        p = PersonalityProcessor(traits={"warmth": 0.8})
        msgs = [{"role": "system", "content": "Be nice."}, {"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        sys_msgs = [m for m in result if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert "Be nice." in sys_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_process_inserts_system_when_missing(self):
        p = PersonalityProcessor(traits={"warmth": 0.8})
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert result[0]["role"] == "system"

    def test_describe_trait_low(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("warmth", 0.0)
        assert desc == "neutral"

    def test_describe_trait_mid(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("warmth", 0.5)
        assert desc == "friendly"

    def test_describe_trait_high(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("warmth", 0.9)
        assert "warm" in desc

    def test_describe_trait_unknown(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("nonexistent", 0.5)
        assert desc == ""

    def test_all_trait_adjectives_keys(self):
        assert len(PersonalityProcessor.TRAIT_ADJECTIVES) == 10

    @pytest.mark.asyncio
    async def test_traits_below_threshold(self):
        p = PersonalityProcessor(traits={"warmth": 0.1})
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "neutral" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_multiple_traits_combined(self):
        p = PersonalityProcessor(traits={"warmth": 0.8, "humor": 0.1})
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        content = result[0]["content"]
        assert "warm" in content
        assert "serious" in content

    def test_describe_creativity(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("creativity", 0.9)
        assert "creative" in desc

    def test_describe_empathy(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("empathy", 0.9)
        assert "empathetic" in desc

    def test_describe_patience(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("patience", 0.0)
        assert "brisk" in desc

    def test_describe_confidence(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("confidence", 0.9)
        assert "confident" in desc

    def test_describe_curiosity(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("curiosity", 0.9)
        assert "curious" in desc

    def test_describe_directness(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("directness", 0.9)
        assert "direct" in desc

    def test_describe_optimism(self):
        p = PersonalityProcessor()
        desc = p._describe_trait("optimism", 0.9)
        assert "optimistic" in desc


# =============================================================================
# StyleProcessor
# =============================================================================

class TestStyleProcessor:
    def test_defaults(self):
        p = StyleProcessor()
        assert p._formality == 0.5
        assert p._directness == 0.5
        assert p._verbosity == 0.5

    def test_custom_values(self):
        p = StyleProcessor(formality=0.8, directness=0.2, verbosity=0.9)
        assert p._formality == 0.8
        assert p._directness == 0.2
        assert p._verbosity == 0.9

    @pytest.mark.asyncio
    async def test_process_neutral_no_injection(self):
        p = StyleProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_formality_high(self):
        p = StyleProcessor(formality=0.8)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "formal" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_formality_low(self):
        p = StyleProcessor(formality=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "casual" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_directness_high(self):
        p = StyleProcessor(directness=0.8)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "direct" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_directness_low(self):
        p = StyleProcessor(directness=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "thorough" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_verbosity_high(self):
        p = StyleProcessor(verbosity=0.8)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "detailed" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_verbosity_low(self):
        p = StyleProcessor(verbosity=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "brief" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_appends_to_system(self):
        p = StyleProcessor(formality=0.8)
        msgs = [{"role": "system", "content": "original"}, {"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        sys_msgs = [m for m in result if m.get("role") == "system"]
        assert len(sys_msgs) == 1
        assert "original" in sys_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_inserts_system_when_missing(self):
        p = StyleProcessor(formality=0.8)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert result[0]["role"] == "system"

    def test_set_style(self):
        p = StyleProcessor()
        p.set_style(formality=0.9, directness=0.1, verbosity=0.8)
        assert p._formality == 0.9
        assert p._directness == 0.1
        assert p._verbosity == 0.8

    @pytest.mark.asyncio
    async def test_all_instructions(self):
        p = StyleProcessor(formality=0.8, directness=0.8, verbosity=0.8)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "formal" in result[0]["content"].lower()
        assert "direct" in result[0]["content"].lower()
        assert "detailed" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_mixed_style(self):
        p = StyleProcessor(formality=0.1, directness=0.8, verbosity=0.1)
        msgs = [{"role": "user", "content": "hi"}]
        result = await p.process(msgs)
        assert "casual" in result[0]["content"].lower()
        assert "brief" in result[0]["content"].lower()

    def test_set_style_defaults(self):
        p = StyleProcessor(formality=0.9, directness=0.9, verbosity=0.9)
        p.set_style()
        assert p._formality == 0.5
        assert p._directness == 0.5
        assert p._verbosity == 0.5


# =============================================================================
# ProviderRegistry
# =============================================================================

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

    def test_overwrite(self):
        register_provider("ow", "old")
        register_provider("ow", "new")
        assert get_provider("ow") == "new"

    def test_clear_does_not_crash(self):
        clear_providers()
        clear_providers()

    def test_register_various_types(self):
        register_provider("p_str", "string")
        register_provider("p_int", 42)
        register_provider("p_none", None)
        assert get_provider("p_str") == "string"
        assert get_provider("p_int") == 42
        assert get_provider("p_none") is None

    def test_list_after_clear(self):
        register_provider("a", 1)
        clear_providers()
        assert "a" not in list_providers()


# =============================================================================
# _softmax (inline implementation from provider module)
# =============================================================================

def _softmax_test(x, axis=-1):
    import numpy as np
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-10)


class TestSoftmax:
    def test_basic(self):
        import numpy as np
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax_test(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_uniform(self):
        import numpy as np
        x = np.array([1.0, 1.0, 1.0])
        result = _softmax_test(x)
        assert all(abs(v - 1.0 / 3.0) < 1e-6 for v in result)

    def test_large_values(self):
        import numpy as np
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax_test(x)
        assert abs(result.sum() - 1.0) < 1e-4

    def test_2d(self):
        import numpy as np
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax_test(x, axis=-1)
        assert result.shape == (2, 2)
        assert all(abs(row.sum() - 1.0) < 1e-6 for row in result)


# =============================================================================
# attach_process_guard_to_provider
# =============================================================================

class TestAttachProcessGuardToProvider:
    def test_returns_false_when_no_provider(self):
        clear_providers()
        result = attach_process_guard_to_provider(MagicMock())
        assert result is False

    def test_returns_false_when_provider_has_no_server(self):
        clear_providers()
        provider = MagicMock(spec=[])  # no get_server
        register_provider("slonet-native", provider)
        result = attach_process_guard_to_provider(MagicMock())
        assert result is False

    def test_returns_false_when_server_returns_none(self):
        clear_providers()
        provider = MagicMock()
        provider.get_server.return_value = None
        register_provider("slonet-native", provider)
        result = attach_process_guard_to_provider(MagicMock())
        assert result is False

    def test_returns_false_when_server_has_no_set_process_guard(self):
        clear_providers()
        server = MagicMock(spec=[])  # no set_process_guard
        provider = MagicMock()
        provider.get_server.return_value = server
        register_provider("slonet-native", provider)
        result = attach_process_guard_to_provider(MagicMock())
        assert result is False

    def test_attaches_guard_successfully(self):
        clear_providers()
        server = MagicMock()
        provider = MagicMock()
        provider.get_server.return_value = server
        register_provider("slonet-native", provider)
        guard = MagicMock()
        result = attach_process_guard_to_provider(guard)
        assert result is True
        server.set_process_guard.assert_called_once_with(guard)

    def test_none_guard_detach(self):
        clear_providers()
        server = MagicMock()
        provider = MagicMock()
        provider.get_server.return_value = server
        register_provider("slonet-native", provider)
        result = attach_process_guard_to_provider(None)
        assert result is True
        server.set_process_guard.assert_called_once_with(None)


# =============================================================================
# update_personality_traits
# =============================================================================

class TestUpdatePersonalityTraits:
    def test_noop_when_no_default_router(self):
        clear_providers()
        update_personality_traits({"warmth": 0.9})  # should not raise

    def test_updates_personality_processor(self):
        clear_providers()
        router = ProviderRouter()
        pp = PersonalityProcessor()
        router.add_processor(pp)
        register_provider("default", router)

        update_personality_traits({"warmth": 0.9, "humor": 0.3})
        assert pp._traits == {"warmth": 0.9, "humor": 0.3}
        clear_providers()

    def test_updates_style_processor(self):
        clear_providers()
        router = ProviderRouter()
        sp = StyleProcessor()
        router.add_processor(sp)
        register_provider("default", router)

        update_personality_traits({"formality": 0.8, "directness": 0.2})
        assert sp._formality == 0.8
        assert sp._directness == 0.2
        clear_providers()

    def test_noop_when_default_is_not_router(self):
        clear_providers()
        register_provider("default", "not_a_router")
        update_personality_traits({"warmth": 0.9})
        assert get_provider("default") == "not_a_router"
        clear_providers()

    def test_noop_when_no_personality_processor(self):
        clear_providers()
        router = ProviderRouter()
        router.add_processor(StyleProcessor())
        register_provider("default", router)
        update_personality_traits({"warmth": 0.9})
        clear_providers()

    def test_default_style_values_when_traits_missing(self):
        clear_providers()
        router = ProviderRouter()
        sp = StyleProcessor()
        router.add_processor(sp)
        register_provider("default", router)

        update_personality_traits({})
        assert sp._formality == 0.5
        assert sp._directness == 0.5
        clear_providers()


# =============================================================================
# _server_from_provider
# =============================================================================

class TestServerFromProvider:
    def test_returns_none_when_no_to_server(self):
        provider = MagicMock(spec=[])  # no to_server
        result = _server_from_provider(provider, MagicMock())
        assert result is None

    def test_returns_none_on_exception(self):
        provider = MagicMock()
        provider.to_server.side_effect = RuntimeError("fail")
        result = _server_from_provider(provider, MagicMock())
        assert result is None

    def test_calls_to_server(self):
        provider = MagicMock()
        expected = MagicMock()
        provider.to_server.return_value = expected
        guard = MagicMock()
        result = _server_from_provider(provider, guard)
        assert result is expected
        provider.to_server.assert_called_once_with(process_guard=guard)
