"""Tests for domains/models/provider.py."""

import pytest
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from domains.models.provider import (
    ModelCapabilities,
    register_provider,
    get_provider,
    list_providers,
    register_processor,
    get_processor,
    list_processors,
    apply_processors,
    VisionProcessor,
    KnowledgeProcessor,
    ToolDef,
    ToolUseProcessor,
    PersonalityProcessor,
    StyleProcessor,
    ProviderRouter,
    update_personality_traits,
    MessageProcessor,
    ChatMessage,
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
        assert c.functions is False

    def test_chat_only(self):
        c = ModelCapabilities(chat=True)
        assert c.chat is True
        assert c.streaming is False

    def test_all_true(self):
        c = ModelCapabilities(chat=True, streaming=True, embedding=True, vision=True, functions=True)
        assert all([c.chat, c.streaming, c.embedding, c.vision, c.functions])


# =============================================================================
# Provider Registry
# =============================================================================

class TestProviderRegistry:
    def setup_method(self):
        _clear_providers()

    def test_register_and_get(self):
        register_provider("test", "mock_provider")
        assert get_provider("test") == "mock_provider"

    def test_get_nonexistent(self):
        assert get_provider("nonexistent") is None

    def test_list_providers(self):
        _clear_providers()
        assert list_providers() == []
        register_provider("a", 1)
        register_provider("b", 2)
        assert set(list_providers()) == {"a", "b"}

    def test_register_overwrite(self):
        register_provider("x", "first")
        register_provider("x", "second")
        assert get_provider("x") == "second"


# =============================================================================
# Processor Registry
# =============================================================================

class TestProcessorRegistry:
    def setup_method(self):
        _clear_processors()

    def test_register_and_get(self):
        register_processor("vision", "vp")
        assert get_processor("vision") == "vp"

    def test_get_nonexistent(self):
        assert get_processor("nonexistent") is None

    def test_list_processors(self):
        _clear_processors()
        assert list_processors() == []
        register_processor("a", 1)
        register_processor("b", 2)
        assert set(list_processors()) == {"a", "b"}


# =============================================================================
# apply_processors
# =============================================================================

class TestApplyProcessors:
    async def test_empty_processors(self):
        result = await apply_processors([{"role": "user", "content": "hi"}], [])
        assert result == [{"role": "user", "content": "hi"}]

    async def test_single_processor(self):
        class UpperProcessor:
            async def process(self, msgs):
                return [{"role": m["role"], "content": m["content"].upper()} for m in msgs]
        result = await apply_processors([{"role": "user", "content": "hello"}], [UpperProcessor()])
        assert result[0]["content"] == "HELLO"

    async def test_multiple_processors_chain(self):
        class AddExclamation:
            async def process(self, msgs):
                return [{"role": m["role"], "content": m["content"] + "!"} for m in msgs]
        class AddQuestion:
            async def process(self, msgs):
                return [{"role": m["role"], "content": m["content"] + "?"} for m in msgs]
        result = await apply_processors([{"role": "user", "content": "hi"}], [AddExclamation(), AddQuestion()])
        assert result[0]["content"] == "hi!?"

    async def test_processor_failure_logged(self):
        class FailingProcessor:
            async def process(self, msgs):
                raise ValueError("boom")
        result = await apply_processors([{"role": "user", "content": "hi"}], [FailingProcessor()])
        assert result == [{"role": "user", "content": "hi"}]


# =============================================================================
# VisionProcessor
# =============================================================================

class TestVisionProcessor:
    def test_init(self):
        vp = VisionProcessor("multimodal")
        assert vp._provider_name == "multimodal"

    def test_extract_images_empty(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": "no images"}]
        assert vp._extract_images(msgs) == []

    def test_extract_images_string_content(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": "data:image/png;base64,abc123"}]
        images = vp._extract_images(msgs)
        assert len(images) == 1

    def test_extract_images_list_content(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:img"}}]}]
        images = vp._extract_images(msgs)
        assert len(images) == 1

    def test_extract_images_no_match(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        assert vp._extract_images(msgs) == []

    def test_extract_images_mixed_content(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "img1"}}]}]
        images = vp._extract_images(msgs)
        assert len(images) == 1

    async def test_process_no_images(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": "plain text"}]
        result = await vp.process(msgs)
        assert result == msgs


# =============================================================================
# KnowledgeProcessor
# =============================================================================

class TestKnowledgeProcessor:
    async def test_init_no_knowledge(self):
        kp = KnowledgeProcessor()
        assert kp._knowledge == []

    async def test_init_with_knowledge(self):
        kp = KnowledgeProcessor(knowledge=["fact1"])
        assert kp._knowledge == ["fact1"]

    async def test_process_no_knowledge_returns_original(self):
        kp = KnowledgeProcessor()
        msgs = [{"role": "user", "content": "hello"}]
        result = await kp.process(msgs)
        assert result == msgs

    async def test_process_with_knowledge_prepends(self):
        kp = KnowledgeProcessor(knowledge=["Earth is round"])
        msgs = [{"role": "user", "content": "hello"}]
        result = await kp.process(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Earth is round" in result[0]["content"]

    async def test_set_knowledge(self):
        kp = KnowledgeProcessor()
        kp.set_knowledge(["new fact"])
        assert kp._knowledge == ["new fact"]

    async def test_process_multiple_facts(self):
        kp = KnowledgeProcessor(knowledge=["a", "b", "c"])
        result = await kp.process([])
        assert len(result) == 1
        for fact in ["a", "b", "c"]:
            assert fact in result[0]["content"]


# =============================================================================
# ToolDef
# =============================================================================

class TestToolDef:
    def test_minimal(self):
        t = ToolDef(name="describe_image", provider_name="multimodal")
        assert t.name == "describe_image"
        assert t.provider_name == "multimodal"
        assert t.description == ""

    def test_full(self):
        t = ToolDef(name="t", provider_name="p", description="desc")
        assert t.description == "desc"


# =============================================================================
# ToolUseProcessor
# =============================================================================

class TestToolUseProcessor:
    def test_init_default_tools(self):
        tp = ToolUseProcessor()
        assert len(tp._tools) == 1
        assert tp._tools[0].name == "describe_image"

    def test_init_custom_tools(self):
        tp = ToolUseProcessor(tools=[ToolDef(name="my_tool", provider_name="p")])
        assert len(tp._tools) == 1
        assert tp._tools[0].name == "my_tool"

    async def test_process_adds_tool_prompt_no_system(self):
        tp = ToolUseProcessor()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        result = await tp.process(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "describe_image" in result[0]["content"]

    async def test_process_appends_to_existing_system(self):
        tp = ToolUseProcessor()
        msgs = [{"role": "system", "content": "Be helpful."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        result = await tp.process(msgs)
        assert result[0]["role"] == "system"
        assert "Be helpful." in result[0]["content"]
        assert "describe_image" in result[0]["content"]

    async def test_process_unmodified(self):
        tp = ToolUseProcessor(tools=[])
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        result = await tp.process(msgs)
        assert len(result) == 2  # still adds empty prompt

    async def test_process_text_only_skips_tool_prompt(self):
        tp = ToolUseProcessor()
        msgs = [{"role": "user", "content": "hello"}]
        result = await tp.process(msgs)
        assert len(result) == 1
        assert "describe_image" not in result[0]["content"]

    def test_match_tool_found(self):
        tp = ToolUseProcessor()
        match = tp.match_tool("some text [[TOOL: describe_image]] data:base64,abc")
        assert match is not None
        name, arg, full = match
        assert name == "describe_image"
        assert "base64" in arg

    def test_match_tool_rejects_placeholder_arg(self):
        tp = ToolUseProcessor()
        assert tp.match_tool("[[TOOL: describe_image]] <base64_image_data>") is None
        assert tp.match_tool("[[TOOL: describe_image]] <base64_encode_data>") is None

    def test_match_tool_not_found(self):
        tp = ToolUseProcessor()
        match = tp.match_tool("no tool call here")
        assert match is None

    def test_match_tool_unknown_name(self):
        tp = ToolUseProcessor()
        match = tp.match_tool("[[TOOL: unknown_tool]] arg")
        assert match is None

    def test_tool_regex(self):
        tp = ToolUseProcessor()
        assert tp.TOOL_RE.search("[[TOOL: describe_image]] arg") is not None
        assert tp.TOOL_RE.search("no brackets") is None
        assert tp.TOOL_RE.search("[[TOOL:]]") is None


# =============================================================================
# PersonalityProcessor
# =============================================================================

class TestPersonalityProcessor:
    def test_init_empty(self):
        pp = PersonalityProcessor()
        assert pp._traits == {}

    def test_init_with_traits(self):
        pp = PersonalityProcessor(traits={"warmth": 0.8})
        assert pp._traits["warmth"] == 0.8

    def test_set_traits(self):
        pp = PersonalityProcessor()
        pp.set_traits({"creativity": 0.9})
        assert pp._traits["creativity"] == 0.9

    def test_describe_trait_known(self):
        pp = PersonalityProcessor()
        desc = pp._describe_trait("warmth", 0.8)
        assert desc == "warm"

    def test_describe_trait_low(self):
        pp = PersonalityProcessor()
        desc = pp._describe_trait("warmth", 0.1)
        assert desc == "neutral"

    def test_describe_trait_exact_match(self):
        pp = PersonalityProcessor()
        desc = pp._describe_trait("warmth", 0.5)
        assert desc == "friendly"

    def test_describe_trait_unknown(self):
        pp = PersonalityProcessor()
        desc = pp._describe_trait("unknown_trait", 0.5)
        assert desc == ""

    async def test_process_no_traits(self):
        pp = PersonalityProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = await pp.process(msgs)
        assert result == msgs

    async def test_process_injects_personality(self):
        pp = PersonalityProcessor(traits={"warmth": 0.8})
        msgs = [{"role": "user", "content": "hi"}]
        result = await pp.process(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "warm" in result[0]["content"]

    async def test_process_appends_to_existing_system(self):
        pp = PersonalityProcessor(traits={"directness": 0.9})
        msgs = [{"role": "system", "content": "Base instruction."}, {"role": "user", "content": "hi"}]
        result = await pp.process(msgs)
        assert "Base instruction" in result[0]["content"]
        assert "direct" in result[0]["content"]

    async def test_process_all_10_traits(self):
        traits_all_high = {
            "warmth": 0.9, "creativity": 0.9, "empathy": 0.9, "formality": 0.9,
            "humor": 0.9, "patience": 0.9, "confidence": 0.9, "curiosity": 0.9,
            "directness": 0.9, "optimism": 0.9,
        }
        pp = PersonalityProcessor(traits=traits_all_high)
        result = await pp.process([{"role": "user", "content": "hi"}])
        assert len(result) == 2
        for trait_name in traits_all_high:
            adj = pp._describe_trait(trait_name, 0.9)
            assert adj in result[0]["content"]


class TestPersonalityProcessorTraitAdjectives:
    """Verify specific trait→adjective mappings."""

    @pytest.mark.parametrize("trait,value,expected", [
        ("warmth", 0.0, "neutral"),
        ("warmth", 0.2, "neutral"),
        ("warmth", 0.3, "reserved"),
        ("warmth", 0.4, "reserved"),
        ("warmth", 0.5, "friendly"),
        ("warmth", 0.6, "friendly"),
        ("warmth", 0.7, "warm"),
        ("warmth", 0.8, "warm"),
        ("warmth", 0.9, "very warm and empathetic"),
        ("warmth", 1.0, "very warm and empathetic"),
        ("creativity", 0.0, "factual"),
        ("creativity", 0.5, "balanced"),
        ("creativity", 0.9, "highly creative and imaginative"),
        ("formality", 0.0, "casual"),
        ("formality", 0.5, "professional"),
        ("formality", 0.9, "highly formal and precise"),
        ("humor", 0.0, "serious"),
        ("humor", 0.5, "witty"),
        ("humor", 0.9, "very humorous and playful"),
        ("confidence", 0.0, "cautious"),
        ("confidence", 0.5, "confident"),
        ("confidence", 0.9, "very confident and decisive"),
        ("curiosity", 0.0, "direct"),
        ("curiosity", 0.5, "curious"),
        ("curiosity", 0.9, "deeply curious and exploratory"),
        ("optimism", 0.0, "realistic"),
        ("optimism", 0.5, "optimistic"),
        ("optimism", 0.9, "very optimistic and encouraging"),
    ])
    def test_trait_mapping(self, trait, value, expected):
        pp = PersonalityProcessor()
        assert pp._describe_trait(trait, value) == expected


# =============================================================================
# StyleProcessor
# =============================================================================

class TestStyleProcessor:
    def test_init_defaults(self):
        sp = StyleProcessor()
        assert sp._formality == 0.5
        assert sp._directness == 0.5
        assert sp._verbosity == 0.5

    def test_set_style(self):
        sp = StyleProcessor()
        sp.set_style(formality=0.9, directness=0.1, verbosity=0.9)
        assert sp._formality == 0.9
        assert sp._directness == 0.1
        assert sp._verbosity == 0.9

    async def test_process_mid_values_no_change(self):
        sp = StyleProcessor(0.5, 0.5, 0.5)
        msgs = [{"role": "user", "content": "hi"}]
        result = await sp.process(msgs)
        assert result == msgs

    async def test_process_high_formality(self):
        sp = StyleProcessor(formality=0.8)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "formal language" in result[0]["content"]

    async def test_process_low_formality(self):
        sp = StyleProcessor(formality=0.2)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "casual" in result[0]["content"]

    async def test_process_high_directness(self):
        sp = StyleProcessor(directness=0.8)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "direct and concise" in result[0]["content"]

    async def test_process_low_directness(self):
        sp = StyleProcessor(directness=0.2)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "thorough and provide context" in result[0]["content"]

    async def test_process_high_verbosity(self):
        sp = StyleProcessor(verbosity=0.8)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "detailed" in result[0]["content"]

    async def test_process_low_verbosity(self):
        sp = StyleProcessor(verbosity=0.2)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "brief" in result[0]["content"]

    async def test_process_all_high(self):
        sp = StyleProcessor(0.8, 0.8, 0.8)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "formal language" in result[0]["content"]
        assert "direct and concise" in result[0]["content"]
        assert "detailed" in result[0]["content"]

    async def test_process_appends_to_existing_system(self):
        sp = StyleProcessor(formality=0.8)
        msgs = [{"role": "system", "content": "Base."}, {"role": "user", "content": "hi"}]
        result = await sp.process(msgs)
        assert "Base" in result[0]["content"]
        assert "formal" in result[0]["content"]


# =============================================================================
# ProviderRouter
# =============================================================================

class TestProviderRouter:
    def setup_method(self):
        _clear_providers()

    def test_init(self):
        router = ProviderRouter()
        assert router._processors == []
        assert router._text_name is None
        assert router.model_id == "router-v1"

    def test_add_processor_returns_self(self):
        router = ProviderRouter()
        result = router.add_processor(KnowledgeProcessor())
        assert result is router

    def test_set_text_provider(self):
        router = ProviderRouter()
        router.set_text_provider("test")
        assert router._text_name == "test"

    def test_capabilities(self):
        router = ProviderRouter()
        caps = router.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.vision is True

    def test_metadata(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.set_text_provider("p")
        meta = router.metadata
        assert "processors" in meta
        assert meta["text_provider"] == "p"

    def test_embed_returns_empty(self):
        router = ProviderRouter()
        assert router.embed("text") == []

    def test_find_tool_processor_none(self):
        router = ProviderRouter()
        assert router._find_tool_processor() is None

    def test_find_tool_processor_found(self):
        router = ProviderRouter()
        tp = ToolUseProcessor()
        router.add_processor(tp)
        assert router._find_tool_processor() is tp

    async def test_chat_stream_no_text_provider(self):
        router = ProviderRouter()
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "No text model configured" in text

    async def test_chat_stream_text_provider_not_found(self):
        register_provider("nonexistent", None)
        router = ProviderRouter()
        router.set_text_provider("nonexistent")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "not available" in text

    async def test_chat(self):
        router = ProviderRouter()
        result = await router.chat([{"role": "user", "content": "hi"}])
        assert "No text model configured" in result


class TestProviderRouterWithMockProvider:
    async def test_chat_stream_with_provider(self):
        _clear_providers()
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "hello from mock" in text

    async def test_chat_with_provider(self):
        _clear_providers()
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        text = await router.chat([{"role": "user", "content": "hi"}])
        assert "hello from mock" in text

    async def test_processors_applied(self):
        _clear_providers()
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.add_processor(UppercaseProcessor())
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hello"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "hello from mock" in text

    async def test_cancel_event_stops_stream(self):
        import threading
        _clear_providers()
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        cancel = threading.Event()
        cancel.set()
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}], cancel_event=cancel):
            tokens.append(token)
        assert len(tokens) == 0

    async def test_tool_loop_detects_and_continues(self):
        _clear_providers()
        register_provider("mock_text", MockToolTextProvider())
        router = ProviderRouter()
        router.add_processor(ToolUseProcessor())
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "describe this"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert len(tokens) > 0


# =============================================================================
# update_personality_traits
# =============================================================================

class TestUpdatePersonalityTraits:
    def setup_method(self):
        _clear_providers()

    def test_no_router(self):
        update_personality_traits({"warmth": 0.9})

    def test_updates_personality_processor(self):
        pp = PersonalityProcessor()
        router = ProviderRouter()
        router.add_processor(pp)
        register_provider("default", router)
        update_personality_traits({"warmth": 0.9, "creativity": 0.5})
        assert pp._traits["warmth"] == 0.9
        assert pp._traits["creativity"] == 0.5

    def test_updates_style_processor(self):
        sp = StyleProcessor()
        router = ProviderRouter()
        router.add_processor(sp)
        register_provider("default", router)
        update_personality_traits({"formality": 0.9, "directness": 0.1})
        assert sp._formality == 0.9
        assert sp._directness == 0.1

    def test_non_router_default_does_nothing(self):
        register_provider("default", "string")
        update_personality_traits({"warmth": 0.9})


# =============================================================================
# Helpers
# =============================================================================

def _clear_providers():
    import domains.models.provider as p
    p._providers.clear()

def _clear_processors():
    import domains.models.provider as p
    p._processors.clear()


class MockTextProvider:
    @property
    def model_id(self):
        return "mock"
    async def chat_stream(self, messages, **kwargs):
        yield "hello from mock"
    async def chat(self, messages, **kwargs):
        return "hello from mock"


class MockToolTextProvider:
    def __init__(self):
        self._call_count = 0
    @property
    def model_id(self):
        return "mock_tool"
    async def chat_stream(self, messages, **kwargs):
        self._call_count += 1
        if self._call_count <= 1:
            yield "Let me [[TOOL: describe_image]] data:image/png;base64,abc"
        else:
            yield "The image is a cute cat sitting on a chair."
    async def chat(self, messages, **kwargs):
        return "mock"


class UppercaseProcessor:
    async def process(self, messages):
        return [{"role": m["role"], "content": m["content"].upper()} for m in messages]
