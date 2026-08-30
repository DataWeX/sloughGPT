"""Tests for domains/models/provider.py."""

import pytest
import threading
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from domains.models.provider import (
    ModelCapabilities,
    ModelProvider,
    register_provider,
    get_provider,
    list_providers,
    clear_providers,
    register_processor,
    get_processor,
    list_processors,
    apply_processors,
    attach_process_guard_to_provider,
    VisionProcessor,
    KnowledgeProcessor,
    ToolDef,
    ToolUseProcessor,
    PersonalityProcessor,
    StyleProcessor,
    ProviderRouter,
    SloTransformerProvider,
    update_personality_traits,
    MessageProcessor,
    ChatMessage,
)


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

    @property
    def capabilities(self):
        return ModelCapabilities(chat=True, streaming=True)

    @property
    def metadata(self):
        return {"type": "mock"}

    async def chat_stream(self, messages, **kwargs):
        yield "hello from mock"

    async def chat(self, messages, **kwargs):
        return "hello from mock"

    def embed(self, text):
        return [0.1, 0.2]


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


class FailingProcessor:
    async def process(self, messages):
        raise ValueError("boom")


class PassthroughProcessor:
    async def process(self, messages):
        return messages


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

    def test_equality(self):
        a = ModelCapabilities(chat=True, streaming=True)
        b = ModelCapabilities(chat=True, streaming=True)
        assert a == b

    def test_inequality(self):
        a = ModelCapabilities(chat=True)
        b = ModelCapabilities(chat=False)
        assert a != b

    def test_repr(self):
        c = ModelCapabilities(chat=True)
        assert "chat=True" in repr(c)


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

    def test_clear_providers(self):
        register_provider("a", 1)
        register_provider("b", 2)
        clear_providers()
        assert list_providers() == []
        assert get_provider("a") is None

    def test_clear_providers_empty(self):
        clear_providers()
        assert list_providers() == []

    def test_register_none_provider(self):
        register_provider("null", None)
        assert get_provider("null") is None

    def test_register_complex_object(self):
        class FakeProvider:
            def __init__(self, v):
                self.v = v
        p = FakeProvider(42)
        register_provider("complex", p)
        assert get_provider("complex").v == 42

    def test_list_providers_order(self):
        _clear_providers()
        for name in ["z", "a", "m"]:
            register_provider(name, name)
        result = list_providers()
        assert len(result) == 3

    def test_providers_isolated_dicts(self):
        register_provider("a", 1)
        clear_providers()
        assert get_provider("a") is None


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

    def test_register_overwrite(self):
        register_processor("x", "first")
        register_processor("x", "second")
        assert get_processor("x") == "second"

    def test_clear_processors(self):
        register_processor("a", 1)
        _clear_processors()
        assert list_processors() == []

    def test_register_none_processor(self):
        register_processor("null", None)
        assert get_processor("null") is None


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
        result = await apply_processors([{"role": "user", "content": "hi"}], [FailingProcessor()])
        assert result == [{"role": "user", "content": "hi"}]

    async def test_processor_failure_continues_chain(self):
        class AppendAfter:
            async def process(self, msgs):
                return [{"role": m["role"], "content": m["content"] + "+after"} for m in msgs]
        result = await apply_processors(
            [{"role": "user", "content": "hi"}],
            [FailingProcessor(), AppendAfter()],
        )
        assert result[0]["content"] == "hi+after"

    async def test_empty_messages(self):
        class Echo:
            async def process(self, msgs):
                return msgs
        result = await apply_processors([], [Echo()])
        assert result == []

    async def test_passthrough_preserves_messages(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        result = await apply_processors(msgs, [PassthroughProcessor()])
        assert result == msgs


# =============================================================================
# attach_process_guard_to_provider
# =============================================================================

class TestAttachProcessGuardToProvider:
    def setup_method(self):
        _clear_providers()

    def test_no_provider_registered(self):
        assert attach_process_guard_to_provider(None) is False

    def test_provider_without_get_server(self):
        class FakeProvider:
            pass
        register_provider("slonet-native", FakeProvider())
        assert attach_process_guard_to_provider(None) is False

    def test_provider_with_get_server_returning_none(self):
        class FakeProvider:
            def get_server(self):
                return None
        register_provider("slonet-native", FakeProvider())
        assert attach_process_guard_to_provider(None) is False

    def test_server_without_set_process_guard(self):
        class FakeServer:
            pass
        class FakeProvider:
            def get_server(self):
                return FakeServer()
        register_provider("slonet-native", FakeProvider())
        assert attach_process_guard_to_provider(None) is False

    def test_successful_attach(self):
        attached_guard = [None]
        class FakeServer:
            def set_process_guard(self, guard):
                attached_guard[0] = guard
        class FakeProvider:
            def get_server(self):
                return FakeServer()
        register_provider("slonet-native", FakeProvider())
        mock_guard = object()
        result = attach_process_guard_to_provider(mock_guard)
        assert result is True
        assert attached_guard[0] is mock_guard

    def test_wrong_provider_name(self):
        register_provider("other", "something")
        assert attach_process_guard_to_provider(None) is False


# =============================================================================
# VisionProcessor
# =============================================================================

class TestVisionProcessor:
    def test_init(self):
        vp = VisionProcessor("multimodal")
        assert vp._provider_name == "multimodal"

    def test_init_default(self):
        vp = VisionProcessor()
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
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "img1"}}
        ]}]
        images = vp._extract_images(msgs)
        assert len(images) == 1

    def test_extract_images_multiple_messages(self):
        vp = VisionProcessor()
        msgs = [
            {"role": "user", "content": "data:image/png;base64,first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "data:image/jpeg;base64,second"},
        ]
        images = vp._extract_images(msgs)
        assert len(images) == 2

    def test_extract_images_string_base64_pattern(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": "text data:image/gif;base64,AAA more text"}]
        images = vp._extract_images(msgs)
        assert len(images) == 1

    def test_extract_images_list_with_missing_url(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        images = vp._extract_images(msgs)
        assert len(images) == 1
        assert images[0] == ""

    def test_extract_images_list_non_dict_part(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": ["not a dict"]}]
        assert vp._extract_images(msgs) == []

    def test_extract_images_empty_string_content(self):
        vp = VisionProcessor()
        msgs = [{"role": "user", "content": ""}]
        assert vp._extract_images(msgs) == []

    def test_extract_images_no_content_key(self):
        vp = VisionProcessor()
        msgs = [{"role": "user"}]
        assert vp._extract_images(msgs) == []

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

    async def test_process_preserves_original_messages(self):
        kp = KnowledgeProcessor(knowledge=["k1"])
        msgs = [{"role": "user", "content": "original"}]
        result = await kp.process(msgs)
        assert result[1]["content"] == "original"
        assert result[1]["role"] == "user"

    async def test_process_empty_knowledge_list(self):
        kp = KnowledgeProcessor(knowledge=[])
        msgs = [{"role": "user", "content": "hi"}]
        result = await kp.process(msgs)
        assert result == msgs

    async def test_set_knowledge_replaces(self):
        kp = KnowledgeProcessor(knowledge=["old"])
        kp.set_knowledge(["new"])
        result = await kp.process([{"role": "user", "content": "hi"}])
        assert "new" in result[0]["content"]
        assert "old" not in result[0]["content"]

    async def test_process_system_message_preserved(self):
        kp = KnowledgeProcessor(knowledge=["k1"])
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
        ]
        result = await kp.process(msgs)
        assert result[0]["role"] == "system"
        assert "k1" in result[0]["content"]
        assert result[1]["role"] == "system"
        assert "Be helpful." in result[1]["content"]


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

    def test_equality(self):
        a = ToolDef(name="a", provider_name="p")
        b = ToolDef(name="a", provider_name="p")
        assert a == b

    def test_inequality(self):
        a = ToolDef(name="a", provider_name="p")
        b = ToolDef(name="b", provider_name="p")
        assert a != b


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
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
        ]
        result = await tp.process(msgs)
        assert result[0]["role"] == "system"
        assert "Be helpful." in result[0]["content"]
        assert "describe_image" in result[0]["content"]

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

    def test_has_image_with_image_url(self):
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        assert ToolUseProcessor._has_image(msgs) is True

    def test_has_image_with_base64_string(self):
        msgs = [{"role": "user", "content": "data:image/png;base64,abc"}]
        assert ToolUseProcessor._has_image(msgs) is True

    def test_has_image_no_image(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert ToolUseProcessor._has_image(msgs) is False

    def test_has_image_list_content_non_image_type(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert ToolUseProcessor._has_image(msgs) is False

    def test_has_image_empty_messages(self):
        assert ToolUseProcessor._has_image([]) is False

    def test_has_image_string_content_without_data_image(self):
        msgs = [{"role": "user", "content": "no image here"}]
        assert ToolUseProcessor._has_image(msgs) is False

    def test_has_image_multiple_messages_one_with_image(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
        ]
        assert ToolUseProcessor._has_image(msgs) is True

    def test_match_tool_with_spaces_in_arg(self):
        tp = ToolUseProcessor()
        match = tp.match_tool("[[TOOL: describe_image]] some_file.png")
        assert match is not None
        assert match[1] == "some_file.png"

    def test_placeholder_regex(self):
        tp = ToolUseProcessor()
        assert tp._PLACEHOLDER_ARG_RE.match("<base64_image_data>") is not None
        assert tp._PLACEHOLDER_ARG_RE.match("<some_placeholder>") is not None
        assert tp._PLACEHOLDER_ARG_RE.match("real_arg") is None
        assert tp._PLACEHOLDER_ARG_RE.match("123") is None

    async def test_process_no_tools_empty_list(self):
        tp = ToolUseProcessor(tools=[])
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
        result = await tp.process(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"


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

    def test_describe_trait_threshold_boundaries(self):
        pp = PersonalityProcessor()
        assert pp._describe_trait("warmth", 0.0) == "neutral"
        assert pp._describe_trait("warmth", 0.29) == "neutral"
        assert pp._describe_trait("warmth", 0.3) == "reserved"
        assert pp._describe_trait("warmth", 0.49) == "reserved"
        assert pp._describe_trait("warmth", 0.5) == "friendly"
        assert pp._describe_trait("warmth", 0.69) == "friendly"
        assert pp._describe_trait("warmth", 0.7) == "warm"
        assert pp._describe_trait("warmth", 0.89) == "warm"
        assert pp._describe_trait("warmth", 0.9) == "very warm and empathetic"
        assert pp._describe_trait("warmth", 1.0) == "very warm and empathetic"

    def test_describe_all_traits(self):
        pp = PersonalityProcessor()
        for trait in PersonalityProcessor.TRAIT_ADJECTIVES:
            for val in [0.0, 0.5, 0.9]:
                desc = pp._describe_trait(trait, val)
                assert desc != "", f"Empty description for {trait}={val}"

    async def test_set_traits_replaces_old(self):
        pp = PersonalityProcessor(traits={"warmth": 0.1})
        pp.set_traits({"warmth": 0.9})
        result = await pp.process([{"role": "user", "content": "hi"}])
        assert "very warm and empathetic" in result[0]["content"]
        assert "neutral" not in result[0]["content"]

    async def test_process_preserves_original_messages(self):
        pp = PersonalityProcessor(traits={"warmth": 0.5})
        msgs = [{"role": "user", "content": "test"}]
        original_content = msgs[0]["content"]
        result = await pp.process(msgs)
        assert result[1]["content"] == original_content


class TestPersonalityProcessorTraitAdjectives:
    """Verify specific trait->adjective mappings."""

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
        ("empathy", 0.0, "detached"),
        ("empathy", 0.5, "understanding"),
        ("empathy", 0.9, "deeply empathetic and compassionate"),
        ("patience", 0.0, "brisk"),
        ("patience", 0.5, "patient"),
        ("patience", 0.9, "extremely patient and methodical"),
        ("directness", 0.0, "indirect"),
        ("directness", 0.5, "balanced"),
        ("directness", 0.9, "very direct and to the point"),
    ])
    def test_trait_mapping(self, trait, value, expected):
        pp = PersonalityProcessor()
        assert pp._describe_trait(trait, value) == expected

    def test_all_traits_have_adjective_tables(self):
        expected_traits = {
            "warmth", "creativity", "empathy", "formality", "humor",
            "patience", "confidence", "curiosity", "directness", "optimism",
        }
        assert set(PersonalityProcessor.TRAIT_ADJECTIVES.keys()) == expected_traits


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

    async def test_process_preserves_original_messages(self):
        sp = StyleProcessor(formality=0.8)
        msgs = [{"role": "user", "content": "original"}]
        result = await sp.process(msgs)
        assert result[1]["content"] == "original"

    def test_set_style_replaces(self):
        sp = StyleProcessor(formality=0.2, directness=0.2, verbosity=0.2)
        sp.set_style(formality=0.9, directness=0.9, verbosity=0.9)
        assert sp._formality == 0.9
        assert sp._directness == 0.9
        assert sp._verbosity == 0.9

    def test_init_custom_values(self):
        sp = StyleProcessor(formality=0.1, directness=0.9, verbosity=0.3)
        assert sp._formality == 0.1
        assert sp._directness == 0.9
        assert sp._verbosity == 0.3

    async def test_process_boundary_0_3(self):
        sp = StyleProcessor(formality=0.3)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert result[0]["content"] == "hi"

    async def test_process_boundary_0_7(self):
        sp = StyleProcessor(formality=0.7)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert result[0]["content"] == "hi"

    async def test_process_all_low(self):
        sp = StyleProcessor(0.2, 0.2, 0.2)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert "casual" in result[0]["content"]
        assert "thorough" in result[0]["content"]
        assert "brief" in result[0]["content"]


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

    def test_add_multiple_processors(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.add_processor(ToolUseProcessor())
        assert len(router._processors) == 2

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
        assert caps.embedding is False

    def test_metadata(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.set_text_provider("p")
        meta = router.metadata
        assert "processors" in meta
        assert meta["text_provider"] == "p"
        assert "max_tool_rounds" in meta
        assert meta["max_tool_rounds"] == 3

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

    def test_find_tool_processor_not_first(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
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

    async def test_chat_text_provider_not_found(self):
        register_provider("nonexistent", None)
        router = ProviderRouter()
        router.set_text_provider("nonexistent")
        result = await router.chat([{"role": "user", "content": "hi"}])
        assert "not available" in result

    def test_max_tool_rounds_default(self):
        router = ProviderRouter()
        assert router._max_tool_rounds == 3

    def test_metadata_processors_list(self):
        router = ProviderRouter()
        router.add_processor(VisionProcessor())
        router.add_processor(ToolUseProcessor())
        meta = router.metadata
        assert "VisionProcessor" in meta["processors"]
        assert "ToolUseProcessor" in meta["processors"]

    def test_model_id_is_router_v1(self):
        router = ProviderRouter()
        assert router.model_id == "router-v1"


class TestProviderRouterWithMockProvider:
    def setup_method(self):
        _clear_providers()

    async def test_chat_stream_with_provider(self):
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "hello from mock" in text

    async def test_chat_with_provider(self):
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        text = await router.chat([{"role": "user", "content": "hi"}])
        assert "hello from mock" in text

    async def test_processors_applied(self):
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
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        cancel = threading.Event()
        cancel.set()
        tokens = []
        async for token in router.chat_stream(
            [{"role": "user", "content": "hi"}], cancel_event=cancel
        ):
            tokens.append(token)
        assert len(tokens) == 0

    async def test_tool_loop_detects_and_continues(self):
        register_provider("mock_text", MockToolTextProvider())
        router = ProviderRouter()
        router.add_processor(ToolUseProcessor())
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream(
            [{"role": "user", "content": "describe this"}]
        ):
            tokens.append(token)
        text = "".join(tokens)
        assert len(tokens) > 0

    async def test_chat_stream_passes_kwargs(self):
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream(
            [{"role": "user", "content": "hi"}],
            max_tokens=100,
            temperature=0.5,
        ):
            tokens.append(token)
        assert len(tokens) > 0

    async def test_chat_stream_passes_session_id(self):
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream(
            [{"role": "user", "content": "hi"}], session_id="test-session"
        ):
            tokens.append(token)
        assert len(tokens) > 0

    async def test_multiple_processors_all_applied(self):
        class AppendA:
            async def process(self, msgs):
                return [
                    {"role": m["role"], "content": m["content"] + "A"} for m in msgs
                ]

        class AppendB:
            async def process(self, msgs):
                return [
                    {"role": m["role"], "content": m["content"] + "B"} for m in msgs
                ]

        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.add_processor(AppendA())
        router.add_processor(AppendB())
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "hello from mock" in text

    async def test_processor_failure_doesnt_break_stream(self):
        register_provider("mock_text", MockTextProvider())
        router = ProviderRouter()
        router.add_processor(FailingProcessor())
        router.set_text_provider("mock_text")
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        text = "".join(tokens)
        assert "hello from mock" in text


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

    def test_updates_both_personality_and_style(self):
        pp = PersonalityProcessor()
        sp = StyleProcessor()
        router = ProviderRouter()
        router.add_processor(pp)
        router.add_processor(sp)
        register_provider("default", router)
        update_personality_traits({"warmth": 0.9, "formality": 0.1, "directness": 0.9})
        assert pp._traits["warmth"] == 0.9
        assert sp._formality == 0.1
        assert sp._directness == 0.9

    def test_style_defaults_when_trait_missing(self):
        sp = StyleProcessor()
        router = ProviderRouter()
        router.add_processor(sp)
        register_provider("default", router)
        update_personality_traits({"warmth": 0.9})
        assert sp._formality == 0.5
        assert sp._directness == 0.5

    def test_no_default_provider(self):
        update_personality_traits({"warmth": 0.9})


# =============================================================================
# SloTransformerProvider
# =============================================================================

class TestSloTransformerProviderEncodeDecode:
    """Test encode/decode logic without needing a real model."""

    def _make_provider(self):
        stoi = {"<PAD>": 0, "<UNK>": 1, " ": 2, "a": 3, "b": 4, "c": 5, "h": 6, "i": 7, "o": 8}
        itos = {0: "<PAD>", 1: "<UNK>", 2: " ", 3: "a", 4: "b", 5: "c", 6: "h", 7: "i", 8: "o"}
        model = type("FakeModel", (), {"n_layer": 2, "n_embed": 64, "n_head": 2})()

        class FakeProvider(SloTransformerProvider):
            def __init__(self):
                self._model = model
                self._stoi = stoi
                self._itos = itos
                self._model_id_str = "test-model"
                self._bos = stoi.get(" ", 0)
                self._eos = stoi.get("<PAD>", 0)

        return FakeProvider()

    def test_encode_basic(self):
        p = self._make_provider()
        ids = p._encode("ab")
        assert ids == [3, 4]

    def test_encode_lowercases(self):
        p = self._make_provider()
        ids = p._encode("AB")
        assert ids == [3, 4]

    def test_encode_unknown_char(self):
        p = self._make_provider()
        ids = p._encode("x")
        assert ids == [2]  # falls back to BOS = stoi[" "]

    def test_encode_empty(self):
        p = self._make_provider()
        ids = p._encode("")
        assert ids == []

    def test_encode_space(self):
        p = self._make_provider()
        ids = p._encode(" ")
        assert ids == [2]

    def test_decode_basic(self):
        p = self._make_provider()
        text = p._decode([3, 4, 5])
        assert text == "abc"

    def test_decode_unknown_id(self):
        p = self._make_provider()
        text = p._decode([99])
        assert text == "?"

    def test_decode_empty(self):
        p = self._make_provider()
        text = p._decode([])
        assert text == ""

    def test_encode_decode_roundtrip(self):
        p = self._make_provider()
        original = "abc"
        encoded = p._encode(original)
        decoded = p._decode(encoded)
        assert decoded == original

    def test_messages_to_prompt_user(self):
        p = self._make_provider()
        msgs = [{"role": "user", "content": "hello"}]
        prompt = p._messages_to_prompt(msgs)
        assert "User: hello" in prompt
        assert prompt.endswith("Assistant:")

    def test_messages_to_prompt_system(self):
        p = self._make_provider()
        msgs = [{"role": "system", "content": "Be nice"}]
        prompt = p._messages_to_prompt(msgs)
        assert "System: Be nice" in prompt

    def test_messages_to_prompt_assistant(self):
        p = self._make_provider()
        msgs = [{"role": "assistant", "content": "I am helpful"}]
        prompt = p._messages_to_prompt(msgs)
        assert "Assistant: I am helpful" in prompt

    def test_messages_to_prompt_multi_turn(self):
        p = self._make_provider()
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = p._messages_to_prompt(msgs)
        assert "System:" in prompt
        assert "User: Hi" in prompt
        assert "Assistant: Hello" in prompt
        assert "User: How are you?" in prompt
        assert prompt.endswith("Assistant:")

    def test_messages_to_prompt_no_role(self):
        p = self._make_provider()
        msgs = [{"content": "hi"}]
        prompt = p._messages_to_prompt(msgs)
        assert "User: hi" in prompt

    def test_model_id(self):
        p = self._make_provider()
        assert p.model_id == "test-model"

    def test_capabilities(self):
        p = self._make_provider()
        caps = p.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.vision is False
        assert caps.embedding is False

    def test_embed_returns_empty(self):
        p = self._make_provider()
        assert p.embed("hello") == []

    def test_metadata(self):
        p = self._make_provider()
        meta = p.metadata
        assert meta["model_id"] == "test-model"
        assert meta["vocab_size"] == 9
        assert meta["type"] == "soultransformer"
        assert meta["n_layer"] == 2
        assert meta["n_embed"] == 64
        assert meta["n_head"] == 2

    def test_itos_converts_keys_to_int(self):
        p = self._make_provider()
        assert all(isinstance(k, int) for k in p._itos.keys())


# =============================================================================
# Protocol compliance checks
# =============================================================================

class TestProtocolCompliance:
    def test_provider_router_implements_model_provider(self):
        router = ProviderRouter()
        assert isinstance(router, ModelProvider)

    def test_vision_processor_implements_message_processor(self):
        vp = VisionProcessor()
        assert isinstance(vp, MessageProcessor)

    def test_knowledge_processor_implements_message_processor(self):
        kp = KnowledgeProcessor()
        assert isinstance(kp, MessageProcessor)

    def test_tool_use_processor_implements_message_processor(self):
        tp = ToolUseProcessor()
        assert isinstance(tp, MessageProcessor)

    def test_personality_processor_implements_message_processor(self):
        pp = PersonalityProcessor()
        assert isinstance(pp, MessageProcessor)

    def test_style_processor_implements_message_processor(self):
        sp = StyleProcessor()
        assert isinstance(sp, MessageProcessor)


# =============================================================================
# Regression / edge cases
# =============================================================================

class TestEdgeCases:
    def setup_method(self):
        _clear_providers()

    def test_register_and_clear_cycle(self):
        register_provider("a", 1)
        register_provider("b", 2)
        clear_providers()
        register_provider("c", 3)
        assert list_providers() == ["c"]

    def test_processor_registry_not_affected_by_clear_providers(self):
        register_processor("vision", "vp")
        clear_providers()
        assert get_processor("vision") == "vp"
        _clear_processors()

    def test_get_provider_returns_same_object(self):
        obj = MockTextProvider()
        register_provider("same", obj)
        assert get_provider("same") is obj

    def test_router_embed_always_empty(self):
        router = ProviderRouter()
        for text in ["", "hello", "a" * 1000]:
            assert router.embed(text) == []

    async def test_apply_processors_returns_new_list(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [])
        assert result == msgs

    async def test_apply_processors_chain_preserves_structure(self):
        class AddField:
            async def process(self, msgs):
                return [{**m, "extra": True} for m in msgs]

        result = await apply_processors(
            [{"role": "user", "content": "hi"}],
            [AddField()],
        )
        assert result[0]["extra"] is True
        assert result[0]["role"] == "user"

    def test_tool_use_processor_tool_re_compiles(self):
        tp = ToolUseProcessor()
        m = tp.TOOL_RE.search("[[TOOL: describe_image]] abc")
        assert m is not None
        assert m.group(1) == "describe_image"
        assert m.group(2) == "abc"

    def test_personality_describe_trait_at_threshold(self):
        pp = PersonalityProcessor()
        assert pp._describe_trait("warmth", 0.3) == "reserved"
        assert pp._describe_trait("warmth", 0.5) == "friendly"
        assert pp._describe_trait("warmth", 0.7) == "warm"
        assert pp._describe_trait("warmth", 0.9) == "very warm and empathetic"

    async def test_style_at_boundary_no_instruction(self):
        sp = StyleProcessor(formality=0.3, directness=0.3, verbosity=0.3)
        result = await sp.process([{"role": "user", "content": "hi"}])
        assert result == [{"role": "user", "content": "hi"}]

    def test_vision_processor_multiple_images_extracted(self):
        vp = VisionProcessor()
        msgs = [
            {"role": "user", "content": "data:image/png;base64,first"},
            {"role": "user", "content": "data:image/jpeg;base64,second"},
        ]
        images = vp._extract_images(msgs)
        assert len(images) == 2

