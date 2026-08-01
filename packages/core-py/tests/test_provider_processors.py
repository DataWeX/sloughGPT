"""Tests for provider module — protocols, processors, registries, ProviderRouter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from domains.models.provider import (
    ChatMessage,
    ModelCapabilities,
    ModelProvider,
    MessageProcessor,
    register_provider,
    get_provider,
    list_providers,
    register_processor,
    get_processor,
    list_processors,
    apply_processors,
    VisionProcessor,
    KnowledgeProcessor,
    ToolUseProcessor,
    ToolDef,
    PersonalityProcessor,
    StyleProcessor,
    ProviderRouter,
    SloTransformerProvider,
    setup_providers,
    update_personality_traits,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear provider/processor registries before each test."""
    import domains.models.provider as mod
    mod._providers.clear()
    mod._processors.clear()
    yield
    mod._providers.clear()
    mod._processors.clear()


def _make_mock_provider(model_id="test-model"):
    """Create a minimal mock provider satisfying ModelProvider protocol."""
    p = AsyncMock()
    p.model_id = model_id
    p.capabilities = ModelCapabilities(chat=True, streaming=True)
    p.metadata = {"type": "mock"}
    p.embed = MagicMock(return_value=[0.1, 0.2])
    return p


# ---------------------------------------------------------------------------
# Registry: providers
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_register_and_get(self):
        prov = _make_mock_provider()
        register_provider("test", prov)
        assert get_provider("test") is prov

    def test_get_nonexistent_returns_none(self):
        assert get_provider("nope") is None

    def test_list_providers(self):
        register_provider("a", _make_mock_provider("a"))
        register_provider("b", _make_mock_provider("b"))
        assert set(list_providers()) == {"a", "b"}

    def test_overwrite(self):
        p1 = _make_mock_provider("p1")
        p2 = _make_mock_provider("p2")
        register_provider("x", p1)
        register_provider("x", p2)
        assert get_provider("x") is p2


# ---------------------------------------------------------------------------
# Registry: processors
# ---------------------------------------------------------------------------

class TestProcessorRegistry:
    def test_register_and_get(self):
        proc = KnowledgeProcessor(knowledge=["fact1"])
        register_processor("k", proc)
        assert get_processor("k") is proc

    def test_get_nonexistent_returns_none(self):
        assert get_processor("nope") is None

    def test_list_processors(self):
        register_processor("a", KnowledgeProcessor())
        register_processor("b", StyleProcessor())
        assert set(list_processors()) == {"a", "b"}


# ---------------------------------------------------------------------------
# apply_processors
# ---------------------------------------------------------------------------

class TestApplyProcessors:
    @pytest.mark.asyncio
    async def test_empty_list(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [])
        assert result == msgs

    @pytest.mark.asyncio
    async def test_single_processor(self):
        proc = KnowledgeProcessor(knowledge=["fact1"])
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [proc])
        assert any("Knowledge context" in m.get("content", "") for m in result)

    @pytest.mark.asyncio
    async def test_exception_in_processor_is_caught(self):
        """Failing processor should not crash the pipeline."""
        class BadProcessor:
            async def process(self, messages):
                raise RuntimeError("boom")

        good = KnowledgeProcessor(knowledge=["ok"])
        msgs = [{"role": "user", "content": "hi"}]
        result = await apply_processors(msgs, [BadProcessor(), good])
        assert any("ok" in m.get("content", "") for m in result)


# ---------------------------------------------------------------------------
# KnowledgeProcessor
# ---------------------------------------------------------------------------

class TestKnowledgeProcessor:
    @pytest.mark.asyncio
    async def test_empty_knowledge_noop(self):
        proc = KnowledgeProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_injects_knowledge(self):
        proc = KnowledgeProcessor(knowledge=["fact1", "fact2"])
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        assert result[0]["role"] == "system"
        assert "fact1" in result[0]["content"]
        assert "fact2" in result[0]["content"]

    def test_set_knowledge(self):
        proc = KnowledgeProcessor()
        assert proc._knowledge == []
        proc.set_knowledge(["new"])
        assert proc._knowledge == ["new"]


# ---------------------------------------------------------------------------
# PersonalityProcessor
# ---------------------------------------------------------------------------

class TestPersonalityProcessor:
    @pytest.mark.asyncio
    async def test_empty_traits_noop(self):
        proc = PersonalityProcessor()
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_injects_personality(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9, "creativity": 0.8})
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = await proc.process(msgs)
        assert "warm" in result[0]["content"].lower()
        assert "creative" in result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_no_system_creates_one(self):
        proc = PersonalityProcessor(traits={"warmth": 0.9})
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        assert result[0]["role"] == "system"
        assert "warm" in result[0]["content"].lower()

    def test_set_traits(self):
        proc = PersonalityProcessor()
        proc.set_traits({"confidence": 0.9})
        assert proc._traits == {"confidence": 0.9}

    @pytest.mark.asyncio
    async def test_all_ten_traits(self):
        """All 10 PersonalityCore traits should produce descriptions."""
        traits = {
            "warmth": 0.9, "creativity": 0.8, "empathy": 0.7,
            "formality": 0.6, "humor": 0.9, "patience": 0.5,
            "confidence": 0.4, "curiosity": 0.9, "directness": 0.2,
            "optimism": 0.8,
        }
        proc = PersonalityProcessor(traits=traits)
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        # Should create a system message with personality description
        assert result[0]["role"] == "system"
        content = result[0]["content"]
        # Each trait should contribute a word
        assert "warm" in content.lower()
        assert "creative" in content.lower()
        assert "empath" in content.lower()
        assert "humor" in content.lower()
        assert "curious" in content.lower()
        assert "positive" in content.lower()


# ---------------------------------------------------------------------------
# StyleProcessor
# ---------------------------------------------------------------------------

class TestStyleProcessor:
    @pytest.mark.asyncio
    async def test_balanced_noop(self):
        proc = StyleProcessor(formality=0.5, directness=0.5, verbosity=0.5)
        msgs = [{"role": "user", "content": "hi"}]
        result = await proc.process(msgs)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_formal_injects(self):
        proc = StyleProcessor(formality=0.9)
        msgs = [{"role": "system", "content": "Be nice."}]
        result = await proc.process(msgs)
        assert "formal" in result[0]["content"].lower()

    def test_set_style(self):
        proc = StyleProcessor()
        proc.set_style(formality=0.1, directness=0.9, verbosity=0.1)
        assert proc._formality == 0.1
        assert proc._directness == 0.9
        assert proc._verbosity == 0.1


# ---------------------------------------------------------------------------
# ToolUseProcessor
# ---------------------------------------------------------------------------

class TestToolUseProcessor:
    @pytest.mark.asyncio
    async def test_injects_tool_prompt(self):
        proc = ToolUseProcessor()
        msgs = [{"role": "system", "content": "You are helpful."}]
        result = await proc.process(msgs)
        assert "tools" in result[0]["content"].lower() or "TOOL" in result[0]["content"]

    def test_match_tool_found(self):
        proc = ToolUseProcessor()
        text = 'I need to see the image. [[TOOL: describe_image]] abc123'
        match = proc.match_tool(text)
        assert match is not None
        assert match[0] == "describe_image"
        assert match[1] == "abc123"

    def test_match_tool_not_found(self):
        proc = ToolUseProcessor()
        assert proc.match_tool("no tool here") is None


# ---------------------------------------------------------------------------
# ProviderRouter
# ---------------------------------------------------------------------------

class TestProviderRouter:
    def test_metadata(self):
        router = ProviderRouter()
        router.add_processor(KnowledgeProcessor())
        router.set_text_provider("test")
        meta = router.metadata
        assert "KnowledgeProcessor" in meta["processors"]
        assert meta["text_provider"] == "test"

    def test_capabilities(self):
        router = ProviderRouter()
        caps = router.capabilities
        assert caps.chat is True
        assert caps.streaming is True

    @pytest.mark.asyncio
    async def test_chat_stream_no_provider(self):
        router = ProviderRouter()
        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        assert "No text model" in tokens[0]

    @pytest.mark.asyncio
    async def test_chat_stream_with_provider(self):
        async def _stream(messages, **kwargs):
            yield "hello"
            yield " world"

        prov = _make_mock_provider()
        prov.chat_stream = _stream

        router = ProviderRouter()
        router.set_text_provider("test")
        register_provider("test", prov)

        tokens = []
        async for token in router.chat_stream([{"role": "user", "content": "hi"}]):
            tokens.append(token)
        assert tokens == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_delegates_to_stream(self):
        async def _stream(messages, **kwargs):
            yield "hi"

        prov = _make_mock_provider()
        prov.chat_stream = _stream

        router = ProviderRouter()
        router.set_text_provider("test")
        register_provider("test", prov)

        result = await router.chat([{"role": "user", "content": "hi"}])
        assert result == "hi"

    @pytest.mark.asyncio
    async def test_processors_run_in_order(self):
        order = []

        class TrackProc:
            async def process(self, messages):
                order.append("first")
                return messages

        class TrackProc2:
            async def process(self, messages):
                order.append("second")
                return messages

        async def _empty_stream(messages, **kwargs):
            return
            yield  # make it a generator

        prov = _make_mock_provider()
        prov.chat_stream = _empty_stream

        router = ProviderRouter()
        router.add_processor(TrackProc())
        router.add_processor(TrackProc2())
        router.set_text_provider("test")
        register_provider("test", prov)

        async for _ in router.chat_stream([{"role": "user", "content": "hi"}]):
            pass
        assert order == ["first", "second"]


# ---------------------------------------------------------------------------
# update_personality_traits
# ---------------------------------------------------------------------------

class TestUpdatePersonalityTraits:
    def test_updates_router_processor(self):
        router = ProviderRouter()
        proc = PersonalityProcessor(traits={"warmth": 0.3})
        router.add_processor(proc)
        register_provider("default", router)

        update_personality_traits({"warmth": 0.9, "creativity": 0.8})
        assert proc._traits == {"warmth": 0.9, "creativity": 0.8}

    def test_no_router_noop(self):
        # Should not raise
        update_personality_traits({"warmth": 0.9})

    def test_no_personality_proc_noop(self):
        router = ProviderRouter()
        register_provider("default", router)
        # Should not raise
        update_personality_traits({"warmth": 0.9})


# ---------------------------------------------------------------------------
# setup_providers
# ---------------------------------------------------------------------------

class TestSetupProviders:
    def test_registers_default_router(self):
        setup_providers()
        router = get_provider("default")
        assert isinstance(router, ProviderRouter)

    def test_registers_all_processors(self):
        setup_providers()
        assert get_processor("vision") is not None
        assert get_processor("tool_use") is not None
        assert get_processor("personality") is not None
        assert get_processor("style") is not None
        # KnowledgeProcessor is per-request, not in the router pipeline
        assert get_processor("knowledge") is None

    def test_personality_traits_passed_through(self):
        setup_providers(personality_traits={"warmth": 0.9})
        proc = get_processor("personality")
        assert proc._traits == {"warmth": 0.9}

    def test_skips_router_if_slonet_registered(self):
        # setup_providers only skips if existing provider is SloNet/SloTransformer type
        class SloNetChatProvider:
            model_id = "slonet"
            capabilities = ModelCapabilities(chat=True, streaming=True)
            metadata = {}
            def embed(self, text): return []
            async def chat_stream(self, messages, **kwargs):
                yield ""
            async def chat(self, messages, **kwargs):
                return ""

        existing = SloNetChatProvider()
        register_provider("default", existing)
        setup_providers()
        # Should not overwrite — type name matches
        assert get_provider("default") is existing

    def test_process_guard_builds_and_attaches_server(self):
        from domains.infrastructure.slonet_server import SloNetServer

        class Guard:
            @property
            def alive(self):
                return True
            def health(self):
                return {"alive": True}
            def on_crash(self, cb):
                pass
            def on_restart(self, cb):
                pass

        guard = Guard()

        class FakeProvider:
            def __init__(self):
                self._server = None
                self._model_id = "fake-slo"
            def set_server(self, server):
                self._server = server
            def get_server(self):
                return self._server
            def to_server(self, process_guard=None, **kwargs):
                return SloNetServer(
                    model=MagicMock(),
                    tokenizer=MagicMock(),
                    model_id="fake-slo",
                    enable_warmup=False,
                    process_guard=process_guard,
                )

        provider = FakeProvider()
        setup_providers(slonet_provider=provider, process_guard=guard)
        server = provider.get_server()
        assert isinstance(server, SloNetServer)
        assert server._process_guard is guard

    def test_process_guard_skipped_without_to_server(self):
        class PlainProvider:
            def __init__(self):
                self._server = None
            def set_server(self, server):
                self._server = server

        provider = PlainProvider()
        setup_providers(slonet_provider=provider, process_guard=MagicMock())
        assert provider._server is None


# ---------------------------------------------------------------------------
# ModelCapabilities
# ---------------------------------------------------------------------------

class TestModelCapabilities:
    def test_defaults(self):
        caps = ModelCapabilities()
        assert caps.chat is False
        assert caps.streaming is False
        assert caps.embedding is False
        assert caps.vision is False
        assert caps.functions is False

    def test_custom(self):
        caps = ModelCapabilities(chat=True, streaming=True, vision=True)
        assert caps.chat is True
        assert caps.vision is True


# ---------------------------------------------------------------------------
# update_personality_traits
# ---------------------------------------------------------------------------

class TestUpdatePersonalityTraits:
    def test_updates_personality_processor(self):
        setup_providers()
        update_personality_traits({"warmth": 0.9, "creativity": 0.8})
        proc = get_processor("personality")
        assert proc is not None
        assert proc._traits["warmth"] == 0.9
        assert proc._traits["creativity"] == 0.8

    def test_updates_style_processor_from_formality_directness(self):
        setup_providers()
        update_personality_traits({"formality": 0.9, "directness": 0.1, "warmth": 0.5})
        style = get_processor("style")
        assert style is not None
        assert style._formality == 0.9
        assert style._directness == 0.1

    def test_style_defaults_when_traits_absent(self):
        setup_providers()
        update_personality_traits({"warmth": 0.7})
        style = get_processor("style")
        assert style is not None
        assert style._formality == 0.5
        assert style._directness == 0.5

    def test_noop_when_no_default_router(self):
        import domains.models.provider as prov
        old = prov._providers.pop("default", None)
        try:
            update_personality_traits({"warmth": 0.5})
        finally:
            if old is not None:
                prov._providers["default"] = old
