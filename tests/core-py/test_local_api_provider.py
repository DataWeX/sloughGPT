"""
Local dispatch must pick the prompt-level provider, not the raw model.

After materialization the parent-preload writer (startup.py) stores the RAW
token-level model in ServerState.model while ServerState.provider keeps the
SloNetChatProvider handle. ``_current_provider()`` read the model slot first,
so every local generate/chat call reached ``raw.generate(input_ids, ...)``
and died in ``np.array("<prompt>", dtype=np.int64)`` with
``invalid literal for int() with base 10: '<prompt>'`` (card 20261125_122006
warm walk; FastAPI routes never saw it — they use get_provider("default")).

Run: ./scripts/python -m pytest tests/core-py/test_local_api_provider.py -x -q
"""

import numpy as np
import pytest

from domain.infrastructure.server_state import get_server_state
from domain.shell._internal import local_api


class _RawModelStub:
    """Token-level raw model — mirrors SloTransformer.generate(input_ids, ...)."""

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, **kwargs):
        # Exact production failure: slonet.py SloTransformer.generate else-branch.
        np.array(input_ids, dtype=np.int64)
        return "raw-should-never-be-used"


class _ProviderStub:
    """Prompt-level chat provider — mirrors SloNetChatProvider.generate/chat."""

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, **kwargs):
        return "provider-text"

    async def chat(self, messages, max_tokens=512, temperature=0.7, **kwargs):
        return "provider-chat"


@pytest.fixture
def state_slots():
    """Save/restore the shared ServerState slots (singleton — never leak)."""
    st = get_server_state()
    saved = (st.model.get(), st.provider.get(), st.model_type.get())
    yield st
    st.model.set(saved[0])
    st.provider.set(saved[1])
    st.model_type.set(saved[2])


class TestProviderSlotPreference:
    def test_prefers_provider_slot_over_raw_model(self, state_slots):
        """Both slots set (the post-materialization state) → provider wins."""
        st = state_slots
        raw = _RawModelStub()
        provider = _ProviderStub()
        st.model.set(raw)
        st.provider.set(provider)
        assert local_api._current_provider() is provider

    def test_falls_back_to_model_slot_when_provider_empty(self, state_slots):
        """Provider slot empty → the model slot still serves (old publish paths)."""
        st = state_slots
        raw = _RawModelStub()
        st.provider.set(None)
        st.model.set(raw)
        assert local_api._current_provider() is raw


class TestDispatchAfterMaterialization:
    """End-to-end: the exact ServerState shape the warm server ends up in."""

    def test_generate_routes_to_provider(self, state_slots):
        st = state_slots
        st.model.set(_RawModelStub())  # what parent-preload writes
        st.provider.set(_ProviderStub())  # what lazy boot writes
        out = local_api.dispatch_post(
            "/inference/generate",
            {
                "prompt": "You are a research agent. Find things.",
                "max_new_tokens": 300,
                "temperature": 0.7,
            },
        )
        assert out.get("text") == "provider-text", out

    def test_chat_routes_to_provider(self, state_slots):
        st = state_slots
        st.model.set(_RawModelStub())  # raw model has no .chat at all
        st.provider.set(_ProviderStub())
        out = local_api.dispatch_post("/chat", {"messages": [{"role": "user", "content": "hi"}]})
        assert out.get("reply") == "provider-chat", out
