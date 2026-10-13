"""Chat-seam integration tests for the unified response cache.

Verifies the cache is consulted on the real serving path (ChatDomain.respond):
a deterministic (low-temperature) request generates once then replays from the
exact-key tier, while a stochastic request always regenerates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from domain.chat._internal.domain import ChatDomain
from domain.inference._internal.response_cache import (
    bump_cache_generation,
    get_response_cache,
    reset_response_cache_for_tests,
)


@pytest.fixture
def cache():
    reset_response_cache_for_tests()
    yield get_response_cache()
    reset_response_cache_for_tests()


def _mock_provider(reply: str = "Hello there!"):
    p = AsyncMock()
    p.chat.return_value = reply
    return p


class TestChatSeamCache:
    @pytest.mark.asyncio
    async def test_deterministic_hit_skips_provider(self, cache):
        domain = ChatDomain()
        provider = _mock_provider("The capital of France is Paris.")

        with patch("domain.models._internal.provider.get_provider", return_value=provider):
            first = await domain.respond(
                messages=[{"role": "user", "content": "capital of France?"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )
            calls_after_first = provider.chat.await_count

            second = await domain.respond(
                messages=[{"role": "user", "content": "capital of France?"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )

        assert first.text == second.text
        # second call was served from cache — provider was not invoked again
        assert provider.chat.await_count == calls_after_first

    @pytest.mark.asyncio
    async def test_stochastic_always_regenerates(self, cache):
        domain = ChatDomain()
        provider = _mock_provider("random reply")

        with patch("domain.models._internal.provider.get_provider", return_value=provider):
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.9,
                max_tokens=16,
            )
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.9,
                max_tokens=16,
            )

        # both calls hit the provider (never cached)
        assert provider.chat.await_count == 2
        assert cache.get_stats()["bypass_stochastic"] >= 2

    @pytest.mark.asyncio
    async def test_param_change_forces_regenerate(self, cache):
        domain = ChatDomain()
        provider = _mock_provider("reply")

        with patch("domain.models._internal.provider.get_provider", return_value=provider):
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=32,  # different max_tokens
            )

        # different key -> regenerated
        assert provider.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_model_swap_invalidates(self, cache):
        domain = ChatDomain()
        provider = _mock_provider("reply")

        with patch("domain.models._internal.provider.get_provider", return_value=provider):
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )
            bump_cache_generation()  # simulate model swap
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )

        # after a swap the stale reply is not replayed
        assert provider.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_error_response_not_cached(self, cache):
        domain = ChatDomain()
        provider = AsyncMock()
        provider.chat.return_value = "[Error: Generation failed]"

        with patch("domain.models._internal.provider.get_provider", return_value=provider):
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )
            await domain.respond(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
                temperature=0.0,
                max_tokens=16,
            )

        # error strings are never cached, so the provider is retried
        assert provider.chat.await_count == 2
