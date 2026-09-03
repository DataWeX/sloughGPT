"""Tests for correlation ID context variable."""
from __future__ import annotations

import asyncio

from domains.infrastructure.correlation import get_correlation_id, set_correlation_id


class TestCorrelationId:
    def test_default_is_none(self):
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_set_and_get(self):
        set_correlation_id("abc-123")
        assert get_correlation_id() == "abc-123"

    def test_overwrite(self):
        set_correlation_id("first")
        set_correlation_id("second")
        assert get_correlation_id() == "second"

    def test_context_var_isolation(self):
        """Different contexts see different values."""
        set_correlation_id("in-main")
        # contextvars are task-local for asyncio
        async def child():
            assert get_correlation_id() == "in-main"
            set_correlation_id("in-child")
            assert get_correlation_id() == "in-child"
        asyncio.run(child())
        # Main context still has original value
        assert get_correlation_id() == "in-main"
