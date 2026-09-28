"""sloughGPT journey tests — 8 page journeys as Arken Tasks.

Each journey goes to a page and performs the page's core action.
Tests run against FakeBackend (no browser); the same Task objects
drive live runs via scripts/run_ux_flows.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

from avion import Arken, ElementLocator
from avion.core.task import Task, TaskStatus, TaskStep
from test_autoclicker import FakeBackend


def run(coro):
    return asyncio.run(coro)


def _goto(page: str):
    async def step(ctx: dict[str, Any]) -> None:
        a: Arken = ctx["arken"]
        ok = await a.goto(page)
        assert ok, f"goto {page} failed"
        assert page in a.current_url, f"unexpected url {a.current_url}"

    return step


def chat_journey() -> Task:
    async def open_chat(ctx):
        await _goto("/chat")(ctx)
        await ctx["arken"].click_text("Start")

    return Task("chat", "open chat and start", [TaskStep("open", open_chat)])


def souls_journey() -> Task:
    return Task("souls", "open souls page", [TaskStep("open", _goto("/souls"))])


def training_journey() -> Task:
    return Task("training", "open training page", [TaskStep("open", _goto("/training"))])


def datasets_journey() -> Task:
    async def search(ctx):
        await _goto("/datasets")(ctx)
        await ctx["arken"].fill(ElementLocator.css("input"), "shakespeare")

    return Task("datasets", "open datasets and search", [TaskStep("search", search)])


def models_journey() -> Task:
    return Task("models", "open models page", [TaskStep("open", _goto("/models"))])


def knowledge_journey() -> Task:
    return Task("knowledge", "open knowledge page", [TaskStep("open", _goto("/knowledge"))])


def settings_journey() -> Task:
    return Task("settings", "open settings page", [TaskStep("open", _goto("/settings"))])


def agents_journey() -> Task:
    return Task("agents", "open agents page", [TaskStep("open", _goto("/agents"))])


ALL_JOURNEYS = [
    chat_journey,
    souls_journey,
    training_journey,
    datasets_journey,
    models_journey,
    knowledge_journey,
    settings_journey,
    agents_journey,
]


def make_session() -> Arken:
    a = Arken(base_url="http://x")
    run(a.start(backend=FakeBackend()))
    return a


class TestJourneys:
    def test_all_eight_pass(self):
        for build in ALL_JOURNEYS:
            a = make_session()
            try:
                result = run(a.run_task(build()))
                assert result.status == TaskStatus.PASSED, build().name
            finally:
                run(a.stop())

    def test_report_lists_all(self):
        a = make_session()
        try:
            for build in ALL_JOURNEYS:
                run(a.run_task(build()))
            assert a.reporter.summary() == {"total": 8, "passed": 8, "failed": 0}
        finally:
            run(a.stop())

    def test_failing_journey_reported(self):
        async def boom(ctx):
            raise RuntimeError("page down")

        a = make_session()
        try:
            result = run(a.run_task(Task("broken", steps=[TaskStep("x", boom)])))
            assert result.status == TaskStatus.FAILED
            assert "[FAIL] broken" in a.report()
        finally:
            run(a.stop())
