"""Playwright live tests — tabs, network mock, find/click/type.

Needs Chromium (playwright install). Slower than unit tests; exercises
the real browser path Arken drives in production.
"""

import asyncio

import pytest
from arken import Arken, ElementLocator
from arken.backends.playwright import PlaywrightBackend
from arken.network import MockResponse, MockRule, NetworkMocker
from arken.tabs import TabManager

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("playwright") is None,
    reason="playwright not installed",
)

PAGE_A = "data:text/html,<html><head><title>A</title></head><body><button>Go</button></body></html>"
PAGE_B = "data:text/html,<html><head><title>B</title></head><body><input name=q></body></html>"


def run(coro):
    return asyncio.run(coro)


class TestLiveBasics:
    def test_goto_find_click_fill(self):
        async def main():
            async with Arken(headless=True) as a:
                assert await a.goto(PAGE_A)
                assert "data:text/html" in a.current_url
                await a.click_text("Go")
                assert await a.goto(PAGE_B)
                await a.fill(ElementLocator.css("input"), "hello")
                shot = await a.screenshot()
                assert len(shot) > 1000

        run(main())

    def test_run_task_live(self):
        from arken.core.task import Task, TaskStep

        async def main():
            async with Arken(headless=True) as a:

                async def open_a(ctx):
                    assert await ctx["arken"].goto(PAGE_A)

                result = await a.run_task(Task("live", steps=[TaskStep("open", open_a)]))
                assert result.status.value == "passed"
                assert a.reporter.summary()["passed"] == 1

        run(main())


class TestLiveTabs:
    def test_manager_over_backend(self):
        async def main():
            backend = PlaywrightBackend(headless=True)
            await backend.start()
            try:
                mgr = TabManager(backend)
                await mgr.open(PAGE_A)
                tab_b = await mgr.open(PAGE_B)
                assert len(mgr.open_tabs) == 3  # start page + 2 opened
                assert mgr.active.id == tab_b.id
                found = mgr.find_by_title("A")
                assert found is not None
                await mgr.switch(found.id)
                assert mgr.active.id == found.id
                assert await mgr.close(tab_b.id)
                assert len(mgr.open_tabs) == 2
            finally:
                await backend.stop()

        run(main())


class TestLiveNetworkMock:
    def test_mocked_response_served(self):
        async def main():
            backend = PlaywrightBackend(headless=True)
            await backend.start()
            try:
                mocker = NetworkMocker()
                mocker.add_rule(
                    MockRule(
                        "stub", "example.invalid", MockResponse(status=200, body={"mocked": True})
                    )
                )
                await backend.enable_network_mock(mocker)
                await backend.navigate("http://example.invalid/page")
                text = await backend.evaluate("document.body.innerText")
                assert "mocked" in text
                assert mocker.stats().interceptions >= 1
                await backend.disable_network_mock()
            finally:
                await backend.stop()

        run(main())
