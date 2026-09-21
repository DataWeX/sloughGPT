"""Tabs tests — open, switch, find, close with a fake provider."""

import asyncio

from arken.tabs import TabManager, TabState


def run(coro):
    return asyncio.run(coro)


class FakeProvider:
    def __init__(self):
        self.pages: dict[str, dict] = {}
        self.seq = 0

    async def tab_open(self, url):
        self.seq += 1
        tid = f"t{self.seq}"
        self.pages[tid] = {"id": tid, "url": url, "title": f"Title {url}"}
        return tid

    async def tab_list(self):
        return list(self.pages.values())

    async def tab_activate(self, tab_id):
        if tab_id not in self.pages:
            raise KeyError(tab_id)

    async def tab_close(self, tab_id):
        self.pages.pop(tab_id, None)


class TestTabs:
    def test_open_becomes_active(self):
        mgr = TabManager(FakeProvider())
        tab = run(mgr.open("http://x/chat"))
        assert mgr.active.id == tab.id
        assert tab.state == TabState.ACTIVE

    def test_switch_moves_active(self):
        mgr = TabManager(FakeProvider())
        a = run(mgr.open("http://x/a"))
        b = run(mgr.open("http://x/b"))
        assert run(mgr.switch(a.id)) is True
        assert mgr.active.id == a.id
        assert mgr._tabs[b.id].state == TabState.BACKGROUND

    def test_switch_unknown_false(self):
        mgr = TabManager(FakeProvider())
        assert run(mgr.switch("nope")) is False

    def test_find_by_url_and_title(self):
        mgr = TabManager(FakeProvider())
        run(mgr.open("http://x/chat"))
        assert mgr.find_by_url("/chat") is not None
        assert mgr.find_by_title("chat") is not None
        assert mgr.find_by_url("/missing") is None

    def test_close_falls_back_to_remaining(self):
        mgr = TabManager(FakeProvider())
        a = run(mgr.open("http://x/a"))
        run(mgr.open("http://x/b"))
        assert run(mgr.close(a.id)) is True
        assert mgr._tabs[a.id].state == TabState.CLOSED
        assert run(mgr.close(a.id)) is False

    def test_close_active_reactivates_other(self):
        mgr = TabManager(FakeProvider())
        run(mgr.open("http://x/a"))
        b = run(mgr.open("http://x/b"))
        run(mgr.close(b.id))
        assert mgr.active is not None
        assert mgr.active.url == "http://x/a"

    def test_close_others(self):
        mgr = TabManager(FakeProvider())
        run(mgr.open("http://x/a"))
        run(mgr.open("http://x/b"))
        run(mgr.open("http://x/c"))
        assert run(mgr.close_others()) == 2
        assert len(mgr.open_tabs) == 1

    def test_refresh_marks_vanished_closed(self):
        provider = FakeProvider()
        mgr = TabManager(provider)
        run(mgr.open("http://x/a"))
        provider.pages.clear()
        run(mgr.refresh())
        assert mgr.open_tabs == []
