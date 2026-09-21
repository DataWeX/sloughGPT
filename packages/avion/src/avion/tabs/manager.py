"""Tab manager — track browser tabs, switch, find, close.

Works against any provider implementing open/list/activate/close.
The Playwright backend gains tab support separately; until then this
manages tab state for backends that expose multiple pages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class TabState(Enum):
    """Lifecycle state of a tab."""

    ACTIVE = "active"
    BACKGROUND = "background"
    CLOSED = "closed"


@dataclass
class TabInfo:
    """A tracked tab."""

    id: str
    url: str = ""
    title: str = ""
    state: TabState = TabState.BACKGROUND
    opened_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "state": self.state.value,
        }


@runtime_checkable
class TabProvider(Protocol):
    """What a multi-tab backend must implement."""

    async def tab_open(self, url: str) -> str: ...
    async def tab_list(self) -> list[dict[str, Any]]: ...
    async def tab_activate(self, tab_id: str) -> None: ...
    async def tab_close(self, tab_id: str) -> None: ...


class TabManager:
    """Tracks tabs and switches between them.

    Usage::

        mgr = TabManager(provider)
        tab = await mgr.open("http://localhost:3000/chat")
        await mgr.switch(tab.id)
        found = mgr.find_by_url("/chat")
    """

    def __init__(self, provider: TabProvider):
        self._provider = provider
        self._tabs: dict[str, TabInfo] = {}
        self._active_id: str | None = None
        self._seq = 0

    @property
    def active(self) -> TabInfo | None:
        if self._active_id is None:
            return None
        return self._tabs.get(self._active_id)

    @property
    def tabs(self) -> list[TabInfo]:
        return list(self._tabs.values())

    @property
    def open_tabs(self) -> list[TabInfo]:
        return [t for t in self._tabs.values() if t.state != TabState.CLOSED]

    async def open(self, url: str, title: str = "") -> TabInfo:
        tab_id = await self._provider.tab_open(url)
        self._tabs[tab_id] = TabInfo(id=tab_id, url=url, title=title)
        await self.switch(tab_id)
        await self.refresh()
        info = self._tabs[tab_id]
        if title:
            info.title = title
        return info

    async def refresh(self) -> list[TabInfo]:
        """Sync tracked state with the provider."""
        seen: set[str] = set()
        for raw in await self._provider.tab_list():
            tid = str(raw.get("id", ""))
            seen.add(tid)
            if tid in self._tabs:
                self._tabs[tid].url = str(raw.get("url", self._tabs[tid].url))
                self._tabs[tid].title = str(raw.get("title", self._tabs[tid].title))
            else:
                self._seq += 1
                self._tabs[tid] = TabInfo(
                    id=tid,
                    url=str(raw.get("url", "")),
                    title=str(raw.get("title", "")),
                )
        for tid, info in self._tabs.items():
            if tid not in seen and info.state != TabState.CLOSED:
                info.state = TabState.CLOSED
        return self.tabs

    async def switch(self, tab_id: str) -> bool:
        if tab_id not in self._tabs or self._tabs[tab_id].state == TabState.CLOSED:
            return False
        await self._provider.tab_activate(tab_id)
        for tid, info in self._tabs.items():
            info.state = (
                TabState.ACTIVE
                if tid == tab_id
                else (TabState.BACKGROUND if info.state != TabState.CLOSED else TabState.CLOSED)
            )
        self._active_id = tab_id
        return True

    def find_by_url(self, fragment: str) -> TabInfo | None:
        for tab in self.open_tabs:
            if fragment in tab.url:
                return tab
        return None

    def find_by_title(self, fragment: str) -> TabInfo | None:
        fragment = fragment.lower()
        for tab in self.open_tabs:
            if fragment in tab.title.lower():
                return tab
        return None

    async def close(self, tab_id: str) -> bool:
        if tab_id not in self._tabs or self._tabs[tab_id].state == TabState.CLOSED:
            return False
        await self._provider.tab_close(tab_id)
        self._tabs[tab_id].state = TabState.CLOSED
        if self._active_id == tab_id:
            self._active_id = None
            remaining = self.open_tabs
            if remaining:
                await self.switch(remaining[0].id)
        return True

    async def close_others(self) -> int:
        keep = self._active_id
        closed = 0
        for tid in [t.id for t in self.open_tabs if t.id != keep]:
            if await self.close(tid):
                closed += 1
        return closed
