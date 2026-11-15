"""Core search types — the normalized output contract for every store.

One search across the system means one shape for what comes back.
Adapters may use any matcher they like (substring, fuzzy, vector); they
all emit :class:`SearchHit` and the aggregator owns fan-out, limits,
timeouts and visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchHit:
    """One result from one store, normalized.

    Attributes:
        id: Store-native identifier (opaque to callers).
        store: Adapter store name (e.g. ``members``, ``knowledge``).
        title: Primary display text.
        detail: Secondary display text (role, status, topic...).
        score: Relevance in ``[0, 1]`` — higher wins. Aggregator sorts on it.
        locator: Jump target — ``route:/path``, ``file:path:line``,
            ``anchor:section``. What makes a hit navigable instead of
            merely readable.
    """

    id: str
    store: str
    title: str
    detail: str = ""
    score: float = 0.0
    locator: str = ""

    def __post_init__(self) -> None:
        if not self.store:
            raise ValueError("SearchHit.store is required")
        if not self.title:
            raise ValueError("SearchHit.title is required")


@dataclass(slots=True)
class SearchContext:
    """Who is searching and in what scope.

    The aggregator uses this for visibility: workspace-scoped adapters
    never run without a ``workspace_id``, so a hit the caller could not
    open is never returned.
    """

    workspace_id: str = ""
    user_id: str = ""
    is_admin: bool = False


@dataclass(slots=True)
class SearchResult:
    """Aggregated outcome of a multi-store search.

    Attributes:
        hits: Merged results, sorted by score desc then title.
        partial: Stores that failed or timed out — never silent.
        skipped: Stores not queried (e.g. workspace-scoped stores when
            no workspace is in context) — a deliberate skip, not a failure.
    """

    hits: list[SearchHit] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
