"""The Searchable protocol — what every data store implements.

Deliberately narrow: a store decides *how* it matches (substring over
rows, FTS, embeddings) and only promises normalized output plus honest
capability metadata. The contract test suite in
``tests/api-server/`` validates every registered adapter
against the same expectations — that suite, not this interface, is what
makes "one search over the whole system" true.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from domain.search.types import SearchContext, SearchHit

Capability = Literal["live", "indexed"]
"""``live`` queries the store on demand; ``indexed`` reads a prebuilt index."""


@runtime_checkable
class Searchable(Protocol):
    """A data store that can be searched."""

    store: str
    """Unique store name — grouping key for results and partial reporting."""

    capability: Capability
    """How the adapter answers queries (live query vs prebuilt index)."""

    workspace_scoped: bool
    """True if results depend on workspace membership — the aggregator
    will not run this adapter without a workspace in context."""

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        """Return matches for ``q``, at most ``limit`` hits.

        Implementations must:
          - match case-insensitively;
          - return ``[]`` (not raise) when nothing matches;
          - raise on store failure so the aggregator can report
            the store in ``partial`` instead of pretending it was empty.
        """
        ...


@runtime_checkable
class Sweepable(Protocol):
    """Optional capability: a store that can refresh its own index on demand.

    The periodic sweeper (``domain.search.sweeper``) calls ``sweep()`` to pull
    an index up to date out-of-band, so a doc nobody queries stops drifting
    behind the source. Not every store has a meaningful refresh (a pure
    ``live`` store always answers from the source), so this is a separate
    protocol — adapters opt in by implementing it, and the sweeper detects
    support with ``isinstance(adapter, Sweepable)``.
    """

    def sweep(self) -> dict[str, int]:
        """Refresh the index; return per-outcome counts.

        Implementations must:
          - bring the index to the source's current state (absorb new
            entries, refresh changed ones, drop deleted ones);
          - return a dict of counts keyed by outcome (e.g. ``new``,
            ``refreshed``, ``evicted``, ``unchanged``);
          - never raise for an absent or empty source (sweep to zeroes).
        """
        ...
