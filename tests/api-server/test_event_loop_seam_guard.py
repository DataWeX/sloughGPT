"""Static guard: blocking store writes must leave the async event loop.

The host is async, the storage core is sync and blocking (AGENTS.md
"Execution Philosophy": *the seam is async and always explicit — one-off
blocking calls go through ``await asyncio.to_thread(...)``*). A MogDB
journal append takes a cross-process ``flock`` and fsyncs: measured
**8-42 ms per append** under load versus 0.019 ms for the raw
``open(...).write(...)`` it replaced, and ``compact()`` holds the same lock
while it rewrites and fsyncs the whole snapshot (~440 ms for a 49 MB
journal). Run inline inside an ``async def`` handler, that stalls *every*
connection the process serves — not just the request that triggered it.

Structural counterpart to the runtime measurements in card faa1cfa7: the
durability crash battery proves the primitive survives SIGKILL; this stops
new handlers from reintroducing the inline pattern that the phase-2 fsync
change exposed in ``experiments.py``/``companion.py``/``mobile.py``.
Exemptions are (file, qualname)-scoped with a reason and must match a real
site, so stale exemptions fail too.

Scope: the async handler plus, transitively, the same-file helpers it calls —
the shape that hid a write from a handler-local scan (``MogDBVectorStore``
delegates its ``insert_one``/``update_one`` to ``upsert_sync``). A write
inside a helper the handler hands to ``to_thread``/``run_in_executor`` is the
compliant form and is not followed (the ``errors.py`` pattern). Resolving
this found six real inline writes in ``api_keys.py`` and ``companion.py``
that the first version of this guard reported as clean.

Two blind spots stay unguarded:

* cross-file delegation. Resolving ``store.delete_pair()`` would need the
  domain package's call graph, so ``WRITE_METHODS`` also lists the handful of
  domain wrapper names that journal through a ``MobileTrainingStore`` method
  rather than a ``Collection`` one;
* pure reads. First touch of a collection replays the journal under the lock,
  so a cold read can still block — but reads were already blocking before the
  fsync change and are memory-fast once loaded, so flagging them would bury
  the write violations behind a dozen exemptions.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Whole server tree, not just the routers: ``training/`` (16 files with async
# handlers) and ``infrastructure/`` (lifespan, middleware, startup hooks) run on
# the same loop and would otherwise be outside the guard for no good reason.
SCAN_ROOTS = (
    REPO / "apps" / "api" / "server",
)

# Collection methods that journal (and therefore fsync under the store lock).
COLLECTION_WRITES = frozenset(
    {
        "insert_one",
        "insert_many",
        "update_one",
        "update_many",
        "delete_one",
        "delete_many",
        "find_one_and_update",
        "find_one_and_replace",
        "find_one_and_delete",
        "compact",
        "drop",
    }
)

# Collection methods that are pure reads: they never append to the journal, so
# running one inline on the loop costs memory, not a disk stall.
COLLECTION_READS = frozenset(
    {"aggregate", "count", "find", "find_by_ids", "find_one"}
)

# Index maintenance mutates process memory only — no ``append_record``/
# ``atomic_write`` in any of them — so it is not an event-loop stall. Listed
# separately (rather than lumped in with reads) so that a future version which
# starts persisting an index fails ``test_write_set_tracks_the_mogdb_api`` and
# forces reclassification.
COLLECTION_IN_MEMORY = frozenset(
    {
        "create_index",
        "create_sorted_index",
        "create_ttl_index",
        "drop_index",
    }
)

# MogDB itself writes to disk through these, and the receiver is a Database
# rather than a Collection: ``compact_all`` compacts every collection,
# ``sync_all`` flushes every synced JSON store, and ``drop_collection`` deletes
# the journal and the compacted file. A scan keyed only to ``COLLECTION_WRITES``
# would never see the call site, because it names ``sync_all`` rather than
# ``sync``.
DATABASE_WRITES = frozenset({"compact_all", "drop_collection", "sync_all"})

# A real write that the scan deliberately does *not* flag. ``MogDB.close()``
# syncs and possibly compacts, so calling it inline would stall the loop — but
# the bare name collides with ``socket.close()``, ``WebSocket.close()``,
# ``loop.close()`` and every other closer in the process, and this guard keys
# on the attribute name alone, so honouring it would bury the genuine
# violations under false positives (it fired on four such calls when tried).
# Listed here so the API-drift test still counts it as classified, and so the
# hole is named rather than invisible.
AMBIGUOUS_WRITES = frozenset({"close"})

DATABASE_READS = frozenset({"collection", "list_collections"})

# Domain wrappers whose bodies call one of the above (MobileTrainingStore):
# these never name a Collection method at the call site, so a scan keyed only
# on COLLECTION_WRITES would miss them.
WRAPPER_WRITES = frozenset(
    {
        "delete_pair",
        "delete_synced",
        "update_quality",
    }
)

WRITE_METHODS = COLLECTION_WRITES | DATABASE_WRITES | WRAPPER_WRITES

# How a handler hands the blocking work to a worker thread.
OFFLOAD_MARKERS = ("to_thread", "run_in_executor")

# (relpath, enclosing qualname) -> reason. Empty: every current violation is
# fixed rather than exempted. Keep it that way unless the block is genuinely
# harmless (e.g. a store that is provably empty and never fsyncs).
EXEMPT: dict[tuple[str, str], str] = {}


def _direct_writes(fn: ast.AST) -> set[str]:
    """Write methods called syntactically inside *fn*."""
    return {
        call.func.attr
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in WRITE_METHODS
    }


def _called_names(fn: ast.AST) -> set[str]:
    """Every function/method name *fn* invokes (attribute or bare)."""
    names: set[str] = set()
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        if isinstance(target, ast.Attribute):
            names.add(target.attr)
        elif isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _offloaded_names(fn: ast.AST) -> set[str]:
    """Names handed to ``to_thread``/``run_in_executor`` as arguments.

    These are *not* performed by the handler — a worker thread runs them, which
    is exactly the seam the rule asks for (``errors.py`` wraps its whole store
    helper this way).
    """
    off: set[str] = set()
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        target = call.func
        marker = target.attr if isinstance(target, ast.Attribute) else None
        if marker is None and isinstance(target, ast.Name):
            marker = target.id
        if marker not in OFFLOAD_MARKERS:
            continue
        for arg in call.args:
            if isinstance(arg, ast.Attribute):
                off.add(arg.attr)
            elif isinstance(arg, ast.Name):
                off.add(arg.id)
    return off


def _handler_violations(path: Path, rel: str) -> list[tuple[str, int, str, str]]:
    """Writes an async handler performs inline: (qualname, lineno, what).

    Resolves calls into same-file helpers (transitively), because the write
    frequently lives in a sync method the handler delegates to — the shape
    that made ``MogDBVectorStore.upsert`` invisible to a handler-local scan.
    Helpers handed to ``to_thread``/``run_in_executor`` are followed no
    further: offloading them is the compliant form.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    funcs: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, node)

    found: list[tuple[str, int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        src = ast.unparse(node)
        if any(marker in src for marker in OFFLOAD_MARKERS):
            continue
        # Walk the handler's same-file callees, skipping offloaded ones.
        seen: set[str] = {node.name}
        stack: list[ast.AST] = [node]
        writes: set[str] = set()
        while stack:
            fn = stack.pop()
            writes |= _direct_writes(fn)
            off = _offloaded_names(fn)
            for name in _called_names(fn):
                if name in off or name in seen:
                    continue
                target = funcs.get(name)
                if target is not None:
                    seen.add(name)
                    stack.append(target)
        for name in sorted(writes):
            found.append((node.name, node.lineno, name, rel))
    return found


def _scan() -> dict[str, list[tuple[str, int, str, str]]]:
    """Inline write sites per file (relative posix path)."""
    out: dict[str, list[tuple[str, int, str, str]]] = {}
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            sites = _handler_violations(path, rel)
            if sites:
                out[rel] = sites
    return out


def test_scan_roots_exist() -> None:
    """The guard is worthless if its roots silently stop matching."""
    for root in SCAN_ROOTS:
        assert root.is_dir(), f"scan root missing: {root}"


def test_write_set_tracks_the_mogdb_api() -> None:
    """The hand-maintained write list must not drift from mogdb's real API.

    ``COLLECTION_WRITES`` is a literal, so a mutating method added to ``Collection``
    would fall outside the scan *silently* — and a guard that reports zero
    violations because it is blind looks exactly like one that found none.
    Classify every public method explicitly, then assert the sets tile the real
    API in both directions: unclassified names mean the list is stale and the
    new method needs a decision; names the guard claims but mogdb no longer
    defines were guessed rather than derived and are quietly covering for a real
    method that was never listed.
    """
    import inspect

    from mogdb.collection import Collection
    from mogdb.database import MogDB as Database

    for label, cls, writes, non_writes in (
        (
            "Collection",
            Collection,
            COLLECTION_WRITES,
            COLLECTION_READS | COLLECTION_IN_MEMORY,
        ),
        ("Database", Database, DATABASE_WRITES, DATABASE_READS | AMBIGUOUS_WRITES),
    ):
        public = {
            name
            for name, member in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith("_")
        }
        unclassified = sorted(public - (writes | non_writes))
        assert not unclassified, (
            f"mogdb {label} defines methods the seam guard does not classify: "
            f"{unclassified}. For each, decide whether it journals to disk "
            f"(add it to {label.upper()}_WRITES so inline use is flagged) or is "
            f"a read/in-memory op, and add it to the matching read set."
        )
        phantom = sorted((writes | non_writes) - public)
        assert not phantom, (
            f"mogdb {label} no longer defines {phantom}; the guard guards "
            f"method names that do not exist, which is how a real method goes "
            f"unnoticed when the API changes shape."
        )
    # The ambiguity is deliberate and must stay honoured: an ambiguous name
    # listed in WRITE_METHODS turns the guard into a false-positive machine.
    assert not (AMBIGUOUS_WRITES & WRITE_METHODS), (
        f"ambiguous write names are in WRITE_METHODS: "
        f"{sorted(AMBIGUOUS_WRITES & WRITE_METHODS)}"
    )


def test_scan_finds_the_wrappers_it_claims_to() -> None:
    """Prove the scan can see a write before trusting its silence.

    A scanner that silently matches nothing also reports zero violations, so
    feed it a synthetic handler that writes inline and assert it is caught.
    The second fixture covers delegation: a handler that calls a helper which
    journals must be caught, while one handing the same helper to
    ``to_thread`` must not be.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        direct = Path(tmp) / "fake_router.py"
        direct.write_text(
            "async def bad_handler():\n"
            "    store = get_training_store()\n"
            "    store.delete_pair('x')\n"
            "    col = db.collection('c')\n"
            "    col.insert_one({'_id': 'y'})\n",
            encoding="utf-8",
        )
        delegated = Path(tmp) / "fake_delegated.py"
        delegated.write_text(
            "class Store:\n"
            "    def _journal(self):\n"
            "        self._col.insert_one({'_id': 'z'})\n"
            "\n"
            "    async def bad_handler(self):\n"
            "        return self._journal()\n"
            "\n"
            "    async def ok_handler(self):\n"
            "        return await asyncio.to_thread(self._journal)\n",
            encoding="utf-8",
        )
        direct_sites = _handler_violations(direct, "fake_router.py")
        delegated_sites = _handler_violations(delegated, "fake_delegated.py")

    assert {what for _, _, what, _ in direct_sites} == {"delete_pair", "insert_one"}
    by_handler: dict[str, set[str]] = {}
    for qual, _, what, _ in delegated_sites:
        by_handler.setdefault(qual, set()).add(what)
    assert by_handler.get("bad_handler") == {"insert_one"}, f"delegation blind: {by_handler}"
    assert "ok_handler" not in by_handler, f"offloaded helper flagged: {by_handler}"


def test_exemptions_match_real_sites() -> None:
    """No stale exemptions: every (file, qualname) must still write inline."""
    found = _scan()
    for (rel, qual), _reason in EXEMPT.items():
        assert (REPO / rel).is_file(), f"exemption file gone: {rel}"
        sites = found.get(rel, [])
        assert any(q == qual for q, _, _, _ in sites), (
            f"stale exemption {rel}:{qual} — no inline write remains there; remove it"
        )


def test_store_writes_leave_the_event_loop() -> None:
    """No blocking store write runs inline inside an async handler."""
    found = _scan()
    violations: list[str] = []
    for rel, sites in found.items():
        for qual, lineno, what, _rel in sites:
            if (rel, qual) in EXEMPT:
                continue
            violations.append(f"{rel}:{lineno} [{qual}] {what}()")
    assert not violations, (
        "blocking store writes inline on the event loop — wrap them in "
        "`await asyncio.to_thread(...)` (the core stays sync; fix the seam):\n"
        + "\n".join(violations)
    )
