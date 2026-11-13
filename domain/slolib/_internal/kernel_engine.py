"""Kernel Engine — selection policy over cores (layer 2 of 3).

engine = policy · ops = contract · core = capability. This module answers
one question — WHICH core runs this op at this shape on THIS machine —
and nothing else. It never executes a kernel, never holds weights
(ownership follows the constructor; the model owns weights, cores are
stateless), and never leaks device vocabulary to callers: selection is
over registered CORES, device is metadata on their declaration.

Selection key: (op, shape, machine-fingerprint). The fingerprint
includes the thread budget because the winner genuinely flips with it
(measured on this box: c32 1.3-1.7x vs numpy at threads=1, 0.03x at
threads=8 — OpenMP team spin-up per call). A static routing table would
be wrong on any other PC; autotuning is what makes the routing true.

Policy, in order:
  1. memoized per-process — one pick per key, ever (determinism:
     run-to-run equality pins engines chosen once, never re-picked
     mid-run; generation stability depends on it).
  2. persisted tuning cache keyed by fingerprint — a new process on the
     same machine starts already-tuned.
  3. safe default = numpy — unknown machine, empty cache, stale or
     vanished core name: fall back to the oracle, never raise.

The cache path is a function (``_cache_path``) so tests can redirect it.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from pathlib import Path

from domain.slolib._internal import kernel_core

logger = logging.getLogger("slo.kernel.engine")

DEFAULT_CORE = "numpy"

# Per-process memo: key -> core name. One decision per key, forever.
_MEMO: dict[str, str] = {}

# Fingerprint cache (cheap to compute, read on every cache load).
_FP: str | None = None


def _fp() -> str:
    """Computed fingerprint (private: ``load``/``save`` take a
    fingerprint PARAMETER that shadows the public function)."""
    global _FP
    if _FP is None:
        parts = [
            platform.machine(),
            platform.processor() or platform.machine(),
            str(os.cpu_count() or 1),
            f"oblas={os.environ.get('OPENBLAS_NUM_THREADS', '?')}",
            f"omp={os.environ.get('OMP_NUM_THREADS', '?')}",
        ]
        _FP = "|".join(parts)
    return _FP


def fingerprint() -> str:
    """Machine identity for the tuning cache — thread budget included.

    The thread budget is part of identity, not a runtime detail: the
    c32-vs-numpy winner flips between OPENBLAS/OMP budgets (see module
    doc). Same box, different budget = different machine for tuning.
    """
    return _fp()


def _cache_path() -> Path:
    """Where tuning decisions persist. SLO_KERNEL_CACHE redirects (tests)."""
    env = os.environ.get("SLO_KERNEL_CACHE")
    if env:
        return Path(env)
    return Path(os.environ.get("HOME", "/tmp")) / ".cache" / "slo" / "kernel_engine.json"


def _key(op: str, shape) -> str:
    """Cache key: op@exact-shape (e.g. ``softmax@4096-32``).

    Exact shapes, not rounded buckets: two nearby-but-different shapes
    can genuinely have different winners (compute vs dispatch crossover),
    and over-sharing a decision would pick wrong for one of them. Keys
    stay few because training repeats the same shapes every step.
    """
    dims = "-".join(str(int(d)) for d in shape)
    return f"{op}@{dims}"


def _valid(op: str, name: str) -> str:
    """``name`` if it can run ``op`` here, else the oracle (never raise)."""
    core = kernel_core.get(name)
    if core is None:
        return DEFAULT_CORE
    try:
        if core.available() and core.supports(op):
            return name
    except Exception as exc:  # noqa: BLE001 - a broken core falls back
        logger.info("core %s failed availability (%s); using %s", name, exc, DEFAULT_CORE)
    return DEFAULT_CORE


def pick(op: str, shape, tuning: dict | None = None) -> str:
    """Which core runs ``op`` at ``shape``. Memoized; safe default numpy.

    ``tuning`` (op@shape -> core name) overrides the persisted cache and
    the memo — an explicit argument beats cached state (used by tests and
    by the tuning harness when it records results), and it re-anchors the
    memo so the default path sees the same decision afterwards.
    """
    key = _key(op, shape)
    if tuning is None:
        if key in _MEMO:
            return _MEMO[key]
        tuning = load()
    chosen = _valid(op, tuning.get(key, DEFAULT_CORE))
    _MEMO[key] = chosen  # one decision per key per process (determinism)
    return chosen


def load(fingerprint: str | None = None) -> dict:
    """Persisted decisions for this machine. {} on any problem — the
    engine degrades to safe-default, never to an exception."""
    path = _cache_path()
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    fp = fingerprint if fingerprint is not None else _fp()
    entry = doc.get(fp)
    return dict(entry) if isinstance(entry, dict) else {}


def save(tuning: dict, fingerprint: str | None = None) -> None:
    """Persist decisions keyed by fingerprint. Other machines' entries
    are preserved — the file is a multi-machine table, not a monologue."""
    path = _cache_path()
    try:
        doc = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        doc = {}
    fp = fingerprint if fingerprint is not None else _fp()
    doc[fp] = dict(tuning)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    except OSError as exc:  # a read-only cache must not break training
        logger.info("kernel tuning cache not writable (%s)", exc)


def reset() -> None:
    """Clear the per-process memo (tests; never called in hot paths —
    a re-pick mid-run is exactly what determinism forbids)."""
    global _FP
    _MEMO.clear()
    _FP = None
