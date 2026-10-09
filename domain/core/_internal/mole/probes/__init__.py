"""Mole probe registry — each probe owns one seam of the live site.

A probe is a read-only observer: GET requests, SSE consumption, file
reads, and (opt-in) the journey browser sweep. Every probe returns a
:class:`ProbeResult`; the mole never raises out of a probe.

``PROBES`` is the execution registry (ordered fast → slow); the mole
dispatches keyword arguments to a probe by its registered name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from domain.core._internal.mole.models import Finding

__all__ = ["PROBES", "ProbeResult"]


@dataclass
class ProbeResult:
    """Uniform probe return: findings plus probe health and raw payload."""

    name: str
    ok: bool = True
    findings: list[Finding] = field(default_factory=list)
    error: str = ""
    raw: dict = field(default_factory=dict)


# Submodules resolve ProbeResult from this package at module scope, so
# they may only be imported after the dataclass above exists.
from . import benchmarks, gates, http, journey, sse  # noqa: E402

PROBES: list[dict] = [
    {"name": "gates", "run": gates.run_probe},
    {"name": "benchmarks", "run": benchmarks.run_probe},
    {"name": "http", "run": http.run_probe},
    {"name": "sse", "run": sse.run_probe},
    {"name": "journey", "run": journey.run_probe},
]
