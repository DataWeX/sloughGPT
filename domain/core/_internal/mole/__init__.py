"""Mole — the always-on, no-AI watcher over the site-doctor seams.

Thin timing layer on the existing doctor routines: cadence, findings
fingerprint (delta dedupe), append-only journal, and change-only events.
Read-only by construction; suggests, never applies.
"""

from __future__ import annotations

from domain.core._internal.mole.watch import (
    default_journal_path,
    fingerprint,
    load_context,
    run_watch,
)

__all__ = ["default_journal_path", "fingerprint", "load_context", "run_watch"]
