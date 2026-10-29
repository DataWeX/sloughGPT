"""Deterministic timezone tests for Note.date_str (card 7e708a68).

``date_str`` must yield the LOCAL calendar day of an aware (UTC-stamped)
``created_at``, because stats/filters compare it against
``date.today()`` (local).  Raw-slicing the UTC stamp made ``today``
report 0 during the first hour after local midnight, when the UTC date
is still yesterday — ``test_stats_endpoint`` failed only in that window
(a midnight-only flake invisible to daytime CI).

Timezone is forced process-wide via ``TZ`` + ``time.tzset()`` so these
tests are deterministic regardless of the wall clock.
"""

from __future__ import annotations

import os
import time

import pytest
from app_planner.core import Note


@pytest.fixture
def set_tz():
    """Force a process-wide timezone; restore it (and re-apply) afterwards."""
    saved = os.environ.get("TZ")

    def _set(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield _set

    if saved is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = saved
    time.tzset()


def test_aware_utc_stamp_converts_to_previous_local_day(set_tz):
    # 00:30 UTC is still the previous evening in Los Angeles (UTC-7):
    # the exact shape of the post-midnight window, forced deterministically.
    set_tz("America/Los_Angeles")
    note = Note(created_at="2026-10-06T00:30:00+00:00")
    assert note.date_str == "2026-10-05"  # UTC slice would say 2026-10-06


def test_aware_utc_stamp_converts_to_next_local_day(set_tz):
    # 23:30 UTC is already the next morning in Tokyo (UTC+9).
    set_tz("Asia/Tokyo")
    note = Note(created_at="2026-10-05T23:30:00+00:00")
    assert note.date_str == "2026-10-06"  # UTC slice would say 2026-10-05


def test_aware_stamp_unchanged_in_utc_env(set_tz):
    set_tz("Etc/UTC")
    note = Note(created_at="2026-10-06T06:00:00+00:00")
    assert note.date_str == "2026-10-06" == note.created_at[:10]


def test_naive_stamp_keeps_historical_slice(set_tz):
    # Naive stamps are assumed local already — no conversion, same as before.
    set_tz("America/Los_Angeles")
    note = Note(created_at="2026-10-06T06:00:00")
    assert note.date_str == "2026-10-06"


def test_malformed_stamp_falls_back_to_slice(set_tz):
    set_tz("America/Los_Angeles")
    note = Note(created_at="not-a-date-xyz")
    assert note.date_str == "not-a-date-xyz"[:10]


def test_empty_created_at(set_tz):
    set_tz("America/Los_Angeles")
    assert Note(created_at="").date_str == ""
    assert Note().date_str == ""
