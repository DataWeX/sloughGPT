"""Tests for domain.shared timestamp helpers — the canonical time library."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from domain.shared import is_valid_iso, normalize_iso, parse_iso, to_iso, utc_now_iso


class TestUtcNowIso:
    def test_ends_with_z_not_offset(self):
        ts = utc_now_iso()
        assert ts.endswith("Z")
        assert "+00:00" not in ts

    def test_js_date_parses_it(self):
        """The exact bug: '+00:00Z' is rejected by JS new Date()."""
        from datetime import datetime as dt

        ts = utc_now_iso()
        parsed = dt.fromisoformat(ts)  # py3.11+ accepts 'Z'
        assert parsed.tzinfo is not None

    def test_microsecond_precision(self):
        assert len(utc_now_iso().split(".")[1].rstrip("Z")) == 6

    def test_monotonic(self):
        first = utc_now_iso()
        second = utc_now_iso()
        assert second >= first


class TestToIso:
    def test_aware_converted_to_utc_z(self):
        aware = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        assert to_iso(aware) == "2026-09-24T06:30:00.000000Z"

    def test_naive_assumed_utc(self):
        assert to_iso(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05.000000Z"

    def test_already_utc(self):
        assert to_iso(datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)) == "2026-01-02T03:04:05.000000Z"


class TestParseIso:
    @pytest.mark.parametrize(
        "raw",
        [
            "2026-09-24T09:47:33.835728Z",
            "2026-09-24T09:47:33.835728+00:00",
            "2026-09-24T09:47:33.835728+00:00Z",  # legacy bug
            "2026-09-24T09:47:33Z",
            "2026-09-24T15:17:33+05:30",
            "2026-09-24 09:47:33",
            "2026-09-24",
        ],
    )
    def test_accepts_valid_and_legacy_forms(self, raw):
        assert parse_iso(raw) is not None

    @pytest.mark.parametrize(
        "bad", [None, "", "   ", "not-a-date", 1712345678, [], {}, "Invalid Date"]
    )
    def test_rejects_garbage_without_raising(self, bad):
        assert parse_iso(bad) is None

    def test_datetime_passthrough(self):
        naive = datetime(2026, 1, 1)
        assert parse_iso(naive) == naive.replace(tzinfo=UTC)

    def test_legacy_plus_offset_z_keeps_instant(self):
        legacy = parse_iso("2026-09-24T09:47:33.835728+00:00Z")
        good = parse_iso("2026-09-24T09:47:33.835728Z")
        assert legacy == good

    def test_offset_preserved(self):
        parsed = parse_iso("2026-09-24T15:17:33+05:30")
        assert parsed.utcoffset() == timedelta(hours=5, minutes=30)


class TestNormalizeIso:
    def test_repairs_legacy_bug(self):
        assert normalize_iso("2026-09-24T09:47:33.835728+00:00Z") == "2026-09-24T09:47:33.835728Z"

    def test_passes_through_valid(self):
        assert normalize_iso("2026-09-24T09:47:33.835728Z") == "2026-09-24T09:47:33.835728Z"

    def test_rewrites_offset_to_z(self):
        assert normalize_iso("2026-09-24T15:17:33+05:30") == "2026-09-24T09:47:33.000000Z"

    @pytest.mark.parametrize("bad", [None, "", "nope", 42, "Invalid Date"])
    def test_garbage_becomes_empty_string(self, bad):
        assert normalize_iso(bad) == ""

    def test_result_is_always_js_parseable(self):
        for raw in ["2026-09-24T09:47:33.835728+00:00Z", "2026-09-24", "junk", None]:
            out = normalize_iso(raw)
            assert out == "" or (out.endswith("Z") and "+" not in out)


class TestIsValidIso:
    def test_valid(self):
        assert is_valid_iso("2026-09-24T09:47:33Z") is True

    def test_legacy_form_is_repaired_not_rejected(self):
        assert is_valid_iso("2026-09-24T09:47:33+00:00Z") is True

    def test_invalid(self):
        assert is_valid_iso("banana") is False
        assert is_valid_iso(None) is False


class TestCheckpointReadSideRepair:
    """Legacy .soul files store born_at as '...+00:00Z' — must be repaired on read."""

    @staticmethod
    def _write_soul(tmp_path, **meta):
        import json

        fp = tmp_path / "legacy.soul"
        fp.write_bytes(b"SLNP")
        (tmp_path / "legacy.soul.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return fp

    def test_legacy_born_at_repaired(self, tmp_path):
        from domain.training._internal.checkpoints import _load_soul_from_path

        fp = self._write_soul(
            tmp_path,
            soul_name="legacy",
            born_at="2026-09-24T09:47:33.835728+00:00Z",
            created_at="2026-09-24T09:47:33.835728+00:00Z",
        )
        row = _load_soul_from_path(fp)
        assert row["born_at"] == "2026-09-24T09:47:33.835728Z"
        assert row["created_at"] == "2026-09-24T09:47:33.835728Z"

    def test_unparseable_born_at_dropped(self, tmp_path):
        from domain.training._internal.checkpoints import _load_soul_from_path

        fp = self._write_soul(tmp_path, soul_name="broken", born_at="Invalid Date")
        row = _load_soul_from_path(fp)
        assert "born_at" not in row

    def test_valid_born_at_untouched(self, tmp_path):
        from domain.training._internal.checkpoints import _load_soul_from_path

        fp = self._write_soul(tmp_path, soul_name="ok", born_at="2026-09-24T09:47:33Z")
        row = _load_soul_from_path(fp)
        assert row["born_at"] == "2026-09-24T09:47:33.000000Z"
