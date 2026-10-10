"""Tests for scripts/prune_test_fixtures.py.

The safety property under test is narrow but load-bearing: the matcher must
fire on exactly the shape tests mint (``^(test|dup|del)-\\d{13}$``) and on
nothing else.  A looser matcher would silently delete a real record somebody
named ``deltoid`` or ``test-run``, which is why the prefix-match regression
cases below matter more than the happy path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "packages" / "mogdb" / "src"))

import prune_test_fixtures as ptf  # noqa: E402

TS = "1791539561968"  # 13 digits, the shape fixtures are minted in


class TestIsFixture:
    """The matcher — the only thing standing between a prune and real data."""

    @pytest.mark.parametrize("prefix", ["test", "dup", "del"])
    def test_matches_the_three_minted_prefixes(self, prefix: str) -> None:
        assert ptf._is_fixture({"id": f"{prefix}-{TS}"})

    @pytest.mark.parametrize(
        "doc",
        [
            # The docstring's own worked example: a prefix match would eat this.
            {"id": "deltoid"},
            {"id": "testing"},
            {"id": "del"},
            {"id": "test-run"},
            {"id": "del-2026-10-09"},
            # Timestamp shape but wrong width — the 13 is load-bearing.
            {"id": f"test-{TS[:-1]}"},
            {"id": f"test-{TS}0"},
            {"id": f"del-{TS[:8]}"},
            # Wrong case and stray whitespace.
            {"id": f"Test-{TS}"},
            {"id": f"TEST-{TS}"},
            {"id": f" test-{TS}"},
            # No id at all, or a non-string id.
            {},
            {"id": ""},
            {"id": None},
            {"id": 1791539561968},
        ],
    )
    def test_rejects_everything_else(self, doc: dict) -> None:
        """A near-miss must never match — over-matching deletes real data."""
        assert not ptf._is_fixture(doc)


class TestCollections:
    def test_lists_mogdb_stems_sorted(self, tmp_path: Path) -> None:
        for name in ("zeta", "alpha", "mid"):
            (tmp_path / f"{name}.mogdb").touch()
        (tmp_path / "ignore.txt").touch()
        (tmp_path / "alpha.json").touch()

        assert ptf._collections(tmp_path) == ["alpha", "mid", "zeta"]

    def test_discovers_collections_that_have_only_a_journal(self, tmp_path: Path) -> None:
        """Regression: a never-compacted store has a journal but no snapshot.

        Keying on ``*.mogdb`` alone returned ``[]`` here, so prune_store never
        opened the collection and reported "0 fixtures" for a store full of
        them.  The journal is the authoritative append-only log and is fully
        readable via ``collection(name).find({})``.
        """
        (tmp_path / "presets.journal.jsonl").touch()
        (tmp_path / "presets.lock").touch()

        assert ptf._collections(tmp_path) == ["presets"]

    def test_unions_snapshot_and_journal_names(self, tmp_path: Path) -> None:
        (tmp_path / "presets.mogdb").touch()
        (tmp_path / "traits.journal.jsonl").touch()

        assert ptf._collections(tmp_path) == ["presets", "traits"]

    def test_empty_for_a_dir_with_no_stores(self, tmp_path: Path) -> None:
        assert ptf._collections(tmp_path) == []


def _make_store(tmp_path: Path, docs: list[dict], name: str = "fixture") -> Path:
    """Build a real MogDB store containing ``docs`` and return its dir."""
    from mogdb import MogDB

    store = tmp_path / f"{name}_mogdb"
    store.mkdir(parents=True, exist_ok=True)
    db = MogDB(str(store), compact_on_close=False)
    try:
        col = db.collection("presets")
        for d in docs:
            col.insert_one(dict(d))
    finally:
        db.close()
    return store


def _ids_in(store: Path, collection: str = "presets") -> set[str]:
    from mogdb import MogDB

    db = MogDB(str(store), compact_on_close=False)
    try:
        return {str(d.get("id")) for d in db.collection(collection).find({})}
    finally:
        db.close()


class TestPruneStore:
    def test_dry_run_removes_nothing_and_reports_zero(self, tmp_path: Path) -> None:
        store = _make_store(
            tmp_path,
            [{"id": f"test-{TS}"}, {"id": "balanced"}, {"id": "curious"}],
        )

        assert ptf.prune_store(store, apply=False) == 0
        assert _ids_in(store) == {f"test-{TS}", "balanced", "curious"}

    def test_apply_removes_only_fixtures(self, tmp_path: Path) -> None:
        store = _make_store(
            tmp_path,
            [
                {"id": f"test-{TS}"},
                {"id": f"dup-{TS}"},
                {"id": f"del-{TS}"},
                {"id": "balanced"},
                {"id": "curious"},
                {"id": "deltoid"},  # must survive
            ],
        )

        assert ptf.prune_store(store, apply=True) == 3
        assert _ids_in(store) == {"balanced", "curious", "deltoid"}

    def test_apply_resyncs_the_json_mirror(self, tmp_path: Path) -> None:
        """The subtle one the script comments on.

        Opening without ``sync_dir`` leaves the derived ``*_json`` mirror
        holding every deleted doc, and ``_bootstrap_from_json`` would
        resurrect them all if the snapshot were ever rebuilt from it.
        """
        store = _make_store(tmp_path, [{"id": f"test-{TS}"}, {"id": "balanced"}])

        assert ptf.prune_store(store, apply=True) == 1

        mirror = tmp_path / "fixture_json"
        assert mirror.is_dir(), "apply must create the mirror"
        blob = "\n".join(p.read_text() for p in mirror.rglob("*.json"))
        assert "balanced" in blob
        assert f"test-{TS}" not in blob, "mirror must not retain the pruned doc"

    def test_returns_zero_for_a_store_with_no_fixtures(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path, [{"id": "balanced"}])

        assert ptf.prune_store(store, apply=True) == 0
        assert _ids_in(store) == {"balanced"}


class TestMain:
    def test_missing_data_dir_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        argv = ["prune_test_fixtures.py", "--data-dir", str(tmp_path / "nope")]
        old = sys.argv
        sys.argv = argv
        try:
            assert ptf.main() == 1
        finally:
            sys.argv = old
        assert "no data dir" in capsys.readouterr().err

    def test_dry_run_is_the_default_and_writes_nothing(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        store = _make_store(data, [{"id": f"test-{TS}"}, {"id": "balanced"}])
        before = _ids_in(store)

        old = sys.argv
        sys.argv = ["prune_test_fixtures.py", "--data-dir", str(data)]
        try:
            assert ptf.main() == 0
        finally:
            sys.argv = old

        assert _ids_in(store) == before
        assert not list(data.glob("_pre_prune_backup_*")), "dry-run must not back up"

    def test_apply_creates_a_backup_then_prunes(self, tmp_path: Path) -> None:
        data = tmp_path / "data"
        store = _make_store(data, [{"id": f"test-{TS}"}, {"id": "balanced"}])

        old = sys.argv
        sys.argv = ["prune_test_fixtures.py", "--data-dir", str(data), "--apply"]
        try:
            assert ptf.main() == 0
        finally:
            sys.argv = old

        assert _ids_in(store) == {"balanced"}

        backups = list(data.glob("_pre_prune_backup_*"))
        assert len(backups) == 1

        # The backup must retain what we just deleted, so the prune is reversible.
        backup_store = backups[0] / "fixture_mogdb"
        assert backup_store.is_dir()
        blob = "".join(p.read_text() for p in backup_store.iterdir() if p.is_file())
        assert f"test-{TS}" in blob
