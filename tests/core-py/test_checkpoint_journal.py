"""Tests for domain.training._internal.checkpoint_journal — MogDB checkpoint metadata."""

from pathlib import Path

from domain.training._internal import checkpoint_journal as cj


class TestJournalRoundtrip:
    def test_upsert_and_list(self, tmp_path, monkeypatch):
        """journal_checkpoint upserts; list_journal_checkpoints returns it."""
        monkeypatch.setattr(cj, "_JOURNAL_DB", tmp_path / "journal")
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        npz = ckpt_dir / "best.npz"
        npz.write_bytes(b"fake")

        meta = {"name": "best", "epoch": 2, "step": 50, "loss": 0.25, "vocab_size": 100}
        cj.journal_checkpoint(meta, npz)

        listed = cj.list_journal_checkpoints(ckpt_dir)
        assert listed is not None
        assert len(listed) == 1
        entry = listed[0]
        assert entry["name"] == "best"
        assert entry["epoch"] == 2
        assert entry["step"] == 50
        assert entry["loss"] == 0.25
        assert entry["path"] == str(npz)
        assert entry["size_mb"] >= 0

    def test_upsert_replaces_same_name(self, tmp_path, monkeypatch):
        """Saving the same checkpoint name twice keeps one doc (upsert)."""
        monkeypatch.setattr(cj, "_JOURNAL_DB", tmp_path / "journal")
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        npz = ckpt_dir / "best.npz"
        npz.write_bytes(b"v1")

        cj.journal_checkpoint({"name": "best", "epoch": 1, "loss": 1.0}, npz)
        npz.write_bytes(b"v2-longer")
        cj.journal_checkpoint({"name": "best", "epoch": 2, "loss": 0.5}, npz)

        listed = cj.list_journal_checkpoints(ckpt_dir)
        assert listed is not None
        assert len(listed) == 1
        assert listed[0]["epoch"] == 2
        assert listed[0]["loss"] == 0.5

    def test_sorts_by_epoch_desc(self, tmp_path, monkeypatch):
        """Newest checkpoint first."""
        monkeypatch.setattr(cj, "_JOURNAL_DB", tmp_path / "journal")
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        for name, epoch in [("old", 1), ("new", 3), ("mid", 2)]:
            npz = ckpt_dir / f"{name}.npz"
            npz.write_bytes(b"x")
            cj.journal_checkpoint({"name": name, "epoch": epoch, "loss": 0.1}, npz)

        listed = cj.list_journal_checkpoints(ckpt_dir)
        assert listed is not None
        assert [c["name"] for c in listed] == ["new", "mid", "old"]


class TestDiskValidation:
    def test_deleted_npz_drops_entry_and_prunes_doc(self, tmp_path, monkeypatch):
        """A checkpoint deleted behind the journal's back never resurrects."""
        monkeypatch.setattr(cj, "_JOURNAL_DB", tmp_path / "journal")
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        npz = ckpt_dir / "gone.npz"
        npz.write_bytes(b"x")
        cj.journal_checkpoint({"name": "gone", "epoch": 1, "loss": 1.0}, npz)

        npz.unlink()  # router deletes files directly

        listed = cj.list_journal_checkpoints(ckpt_dir)
        assert listed == []
        # Stale doc pruned from the journal itself.
        coll = cj._collection()
        assert coll.count() == 0

    def test_empty_journal_returns_none(self, tmp_path, monkeypatch):
        """No journal docs → None, so the caller falls back to *_meta.json."""
        monkeypatch.setattr(cj, "_JOURNAL_DB", tmp_path / "journal")
        assert cj.list_journal_checkpoints(tmp_path / "checkpoints") is None

    def test_collection_failure_returns_none(self, monkeypatch):
        """MogDB unavailable → None (file fallback), never raises."""
        monkeypatch.setattr(cj, "_collection", lambda: None)
        assert cj.list_journal_checkpoints(Path("/nonexistent")) is None

    def test_journal_write_is_best_effort(self, tmp_path, monkeypatch):
        """A failing journal never breaks the save path."""
        monkeypatch.setattr(cj, "_collection", lambda: (_ for _ in ()).throw(OSError("boom")))
        cj.journal_checkpoint({"name": "x"}, tmp_path / "x.npz")  # must not raise
