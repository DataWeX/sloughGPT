"""Recovery eligibility + checkpoint-scan scoping.

Regression suite for a job row with no dataset being offered as a "Recoverable
Job". Recovering it spawned a worker that died on the first data load:

    Data file not found: '' (tried '.' and 'data/input.txt')

and, because the job recorded no checkpoint path, the resume scan fell back to
"newest .soul in models/auto-training" — which belonged to a *different* job.
"""

from __future__ import annotations

import os
from pathlib import Path

from training.job_store import JobStore
from training.recovery_policy import (
    NON_RECOVERABLE_TYPES,
    data_path_resolvable,
    job_type,
    recovery_incompatibility,
)


def _job(**overrides) -> dict:
    job: dict = {"id": "job-1", "name": "run", "status": "interrupted", "config": {}}
    job.update(overrides)
    return job


def _corpus(tmp_path: Path) -> Path:
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("hello world\n" * 10, encoding="utf-8")
    return data_file


# ── job_type ────────────────────────────────────────────────────────────────


class TestJobType:
    def test_reads_top_level_marker(self):
        assert job_type(_job(type="distill")) == "distill"

    def test_falls_back_to_config_marker(self):
        assert job_type(_job(config={"type": "lora"})) == "lora"

    def test_empty_for_rows_that_predate_markers(self):
        assert job_type(_job()) == ""


# ── data_path_resolvable ────────────────────────────────────────────────────


class TestDataPathResolvable:
    def test_existing_file(self, tmp_path):
        assert data_path_resolvable(str(_corpus(tmp_path)))

    def test_existing_directory(self, tmp_path):
        (tmp_path / "corpus_dir").mkdir()
        assert data_path_resolvable(str(tmp_path / "corpus_dir"))

    def test_dataset_name_resolves_under_data(self, tmp_path, monkeypatch):
        (tmp_path / "data" / "wikitext").mkdir(parents=True)
        (tmp_path / "data" / "wikitext" / "input.txt").write_text("x", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert data_path_resolvable("wikitext")

    def test_missing_path(self, tmp_path):
        assert not data_path_resolvable(str(tmp_path / "gone.txt"))

    def test_unknown_dataset_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert not data_path_resolvable("no-such-dataset")


# ── recovery_incompatibility ────────────────────────────────────────────────


class TestRecoveryIncompatibility:
    def test_plain_job_with_live_dataset_is_recoverable(self, tmp_path):
        assert recovery_incompatibility(_job(data_path=str(_corpus(tmp_path)))) is None

    def test_empty_data_path_is_rejected(self):
        reason = recovery_incompatibility(_job(data_path=""))
        assert reason is not None
        assert "No dataset recorded" in reason

    def test_missing_data_path_column_is_rejected(self):
        # The exact shape of the phantom row: no data_path key at all.
        reason = recovery_incompatibility(_job(data_path=None))
        assert reason is not None
        assert "No dataset recorded" in reason

    def test_whitespace_only_data_path_is_rejected(self):
        assert recovery_incompatibility(_job(data_path="   ")) is not None

    def test_dangling_data_path_is_rejected(self, tmp_path):
        reason = recovery_incompatibility(_job(data_path=str(tmp_path / "gone.txt")))
        assert reason is not None
        assert "Dataset not found" in reason

    def test_non_recoverable_types_are_rejected(self):
        for marker in sorted(NON_RECOVERABLE_TYPES):
            reason = recovery_incompatibility(_job(type=marker, data_path="whatever"))
            assert reason is not None, marker
            assert f"'{marker}'" in reason

    def test_rows_without_a_marker_are_allowed(self, tmp_path):
        # Legacy rows predate the marker; they used SloughGPTTrainer, so they
        # must stay recoverable.
        assert recovery_incompatibility(_job(data_path=str(_corpus(tmp_path)))) is None


# ── checkpoint scan scoping ─────────────────────────────────────────────────


class TestCheckpointScanScoping:
    def test_candidates_filtered_to_the_jobs_stem(self, tmp_path):
        from domain.training._internal.train_pipeline import CheckpointManager

        (tmp_path / "api_conversations_1790243253.soul").write_bytes(b"other job")
        (tmp_path / "corpus_200.soul").write_bytes(b"mine")
        (tmp_path / "corpus_99.soul").write_bytes(b"mine, older")

        manager = CheckpointManager(str(tmp_path))
        names = [p.name for p in manager._candidates_newest_first("corpus")]
        assert sorted(names) == ["corpus_200.soul", "corpus_99.soul"]
        assert manager._candidates_newest_first("no-such-stem") == []

    def test_unscoped_scan_still_returns_everything(self, tmp_path):
        from domain.training._internal.train_pipeline import CheckpointManager

        (tmp_path / "a.soul").write_bytes(b"1")
        (tmp_path / "b.soul").write_bytes(b"2")

        manager = CheckpointManager(str(tmp_path))
        assert {p.name for p in manager._candidates_newest_first()} == {"a.soul", "b.soul"}

    def test_load_latest_ignores_a_foreign_checkpoint(self, tmp_path):
        from domain.training._internal.train_pipeline import CheckpointManager

        (tmp_path / "api_conversations_1790243253.soul").write_bytes(b"another job's weights")

        manager = CheckpointManager(str(tmp_path))
        assert manager.load_latest_with_path(stem_prefix="corpus") == (None, None)


# ── recoverable list hides what the endpoint would reject ───────────────────


class TestRecoverableListFiltering:
    def test_ineligible_jobs_are_not_offered(self, tmp_path):
        store = JobStore(str(tmp_path / "jobs.db"))

        store.create("good", "run", {}, "ds")
        store.update("good", status="interrupted", data_path=str(_corpus(tmp_path)))

        store.create("phantom", "distill", {})
        store.update("phantom", status="interrupted", data_path="")

        store.create("distill_job", "Distill", {"type": "distill"})
        store.update("distill_job", status="failed", data_path=str(_corpus(tmp_path)))

        offered = {j["id"] for j in store.get_recoverable_jobs()}
        assert offered == {"good"}


# ── tests never touch the repo's job store ──────────────────────────────────


class TestStoreIsolation:
    def test_default_db_path_is_redirected_away_from_the_repo(self):
        env_path = os.environ.get("SLO_TRAINING_JOBS_DB")
        assert env_path, "conftest must redirect the job store for the test session"

        store = JobStore()
        assert str(store.db_path) == env_path
        assert "data" not in store.db_path.parts[-2:-1]
