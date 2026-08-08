"""Tests for TrainingController."""
import pytest
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.training import TrainingController


@pytest.fixture
def ctrl(tmp_path):
    return TrainingController(tmp_path)


class TestCreateJob:
    def test_creates_job(self, ctrl):
        job = ctrl.create_job("test-job", {"epochs": 10})
        assert job["name"] == "test-job"
        assert job["status"] == "pending"
        assert "id" in job

    def test_persists_job(self, ctrl):
        job = ctrl.create_job("persist", {"epochs": 5})
        loaded = ctrl.get_job(job["id"])
        assert loaded is not None
        assert loaded["name"] == "persist"

    def test_unique_ids(self, ctrl):
        j1 = ctrl.create_job("a", {})
        j2 = ctrl.create_job("b", {})
        assert j1["id"] != j2["id"]


class TestGetJob:
    def test_get_existing(self, ctrl):
        job = ctrl.create_job("findme", {})
        result = ctrl.get_job(job["id"])
        assert result["name"] == "findme"

    def test_get_nonexistent(self, ctrl):
        assert ctrl.get_job("no-such-id") is None


class TestListJobs:
    def test_empty(self, ctrl):
        assert ctrl.list_jobs() == []

    def test_multiple(self, ctrl):
        ctrl.create_job("a", {})
        ctrl.create_job("b", {})
        result = ctrl.list_jobs()
        assert len(result) == 2


class TestUpdateJobStatus:
    def test_update_status(self, ctrl):
        job = ctrl.create_job("updatable", {})
        ctrl.update_job_status(job["id"], "running")
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "running"

    def test_update_nonexistent(self, ctrl):
        result = ctrl.update_job_status("ghost", "done")
        assert result == {}

    def test_update_with_kwargs(self, ctrl):
        job = ctrl.create_job("extra", {})
        ctrl.update_job_status(job["id"], "done", loss=0.5)
        loaded = ctrl.get_job(job["id"])
        assert loaded["loss"] == 0.5


class TestStartStopJob:
    def test_start_sets_running(self, ctrl):
        job = ctrl.create_job("starter", {})
        ctrl.start_job(job["id"])
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "running"
        assert "started_at" in loaded

    def test_stop_sets_stopped(self, ctrl):
        job = ctrl.create_job("stopper", {})
        ctrl.start_job(job["id"])
        ctrl.stop_job(job["id"])
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "stopped"

    def test_update_status_to_completed(self, ctrl):
        job = ctrl.create_job("completer", {})
        ctrl.update_job_status(job["id"], "completed", loss=1.23)
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "completed"
        assert loaded["loss"] == 1.23

    def test_update_status_to_failed(self, ctrl):
        job = ctrl.create_job("failer", {})
        ctrl.update_job_status(job["id"], "failed", error="OOM")
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "failed"
        assert loaded["error"] == "OOM"


class TestGetCheckpoints:
    def test_returns_list(self, ctrl):
        result = ctrl.get_checkpoints()
        assert isinstance(result, list)

    def test_returns_empty(self, ctrl):
        result = ctrl.get_checkpoints()
        assert result == []


class TestGetTrainingDatasets:
    def test_returns_list(self, ctrl):
        result = ctrl.get_training_datasets()
        assert isinstance(result, list)


class TestCheckpointsFilesystem:
    def test_detects_pt_files(self, tmp_path):
        (tmp_path / "models" / "checkpoints").mkdir(parents=True)
        ckpt = tmp_path / "models" / "checkpoints" / "epoch_3.pt"
        ckpt.write_bytes(b"0" * (1024 * 1024))
        ctrl = TrainingController(tmp_path)
        result = ctrl.get_checkpoints()
        assert len(result) == 1
        assert result[0]["name"] == "epoch_3"
        assert result[0]["size_mb"] > 0.9
        assert "created" in result[0]

    def test_multiple_checkpoints(self, tmp_path):
        (tmp_path / "models" / "checkpoints").mkdir(parents=True)
        for name in ["a", "b", "c"]:
            (tmp_path / "models" / "checkpoints" / f"{name}.pt").write_bytes(b"123")
        ctrl = TrainingController(tmp_path)
        names = {c["name"] for c in ctrl.get_checkpoints()}
        assert names == {"a", "b", "c"}

    def test_ignores_non_pt_files(self, tmp_path):
        (tmp_path / "models" / "checkpoints").mkdir(parents=True)
        (tmp_path / "models" / "checkpoints" / "a.pt").write_bytes(b"x")
        (tmp_path / "models" / "checkpoints" / "b.bin").write_bytes(b"x")
        (tmp_path / "models" / "checkpoints" / "notes.txt").write_text("x")
        ctrl = TrainingController(tmp_path)
        assert len(ctrl.get_checkpoints()) == 1


class TestGetTrainingDatasetsFilesystem:
    def test_detects_dataset_dirs(self, tmp_path):
        ds = tmp_path / "data" / "features" / "shakespeare"
        ds.mkdir(parents=True)
        (ds / "train.jsonl").write_text("{}")
        (ds / "val.jsonl").write_text("{}")
        (tmp_path / "data" / "features" / "plain").mkdir(parents=True)
        ctrl = TrainingController(tmp_path)
        result = ctrl.get_training_datasets()
        ds_entry = [d for d in result if d["name"] == "shakespeare"]
        assert len(ds_entry) == 1
        assert ds_entry[0]["file_count"] == 2

    def test_empty_features_dir(self, tmp_path):
        (tmp_path / "data" / "features").mkdir(parents=True)
        ctrl = TrainingController(tmp_path)
        assert ctrl.get_training_datasets() == []


class TestJobsPersistence:
    def test_jobs_persist_across_instances(self, tmp_path):
        ctrl1 = TrainingController(tmp_path)
        job = ctrl1.create_job("durable", {"epochs": 3})

        ctrl2 = TrainingController(tmp_path)
        loaded = ctrl2.get_job(job["id"])
        assert loaded is not None
        assert loaded["name"] == "durable"
        assert loaded["config"] == {"epochs": 3}

    def test_created_at_is_set(self, ctrl):
        job = ctrl.create_job("timestamped", {})
        assert "created_at" in job
        assert job["created_at"] != ""

    def test_config_stored(self, ctrl):
        cfg = {"epochs": 20, "lr": 1e-4, "batch_size": 8}
        job = ctrl.create_job("cfg", cfg)
        loaded = ctrl.get_job(job["id"])
        assert loaded["config"] == cfg

    def test_kwargs_overwrite_status(self, ctrl):
        job = ctrl.create_job("overwrite", {})
        ctrl.update_job_status(job["id"], "running", loss=1.0, progress=50)
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "running"
        assert loaded["loss"] == 1.0
        assert loaded["progress"] == 50

    def test_corrupt_jobs_file_returns_empty(self, tmp_path):
        td = tmp_path / "data" / "training"
        td.mkdir(parents=True)
        (td / "jobs.json").write_text("{not valid json")
        ctrl = TrainingController(tmp_path)
        assert ctrl.list_jobs() == []

    def test_job_survives_status_updates(self, ctrl):
        job = ctrl.create_job("lifecycle", {})
        ctrl.update_job_status(job["id"], "running")
        ctrl.update_job_status(job["id"], "completed", loss=0.5)
        ctrl.update_job_status(job["id"], "failed", error="x")
        loaded = ctrl.get_job(job["id"])
        assert loaded["status"] == "failed"
        assert loaded["error"] == "x"
        assert loaded["loss"] == 0.5
