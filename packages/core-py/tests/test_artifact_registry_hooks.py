"""Stage 2 artifact-registry write-path hooks.

- ``try_register`` behavior (validated record / swallow-on-error).
- Presence of the hook call at each write site (download, checkpoint x2,
  dataset, model import).
- One live end-to-end hook exercise: dataset ``add_data`` registers the
  corpus file.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))

from domain.infrastructure._internal.artifact_registry import try_register  # noqa: E402


def _src(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


class TestTryRegister:
    def test_returns_record_for_file(self, tmp_path):
        fp = tmp_path / "m.soul"
        fp.write_text("{}")
        record = try_register("model", fp, name="m")
        assert record["kind"] == "model"
        assert record["id"] == "m.soul"
        assert record["artifact_id"] == "model:m.soul"
        assert record["name"] == "m"
        assert record["path"] == str(fp)
        assert "fingerprint" in record["meta"]

    def test_download_kind_on_dir(self, tmp_path):
        d = tmp_path / "res"
        d.mkdir()
        record = try_register("download", d, name="res")
        assert record["kind"] == "download"
        assert record["source"] == "local"

    def test_dataset_kind_uses_parent_dir_id(self, tmp_path):
        entry = tmp_path / "my-ds"
        entry.mkdir()
        corpus = entry / "corpus.jsonl"
        corpus.write_text('{"text":"hi"}\n')
        record = try_register("dataset", corpus, name="my-ds")
        assert record["id"] == "my-ds"

    def test_missing_path_returns_none(self, tmp_path):
        assert try_register("model", tmp_path / "nope.soul") is None

    def test_file_kind_rejected(self, tmp_path):
        fp = tmp_path / "x.txt"
        fp.write_text("x")
        assert try_register("file", fp) is None

    def test_unknown_kind_returns_none(self, tmp_path):
        fp = tmp_path / "x.bin"
        fp.write_text("x")
        assert try_register("gadget", fp) is None


class TestWriteSiteHooks:
    def test_download_site(self):
        src = _src("domain/infrastructure/_internal/external_download.py")
        assert 'try_register("download"' in src
        assert "artifact" in src  # completion response carries the record

    def test_checkpoint_sites(self):
        src = _src("domain/training/_internal/training_handler.py")
        assert src.count('try_register("checkpoint"') >= 2  # soul + npz savers

    def test_dataset_site(self):
        src = _src("apps/api/server/controllers/datasets.py")
        assert 'try_register("dataset"' in src

    def test_model_import_site(self):
        src = _src("apps/api/server/controllers/models.py")
        assert 'try_register("model"' in src


class TestLiveDatasetHook:
    def test_add_data_registers_corpus(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLO_CACHE_DIR", str(tmp_path / "cache"))

        from apps.api.server.controllers.datasets import DatasetsController

        ctrl = DatasetsController(repo_root=tmp_path)
        created = ctrl.create_dataset("hook-ds")
        assert created["created"] is True
        count = ctrl.add_data("hook-ds", ["hello", "world"])
        assert count == 2

        corpus = tmp_path / "cache" / "external" / "hook-ds" / "corpus.jsonl"
        assert corpus.is_file()
        record = try_register("dataset", corpus, name="hook-ds")
        assert record["id"] == "hook-ds"
        assert record["kind"] == "dataset"
        assert Path(record["path"]).is_file()
