"""Tests for DatasetsController."""
import pytest
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.datasets import DatasetsController


@pytest.fixture
def tmp_repo(tmp_path):
    return DatasetsController(tmp_path)


@pytest.fixture
def repo_with_datasets(tmp_path):
    ds_dir = tmp_path / "datasets"
    ds_dir.mkdir()
    (ds_dir / "shakespeare").mkdir()
    (ds_dir / "shakespeare" / "input.txt").write_text("hello world")
    (ds_dir / "shakespeare" / "corpus.jsonl").write_text('{"text":"a"}\n{"text":"b"}\n')
    (ds_dir / "code").mkdir()
    (ds_dir / "code" / "input.txt").write_text("def foo(): pass")
    return DatasetsController(tmp_path)


class TestListDatasets:
    def test_empty_dir(self, tmp_repo):
        assert tmp_repo.list_datasets() == []

    def test_returns_datasets(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets()
        names = [d["name"].lower() for d in result]
        assert "shakespeare" in names
        assert "code" in names

    def test_has_required_fields(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets()
        for ds in result:
            assert "name" in ds
            assert "size" in ds
            assert "num_samples" in ds

    def test_search_filter(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets(q="Shakespeare")
        assert len(result) == 1
        assert result[0]["name"] == "Shakespeare"

    def test_search_no_match(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets(q="nonexistent")
        assert len(result) == 0

    def test_corpus_counted(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets()
        shakespeare = next(d for d in result if d["name"] == "Shakespeare")
        assert shakespeare["num_samples"] == 2

    def test_dataset_type_corpus(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets()
        shakespeare = next(d for d in result if d["name"] == "Shakespeare")
        assert shakespeare["type"] == "corpus"

    def test_dataset_type_text(self, repo_with_datasets):
        result = repo_with_datasets.list_datasets()
        code = next(d for d in result if d["name"] == "Code")
        assert code["type"] == "text"

    def test_empty_dir_no_datasets_dir(self, tmp_path):
        ctrl = DatasetsController(tmp_path)
        assert ctrl.list_datasets() == []


class TestGetDataset:
    def test_existing_dataset(self, repo_with_datasets):
        result = repo_with_datasets.get_dataset("shakespeare")
        assert result is not None
        assert result["id"] == "shakespeare"
        assert result["exists"] is True

    def test_nonexistent_dataset(self, repo_with_datasets):
        result = repo_with_datasets.get_dataset("nonexistent")
        assert result is None


class TestGetDatasetStats:
    def test_corpus_stats(self, repo_with_datasets):
        result = repo_with_datasets.get_dataset_stats("shakespeare")
        assert result is not None
        assert result["format"] == "jsonl"
        assert result["samples"] == 2
        assert result["file_type"] == "jsonl"

    def test_text_stats(self, repo_with_datasets):
        result = repo_with_datasets.get_dataset_stats("code")
        assert result is not None
        assert result["format"] == "text"
        assert result["file_type"] == "txt"

    def test_nonexistent_stats(self, repo_with_datasets):
        result = repo_with_datasets.get_dataset_stats("nonexistent")
        assert result is None

    def test_messages_format(self, repo_with_datasets, tmp_path):
        ds_dir = tmp_path / "datasets" / "convo"
        ds_dir.mkdir()
        msgs = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
        ]
        (ds_dir / "corpus.jsonl").write_text("\n".join(json.dumps(m) for m in msgs) + "\n")
        ctrl = DatasetsController(tmp_path)
        result = ctrl.get_dataset_stats("convo")
        assert result["format"] == "messages"
        assert result["has_messages"] is True

    def test_dialogue_format(self, repo_with_datasets, tmp_path):
        ds_dir = tmp_path / "datasets" / "dialog"
        ds_dir.mkdir()
        (ds_dir / "corpus.jsonl").write_text(
            "User: hello\nAssistant: hi there\nUser: how are you\nAssistant: fine\n"
        )
        ctrl = DatasetsController(tmp_path)
        result = ctrl.get_dataset_stats("dialog")
        assert result["format"] == "dialogue"


class TestCreateDataset:
    def test_creates_directory(self, tmp_repo):
        result = tmp_repo.create_dataset("test_ds")
        assert result["created"] is True
        assert result["id"] == "test_ds"
        assert (tmp_repo.datasets_dir / "test_ds").exists()

    def test_creates_with_description(self, tmp_repo):
        result = tmp_repo.create_dataset("test_ds", description="A test dataset")
        assert result["description"] == "A test dataset"

    def test_idempotent_create(self, tmp_repo):
        tmp_repo.create_dataset("test_ds")
        result = tmp_repo.create_dataset("test_ds")
        assert result["created"] is True


class TestUpdateDataset:
    def test_update_description(self, repo_with_datasets):
        result = repo_with_datasets.update_dataset("shakespeare", {"description": "Updated desc"})
        assert result is not None
        assert result["description"] == "Updated desc"

    def test_rename_dataset(self, repo_with_datasets):
        result = repo_with_datasets.update_dataset("shakespeare", {"name": "shakes_v2"})
        assert result is not None
        assert result["name"] == "shakes_v2"
        assert (repo_with_datasets.datasets_dir / "shakes_v2").exists()
        assert not (repo_with_datasets.datasets_dir / "shakespeare").exists()

    def test_rename_to_existing_name(self, repo_with_datasets):
        result = repo_with_datasets.update_dataset("shakespeare", {"name": "code"})
        assert result is None

    def test_nonexistent_dataset(self, repo_with_datasets):
        result = repo_with_datasets.update_dataset("nonexistent", {"description": "x"})
        assert result is None


class TestDeleteDataset:
    def test_delete_existing(self, repo_with_datasets):
        assert repo_with_datasets.delete_dataset("shakespeare") is True
        assert not (repo_with_datasets.datasets_dir / "shakespeare").exists()

    def test_delete_nonexistent(self, repo_with_datasets):
        assert repo_with_datasets.delete_dataset("nonexistent") is False


class TestAddData:
    def test_add_data(self, repo_with_datasets):
        count = repo_with_datasets.add_data("shakespeare", ["line1", "line2", "line3"])
        assert count == 3
        corpus = (repo_with_datasets.datasets_dir / "shakespeare" / "corpus.jsonl")
        lines = [l for l in corpus.read_text().splitlines() if l.strip()]
        assert len(lines) >= 5

    def test_add_data_nonexistent(self, repo_with_datasets):
        result = repo_with_datasets.add_data("nonexistent", ["data"])
        assert result is None


class TestSearchDatasets:
    def test_search(self, repo_with_datasets):
        results = repo_with_datasets.search_datasets("shakes")
        assert "shakespeare" in results

    def test_search_no_match(self, repo_with_datasets):
        results = repo_with_datasets.search_datasets("xyz")
        assert results == []

    def test_search_empty_dir(self, tmp_repo):
        results = tmp_repo.search_datasets("anything")
        assert results == []


class TestPreviewDataset:
    def test_preview_corpus(self, repo_with_datasets):
        result = repo_with_datasets.preview_dataset("shakespeare", limit=1)
        assert result is not None
        assert result["dataset_id"] == "shakespeare"
        assert len(result["samples"]) == 1
        assert result["samples"][0]["language"] == "text"

    def test_preview_text_file(self, repo_with_datasets):
        result = repo_with_datasets.preview_dataset("code", limit=5)
        assert result is not None
        assert len(result["samples"]) == 1

    def test_preview_nonexistent(self, repo_with_datasets):
        result = repo_with_datasets.preview_dataset("nonexistent")
        assert result is None

    def test_preview_visual_dataset(self, repo_with_datasets, tmp_path):
        ds_dir = tmp_path / "datasets" / "vis"
        ds_dir.mkdir()
        entry = {
            "image_path": "/img/test.jpg",
            "conversations": [
                {"from": "human", "value": "What is this?"},
                {"from": "gpt", "value": "A cat sitting on a mat."},
            ]
        }
        (ds_dir / "corpus.jsonl").write_text(json.dumps(entry) + "\n")
        (ds_dir / ".visual_metadata.json").write_text("{}")
        ctrl = DatasetsController(tmp_path)
        result = ctrl.preview_dataset("vis", limit=5)
        assert result is not None
        assert "visual" in result["languages"]
        assert result["samples"][0]["language"] == "visual"


class TestExportDataset:
    def test_export_corpus(self, repo_with_datasets):
        result = repo_with_datasets.export_dataset("shakespeare")
        assert result is not None
        assert result.exists()
        assert result.suffix == ".jsonl"

    def test_export_text_file(self, repo_with_datasets):
        result = repo_with_datasets.export_dataset("code")
        assert result is not None
        assert result.exists()

    def test_export_nonexistent(self, repo_with_datasets):
        result = repo_with_datasets.export_dataset("nonexistent")
        assert result is None

    def test_export_custom_format(self, repo_with_datasets):
        result = repo_with_datasets.export_dataset("shakespeare", format="csv")
        assert result is not None
        assert result.suffix == ".csv"


class TestVersioning:
    def test_create_and_list_versions(self, repo_with_datasets):
        ts = repo_with_datasets.create_version_snapshot("shakespeare")
        assert ts is not None
        versions = repo_with_datasets.list_versions("shakespeare")
        assert ts in versions

    def test_list_versions_empty(self, repo_with_datasets):
        versions = repo_with_datasets.list_versions("shakespeare")
        assert versions == []

    def test_restore_version(self, repo_with_datasets):
        ts = repo_with_datasets.create_version_snapshot("shakespeare")
        result = repo_with_datasets.restore_version("shakespeare", ts)
        assert result is True

    def test_restore_nonexistent_version(self, repo_with_datasets):
        result = repo_with_datasets.restore_version("shakespeare", "99999999")
        assert result is False

    def test_create_version_nonexistent_dataset(self, repo_with_datasets):
        result = repo_with_datasets.create_version_snapshot("nonexistent")
        assert result is None


class TestDescribeDataset:
    def test_describe_small_dataset(self, repo_with_datasets):
        desc = repo_with_datasets._describe_dataset(
            repo_with_datasets.datasets_dir / "code", [], 100
        )
        assert "small dataset" in desc

    def test_describe_dataset_with_content(self, repo_with_datasets, tmp_path):
        ds_dir = tmp_path / "datasets" / "medium"
        ds_dir.mkdir()
        (ds_dir / "input.txt").write_text("hello world " * 200)
        desc = repo_with_datasets._describe_dataset(ds_dir, [], 1024)
        assert "Text" in desc
        assert "words" in desc

    def test_describe_empty_file(self, repo_with_datasets):
        ds_dir = repo_with_datasets.datasets_dir / "empty"
        ds_dir.mkdir()
        desc = repo_with_datasets._describe_dataset(ds_dir, [], 0)
        assert "Dataset" in desc


class TestSingleton:
    def test_singleton_same_instance(self):
        from controllers.datasets import get_datasets_controller
        a = get_datasets_controller()
        b = get_datasets_controller()
        assert a is b
