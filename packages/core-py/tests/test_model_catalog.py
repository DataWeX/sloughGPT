"""Tests for the MogDB-backed model catalog."""

from pathlib import Path

import pytest

from domains.infrastructure import model_catalog as mc


@pytest.fixture
def catalog(tmp_path):
    return mc.ModelCatalog(db_path=str(tmp_path / "catalog.db"))


class TestModelCatalogLifecycle:
    def test_add_returns_model_id(self, catalog):
        assert catalog.add("gpt2") == "gpt2"

    def test_add_stores_metadata(self, catalog):
        catalog.add("qwen-0.5b", source="huggingface", format="safetensors", parameters=500_000_000)
        doc = catalog.get("qwen-0.5b")
        assert doc is not None
        assert doc["source"] == "huggingface"
        assert doc["format"] == "safetensors"
        assert doc["parameters"] == 500_000_000
        assert doc["status"] == "available"
        assert doc["inference_count"] == 0
        assert doc["created_at"] is not None
        assert doc["updated_at"] is not None

    def test_add_with_tags_and_extra(self, catalog):
        catalog.add("m1", tags=["chat", "small"], description="desc", quantized=True, quant_bits=8)
        doc = catalog.get("m1")
        assert doc["tags"] == ["chat", "small"]
        assert doc["description"] == "desc"
        assert doc["quantized"] is True
        assert doc["quant_bits"] == 8

    def test_add_preserves_runtime_fields_on_update(self, catalog):
        catalog.add("m1")
        catalog.mark_loaded("m1")
        catalog.record_inference("m1")
        catalog.add("m1", description="new desc", parameters=123)
        doc = catalog.get("m1")
        assert doc["status"] == "loaded"
        assert doc["description"] == "new desc"
        assert doc["parameters"] == 123
        assert doc["inference_count"] == 1

    def test_add_does_not_duplicate(self, catalog):
        catalog.add("m1")
        catalog.add("m1")
        assert catalog.count() == 1

    def test_mark_loaded(self, catalog):
        catalog.add("m1")
        catalog.mark_loaded("m1", device="cuda")
        doc = catalog.get("m1")
        assert doc["status"] == "loaded"
        assert doc["device"] == "cuda"
        assert doc["loaded_at"] is not None
        assert doc["error"] is None

    def test_mark_unloaded(self, catalog):
        catalog.add("m1")
        catalog.mark_loaded("m1")
        catalog.mark_unloaded("m1")
        doc = catalog.get("m1")
        assert doc["status"] == "available"
        assert doc["loaded_at"] is None

    def test_mark_error(self, catalog):
        catalog.add("m1")
        catalog.mark_error("m1", "oom")
        doc = catalog.get("m1")
        assert doc["status"] == "error"
        assert doc["error"] == "oom"

    def test_record_inference_increments(self, catalog):
        catalog.add("m1")
        catalog.record_inference("m1")
        catalog.record_inference("m1")
        doc = catalog.get("m1")
        assert doc["inference_count"] == 2
        assert doc["last_used"] is not None

    def test_get_missing_returns_none(self, catalog):
        assert catalog.get("ghost") is None

    def test_remove_existing(self, catalog):
        catalog.add("m1")
        assert catalog.remove("m1") is True
        assert catalog.get("m1") is None

    def test_remove_missing_returns_false(self, catalog):
        assert catalog.remove("ghost") is False

    def test_count(self, catalog):
        catalog.add("a")
        catalog.add("b")
        assert catalog.count() == 2


class TestModelCatalogQueries:
    def test_list_all(self, catalog):
        catalog.add("a")
        catalog.add("b")
        ids = sorted(m["model_id"] for m in catalog.list_all())
        assert ids == ["a", "b"]

    def test_list_loaded_only(self, catalog):
        catalog.add("a")
        catalog.add("b")
        catalog.mark_loaded("a")
        assert [m["model_id"] for m in catalog.list_loaded()] == ["a"]

    def test_list_available_excludes_loaded(self, catalog):
        catalog.add("a")
        catalog.add("b")
        catalog.mark_loaded("a")
        ids = sorted(m["model_id"] for m in catalog.list_available())
        assert ids == ["b"]

    def test_list_by_source(self, catalog):
        catalog.add("a", source="local")
        catalog.add("b", source="huggingface")
        assert [m["model_id"] for m in catalog.list_by_source("huggingface")] == ["b"]

    def test_list_by_tag(self, catalog):
        catalog.add("a", tags=["code"])
        catalog.add("b", tags=["chat", "code"])
        catalog.add("c", tags=["chat"])
        ids = sorted(m["model_id"] for m in catalog.list_by_tag("code"))
        assert ids == ["a", "b"]

    def test_list_by_tag_empty(self, catalog):
        catalog.add("a")
        assert catalog.list_by_tag("code") == []

    def test_stats(self, catalog):
        catalog.add("a", parameters=100)
        catalog.add("b", parameters=200)
        catalog.mark_loaded("a")
        catalog.record_inference("a")
        catalog.add("c")
        catalog.mark_error("c", "boom")
        s = catalog.stats()
        assert s["total"] == 3
        assert s["loaded"] == 1
        assert s["available"] == 1
        assert s["errors"] == 1
        assert s["total_parameters"] == 100
        assert s["total_inferences"] == 1
        assert set(s["sources"]) == {"local"}

    def test_stats_empty(self, catalog):
        s = catalog.stats()
        assert s["total"] == 0
        assert s["sources"] == []


class TestModelCatalogSync:
    def test_sync_from_hf_cache(self, catalog, tmp_path, monkeypatch):
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--cool-model"
        slnc_dir = model_dir / "snapshots" / "main"
        slnc_dir.mkdir(parents=True)
        (slnc_dir / "model.slnc").write_bytes(b"SLNC")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/cool-model")
        assert doc is not None
        assert doc["source"] == "huggingface"
        assert doc["format"] == "slnc"
        assert doc["path"].endswith("model.slnc")

    def test_sync_from_hf_cache_no_slnc(self, catalog, tmp_path, monkeypatch):
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--safetensors-only"
        (model_dir / "snapshots" / "main").mkdir(parents=True)
        (model_dir / "snapshots" / "main" / "model.safetensors").write_bytes(b"x")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 0

    def test_sync_skips_non_dir_entries(self, catalog, tmp_path, monkeypatch):
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--cool-model"
        (model_dir / "snapshots" / "main").mkdir(parents=True)
        (model_dir / "snapshots" / "main" / "model.slnc").write_bytes(b"SLNC")
        (hf_cache / "models--orphan.txt").write_text("x")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        assert catalog.get("org/cool-model") is not None

    def test_sync_finds_slnc_in_other_snapshot(self, catalog, tmp_path, monkeypatch):
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--v2-model"
        v2_dir = model_dir / "snapshots" / "v2"
        v2_dir.mkdir(parents=True)
        (v2_dir / "model.slnc").write_bytes(b"SLNC")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/v2-model")
        assert doc is not None
        assert doc["path"].endswith("model.slnc")

    def test_sync_from_custom_dirs(self, catalog, tmp_path):
        cache = tmp_path / "models"
        (cache / "sub").mkdir(parents=True)
        (cache / "my-model.slnc").write_bytes(b"SLNC")
        (cache / "sub" / "other.slnc").write_bytes(b"SLNC")
        (cache / "not-a-model.txt").write_text("x")

        assert catalog.sync_from_disk(cache_dirs=[cache]) == 2
        assert catalog.get("my-model") is not None
        assert catalog.get("other") is not None

    def test_sync_skips_duplicates(self, catalog, tmp_path):
        cache = tmp_path / "models"
        cache.mkdir()
        (cache / "dup.slnc").write_bytes(b"SLNC")
        assert catalog.sync_from_disk(cache_dirs=[cache]) == 1
        assert catalog.sync_from_disk(cache_dirs=[cache]) == 0
        assert catalog.count() == 1

    def test_sync_missing_dir(self, catalog, tmp_path):
        assert catalog.sync_from_disk(cache_dirs=[tmp_path / "nope"]) == 0


class TestModelCatalogSingleton:
    def test_get_model_catalog_singleton(self, tmp_path, monkeypatch):
        from domains.infrastructure import model_catalog as mc_mod

        monkeypatch.setattr(mc_mod, "_catalog", None)
        a = mc_mod.get_model_catalog(str(tmp_path / "a.db"))
        b = mc_mod.get_model_catalog()
        assert a is b
        monkeypatch.setattr(mc_mod, "_catalog", None)
