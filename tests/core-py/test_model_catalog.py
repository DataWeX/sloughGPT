"""Tests for the MogDB-backed model catalog."""

from pathlib import Path

import pytest

from domain.infrastructure._internal import model_catalog as mc


@pytest.fixture
def catalog(tmp_path):
    return mc.ModelCatalog(db_path=str(tmp_path / "catalog.db"))


class TestModelCatalogLifecycle:
    def test_add_returns_model_id(self, catalog):
        assert catalog.add("gpt2") == "gpt2"

    def test_add_defaults_to_not_downloaded(self, catalog):
        """Known id != present on disk; only a disk scan may claim otherwise."""
        catalog.add("some/known-repo")
        assert catalog.get("some/known-repo")["downloaded"] is False

    def test_add_records_downloaded_flag(self, catalog):
        catalog.add("some/local-model", path="/tmp/m.safetensors", downloaded=True)
        assert catalog.get("some/local-model")["downloaded"] is True

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
        assert s["total_parameters"] == 300
        assert s["total_inferences"] == 1
        assert set(s["sources"]) == {"local"}

    def test_stats_empty(self, catalog):
        s = catalog.stats()
        assert s["total"] == 0
        assert s["sources"] == []


class TestModelCatalogSync:
    @pytest.fixture(autouse=True)
    def _empty_home(self, tmp_path, monkeypatch):
        """Never scan the developer's real HF cache.

        The scan is home-rooted, so without this a machine with a 3.8 GB
        model in ``~/.cache`` changes these tests' results — the suite must
        not depend on whose laptop runs it.
        """
        home = tmp_path / "home"
        (home / ".cache" / "huggingface" / "hub").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        # The scan honours HF_HOME (same view as the loader), so an inherited
        # HF_HOME would bypass the isolated home and hit the real cache.
        monkeypatch.delenv("HF_HOME", raising=False)
        return home

    @pytest.fixture(autouse=True)
    def _isolated_home(self, tmp_path, monkeypatch):
        """Never read the developer's real ``~/.cache`` — each test owns a home.

        Without this the HF scan walks the machine's actual cache, so
        results depend on what the developer has downloaded (and the
        assertion counts drift).
        """
        home = tmp_path / "home"
        (home / ".cache" / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        return home

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

    def test_sync_catalogs_complete_safetensors_download(self, catalog, tmp_path, monkeypatch):
        """A finished plain-HF download is a loadable model.

        Regression: sync_from_disk only globbed for ``model.slnc``, so a
        stock ``huggingface-cli download`` left the registry empty
        (``GET /models/catalog`` -> total 0) with the weights sitting on
        disk. Completeness, not file extension, decides.
        """
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--safetensors-only"
        snap = model_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"w" * 2048)
        (snap / "config.json").write_text("{}")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/safetensors-only")
        assert doc is not None
        assert doc["source"] == "huggingface"
        assert doc["format"] == "safetensors"
        assert doc["downloaded"] is True
        assert doc["path"].endswith("model.safetensors")

    def test_sync_prefers_safetensors_over_bin(self, catalog, tmp_path, monkeypatch):
        """Dual-format repo: safetensors wins even when the .bin is larger."""
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--dual-format"
        snap = model_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"s" * 2048)
        (snap / "pytorch_model.bin").write_bytes(b"b" * 8192)
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/dual-format")
        assert doc is not None
        assert doc["format"] == "safetensors"
        assert doc["path"].endswith("model.safetensors")

    def test_sync_skips_incomplete_download(self, catalog, tmp_path, monkeypatch):
        """An interrupted download has a directory but is not a model."""
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--half-downloaded"
        snap = model_dir / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_bytes(b"w" * 2048)
        (snap / "model.safetensors.incomplete").write_bytes(b"")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 0
        assert catalog.get("org/half-downloaded") is None

    def test_sync_skips_refs_only_stub(self, catalog, tmp_path, monkeypatch):
        """Disk-reclaim stub: refs/ kept, snapshots/ deleted. Not loadable.

        The backup dir on this machine
        (``models--Qwen--Qwen2.5-0.5B-Instruct.refs-only-stub.bak-20261027``)
        has exactly this shape: refs but no weights. Cataloguing it would
        report a model that cannot be loaded.
        """
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--stub"
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 0
        assert catalog.get("org/stub") is None

    def test_sync_finds_flat_layout_slnc(self, catalog, tmp_path, monkeypatch):
        """The project-local mirror keeps ``model.slnc`` at the dir root.

        This is the real shape on this machine:
        ``models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc``
        (2.5 GB, no ``snapshots/`` dir at all), symlinked from ``~/.cache``.
        The old scan only looked under ``snapshots/``, so a 3.8 GB model
        catalogued as nothing while the loader happily loaded it.
        """
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--flat"
        model_dir.mkdir(parents=True)
        (model_dir / "model.slnc").write_bytes(b"SLNC")
        (model_dir / "model.safetensors").write_bytes(b"w" * 2048)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/flat")
        assert doc is not None
        assert doc["format"] == "slnc"
        assert doc["path"].endswith("model.slnc")
        assert doc["downloaded"] is True

    def test_sync_honours_hf_home(self, catalog, tmp_path, monkeypatch):
        """Scan the cache the *loader* uses, not a hardcoded home path.

        ``main.py`` points HF_HOME at the shared checkout's mirror, so the
        registry must look there — otherwise the catalog reports zero models
        while the loader happily loads one.
        """
        hf_home = tmp_path / "hf-home"
        model_dir = hf_home / "hub" / "models--org--mirrored"
        model_dir.mkdir(parents=True)
        (model_dir / "model.slnc").write_bytes(b"SLNC")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/mirrored")
        assert doc is not None
        assert doc["format"] == "slnc"
        assert doc["downloaded"] is True

    def test_sync_skips_sub_kilobyte_weights(self, catalog, tmp_path, monkeypatch):
        """A 1-byte placeholder is not a weight file."""
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

    def test_sync_from_custom_dirs(self, catalog, tmp_path, monkeypatch):
        # Isolate from the developer's real HF cache: the scan now catalogues
        # any complete download it finds, so an unpatched home makes this
        # assertion depend on whose machine runs it.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
        cache = tmp_path / "models"
        (cache / "sub").mkdir(parents=True)
        (cache / "my-model.slnc").write_bytes(b"SLNC")
        (cache / "sub" / "other.slnc").write_bytes(b"SLNC")
        (cache / "not-a-model.txt").write_text("x")

        assert catalog.sync_from_disk(cache_dirs=[cache]) == 2
        assert catalog.get("my-model") is not None
        assert catalog.get("other") is not None

    def test_sync_skips_duplicates(self, catalog, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
        cache = tmp_path / "models"
        cache.mkdir()
        (cache / "dup.slnc").write_bytes(b"SLNC")
        assert catalog.sync_from_disk(cache_dirs=[cache]) == 1
        assert catalog.sync_from_disk(cache_dirs=[cache]) == 0
        assert catalog.count() == 1

    def test_sync_missing_dir(self, catalog, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "empty-home"))
        assert catalog.sync_from_disk(cache_dirs=[tmp_path / "nope"]) == 0

    def test_sync_catalogs_flat_layout_slnc(self, catalog, tmp_path, monkeypatch):
        """Project-local mirror shape: model.slnc at the dir root.

        This is the layout actually on disk here —
        ``models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc``
        (2.52 GB) — with no ``snapshots/`` at all. The old scan only looked
        under ``snapshots/``, so a 3.8 GB model catalogued as nothing
        (``GET /models/catalog`` -> total 0).
        """
        hf_cache = tmp_path / "home" / ".cache" / "huggingface" / "hub"
        model_dir = hf_cache / "models--org--flat-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.slnc").write_bytes(b"SLNC")
        (model_dir / "model.safetensors").write_bytes(b"w" * 2048)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "refs").mkdir()
        (model_dir / "refs" / "main").write_text("abc123")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog.sync_from_disk() == 1
        doc = catalog.get("org/flat-model")
        assert doc is not None
        assert doc["format"] == "slnc"
        assert doc["downloaded"] is True
        assert doc["path"].endswith("model.slnc")


class TestModelCatalogSingleton:
    def test_get_model_catalog_singleton(self, tmp_path, monkeypatch):
        from domain.infrastructure._internal import model_catalog as mc_mod

        monkeypatch.setattr(mc_mod, "_catalog", None)
        a = mc_mod.get_model_catalog(str(tmp_path / "a.db"))
        b = mc_mod.get_model_catalog()
        assert a is b
        monkeypatch.setattr(mc_mod, "_catalog", None)
