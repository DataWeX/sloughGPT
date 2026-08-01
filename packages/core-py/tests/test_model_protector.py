"""Tests for Model Protector — read-only protection of local model files."""

import os
import stat

import pytest

from domains.infrastructure import model_protector as mp


@pytest.fixture
def hf_home(tmp_path, monkeypatch):
    """Point HF cache at a temp dir."""
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def model_dir(hf_home):
    """Create a fake model cache dir with some weight files."""
    hub = hf_home / "hub"
    d = hub / "models--gpt2"
    d.mkdir(parents=True)
    (d / "model.slnc").write_bytes(b"\x00" * 16)
    (d / "config.json").write_text("{}")
    return d


def _write_bits(mode):
    return mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


class TestProtectModel:
    def test_protects_explicit_files(self, model_dir):
        files = [model_dir / "model.slnc"]
        result = mp.protect_model("gpt2", files)
        assert result["protected"] == [str(model_dir / "model.slnc")]
        assert result["errors"] == []

    def test_chmod_removes_write_bits(self, model_dir):
        files = [model_dir / "model.slnc"]
        mp.protect_model("gpt2", files)
        mode = os.stat(model_dir / "model.slnc").st_mode
        assert _write_bits(mode) == 0

    def test_drops_marker_file(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        assert (model_dir / ".nomodeldelete").exists()

    def test_marker_content(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        content = (model_dir / ".nomodeldelete").read_text()
        assert "Do not delete" in content

    def test_writes_manifest(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        assert (model_dir / ".sloughgpt-protected").exists()
        manifest = mp._read_manifest(model_dir)
        assert manifest is not None
        assert manifest["model_dir"] == str(model_dir)
        assert manifest["protected_files"][0]["path"] == str(model_dir / "model.slnc")
        assert manifest["protected_files"][0]["size"] == 16

    def test_skips_missing_files(self, model_dir):
        result = mp.protect_model("gpt2", [model_dir / "nope.bin"])
        assert result["protected"] == []

    def test_auto_discovers_weight_files(self, model_dir):
        (model_dir / "weights").mkdir()
        (model_dir / "weights" / "extra.safetensors").write_bytes(b"\x00")
        (model_dir / "model.bin").write_bytes(b"\x00")
        (model_dir / "tokenizer.json").write_text("{}")
        result = mp.protect_model("gpt2")
        protected = {os.path.basename(p) for p in result["protected"]}
        assert "model.slnc" in protected
        assert "model.bin" in protected
        assert "extra.safetensors" in protected
        assert "tokenizer.json" in protected
        assert "config.json" in protected

    def test_protect_is_idempotent(self, model_dir):
        files = [model_dir / "model.slnc"]
        mp.protect_model("gpt2", files)
        result = mp.protect_model("gpt2", files)
        assert result["protected"] == [str(model_dir / "model.slnc")]


class TestUnprotectModel:
    def test_restores_write_permission(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        mp.unprotect_model("gpt2")
        mode = os.stat(model_dir / "model.slnc").st_mode
        assert mode & stat.S_IWUSR != 0

    def test_removes_marker(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        mp.unprotect_model("gpt2")
        assert not (model_dir / ".nomodeldelete").exists()

    def test_removes_manifest(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        mp.unprotect_model("gpt2")
        assert not (model_dir / ".sloughgpt-protected").exists()

    def test_returns_unprotected_count(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        result = mp.unprotect_model("gpt2")
        assert result["unprotected"] == 1
        assert result["errors"] == []

    def test_unprotect_after_check_makes_files_deletable(self, model_dir):
        files = [model_dir / "model.slnc"]
        mp.protect_model("gpt2", files)
        mp.unprotect_model("gpt2")
        os.chmod(model_dir / "model.slnc", stat.S_IWUSR | stat.S_IRUSR)
        os.unlink(model_dir / "model.slnc")
        assert not (model_dir / "model.slnc").exists()

    def test_unprotect_without_manifest_is_safe(self, model_dir):
        result = mp.unprotect_model("gpt2")
        assert result["unprotected"] == 0


class TestCheckModel:
    def test_all_present_returns_empty(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        assert mp.check_model("gpt2") == []

    def test_missing_file_reported(self, model_dir):
        files = [model_dir / "model.slnc", model_dir / "config.json"]
        mp.protect_model("gpt2", files)
        os.unlink(model_dir / "model.slnc")
        missing = mp.check_model("gpt2")
        assert missing == [str(model_dir / "model.slnc")]

    def test_no_manifest_returns_empty(self, model_dir):
        assert mp.check_model("gpt2") == []

    def test_corrupt_manifest_returns_empty(self, model_dir):
        (model_dir / ".sloughgpt-protected").write_text("{not json")
        assert mp.check_model("gpt2") == []

    def test_different_model_isolated(self, model_dir):
        mp.protect_model("gpt2", [model_dir / "model.slnc"])
        other = model_dir.parent / "models--other"
        assert mp.check_model("other") == []


class TestListProtected:
    def test_lists_protected_models(self, hf_home):
        hub = hf_home / "hub"
        gpt2 = hub / "models--gpt2"
        gpt2.mkdir(parents=True)
        (gpt2 / "model.slnc").write_bytes(b"\x00")
        qwen = hub / "models--Qwen--Qwen2.5-0.5B-Instruct"
        qwen.mkdir(parents=True)
        (qwen / "model.slnc").write_bytes(b"\x00")
        mp.protect_model("gpt2", [gpt2 / "model.slnc"])
        mp.protect_model("Qwen/Qwen2.5-0.5B-Instruct", [qwen / "model.slnc"])

        protected = {p["model_id"] for p in mp.list_protected()}
        assert protected == {"gpt2", "Qwen/Qwen2.5-0.5B-Instruct"}

    def test_counts_missing_files(self, hf_home):
        hub = hf_home / "hub"
        gpt2 = hub / "models--gpt2"
        gpt2.mkdir(parents=True)
        f = gpt2 / "model.slnc"
        f.write_bytes(b"\x00")
        mp.protect_model("gpt2", [f])
        os.unlink(f)
        entry = mp.list_protected()[0]
        assert entry["missing"] == 1
        assert entry["missing_files"] == [str(f)]

    def test_skips_unprotected_dirs(self, hf_home):
        hub = hf_home / "hub"
        d = hub / "models--gpt2"
        d.mkdir(parents=True)
        (d / "model.slnc").write_bytes(b"\x00")
        assert mp.list_protected() == []

    def test_empty_hub_returns_empty(self, hf_home):
        assert mp.list_protected() == []


class TestPaths:
    def test_get_model_dir_uses_hf_home(self, hf_home):
        d = mp._get_model_dir("gpt2")
        assert str(d) == str(hf_home / "hub" / "models--gpt2")

    def test_get_model_dir_slashes_become_dashes(self, hf_home):
        d = mp._get_model_dir("Qwen/Qwen2.5-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2.5-0.5B-Instruct"
