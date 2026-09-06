"""Tests for training.checkpoints — find_checkpoint, load_soul, list_checkpoints, etc."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from domains.training.state import CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR
from domains.training.checkpoints import (
    find_checkpoint, load_soul, load_lora_soul, _load_soul_from_path,
    _scan_all_checkpoints, list_checkpoints, delete_checkpoint,
    download_checkpoint_path, checkpoint_info, get_all_checkpoint_data,
    export_all_metrics,
)


def _reset_dirs():
    for d in (CHECKPOINTS_DIR, TURBO_DIR, LORA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _make_soul_file(path: Path, content: str = "x" * 5000):
    path.write_text(content)


def _make_soul_with_meta(path: Path, meta: dict):
    path.write_text("x" * 5000)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta))


# ── find_checkpoint ─────────────────────────────────────────────────────────


class TestFindCheckpoint:

    def setup_method(self):
        _reset_dirs()

    def test_find_soul_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "test_ckpt.soul")
        result = find_checkpoint("test_ckpt.soul")
        assert result is not None
        assert result.name == "test_ckpt.soul"

    def test_find_soul_by_name(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.soul")
        result = find_checkpoint("model")
        assert result is not None

    def test_find_slo_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.slo")
        result = find_checkpoint("model.slo")
        assert result is not None

    def test_find_in_turbo_dir(self):
        _make_soul_file(TURBO_DIR / "turbo_ckpt.soul")
        result = find_checkpoint("turbo_ckpt.soul")
        assert result is not None

    def test_not_found(self):
        result = find_checkpoint("nonexistent_model_xyz")
        assert result is None

    def test_no_path_traversal(self):
        _make_soul_file(CHECKPOINTS_DIR / "evil.soul")
        result = find_checkpoint("../../../etc/evil.soul")
        assert result is None


# ── load_soul ───────────────────────────────────────────────────────────────


class TestLoadSoul:

    def setup_method(self):
        _reset_dirs()

    def test_load_with_meta_json(self):
        meta = {"soul_name": "my-custom-soul", "final_train_loss": 0.5}
        _make_soul_with_meta(CHECKPOINTS_DIR / "test.soul", meta)
        result = load_soul("test")
        assert result is not None
        # soul_name "my-custom-soul" → after replace "-soul" → "my-custom" != "test"
        assert result["soul"] == "my-custom"
        assert result["loss"] == 0.5

    def test_load_small_soul_skipped(self):
        _make_soul_file(CHECKPOINTS_DIR / "tiny.soul", "x" * 100)
        result = load_soul("tiny")
        assert result is None

    def test_load_slo_file(self):
        _make_soul_file(CHECKPOINTS_DIR / "model.slo")
        with patch("domains.training.checkpoints.read_slo_json_header", return_value=None):
            result = load_soul("model")
        assert result is not None

    def test_load_not_found(self):
        result = load_soul("nonexistent_xyz")
        assert result is None

    def test_load_from_turbo_dir(self):
        meta = {"soul_name": "turbo-soul"}
        _make_soul_with_meta(TURBO_DIR / "turbo_test.soul", meta)
        result = load_soul("turbo_test")
        assert result is not None

    def test_load_with_traits(self):
        meta = {
            "soul_name": "trait-soul",
            "personality_traits": {"curious": 0.8, "creative": 0.6},
            "lineage": "slonet",
        }
        _make_soul_with_meta(CHECKPOINTS_DIR / "traits.soul", meta)
        result = load_soul("traits")
        assert result["traits"]["curious"] == 0.8

    def test_load_with_soul_name_same_as_stem(self):
        meta = {"soul_name": "test"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "test.soul", meta)
        result = load_soul("test")
        assert result["soul"] == "unknown"


# ── load_lora_soul ──────────────────────────────────────────────────────────


class TestLoadLoraSoul:

    def setup_method(self):
        LORA_DIR.mkdir(parents=True, exist_ok=True)

    def test_load_lora(self):
        meta = {"soul_name": "lora-soul"}
        _make_soul_with_meta(LORA_DIR / "lora_test.soul", meta)
        result = load_lora_soul("lora_test")
        assert result is not None

    def test_load_lora_not_found(self):
        result = load_lora_soul("nonexistent_lora")
        assert result is None


# ── _load_soul_from_path ───────────────────────────────────────────────────


class TestLoadSoulFromPath:

    def setup_method(self):
        _reset_dirs()

    def test_load_no_meta(self):
        path = CHECKPOINTS_DIR / "nometa.soul"
        _make_soul_file(path)
        with patch("domains.training.checkpoints.read_slo_json_header", return_value=None):
            result = _load_soul_from_path(path)
        assert result is not None
        assert result["soul"] == "unknown"

    def test_load_with_created_at(self):
        meta = {"soul_name": "dated", "created_at": "2024-01-01"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "dated.soul", meta)
        result = _load_soul_from_path(CHECKPOINTS_DIR / "dated.soul")
        assert result["created_at"] == "2024-01-01"


# ── async functions ─────────────────────────────────────────────────────────


class TestAsyncFunctions:

    def setup_method(self):
        _reset_dirs()

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        meta = {"soul_name": "list-test"}
        _make_soul_with_meta(CHECKPOINTS_DIR / "list_test.soul", meta)
        result = await list_checkpoints()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_delete_checkpoint_not_found(self):
        result = await delete_checkpoint("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_delete_checkpoint_invalid_name(self):
        with pytest.raises(ValueError, match="Invalid"):
            await delete_checkpoint("../evil")

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        _make_soul_file(CHECKPOINTS_DIR / "del_test.soul")
        result = await delete_checkpoint("del_test.soul")
        assert "del_test.soul" in result
        assert not (CHECKPOINTS_DIR / "del_test.soul").exists()

    @pytest.mark.asyncio
    async def test_download_checkpoint_path_not_found(self):
        result = await download_checkpoint_path("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_checkpoint_path_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            await download_checkpoint_path("../evil")

    @pytest.mark.asyncio
    async def test_download_checkpoint_path(self):
        _make_soul_file(CHECKPOINTS_DIR / "dl_test.soul")
        result = await download_checkpoint_path("dl_test.soul")
        assert result is not None

    @pytest.mark.asyncio
    async def test_checkpoint_info_not_found(self):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            await checkpoint_info("nonexistent")

    @pytest.mark.asyncio
    async def test_get_all_checkpoint_data(self):
        result = await get_all_checkpoint_data()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_export_all_metrics(self):
        result = await export_all_metrics()
        assert "exported_at" in result
        assert "total_checkpoints" in result
        assert "checkpoints" in result
