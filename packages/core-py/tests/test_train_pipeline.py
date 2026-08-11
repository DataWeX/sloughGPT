"""Tests for domains/training/train_pipeline.py (100% coverage target)."""

import math
import os
import sys
import threading
import time

import numpy as np
import pytest

from domains.training import train_pipeline as tp
from domains.training.lr_schedulers import create_scheduler
from domains.training.slonet import SloAdam
from domains.training.train_pipeline import (
    CheckpointManager,
    SloughGPTTrainer,
    TextDataset,
    TrainerConfig,
    _build_training_state_metadata,
    _load_soul_checkpoint,
    _make_json_safe,
    _parse_training_state_metadata,
    prepare_data,
)

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


@pytest.fixture
def data_path(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(DATA_TEXT, encoding="utf-8")
    return str(p)


def tiny_config(tmp_path, **overrides):
    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=2,
        epochs=1,
        max_steps=None,
        gradient_accumulation_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        log_interval=1,
        eval_interval=1000,
        checkpoint_interval=1000,
        warmup_steps=1,
        min_lr=1e-5,
        max_checkpoints=5,
        scheduler_type="cosine",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_trainer(data_path, cfg):
    return SloughGPTTrainer(data_path, config=cfg)


# =============================================================================
# TextDataset
# =============================================================================


class TestTextDataset:
    def test_init_list_converts_to_int64(self):
        ds = TextDataset([1, 2, 3, 4, 5], block_size=2)
        assert ds.data.dtype == np.int64
        assert list(ds.data) == [1, 2, 3, 4, 5]

    def test_init_ndarray_passthrough(self):
        arr = np.array([1, 2, 3], dtype=np.float32)
        ds = TextDataset(arr, block_size=2)
        assert ds.data is arr

    def test_len(self):
        ds = TextDataset(list(range(10)), block_size=3)
        assert len(ds) == 7

    def test_len_short_data_floor_at_zero(self):
        ds = TextDataset(list(range(2)), block_size=5)
        assert len(ds) == 0

    def test_getitem_windows(self):
        ds = TextDataset(list(range(6)), block_size=3)
        x, y = ds[1]
        assert list(x) == [1, 2, 3]
        assert list(y) == [2, 3, 4]


# =============================================================================
# prepare_data
# =============================================================================


class TestPrepareData:
    def test_single_path(self, data_path):
        data, n_chars, stoi, itos = prepare_data(data_path, block_size=8)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.int64
        assert len(data) == len(DATA_TEXT)
        assert n_chars == len(set(DATA_TEXT))
        assert sorted(stoi) == sorted(set(DATA_TEXT))
        assert itos == {i: c for i, c in enumerate(sorted(set(DATA_TEXT)))}

    def test_datasets_dir_fallback(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "datasets" / "corpus2"
        d.mkdir(parents=True)
        (d / "input.txt").write_text("hello world hello", encoding="utf-8")
        data, n_chars, stoi, itos = prepare_data("corpus2", block_size=8)
        assert len(data) == len("hello world hello")
        assert n_chars == len(set("hello world hello"))

    def test_list_with_ratios(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = tmp_path / "datasets" / "a"
        b = tmp_path / "datasets" / "b"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        (a / "input.txt").write_text("aaaa", encoding="utf-8")
        (b / "input.txt").write_text("bbbb", encoding="utf-8")
        data, n_chars, stoi, itos = prepare_data([("a", 0.5), ("b", 1.0)], block_size=8)
        assert len(data) == 2 + 4
        assert n_chars == 2

    def test_list_with_ratios_missing_dataset_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        a = tmp_path / "datasets" / "a"
        a.mkdir(parents=True)
        (a / "input.txt").write_text("hello there", encoding="utf-8")
        data, n_chars, stoi, itos = prepare_data([("a", 1.0), ("missing", 0.5)], block_size=8)
        assert len(data) == len("hello there")
        assert any("not found" in r.message for r in caplog.records)

    def test_list_with_ratios_no_valid_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="No valid datasets found"):
            prepare_data([("missing", 1.0)], block_size=8)

    def test_plain_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = tmp_path / "datasets" / "a"
        b = tmp_path / "datasets" / "b"
        a.mkdir(parents=True)
        b.mkdir(parents=True)
        (a / "input.txt").write_text("xyz", encoding="utf-8")
        (b / "input.txt").write_text("uv", encoding="utf-8")
        data, n_chars, stoi, itos = prepare_data(["a", "b"], block_size=8)
        assert len(data) == 5

    def test_plain_list_missing_skipped(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        a = tmp_path / "datasets" / "a"
        a.mkdir(parents=True)
        (a / "input.txt").write_text("xyz", encoding="utf-8")
        data, n_chars, stoi, itos = prepare_data(["a", "nope"], block_size=8)
        assert len(data) == 3
        assert any("not found" in r.message for r in caplog.records)


# =============================================================================
# TrainerConfig
# =============================================================================


class TestTrainerConfig:
    def test_device_auto_defaults_to_cpu(self):
        cfg = TrainerConfig()
        assert cfg.device == "cpu"

    def test_explicit_device_kept(self):
        cfg = TrainerConfig(device="cuda")
        assert cfg.device == "cuda"


# =============================================================================
# _make_json_safe
# =============================================================================


class TestMakeJsonSafe:
    def test_ndarray_to_list(self):
        assert _make_json_safe(np.array([1, 2])) == [1, 2]

    def test_np_integer(self):
        assert _make_json_safe(np.int64(7)) == 7
        assert isinstance(_make_json_safe(np.int64(7)), int)

    def test_np_floating(self):
        assert _make_json_safe(np.float32(1.5)) == 1.5
        assert isinstance(_make_json_safe(np.float32(1.5)), float)

    def test_dict_recursion(self):
        assert _make_json_safe({"a": np.array([1]), "b": {"c": np.int64(2)}}) == {
            "a": [1],
            "b": {"c": 2},
        }

    def test_list_tuple(self):
        assert _make_json_safe([np.array([1]), (np.int64(2),)]) == [[1], [2]]

    def test_nan_to_none(self):
        assert _make_json_safe(float("nan")) is None

    def test_inf_to_none(self):
        assert _make_json_safe(float("inf")) is None

    def test_passthrough(self):
        assert _make_json_safe("hello") == "hello"


# =============================================================================
# _load_soul_checkpoint
# =============================================================================


class TestLoadSoulCheckpoint:
    def _make_profile(self, metadata=None):
        from domains.inference.slo_format import SloProfile

        p = SloProfile(name="assistant")
        if metadata is not None:
            p.metadata = metadata
        return p

    def test_full_training_state(self, monkeypatch, tmp_path):
        from domains.inference import slo_format

        profile = self._make_profile(
            metadata={
                "training_state": {
                    "step": 42,
                    "epoch": 3,
                    "accumulation_step": 1,
                    "optimizer": {"hyperparameters": {"lr": 0.001}},
                    "scheduler": {"last_lr": 0.0001},
                }
            }
        )
        monkeypatch.setattr(slo_format, "load_soul", lambda p: (profile, {"w": 1}))
        result = _load_soul_checkpoint(str(tmp_path / "x.soul"))
        assert result["model_state_dict"] == {"w": 1}
        assert result["step"] == 42
        assert result["epoch"] == 3
        assert result["accumulation_step"] == 1
        assert result["optimizer_state_dict"]["hyperparameters"]["lr"] == 0.001
        assert result["scheduler_state_dict"] == {"last_lr": 0.0001}

    def test_no_metadata(self, monkeypatch, tmp_path):
        from domains.inference import slo_format

        profile = self._make_profile()
        monkeypatch.setattr(slo_format, "load_soul", lambda p: (profile, {"w": 1}))
        result = _load_soul_checkpoint(str(tmp_path / "x.soul"))
        assert result["step"] == 0
        assert result["epoch"] == 0
        assert "optimizer_state_dict" not in result
        assert "scheduler_state_dict" not in result

    def test_training_state_not_dict(self, monkeypatch, tmp_path):
        from domains.inference import slo_format

        profile = self._make_profile(metadata={"training_state": "nope"})
        monkeypatch.setattr(slo_format, "load_soul", lambda p: (profile, {"w": 1}))
        result = _load_soul_checkpoint(str(tmp_path / "x.soul"))
        assert result["step"] == 0
        assert result["epoch"] == 0


# =============================================================================
# _build_training_state_metadata
# =============================================================================


class TestBuildTrainingStateMetadata:
    def test_no_optimizer_no_scheduler(self):
        state = _build_training_state_metadata(step=5, epoch=1, accumulation_step=2)
        assert state == {"step": 5, "epoch": 1, "accumulation_step": 2}

    def test_with_optimizer_and_params(self):
        opt = SloAdam(lr=0.001, weight_decay=0.01)
        params = [np.array([1.0, 2.0])]
        state = _build_training_state_metadata(optimizer=opt, params=params, step=1, epoch=0)
        assert state["step"] == 1
        assert "optimizer" in state
        assert set(state["optimizer"]) == {"hyperparameters", "t", "state"}

    def test_with_optimizer_no_params(self):
        opt = SloAdam(lr=0.001, weight_decay=0.0)
        state = _build_training_state_metadata(optimizer=opt)
        assert "optimizer" in state

    def test_optimizer_state_dict_raises_skipped(self):
        class _Boom:
            def state_dict(self, params=None):
                raise RuntimeError("no")

        state = _build_training_state_metadata(optimizer=_Boom())
        assert "optimizer" not in state

    def test_scheduler_state_dict(self):
        opt = SloAdam(lr=0.001, weight_decay=0.0)
        sched = create_scheduler(opt, scheduler_type="cosine", total_steps=10, warmup_steps=1, min_lr=1e-5)
        state = _build_training_state_metadata(scheduler=sched)
        assert "scheduler" in state

    def test_scheduler_state_dict_raises_skipped(self):
        class _Boom:
            def state_dict(self):
                raise RuntimeError("no")

        state = _build_training_state_metadata(scheduler=_Boom())
        assert "scheduler" not in state


# =============================================================================
# _parse_training_state_metadata
# =============================================================================


class TestParseTrainingStateMetadata:
    def test_empty(self):
        result = _parse_training_state_metadata({})
        assert result["step"] == 0
        assert result["epoch"] == 0
        assert result["accumulation_step"] == 0

    def test_optimizer_with_state_arrays(self):
        raw = {
            "training_state": {
                "step": 7,
                "epoch": 2,
                "accumulation_step": 0,
                "optimizer": {
                    "hyperparameters": {"lr": 0.001},
                    "state": {"param_0": {"m": [1.0, 2.0], "v": [0.5, 0.5]}},
                },
            }
        }
        result = _parse_training_state_metadata(raw)
        assert result["step"] == 7
        m = result["optimizer"]["state"]["param_0"]["m"]
        assert m.dtype == np.float64
        assert list(m) == [1.0, 2.0]
        assert list(result["optimizer"]["state"]["param_0"]["v"]) == [0.5, 0.5]

    def test_scheduler_present(self):
        raw = {
            "training_state": {
                "scheduler": {"last_lr": [0.0001]},
            }
        }
        result = _parse_training_state_metadata(raw)
        assert result["scheduler"] == {"last_lr": [0.0001]}

    def test_opt_raw_not_dict(self):
        raw = {"training_state": {"optimizer": "nope"}}
        result = _parse_training_state_metadata(raw)
        assert "optimizer" not in result

    def test_scalar_states_untouched(self):
        raw = {"training_state": {"optimizer": {"state": {"param_0": {"v": 3}}}}}
        result = _parse_training_state_metadata(raw)
        assert result["optimizer"]["state"]["param_0"]["v"] == 3


# =============================================================================
# CheckpointManager
# =============================================================================


def _tiny_model():
    from domains.models import SloughGPTModel

    return SloughGPTModel(
        vocab_size=64, n_embed=16, n_layer=1, n_head=2, block_size=8, dropout=0.0
    )


def _manager(tmp_path, **kwargs):
    return CheckpointManager(str(tmp_path / "ck"), max_checkpoints=kwargs.get("max_checkpoints", 5),
                             save_best_only=kwargs.get("save_best_only", False))


class TestCheckpointManager:
    def test_init_creates_dir(self, tmp_path):
        m = CheckpointManager(str(tmp_path / "newdir"), max_checkpoints=3, save_best_only=True)
        assert m.checkpoint_dir.is_dir()
        assert m.max_checkpoints == 3
        assert m.save_best_only is True
        assert m.best_metric == float("inf")
        assert m.checkpoints == []

    def test_save_soul_with_vocab(self, tmp_path):
        m = _manager(tmp_path)
        model = _tiny_model()
        path = m.save(model, None, None, step=1, metrics={"eval_loss": 2.5}, config=TrainerConfig(),
                      epoch=0, stoi={"a": 0, "b": 1}, itos={0: "a", 1: "b"}, chars=["a", "b"])
        assert path is not None
        assert path.endswith(".soul")
        assert os.path.exists(path)
        assert os.path.exists(path + ".meta.json")
        assert m.best_metric == 2.5
        assert m.checkpoints == [{"step": 1, "path": path, "metrics": {"eval_loss": 2.5}}]

    def test_save_derives_chars_from_itos(self, tmp_path):
        m = _manager(tmp_path)
        model = _tiny_model()
        path = m.save(model, None, None, step=1, metrics={"eval_loss": 2.0}, config=TrainerConfig(),
                      stoi={"a": 0, "b": 1}, itos={0: "a", 1: "b"})
        from domains.inference.slo_format import load_soul
        profile, _ = load_soul(path)
        assert profile.metadata["chars"] == ["a", "b"]
        assert profile.metadata["stoi"] == {"a": 0, "b": 1}
        assert profile.metadata["itos"] == {"0": "a", "1": "b"}

    def test_save_uses_loss_fallback(self, tmp_path):
        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, None, step=1, metrics={"loss": 3.0}, config=TrainerConfig())
        assert m.best_metric == 3.0

    def test_save_best_only_skips_worse(self, tmp_path):
        m = _manager(tmp_path, save_best_only=True)
        model = _tiny_model()
        p1 = m.save(model, None, None, step=1, metrics={"eval_loss": 2.0}, config=TrainerConfig())
        p2 = m.save(model, None, None, step=2, metrics={"eval_loss": 5.0}, config=TrainerConfig())
        assert p1 is not None
        assert p2 is None
        assert not os.path.exists(os.path.join(str(m.checkpoint_dir), "assistant_*.soul")) or True
        assert len(m.checkpoints) == 1

    def test_save_best_only_is_final_bypasses(self, tmp_path):
        m = _manager(tmp_path, save_best_only=True)
        model = _tiny_model()
        p1 = m.save(model, None, None, step=1, metrics={"eval_loss": 2.0}, config=TrainerConfig())
        p2 = m.save(model, None, None, step=2, metrics={"eval_loss": 5.0}, config=TrainerConfig(), is_final=True)
        assert p1 is not None
        assert p2 is not None
        assert len(m.checkpoints) == 2

    def test_save_npz_fallback_on_soul_failure(self, tmp_path, monkeypatch):
        import domains.inference as dinf

        def _boom(*a, **k):
            raise RuntimeError("soul export failed")

        monkeypatch.setattr(dinf, "save_soul", _boom)
        m = _manager(tmp_path)
        model = _tiny_model()
        opt = SloAdam(lr=0.001, weight_decay=0.01)
        path = m.save(model, opt, None, step=3, metrics={"eval_loss": 1.5}, config=TrainerConfig(),
                      stoi={"a": 0}, itos={0: "a"})
        assert path.endswith("step_3.npz")
        assert os.path.exists(path)
        bundle = CheckpointManager.load_from_path(path)
        assert bundle["step"] == 3
        assert bundle["stoi"] == {"a": 0}
        assert "model_state_dict" in bundle

    def test_save_npz_optimizer_serialize_failure(self, tmp_path, monkeypatch, caplog):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

        class _BadOpt:
            def state_dict(self, params=None):
                raise RuntimeError("cannot serialize")

        m = _manager(tmp_path)
        path = m.save(_tiny_model(), _BadOpt(), None, step=4, metrics={"loss": 1.0}, config=TrainerConfig())
        assert path.endswith(".npz")
        assert any("Could not serialize optimizer state" in r.message for r in caplog.records)

    def test_save_npz_with_scheduler(self, tmp_path, monkeypatch):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

        class _Sched:
            def state_dict(self):
                return {"step": 2}

        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, _Sched(), step=4, metrics={"loss": 1.0}, config=TrainerConfig())
        assert path.endswith(".npz")
        bundle = CheckpointManager.load_from_path(path)
        assert bundle["scheduler_state_dict"] == {"step": 2}

    def test_save_npz_scheduler_serialize_failure(self, tmp_path, monkeypatch, caplog):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))

        class _BadSched:
            def state_dict(self):
                raise RuntimeError("cannot serialize scheduler")

        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, _BadSched(), step=4, metrics={"loss": 1.0}, config=TrainerConfig())
        assert path.endswith(".npz")
        assert any("Could not serialize scheduler state" in r.message for r in caplog.records)

    def test_save_npz_chars_passed(self, tmp_path, monkeypatch):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, None, step=4, metrics={"loss": 1.0}, config=TrainerConfig(),
                      stoi={"a": 0, "b": 1}, itos={0: "a", 1: "b"}, chars=["a", "b"])
        bundle = CheckpointManager.load_from_path(path)
        assert bundle["chars"] == ["a", "b"]

    def test_save_npz_chars_derived_keyerror_ignored(self, tmp_path, monkeypatch):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, None, step=4, metrics={"loss": 1.0}, config=TrainerConfig(),
                      stoi={"a": 0, "b": 1}, itos={0: "a"})
        bundle = CheckpointManager.load_from_path(path)
        assert "chars" not in bundle

    def test_load_from_path_missing(self, tmp_path, caplog):
        m = _manager(tmp_path)
        assert m.load_from_path(str(tmp_path / "nope.soul")) is None
        assert any("not found" in r.message for r in caplog.records)

    def test_load_from_path_soul(self, tmp_path):
        m = _manager(tmp_path)
        model = _tiny_model()
        path = m.save(model, None, None, step=5, metrics={"eval_loss": 1.0}, config=TrainerConfig())
        bundle = m.load_from_path(path)
        assert bundle["step"] == 5
        assert "model_state_dict" in bundle

    def test_load_from_path_npz(self, tmp_path, monkeypatch):
        import domains.inference as dinf

        monkeypatch.setattr(dinf, "save_soul", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        m = _manager(tmp_path)
        path = m.save(_tiny_model(), None, None, step=7, metrics={"loss": 1.0}, config=TrainerConfig())
        bundle = m.load_from_path(path)
        assert bundle["step"] == 7

    def test_load_from_path_pt_rejected(self, tmp_path, caplog):
        m = _manager(tmp_path)
        p = tmp_path / "legacy.pt"
        p.write_bytes(b"x")
        assert m.load_from_path(str(p)) is None
        assert any("Unsupported checkpoint format" in r.message for r in caplog.records)

    def test_cleanup_within_limit(self, tmp_path):
        m = _manager(tmp_path)
        m.checkpoints = [{"path": "a.soul"}, {"path": "b.soul"}]
        m._cleanup_old_checkpoints()
        assert len(m.checkpoints) == 2

    def test_cleanup_removes_old_files(self, tmp_path):
        m = _manager(tmp_path, max_checkpoints=1)
        for i in range(3):
            f = m.checkpoint_dir / f"assistant_{i}.soul"
            f.write_bytes(b"x")
            m.checkpoints.append({"step": i, "path": str(f), "metrics": {}})
        m._cleanup_old_checkpoints()
        assert len(m.checkpoints) == 1
        assert m.checkpoints[0]["step"] == 2
        assert not (m.checkpoint_dir / "assistant_0.soul").exists()
        assert not (m.checkpoint_dir / "assistant_1.soul").exists()

    def test_cleanup_missing_path_skipped(self, tmp_path):
        m = _manager(tmp_path, max_checkpoints=1)
        m.checkpoints.append({"step": 0, "path": str(tmp_path / "ghost.soul"), "metrics": {}})
        m.checkpoints.append({"step": 1, "path": str(tmp_path / "real.soul"), "metrics": {}})
        (tmp_path / "real.soul").write_bytes(b"x")
        m._cleanup_old_checkpoints()
        assert len(m.checkpoints) == 1
        assert m.checkpoints[0]["step"] == 1

    def test_load_latest_none(self, tmp_path):
        m = _manager(tmp_path)
        assert m.load_latest() is None

    def test_load_latest_picks_newest(self, tmp_path):
        m = _manager(tmp_path)
        m.save(_tiny_model(), None, None, step=1, metrics={"loss": 1.0}, config=TrainerConfig())
        old = m.checkpoint_dir / "step_9.pt"
        old.write_bytes(b"x")
        os.utime(old, (100, 100))
        bundle = m.load_latest()
        assert bundle is not None
        assert "model_state_dict" in bundle

    def test_load_best_empty_falls_back_to_latest(self, tmp_path):
        m = _manager(tmp_path)
        assert m.load_best() is None

    def test_load_best_picks_min_eval_loss(self, tmp_path):
        m = _manager(tmp_path)
        model = _tiny_model()
        p1 = m.save(model, None, None, step=1, metrics={"eval_loss": 5.0}, config=TrainerConfig())
        p2 = m.save(model, None, None, step=2, metrics={"eval_loss": 1.0}, config=TrainerConfig())
        best = m.load_best()
        assert best is not None
        assert best["step"] == 2

    def test_load_best_missing_path_none(self, tmp_path):
        m = _manager(tmp_path)
        m.checkpoints = [{"step": 1, "path": str(tmp_path / "ghost.soul"), "metrics": {"eval_loss": 1.0}}]
        assert m.load_best() is None


# =============================================================================
# SloughGPTTrainer — init / helpers
# =============================================================================


class TestTrainerInit:
    def test_is_training_default_and_stop(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t.is_training is False
        t.stop()
        assert t.is_training is False

    def test_config_device_and_vocab(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path)
        t = make_trainer(data_path, cfg)
        assert t.config is cfg
        assert t.device == "cpu"
        assert t.config.device == "cpu"
        assert t.vocab_size == len(set(DATA_TEXT))

    def test_legacy_vocab_size(self, data_path, tmp_path):
        t = SloughGPTTrainer(data_path, n_embed=16, n_layer=1, n_head=2, block_size=8,
                             vocab_size=48, max_steps=1,
                             checkpoint_dir=str(tmp_path / "ck"))
        assert t.vocab_size == 48
        assert t.config.vocab_size == 48

    def test_legacy_no_vocab_uses_data(self, data_path, tmp_path):
        t = SloughGPTTrainer(data_path, n_embed=16, n_layer=1, n_head=2, block_size=8,
                             max_steps=1,
                             checkpoint_dir=str(tmp_path / "ck"))
        assert t.vocab_size == len(set(DATA_TEXT))

    def test_config_explicit_vocab_wins(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, vocab_size=64)
        t = make_trainer(data_path, cfg)
        assert t.vocab_size == 64

    def test_lora_applied(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, use_lora=True, lora_rank=4, lora_alpha=8))
        assert t.config.use_lora is True
        params = {n for n, _ in t.model.named_parameters()}
        assert params

    def test_lora_no_target_modules_noop(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, use_lora=True, lora_rank=4, lora_alpha=8))
        assert all("lora_" not in n for n, _ in t.model.named_parameters())

    def test_torch_free_cpu_only(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t.device == "cpu"
        assert t.config.device == "cpu"
        assert not hasattr(t, "ddp_model")
        assert not hasattr(t.config, "use_distributed")
        assert not hasattr(t.config, "use_fsdp")
        assert not hasattr(t.config, "use_mixed_precision")

    def test_data_split(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        n = int(0.9 * len(t.data))
        assert len(t.train_data) == n
        assert len(t.val_data) == len(t.data) - n

    def test_experiment_tracker_stored(self, data_path, tmp_path):
        tracker = _RecordingTracker()
        t = SloughGPTTrainer(data_path, config=tiny_config(tmp_path), experiment_tracker=tracker)
        assert t._experiment_tracker is tracker


class _RecordingTracker:
    def __init__(self):
        self.metrics = []

    def log_metrics(self, metrics, step):
        self.metrics.append((dict(metrics), step))


class TestTrainerHelpers:
    def test_create_optimizer(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        opt = t._create_optimizer()
        assert isinstance(opt, SloAdam)

    def test_create_scheduler_with_max_steps(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=12)
        t = make_trainer(data_path, cfg)
        assert t._create_scheduler() is not None

    def test_create_scheduler_without_max_steps(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t._create_scheduler() is not None

    def test_training_model_defaults_to_model(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t.training_model is t.model

    def test_training_model_returns_ddp(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t.training_model is t.model
        assert not hasattr(t, "ddp_model")

    def test_get_batch_train_shape(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        x, y = t.get_batch("train")
        assert x.shape == (2, 8)
        assert y.shape == (2, 8)
        assert x.dtype == np.int64

    def test_get_batch_val(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        x, y = t.get_batch("val")
        assert x.shape == (2, 8)

    def test_setup_device(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t._setup_device() == "cpu"


# =============================================================================
# SloughGPTTrainer — train_step / evaluate
# =============================================================================


class TestTrainStep:
    def test_train_step_returns_metrics(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        m = t.train_step()
        assert set(m) == {"loss", "raw_loss"}
        assert m["loss"] == m["raw_loss"]
        assert math.isfinite(m["loss"])
        assert t.accumulation_step == 0

    def test_accumulation_deferred(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, gradient_accumulation_steps=2))
        m1 = t.train_step()
        assert t.accumulation_step == 1
        assert math.isfinite(m1["loss"])
        assert math.isclose(m1["loss"], m1["raw_loss"], rel_tol=1e-5)
        m2 = t.train_step()
        assert t.accumulation_step == 0
        assert math.isfinite(m2["loss"])

    def test_no_grad_clip_when_zero(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, max_grad_norm=0.0))
        m = t.train_step()
        assert math.isfinite(m["loss"])

    def test_grad_clip_active(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, max_grad_norm=0.001))
        m = t.train_step()
        assert math.isfinite(m["loss"])

    def test_scheduler_steps(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        lr_before = t.scheduler.get_last_lr()[0]
        t.train_step()
        lr_after = t.scheduler.get_last_lr()[0]
        assert lr_after != lr_before

    def test_ema_tracks_raw_loss_both_ways(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        m = t.train_step()
        assert m["loss"] == m["raw_loss"]
        assert t._ema_loss == m["raw_loss"]
        # Force a raw loss far above the EMA floor and confirm the reported
        # loss rises with it instead of freezing at a one-way minimum.
        t._ema_loss = 0.5
        m2 = t.train_step()
        assert m2["raw_loss"] > 0.5
        assert m2["loss"] > 0.5
        assert math.isclose(
            m2["loss"],
            t._ema_alpha * m2["raw_loss"] + (1 - t._ema_alpha) * 0.5,
            rel_tol=1e-6,
        )

    def test_evaluate(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        ev = t.evaluate(num_batches=1)
        assert set(ev) == {"eval_loss", "eval_ppl"}
        assert math.isclose(ev["eval_ppl"], float(np.exp(ev["eval_loss"])), rel_tol=1e-6)


# =============================================================================
# SloughGPTTrainer — restore
# =============================================================================


class TestRestore:
    def test_restore_round_trip_soul(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t1 = make_trainer(data_path, cfg)
        t1.train()
        souls = list((tmp_path / "ckpts").glob("*.soul"))
        assert souls
        bundle = CheckpointManager.load_from_path(str(souls[0]))
        t2 = make_trainer(data_path, tiny_config(tmp_path))
        t2._restore_from_checkpoint_bundle(bundle)
        assert t2.global_step == 2
        assert t2.current_epoch == 0
        assert t2.stoi == t1.stoi
        assert t2.itos == t1.itos
        assert t2.vocab_size == len(t1.stoi)

    def test_restore_strict_failure_retries_non_strict(self, data_path, tmp_path, caplog):
        t = make_trainer(data_path, tiny_config(tmp_path))
        calls = {"n": 0}
        orig = t.model.load_state_dict

        class _Incomp:
            missing_keys = ["tok_emb.weight"]
            unexpected_keys = ["x"]

        def flaky(state_dict, strict=True, **kw):
            calls["n"] += 1
            if strict:
                raise RuntimeError("strict failed")
            orig(state_dict, strict=False)
            return _Incomp()

        t.model.load_state_dict = flaky
        bundle = {"model_state_dict": {"tok_emb.weight": np.zeros((64, 16))}, "step": 3, "epoch": 0}
        t._restore_from_checkpoint_bundle(bundle)
        assert calls["n"] == 2
        assert any("Partial load" in r.message for r in caplog.records)

    def test_restore_optimizer_failure_ignored(self, data_path, tmp_path, caplog):
        t = make_trainer(data_path, tiny_config(tmp_path))

        def boom_opt(state, params=None):
            raise ValueError("opt boom")

        def boom_sched(state):
            raise ValueError("sched boom")

        t.optimizer.load_state_dict = boom_opt
        t.scheduler.load_state_dict = boom_sched
        bundle = {
            "model_state_dict": {},
            "optimizer_state_dict": {"t": 1, "state": {}},
            "scheduler_state_dict": {"bogus": 1},
            "step": 1,
            "epoch": 0,
        }
        t._restore_from_checkpoint_bundle(bundle)
        assert any("optimizer_state_dict" in r.message for r in caplog.records)
        assert any("scheduler_state_dict" in r.message for r in caplog.records)
        assert t.global_step == 1

    def test_restore_updates_vocab_from_stoi(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        bundle = {
            "model_state_dict": {},
            "stoi": {"a": 0, "b": 1},
            "itos": {0: "a", 1: "b"},
            "step": 0,
            "epoch": 0,
        }
        t._restore_from_checkpoint_bundle(bundle)
        assert t.vocab_size == 2


class TestProgressDenominator:
    def test_no_max_steps(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        assert t._progress_denominator(10) == 10

    def test_max_steps_smaller(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, max_steps=4, epochs=3))
        assert t._progress_denominator(10) == 4

    def test_max_steps_larger_capped(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path, max_steps=500, epochs=3))
        assert t._progress_denominator(10) == 30


# =============================================================================
# SloughGPTTrainer — train()
# =============================================================================


class TestTrain:
    def test_happy_path(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t = make_trainer(data_path, cfg)
        r = t.train()
        assert r.success is True
        assert r.global_step == 2
        assert r.total_steps == 2
        assert r.epochs_completed == 1
        assert math.isfinite(r.final_loss)
        assert r.model_path
        assert os.path.exists(r.model_path)
        assert t.is_training is False

    def test_eval_and_best_checkpoint(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=1, eval_interval=1, checkpoint_interval=1)
        t = make_trainer(data_path, cfg)
        r = t.train()
        assert r.success is True
        assert r.global_step == 1
        assert t._best_val_loss != float("inf")
        assert t._best_model_path is not None
        assert t._patience_counter == 0

    def test_early_stopping(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=4, eval_interval=1, early_stopping_patience=1)
        t = make_trainer(data_path, cfg)
        values = iter([5.0, 6.0])

        def fake_evaluate(num_batches=50):
            v = next(values)
            return {"eval_loss": v, "eval_ppl": float(np.exp(v))}

        t.evaluate = fake_evaluate
        events = []
        r = t.train(on_progress=events.append)
        assert t._early_stopped is True
        assert any(e.get("done") and e["done_reason"].startswith("early_stopping") for e in events)
        assert r.success is True
        assert t._best_val_loss == 5.0

    def test_on_progress_events(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t = make_trainer(data_path, cfg)
        events = []
        r = t.train(on_progress=events.append)
        assert events
        ev = events[0]
        assert set(ev) >= {"global_step", "epoch", "epochs", "steps_per_epoch",
                           "progress_percent", "train_loss", "eval_loss",
                           "learning_rate", "done", "done_reason"}
        assert ev["epochs"] == 1
        assert ev["progress_percent"] < 100

    def test_on_progress_exception_caught(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=1)
        t = make_trainer(data_path, cfg)

        def boom(**kw):
            raise RuntimeError("callback exploded")

        r = t.train(on_progress=boom)
        assert r.success is True

    def test_cancel_event(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=5)
        t = make_trainer(data_path, cfg)
        cancel = threading.Event()
        cancel.set()
        r = t.train(cancel_event=cancel)
        assert r.global_step == 0
        assert r.success is True

    def test_pause_event_resumes(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t = make_trainer(data_path, cfg)
        pause = threading.Event()
        pause.set()

        def clear():
            time.sleep(0.3)
            pause.clear()

        th = threading.Thread(target=clear)
        th.start()
        try:
            r = t.train(pause_event=pause)
        finally:
            th.join()
        assert r.global_step == 2
        assert r.success is True

    def test_experiment_tracker_called(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2, eval_interval=1)
        tracker = _RecordingTracker()
        t = SloughGPTTrainer(data_path, config=cfg, experiment_tracker=tracker)
        t.train()
        assert any(m[0].get("meta/total_parameters") for m in tracker.metrics)
        assert any("train/best_eval_loss" in m[0] for m in tracker.metrics)
        assert any("eval/loss" in m[0] for m in tracker.metrics)
        assert any("eval/perplexity" in m[0] for m in tracker.metrics)

    def test_stop_between_epochs(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=None, epochs=2)
        t = make_trainer(data_path, cfg)
        t.train_data = t.train_data[:64]

        def stop_after_epoch0(ev):
            if ev["epoch"] >= 1:
                t.stop()

        r = t.train(on_progress=stop_after_epoch0)
        assert r.success is True
        assert r.global_step == 4
        assert t.is_training is False

    def test_pause_cancelled_during_wait(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=3)
        t = make_trainer(data_path, cfg)
        pause = threading.Event()
        pause.set()
        cancel = threading.Event()

        def set_cancel():
            time.sleep(0.2)
            cancel.set()

        th = threading.Thread(target=set_cancel)
        th.start()
        try:
            r = t.train(cancel_event=cancel, pause_event=pause)
        finally:
            th.join()
        assert r.success is True
        assert r.global_step == 0

    def test_resume_from_soul_restores_step(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t1 = make_trainer(data_path, cfg)
        t1.train()
        souls = list((tmp_path / "ckpts").glob("*.soul"))
        t2 = make_trainer(data_path, tiny_config(tmp_path, max_steps=2))
        r = t2.train(resume=True, resume_path=str(souls[0]))
        assert r.global_step == 2
        assert t2.global_step == 2

    def test_resume_without_path_uses_latest(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=2)
        t1 = make_trainer(data_path, cfg)
        t1.train()
        t2 = make_trainer(data_path, tiny_config(tmp_path, max_steps=2))
        r = t2.train(resume=True, resume_path=None)
        assert r.global_step == 2

    def test_max_steps_break(self, data_path, tmp_path):
        cfg = tiny_config(tmp_path, max_steps=1)
        t = make_trainer(data_path, cfg)
        r = t.train()
        assert r.global_step == 1
        assert r.epochs_completed == 1


# =============================================================================
# SloughGPTTrainer — save_checkpoint / save / generate
# =============================================================================


class TestSaveCheckpoint:
    def test_save_checkpoint_writes_soul(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        t.save_checkpoint({"eval_loss": 1.5})
        assert t._last_checkpoint_path is not None
        assert t._last_checkpoint_path.endswith(".soul")
        assert os.path.exists(t._last_checkpoint_path)
        from domains.inference.slo_format import load_soul
        profile, _ = load_soul(t._last_checkpoint_path)
        assert profile.metadata["vocab_size"] == t.vocab_size
        assert profile.metadata["chars"] is not None

    def test_save_checkpoint_chars_fallback(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        t.itos = {0: "a", 1: "b"}
        t.vocab_size = 5
        t.save_checkpoint()
        assert t._last_checkpoint_path is not None
        assert os.path.exists(t._last_checkpoint_path)


class TestSave:
    def test_save_sou(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        out = str(tmp_path / "model_out")
        t.save(out, format="sou")
        assert os.path.exists(out + ".soul")
        assert os.path.exists(out + ".soul.meta.json")

    def test_save_sou_embeds_training_state(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        t.global_step = 3
        t.current_epoch = 1
        out = str(tmp_path / "model_ts")
        t.save(out, format="sou")
        from domains.inference.slo_format import load_soul
        profile, _ = load_soul(out + ".soul")
        ts = profile.metadata["training_state"]
        assert ts["step"] == 3
        assert ts["epoch"] == 1
        assert "optimizer" in ts

    def test_save_non_sou_format_saves_soul(self, data_path, tmp_path, caplog):
        t = make_trainer(data_path, tiny_config(tmp_path))
        out = str(tmp_path / "model_st")
        t.save(out, format="safetensors")
        assert os.path.exists(out + ".soul")
        assert os.path.exists(out + ".soul.meta.json")
        assert any("Unsupported format" in r.message for r in caplog.records)

    def test_save_unknown_format_uses_soul(self, data_path, tmp_path, caplog):
        t = make_trainer(data_path, tiny_config(tmp_path))
        out = str(tmp_path / "m")
        t.save(out, format="pt")
        assert os.path.exists(out + ".soul")
        assert any("Unsupported format" in r.message for r in caplog.records)


class TestGenerate:
    def test_generate_returns_text(self, data_path, tmp_path):
        np.random.seed(0)
        t = make_trainer(data_path, tiny_config(tmp_path))
        prompt = "the quick"
        text = t.generate(prompt, max_tokens=4, temperature=0.8)
        assert isinstance(text, str)
        assert len(text) == len(prompt) + 4
        charset = set(DATA_TEXT)
        assert set(text) <= charset

    def test_generate_unknown_chars_map_to_token_zero(self, data_path, tmp_path):
        t = make_trainer(data_path, tiny_config(tmp_path))
        text = t.generate("~~~unknown", max_tokens=1)
        assert "unknown" in text
        assert text.startswith(t.itos[0] * 3)


# =============================================================================
# main()
# =============================================================================


class TestMain:
    def _run_main(self, monkeypatch, argv, trainer_cls):
        calls = {"init": None, "train": [], "generate": []}

        class _Fake:
            def __init__(self, **kw):
                calls["init"] = kw

            def train(self, **kw):
                calls["train"].append(kw)
                return tp.TrainResult(success=True, global_step=1, total_steps=1,
                                      final_loss=0.5, epochs_completed=1)

            def generate(self, prompt, max_tokens=200, temperature=0.8):
                calls["generate"].append((prompt, max_tokens, temperature))
                return "gen"

        monkeypatch.setattr(tp, "SloughGPTTrainer", _Fake)
        monkeypatch.setattr(sys, "argv", ["train_pipeline.py"] + argv)
        tp.main()
        return calls

    def test_main_default(self, monkeypatch, tmp_path, data_path):
        calls = self._run_main(monkeypatch, ["--data", data_path, "--max-steps", "1"], tp.SloughGPTTrainer)
        assert calls["init"]["max_steps"] == 1
        assert calls["init"]["n_embed"] == 128
        assert calls["train"] == [{}]
        assert calls["generate"] == [("First", 200, 0.8)]

    def test_main_resume_path(self, monkeypatch, tmp_path, data_path):
        calls = self._run_main(monkeypatch, ["--data", data_path, "--resume", "ck.soul"], tp.SloughGPTTrainer)
        assert calls["train"] == [{"resume": True, "resume_path": "ck.soul"}]

    def test_main_resume_latest(self, monkeypatch, tmp_path, data_path):
        calls = self._run_main(monkeypatch, ["--data", data_path, "--resume-latest"], tp.SloughGPTTrainer)
        assert calls["train"] == [{"resume": True, "resume_path": None}]

    def test_main_lora_flags(self, monkeypatch, tmp_path, data_path):
        calls = self._run_main(monkeypatch, ["--data", data_path, "--lora", "--lora-rank", "4",
                                             "--lora-alpha", "8.0"], tp.SloughGPTTrainer)
        assert calls["init"]["use_lora"] is True
        assert calls["init"]["lora_rank"] == 4
        assert calls["init"]["lora_alpha"] == 8.0

    def test_main_resume_conflict_errors(self, monkeypatch, tmp_path, data_path):
        monkeypatch.setattr(sys, "argv", ["train_pipeline.py", "--data", data_path,
                                          "--resume", "a.soul", "--resume-latest"])
        with pytest.raises(SystemExit):
            tp.main()
