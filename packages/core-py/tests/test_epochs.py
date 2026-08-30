"""Tests for honest ``epochs_trained`` metadata in .soul checkpoints.

``SloughGPTTrainer.save`` must report the number of epochs actually completed
at save time — not the config's target epoch count. A save before any training
step claims zero epochs rather than a fabricated value, and a run that stops
mid-epoch (e.g. on max_steps) reports the completed epochs, never the entered
count.

Additional tests cover helper utilities: TrainerConfig, TextDataset,
_make_json_safe, _parse_training_state_metadata, CheckpointManager,
and _progress_denominator / _format_eta.
"""

import math
import pytest
import numpy as np

from domains.inference.slo_format import load_soul
from domains.training.train_pipeline import (
    SloughGPTTrainer,
    TrainerConfig,
    TextDataset,
    _make_json_safe,
    _parse_training_state_metadata,
    CheckpointManager,
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
        batch_size=4,
        epochs=3,
        max_steps=50,
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


# ── Integration tests (actual training) ────────────────────────────────


def test_epochs_trained_matches_actual_completed_epochs(data_path, tmp_path):
    cfg = tiny_config(tmp_path)
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    steps_per_epoch = len(t.train_data) // cfg.block_size // cfg.batch_size
    expected_epochs = min(cfg.epochs, cfg.max_steps // steps_per_epoch)

    assert expected_epochs < cfg.epochs  # run stops before the config target
    assert t._completed_epochs == expected_epochs
    assert t._last_train_loss is not None  # at least one step ran

    profile, _ = load_soul(t._last_checkpoint_path)
    assert profile.epochs_trained == expected_epochs
    assert profile.epochs_trained != cfg.epochs  # not the config default


def test_epochs_trained_is_zero_for_mid_epoch_stop(data_path, tmp_path):
    cfg = tiny_config(tmp_path, max_steps=6)  # stops well inside epoch 0
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    assert t.global_step == 6
    assert t._last_train_loss is not None
    assert t._completed_epochs == 0  # no epoch fully completed

    profile, _ = load_soul(t._last_checkpoint_path)
    assert profile.epochs_trained == 0
    assert profile.epochs_trained != cfg.epochs


def test_epochs_trained_is_zero_before_training(data_path, tmp_path):
    t = SloughGPTTrainer(data_path, config=tiny_config(tmp_path))
    out = str(tmp_path / "fresh")
    t.save(out)

    profile, _ = load_soul(out + ".soul")
    assert profile.epochs_trained == 0


def test_completed_epochs_accumulate_across_resume(data_path, tmp_path):
    cfg1 = tiny_config(tmp_path, max_steps=40)
    t1 = SloughGPTTrainer(data_path, config=cfg1)
    t1.train()

    steps_per_epoch = len(t1.train_data) // cfg1.block_size // cfg1.batch_size
    exp1 = min(cfg1.epochs, 40 // steps_per_epoch)
    p1, _ = load_soul(t1._last_checkpoint_path)
    assert p1.epochs_trained == exp1

    cfg2 = tiny_config(tmp_path, max_steps=80)
    t2 = SloughGPTTrainer(data_path, config=cfg2)
    t2.train(resume=True, resume_path=t1._last_checkpoint_path)

    assert t2.global_step == 80
    exp2 = min(cfg2.epochs, 80 // steps_per_epoch)
    p2, _ = load_soul(t2._last_checkpoint_path)
    assert p2.epochs_trained == exp2
    assert p2.epochs_trained > p1.epochs_trained  # tally carried across resume


def test_completed_epochs_embedded_in_training_state(data_path, tmp_path):
    cfg = tiny_config(tmp_path, max_steps=40)
    t = SloughGPTTrainer(data_path, config=cfg)
    t.train()

    profile, _ = load_soul(t._last_checkpoint_path)
    training = profile.metadata["training_state"]
    steps_per_epoch = len(t.train_data) // cfg.block_size // cfg.batch_size
    assert training["completed_epochs"] == min(cfg.epochs, 40 // steps_per_epoch)
    assert training["step"] == 40


# ── TrainerConfig tests ────────────────────────────────────────────────


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.vocab_size == 256
        assert cfg.n_embed == 64
        assert cfg.n_layer == 2
        assert cfg.n_head == 4
        assert cfg.block_size == 64
        assert cfg.dropout == 0.1
        assert cfg.batch_size == 32
        assert cfg.epochs == 10
        assert cfg.max_steps is None
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 0.01
        assert cfg.max_grad_norm == 1.0
        assert cfg.scheduler_type == "cosine"
        assert cfg.warmup_steps == 100
        assert cfg.min_lr == 1e-5
        assert cfg.device == "cpu"

    def test_custom_values(self):
        cfg = TrainerConfig(vocab_size=512, n_embed=128, n_layer=4, n_head=8)
        assert cfg.vocab_size == 512
        assert cfg.n_embed == 128
        assert cfg.n_layer == 4
        assert cfg.n_head == 8

    def test_auto_device_resolves_to_cpu(self):
        cfg = TrainerConfig(device="auto")
        cfg.__post_init__()
        assert cfg.device == "cpu"

    def test_lora_defaults(self):
        cfg = TrainerConfig()
        assert cfg.use_lora is False
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16

    def test_checkpoint_defaults(self):
        cfg = TrainerConfig()
        assert cfg.save_best_only is False
        assert cfg.max_checkpoints == 5
        assert cfg.checkpoint_interval == 500

    def test_early_stopping_default(self):
        cfg = TrainerConfig()
        assert cfg.early_stopping_patience == 0

    def test_gradient_accumulation_default(self):
        cfg = TrainerConfig()
        assert cfg.gradient_accumulation_steps == 1

    def test_device_cpu_unchanged(self):
        cfg = TrainerConfig(device="cpu")
        cfg.__post_init__()
        assert cfg.device == "cpu"

    def test_max_steps_none_default(self):
        cfg = TrainerConfig()
        assert cfg.max_steps is None

    def test_scheduler_types(self):
        for stype in ["cosine", "linear", "constant"]:
            cfg = TrainerConfig(scheduler_type=stype)
            assert cfg.scheduler_type == stype


# ── TextDataset tests ──────────────────────────────────────────────────


class TestTextDataset:
    def test_len(self):
        data = np.arange(100, dtype=np.int64)
        ds = TextDataset(data, block_size=10)
        assert len(ds) == 90

    def test_len_shorter_than_block(self):
        data = np.arange(5, dtype=np.int64)
        ds = TextDataset(data, block_size=10)
        assert len(ds) == 0

    def test_getitem(self):
        data = np.arange(20, dtype=np.int64)
        ds = TextDataset(data, block_size=5)
        x, y = ds[0]
        np.testing.assert_array_equal(x, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(y, [1, 2, 3, 4, 5])

    def test_getitem_shifted(self):
        data = np.arange(20, dtype=np.int64)
        ds = TextDataset(data, block_size=5)
        x, y = ds[3]
        np.testing.assert_array_equal(x, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(y, [4, 5, 6, 7, 8])

    def test_converts_list_to_array(self):
        ds = TextDataset([1, 2, 3, 4, 5], block_size=2)
        assert isinstance(ds.data, np.ndarray)
        assert ds.data.dtype == np.int64

    def test_empty_data(self):
        ds = TextDataset(np.array([], dtype=np.int64), block_size=5)
        assert len(ds) == 0

    def test_getitem_last_valid(self):
        data = np.arange(20, dtype=np.int64)
        ds = TextDataset(data, block_size=5)
        x, y = ds[14]
        np.testing.assert_array_equal(x, [14, 15, 16, 17, 18])
        np.testing.assert_array_equal(y, [15, 16, 17, 18, 19])

    def test_len_exact_block_size(self):
        data = np.arange(10, dtype=np.int64)
        ds = TextDataset(data, block_size=10)
        assert len(ds) == 0

    def test_len_one_more_than_block(self):
        data = np.arange(11, dtype=np.int64)
        ds = TextDataset(data, block_size=10)
        assert len(ds) == 1

    def test_data_is_int64(self):
        ds = TextDataset([1, 2, 3, 4, 5], block_size=2)
        assert ds.data.dtype == np.int64

    def test_single_element_data(self):
        ds = TextDataset(np.array([42], dtype=np.int64), block_size=5)
        assert len(ds) == 0

    def test_large_block_size(self):
        data = np.arange(100, dtype=np.int64)
        ds = TextDataset(data, block_size=99)
        assert len(ds) == 1

    def test_y_is_x_shifted(self):
        data = np.arange(20, dtype=np.int64)
        ds = TextDataset(data, block_size=5)
        x, y = ds[0]
        np.testing.assert_array_equal(y, x + 1)

    def test_negative_data(self):
        data = np.array([-5, -4, -3, -2, -1, 0, 1, 2], dtype=np.int64)
        ds = TextDataset(data, block_size=3)
        x, y = ds[0]
        np.testing.assert_array_equal(x, [-5, -4, -3])
        np.testing.assert_array_equal(y, [-4, -3, -2])

    def test_data_preserved(self):
        data = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        ds = TextDataset(data, block_size=2)
        np.testing.assert_array_equal(ds.data, data)


# ── _make_json_safe tests ──────────────────────────────────────────────


class TestMakeJsonSafe:
    def test_ndarray(self):
        result = _make_json_safe(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_integer(self):
        result = _make_json_safe(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_float(self):
        result = _make_json_safe(np.float64(3.14))
        assert result == 3.14

    def test_dict(self):
        result = _make_json_safe({"a": np.array([1]), "b": np.int64(2)})
        assert result == {"a": [1], "b": 2}

    def test_list(self):
        result = _make_json_safe([np.array([1]), np.int64(2)])
        assert result == [[1], 2]

    def test_tuple(self):
        result = _make_json_safe((np.array([1]),))
        assert result == [[1]]

    def test_nan(self):
        result = _make_json_safe(float("nan"))
        assert result is None

    def test_inf(self):
        result = _make_json_safe(float("inf"))
        assert result is None

    def test_negative_inf(self):
        result = _make_json_safe(float("-inf"))
        assert result is None

    def test_regular_string(self):
        result = _make_json_safe("hello")
        assert result == "hello"

    def test_regular_int(self):
        result = _make_json_safe(42)
        assert result == 42

    def test_nested_nan(self):
        result = _make_json_safe({"loss": float("nan"), "step": 10})
        assert result["loss"] is None
        assert result["step"] == 10

    def test_nested_list_nan(self):
        result = _make_json_safe([float("nan"), 1.0])
        assert result == [None, 1.0]

    def test_nested_dict_deep(self):
        result = _make_json_safe({"a": {"b": {"c": np.array([1])}}})
        assert result == {"a": {"b": {"c": [1]}}}

    def test_list_of_dicts(self):
        result = _make_json_safe([{"x": np.int64(1)}, {"y": np.float64(2.0)}])
        assert result == [{"x": 1}, {"y": 2.0}]

    def test_bool_passthrough(self):
        result = _make_json_safe(True)
        assert result is True

    def test_none_passthrough(self):
        result = _make_json_safe(None)
        assert result is None

    def test_nested_nan_in_list(self):
        result = _make_json_safe([float("nan"), 1, float("inf")])
        assert result == [None, 1, None]

    def test_np_bool(self):
        result = _make_json_safe(np.bool_(True))
        assert result == True

    def test_complex_number(self):
        result = _make_json_safe(complex(1, 2))
        assert isinstance(result, complex)

    def test_empty_dict(self):
        result = _make_json_safe({})
        assert result == {}

    def test_empty_list(self):
        result = _make_json_safe([])
        assert result == []

    def test_nested_empty_containers(self):
        result = _make_json_safe({"a": [], "b": {}})
        assert result == {"a": [], "b": {}}

    def test_np_float32(self):
        result = _make_json_safe(np.float32(1.5))
        assert result == 1.5

    def test_np_int32(self):
        result = _make_json_safe(np.int32(7))
        assert result == 7
        assert isinstance(result, int)

    def test_string_with_nan(self):
        result = _make_json_safe("not a number")
        assert result == "not a number"


# ── _parse_training_state_metadata tests ───────────────────────────────


class TestParseTrainingStateMetadata:
    def test_empty(self):
        result = _parse_training_state_metadata({})
        assert result["step"] == 0
        assert result["epoch"] == 0
        assert result["accumulation_step"] == 0

    def test_step_and_epoch(self):
        result = _parse_training_state_metadata({
            "training_state": {"step": 100, "epoch": 5}
        })
        assert result["step"] == 100
        assert result["epoch"] == 5

    def test_completed_epochs(self):
        result = _parse_training_state_metadata({
            "training_state": {"completed_epochs": 3}
        })
        assert result["completed_epochs"] == 3

    def test_completed_epochs_absent(self):
        result = _parse_training_state_metadata({"training_state": {}})
        assert "completed_epochs" not in result

    def test_optimizer_state_converts_lists(self):
        result = _parse_training_state_metadata({
            "training_state": {
                "optimizer": {
                    "hyperparameters": {"lr": 0.001},
                    "state": {
                        "param0": {"m": [1.0, 2.0], "v": [3.0, 4.0]}
                    }
                }
            }
        })
        opt = result["optimizer"]
        assert opt["hyperparameters"]["lr"] == 0.001
        assert isinstance(opt["state"]["param0"]["m"], np.ndarray)

    def test_scheduler_passthrough(self):
        sched = {"type": "cosine", "initial_lr": 0.001}
        result = _parse_training_state_metadata({
            "training_state": {"scheduler": sched}
        })
        assert result["scheduler"] == sched

    def test_accumulation_step(self):
        result = _parse_training_state_metadata({
            "training_state": {"accumulation_step": 2}
        })
        assert result["accumulation_step"] == 2

    def test_optimizer_none(self):
        result = _parse_training_state_metadata({
            "training_state": {"optimizer": None}
        })
        assert "optimizer" not in result

    def test_full_metadata(self):
        result = _parse_training_state_metadata({
            "training_state": {
                "step": 100,
                "epoch": 5,
                "completed_epochs": 3,
                "accumulation_step": 2,
            }
        })
        assert result["step"] == 100
        assert result["epoch"] == 5
        assert result["completed_epochs"] == 3
        assert result["accumulation_step"] == 2

    def test_optimizer_with_empty_state(self):
        result = _parse_training_state_metadata({
            "training_state": {
                "optimizer": {
                    "hyperparameters": {"lr": 0.001},
                    "state": {}
                }
            }
        })
        assert result["optimizer"]["state"] == {}

    def test_scheduler_all_fields(self):
        sched = {
            "type": "cosine",
            "initial_lr": 0.001,
            "total_steps": 1000,
            "warmup_steps": 100,
        }
        result = _parse_training_state_metadata({
            "training_state": {"scheduler": sched}
        })
        assert result["scheduler"]["type"] == "cosine"
        assert result["scheduler"]["initial_lr"] == 0.001

    def test_none_values(self):
        result = _parse_training_state_metadata({
            "training_state": {
                "optimizer": None,
                "scheduler": None,
            }
        })
        assert "optimizer" not in result
        assert "scheduler" not in result

    def test_missing_all_fields(self):
        result = _parse_training_state_metadata({})
        assert result["step"] == 0
        assert result["epoch"] == 0
        assert result["accumulation_step"] == 0

    def test_large_step(self):
        result = _parse_training_state_metadata({
            "training_state": {"step": 999999}
        })
        assert result["step"] == 999999

    def test_optimizer_not_dict(self):
        result = _parse_training_state_metadata({
            "training_state": {"optimizer": "invalid"}
        })
        assert "optimizer" not in result

    def test_scheduler_not_dict(self):
        result = _parse_training_state_metadata({
            "training_state": {"scheduler": "invalid"}
        })
        assert "scheduler" not in result

    def test_multiple_optimizer_params(self):
        result = _parse_training_state_metadata({
            "training_state": {
                "optimizer": {
                    "hyperparameters": {"lr": 0.001, "weight_decay": 0.01},
                    "state": {
                        "w1": {"m": [1.0], "v": [2.0]},
                        "w2": {"m": [3.0], "v": [4.0]},
                    }
                }
            }
        })
        assert "w1" in result["optimizer"]["state"]
        assert "w2" in result["optimizer"]["state"]
        assert isinstance(result["optimizer"]["state"]["w1"]["m"], np.ndarray)

    def test_zero_step(self):
        result = _parse_training_state_metadata({
            "training_state": {"step": 0}
        })
        assert result["step"] == 0

    def test_negative_accumulation_step(self):
        result = _parse_training_state_metadata({
            "training_state": {"accumulation_step": -1}
        })
        assert result["accumulation_step"] == -1


# ── CheckpointManager tests ────────────────────────────────────────────


class TestCheckpointManager:
    def test_init_creates_dir(self, tmp_path):
        cp_dir = tmp_path / "checkpoints"
        mgr = CheckpointManager(str(cp_dir))
        assert cp_dir.exists()

    def test_latest_path_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "ckpts"))
        assert mgr.latest_path() is None

    def test_is_resumable_false(self, tmp_path):
        assert CheckpointManager.is_resumable(str(tmp_path / "nonexistent.soul")) is False

    def test_is_resumable_soul(self, tmp_path):
        p = tmp_path / "model.soul"
        p.write_text("fake")
        assert CheckpointManager.is_resumable(str(p)) is True

    def test_is_resumable_npz(self, tmp_path):
        p = tmp_path / "model.npz"
        p.write_text("fake")
        assert CheckpointManager.is_resumable(str(p)) is True

    def test_is_resumable_unsupported(self, tmp_path):
        p = tmp_path / "model.txt"
        p.write_text("fake")
        assert CheckpointManager.is_resumable(str(p)) is False

    def test_load_from_path_missing(self, tmp_path):
        result = CheckpointManager.load_from_path(str(tmp_path / "missing.soul"))
        assert result is None

    def test_load_from_path_unsupported(self, tmp_path):
        p = tmp_path / "model.pkl"
        p.write_text("fake")
        result = CheckpointManager.load_from_path(str(p))
        assert result is None

    def test_candidates_excludes_tmp(self, tmp_path):
        cp_dir = tmp_path / "ckpts"
        cp_dir.mkdir()
        (cp_dir / "model.soul").write_text("a")
        (cp_dir / "model.tmp").write_text("b")
        (cp_dir / "model.tmp.npz").write_text("c")
        mgr = CheckpointManager(str(cp_dir))
        candidates = mgr._candidates_newest_first()
        names = [p.name for p in candidates]
        assert "model.soul" in names
        assert "model.tmp" not in names
        assert "model.tmp.npz" not in names

    def test_candidates_soul_and_npz(self, tmp_path):
        cp_dir = tmp_path / "ckpts"
        cp_dir.mkdir()
        (cp_dir / "model.soul").write_text("a")
        (cp_dir / "model.npz").write_text("b")
        mgr = CheckpointManager(str(cp_dir))
        candidates = mgr._candidates_newest_first()
        names = [p.name for p in candidates]
        assert "model.soul" in names
        assert "model.npz" in names

    def test_latest_path_with_checkpoints(self, tmp_path):
        cp_dir = tmp_path / "ckpts"
        cp_dir.mkdir()
        (cp_dir / "model.soul").write_text("a")
        mgr = CheckpointManager(str(cp_dir))
        assert mgr.latest_path() is not None

    def test_latest_valid_path_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "empty"))
        assert mgr.latest_valid_path() is None

    def test_load_latest_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "empty"))
        assert mgr.load_latest() is None

    def test_load_latest_with_path_empty(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path / "empty"))
        path, bundle = mgr.load_latest_with_path()
        assert path is None
        assert bundle is None

    def test_checkpoint_dir_created(self, tmp_path):
        cp_dir = tmp_path / "new_dir" / "subdir"
        mgr = CheckpointManager(str(cp_dir))
        assert cp_dir.exists()

    def test_candidates_excludes_all_tmp(self, tmp_path):
        cp_dir = tmp_path / "ckpts"
        cp_dir.mkdir()
        (cp_dir / "a.soul").write_text("a")
        (cp_dir / "b.tmp").write_text("b")
        (cp_dir / "c.tmp.npz").write_text("c")
        (cp_dir / "d.npz").write_text("d")
        mgr = CheckpointManager(str(cp_dir))
        candidates = mgr._candidates_newest_first()
        names = [p.name for p in candidates]
        assert "a.soul" in names
        assert "d.npz" in names
        assert "b.tmp" not in names
        assert "c.tmp.npz" not in names

    def test_is_resumable_directory(self, tmp_path):
        assert CheckpointManager.is_resumable(str(tmp_path)) is False

    def test_candidates_sorted_newest_first(self, tmp_path):
        import time
        cp_dir = tmp_path / "ckpts"
        cp_dir.mkdir()
        (cp_dir / "old.soul").write_text("old")
        time.sleep(0.05)
        (cp_dir / "new.soul").write_text("new")
        mgr = CheckpointManager(str(cp_dir))
        candidates = mgr._candidates_newest_first()
        assert candidates[0].name == "new.soul"
        assert candidates[1].name == "old.soul"

    def test_multiple_checkpoint_dirs(self, tmp_path):
        d1 = tmp_path / "dir1"
        d2 = tmp_path / "dir2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.soul").write_text("a")
        (d2 / "b.soul").write_text("b")
        mgr1 = CheckpointManager(str(d1))
        mgr2 = CheckpointManager(str(d2))
        assert mgr1.latest_path() is not None
        assert mgr2.latest_path() is not None
        assert "a.soul" in mgr1.latest_path()
        assert "b.soul" in mgr2.latest_path()


# ── _format_eta tests ──────────────────────────────────────────────────


class TestFormatEta:
    def test_none(self):
        assert SloughGPTTrainer._format_eta(None) == "--"

    def test_negative(self):
        assert SloughGPTTrainer._format_eta(-5.0) == "--"

    def test_seconds(self):
        assert SloughGPTTrainer._format_eta(30.0) == "30s"

    def test_minutes(self):
        assert SloughGPTTrainer._format_eta(125.0) == "2m 05s"

    def test_hours(self):
        assert SloughGPTTrainer._format_eta(3661.0) == "1h 01m"

    def test_zero(self):
        assert SloughGPTTrainer._format_eta(0.0) == "0s"

    def test_exactly_one_minute(self):
        assert SloughGPTTrainer._format_eta(60.0) == "1m 00s"

    def test_exactly_one_hour(self):
        assert SloughGPTTrainer._format_eta(3600.0) == "1h 00m"

    def test_large_value(self):
        result = SloughGPTTrainer._format_eta(86400.0)
        assert "h" in result

    def test_fractional_seconds(self):
        result = SloughGPTTrainer._format_eta(0.5)
        assert result == "0s"

    def test_59_seconds(self):
        assert SloughGPTTrainer._format_eta(59.0) == "59s"

    def test_59_minutes_59_seconds(self):
        assert SloughGPTTrainer._format_eta(3599.0) == "59m 59s"

    def test_two_hours(self):
        assert SloughGPTTrainer._format_eta(7200.0) == "2h 00m"

    def test_one_second(self):
        assert SloughGPTTrainer._format_eta(1.0) == "1s"

    def test_119_seconds(self):
        assert SloughGPTTrainer._format_eta(119.0) == "1m 59s"


# ── _progress_denominator tests ────────────────────────────────────────


class TestProgressDenominator:
    def _make_trainer_stub(self, epochs=10, max_steps=None):
        class Stub:
            config = TrainerConfig(epochs=epochs, max_steps=max_steps)
        return Stub()

    def test_no_max_steps(self):
        t = self._make_trainer_stub(epochs=10)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=50)
        assert result == 500

    def test_with_max_steps(self):
        t = self._make_trainer_stub(epochs=10, max_steps=30)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=50)
        assert result == 30

    def test_max_steps_exceeds_epoch_budget(self):
        t = self._make_trainer_stub(epochs=2, max_steps=9999)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=10)
        assert result == 20

    def test_zero_steps_per_epoch(self):
        t = self._make_trainer_stub(epochs=5)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=0)
        assert result >= 1

    def test_one_epoch(self):
        t = self._make_trainer_stub(epochs=1)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=100)
        assert result == 100

    def test_large_epochs(self):
        t = self._make_trainer_stub(epochs=100)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=10)
        assert result == 1000

    def test_max_steps_zero(self):
        t = self._make_trainer_stub(epochs=10, max_steps=0)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=50)
        assert result >= 1

    def test_max_steps_one(self):
        t = self._make_trainer_stub(epochs=10, max_steps=1)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=50)
        assert result == 1

    def test_negative_steps_per_epoch(self):
        t = self._make_trainer_stub(epochs=5)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=-10)
        assert result >= 1

    def test_epochs_one_step_one(self):
        t = self._make_trainer_stub(epochs=1, max_steps=1)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=1)
        assert result == 1

    def test_large_max_steps(self):
        t = self._make_trainer_stub(epochs=10, max_steps=1_000_000)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=50)
        assert result == 500

    def test_small_steps_per_epoch(self):
        t = self._make_trainer_stub(epochs=5)
        result = SloughGPTTrainer._progress_denominator(t, steps_per_epoch=1)
        assert result == 5


# ── Additional TrainerConfig tests ────────────────────────────────────────


class TestTrainerConfigExtra:
    def test_all_fields_settable(self):
        cfg = TrainerConfig(
            vocab_size=512, n_embed=128, n_layer=4, n_head=8,
            block_size=256, dropout=0.2, batch_size=64, epochs=20,
            learning_rate=0.001, weight_decay=0.1, max_grad_norm=2.0,
            scheduler_type="linear", warmup_steps=200, min_lr=1e-6,
            checkpoint_dir="/tmp/ckpts", checkpoint_interval=100,
            save_best_only=True, max_checkpoints=3, use_lora=True,
            lora_rank=16, lora_alpha=32, log_interval=5, eval_interval=50,
            early_stopping_patience=10, gradient_accumulation_steps=4,
        )
        assert cfg.vocab_size == 512
        assert cfg.n_embed == 128
        assert cfg.n_layer == 4
        assert cfg.n_head == 8
        assert cfg.block_size == 256
        assert cfg.dropout == 0.2
        assert cfg.batch_size == 64
        assert cfg.epochs == 20
        assert cfg.learning_rate == 0.001
        assert cfg.weight_decay == 0.1
        assert cfg.max_grad_norm == 2.0
        assert cfg.scheduler_type == "linear"
        assert cfg.warmup_steps == 200
        assert cfg.min_lr == 1e-6
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.checkpoint_interval == 100
        assert cfg.save_best_only is True
        assert cfg.max_checkpoints == 3
        assert cfg.use_lora is True
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.log_interval == 5
        assert cfg.eval_interval == 50
        assert cfg.early_stopping_patience == 10
        assert cfg.gradient_accumulation_steps == 4

    def test_device_auto_to_cpu(self):
        cfg = TrainerConfig(device="auto")
        cfg.__post_init__()
        assert cfg.device == "cpu"
