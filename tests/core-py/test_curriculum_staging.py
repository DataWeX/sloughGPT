"""Curriculum staging: schema, stage mixing, stage-gate judge, single-trainer conductor.

Card 20261115_081635_training-curriculum-staging-recipe-stage-gate-judge-dynamic.

The contract under test:

- ``curriculum.json`` declares ordered stages (datasets + ratios, replay %,
  EWC consolidation, gate criteria) — dynamic, never hard-coded;
- ``run_curriculum`` drives ONE ``SloughGPTTrainer`` through every stage
  (model/optimizer/EWC state carries across boundaries; only data swaps) —
  never N training loops;
- the stage-gate judge (``lm_eval_char`` eval loss + ``quality_scorer``
  sample metrics) decides checkpoint advance: pass → advance, fail →
  per-stage policy (default: warn, mark provisional, advance anyway);
- ``swap_stage_data`` re-tokenizes the new corpus through the EXISTING
  ``stoi`` (unknown chars → token 0) — a corpus-order rebuild would silently
  remap every learned char id, so the vocab must never be rebuilt mid-run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from domain.training._internal import curriculum as cur_mod
from domain.training._internal.curriculum import (
    Curriculum,
    CurriculumError,
    GateCriteria,
    StageGateError,
    StageSpec,
    compute_stage_mix,
    evaluate_stage_gate,
    load_curriculum,
    run_curriculum,
)
from domain.training._internal.train_pipeline import (
    SloughGPTTrainer,
    TrainerConfig,
)

DATA_TEXT = (
    "the quick brown fox jumps over the lazy dog and runs across the meadow "
    "again and again while the dog sleeps soundly in the warm sun all day "
    "long. " * 8
)


def _stage(name: str = "s1", **over):
    stage = {"name": name, "datasets": [{"path": f"data/{name}/input.txt", "ratio": 1.0}]}
    stage.update(over)
    return stage


def _write(tmp_path, doc, name="curriculum.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


# ── Schema ──────────────────────────────────────────────────────────────


class TestLoadCurriculum:
    def test_minimal_valid(self, tmp_path):
        cur = load_curriculum(_write(tmp_path, {"version": 1, "stages": [_stage()]}))
        assert len(cur.stages) == 1
        s = cur.stages[0]
        assert s.name == "s1"
        assert s.epochs == 1
        assert s.replay == 0.0
        assert s.ewc is False
        assert s.gate.is_empty
        assert s.on_gate_fail == "continue"
        assert s.eval_text is None
        assert cur.eval_text is None
        assert cur.stages[0].datasets == (("data/s1/input.txt", 1.0),)

    def test_description_is_exposed(self, tmp_path):
        """JSON has no comments — `description` is how a curriculum marks
        its own placeholder thresholds as placeholders."""
        note = "EXAMPLE: gate thresholds are placeholders; calibrate first."
        cur = load_curriculum(
            _write(tmp_path, {"version": 1, "description": note, "stages": [_stage()]})
        )
        assert cur.description == note

    def test_description_must_be_nonempty_string(self, tmp_path):
        with pytest.raises(CurriculumError, match="description"):
            load_curriculum(
                _write(tmp_path, {"version": 1, "description": 42, "stages": [_stage()]})
            )
        with pytest.raises(CurriculumError, match="description"):
            load_curriculum(
                _write(tmp_path, {"version": 1, "description": "  ", "stages": [_stage()]})
            )

    def test_full_schema(self, tmp_path):
        doc = {
            "version": 1,
            "eval_text": "data/eval.txt",
            "stages": [
                _stage(
                    "a",
                    epochs=5,
                    replay=0.2,
                    ewc=True,
                    gate={"max_eval_loss": 3.5, "min_quality": 2.0, "max_toxicity": 0.1},
                    on_gate_fail="abort",
                    eval_text="data/a_eval.txt",
                ),
                _stage(
                    "b",
                    datasets=[{"path": "x.txt", "ratio": 2.0}, {"path": "y.txt", "ratio": 6.0}],
                ),
            ],
        }
        cur = load_curriculum(_write(tmp_path, doc))
        a = cur.stages[0]
        assert a.epochs == 5
        assert a.replay == 0.2
        assert a.ewc is True
        assert a.gate == GateCriteria(max_eval_loss=3.5, min_quality=2.0, max_toxicity=0.1)
        assert a.on_gate_fail == "abort"
        assert a.eval_text == "data/a_eval.txt"
        assert cur.eval_text == "data/eval.txt"
        b = cur.stages[1]
        assert [r for _, r in b.datasets] == pytest.approx([0.25, 0.75])

    @pytest.mark.parametrize(
        "doc, fragment",
        [
            pytest.param({"version": 1, "stages": []}, "stages", id="empty-stages"),
            pytest.param({"version": 2, "stages": [_stage()]}, "version", id="bad-version"),
            pytest.param(
                {"version": 1, "stages": [_stage()], "stagez": []},
                "stagez",
                id="unknown-top-level",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(epcs=3)]},
                "epcs",
                id="unknown-stage-key",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(gate={"max_eval_loss": 1.0, "min_loss": 2.0})]},
                "min_loss",
                id="unknown-gate-key",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(), _stage()]},
                "duplicate",
                id="duplicate-names",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(datasets=[{"path": "a", "ratio": 0}])]},
                "ratio",
                id="zero-ratio",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(datasets=[{"path": "a", "ratio": -1}])]},
                "ratio",
                id="negative-ratio",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(datasets=[{"path": "a", "ratio": float("inf")}])]},
                "ratio",
                id="infinite-ratio",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(datasets=[])]},
                "datasets",
                id="empty-datasets",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(replay=1.5)]},
                "replay",
                id="replay-too-large",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(replay=-0.1)]},
                "replay",
                id="replay-negative",
            ),
            pytest.param({"version": 1, "stages": [_stage(epochs=0)]}, "epochs", id="zero-epochs"),
            pytest.param(
                {"version": 1, "stages": [_stage(on_gate_fail="explode")]},
                "on_gate_fail",
                id="bad-gate-policy",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(gate={"max_eval_loss": -1.0})]},
                "max_eval_loss",
                id="negative-gate-threshold",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(gate={"max_eval_loss": 2.0})]},
                "eval_text",
                id="eval-loss-gate-without-eval-text",
            ),
            pytest.param(
                {"version": 1, "stages": [_stage(name="")]},
                "name",
                id="empty-name",
            ),
        ],
    )
    def test_rejects(self, tmp_path, doc, fragment):
        with pytest.raises(CurriculumError) as ei:
            load_curriculum(_write(tmp_path, doc))
        assert fragment in str(ei.value)

    def test_missing_file(self, tmp_path):
        with pytest.raises(CurriculumError):
            load_curriculum(str(tmp_path / "nope.json"))

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(CurriculumError):
            load_curriculum(str(p))

    def test_ratio_normalization(self, tmp_path):
        doc = {
            "version": 1,
            "stages": [
                _stage(
                    "a",
                    datasets=[
                        {"path": "x", "ratio": 1.0},
                        {"path": "y", "ratio": 3.0},
                        {"path": "z", "ratio": 4.0},
                    ],
                )
            ],
        }
        cur = load_curriculum(_write(tmp_path, doc))
        ratios = [r for _, r in cur.stages[0].datasets]
        assert ratios == pytest.approx([0.125, 0.375, 0.5])
        assert sum(ratios) == pytest.approx(1.0)


# ── Stage mixing (ratios + replay) ──────────────────────────────────────


class TestComputeStageMix:
    @staticmethod
    def _stage(datasets, replay=0.0):
        return StageSpec(name="s", datasets=tuple(datasets), replay=replay)

    def test_no_replay_own_only(self):
        stage = self._stage([("a.txt", 1.0)])
        assert compute_stage_mix(stage, ["old.txt"]) == [("a.txt", 1.0)]

    def test_replay_splits_over_unique_completed(self):
        stage = self._stage([("a.txt", 1.0)], replay=0.3)
        mix = dict(compute_stage_mix(stage, ["old1.txt", "old2.txt", "old1.txt"]))
        assert mix["a.txt"] == pytest.approx(0.7)
        assert mix["old1.txt"] == pytest.approx(0.15)
        assert mix["old2.txt"] == pytest.approx(0.15)
        assert sum(mix.values()) == pytest.approx(1.0)

    def test_replay_first_stage_is_noop(self):
        stage = self._stage([("a.txt", 1.0)], replay=0.5)
        assert compute_stage_mix(stage, []) == [("a.txt", 1.0)]

    def test_own_ratios_rescaled_with_replay(self):
        stage = self._stage([("a.txt", 3.0), ("b.txt", 1.0)], replay=0.25)
        mix = dict(compute_stage_mix(stage, ["old.txt"]))
        assert mix["a.txt"] == pytest.approx(0.75 * 0.75)
        assert mix["b.txt"] == pytest.approx(0.75 * 0.25)
        assert mix["old.txt"] == pytest.approx(0.25)

    def test_mix_sums_to_one(self):
        stage = self._stage([("a.txt", 2.0), ("b.txt", 6.0)], replay=0.4)
        mix = compute_stage_mix(stage, ["o1", "o2"])
        assert sum(r for _, r in mix) == pytest.approx(1.0)


# ── Stage-gate judge ────────────────────────────────────────────────────


class TestEvaluateStageGate:
    @staticmethod
    def _stage(gate, on_gate_fail="continue"):
        return StageSpec(
            name="s1",
            datasets=(("a.txt", 1.0),),
            gate=gate,
            on_gate_fail=on_gate_fail,
        )

    @pytest.fixture(autouse=True)
    def _stub_metrics(self, monkeypatch):
        self.eval_loss = 1.5
        self.sample = {"avg_quality": 3.5, "toxicity_rate": 0.05}
        monkeypatch.setattr(cur_mod, "_eval_loss", lambda ckpt, eval_text: self.eval_loss)
        monkeypatch.setattr(cur_mod, "_sample_metrics", lambda trainer: self.sample)

    def test_empty_gate_passes_vacuously(self):
        ok, reason, metrics = evaluate_stage_gate(
            MagicMock(), self._stage(GateCriteria()), checkpoint="c.soul", eval_text="e.txt"
        )
        assert ok is True
        assert reason == ""
        assert metrics == {}

    def test_all_criteria_pass(self):
        ok, reason, metrics = evaluate_stage_gate(
            MagicMock(),
            self._stage(GateCriteria(max_eval_loss=2.0, min_quality=2.0, max_toxicity=0.1)),
            checkpoint="c.soul",
            eval_text="e.txt",
        )
        assert ok is True
        assert reason == ""
        assert metrics["eval_loss"] == 1.5
        assert metrics["avg_quality"] == 3.5

    def test_eval_loss_fail(self):
        self.eval_loss = 4.0
        ok, reason, _ = evaluate_stage_gate(
            MagicMock(),
            self._stage(GateCriteria(max_eval_loss=2.0)),
            checkpoint="c.soul",
            eval_text="e.txt",
        )
        assert ok is False
        assert "eval_loss" in reason

    def test_quality_fail(self):
        self.sample = {"avg_quality": 0.5, "toxicity_rate": 0.0}
        ok, reason, _ = evaluate_stage_gate(
            MagicMock(),
            self._stage(GateCriteria(min_quality=2.0)),
            checkpoint="c.soul",
            eval_text="e.txt",
        )
        assert ok is False
        assert "quality" in reason

    def test_toxicity_fail(self):
        self.sample = {"avg_quality": 4.0, "toxicity_rate": 0.9}
        ok, reason, _ = evaluate_stage_gate(
            MagicMock(),
            self._stage(GateCriteria(max_toxicity=0.1)),
            checkpoint="c.soul",
            eval_text="e.txt",
        )
        assert ok is False
        assert "toxicity" in reason

    def test_unmeasurable_sample_fails_honestly(self):
        # No usable model sample → the gate must FAIL, never pass vacuously.
        self.sample = None
        ok, reason, _ = evaluate_stage_gate(
            MagicMock(),
            self._stage(GateCriteria(min_quality=2.0)),
            checkpoint="c.soul",
            eval_text="e.txt",
        )
        assert ok is False
        assert "cannot evaluate" in reason


# ── Runner (one trainer, N stages) ──────────────────────────────────────


def _fake_trainer_factory(created, train_results=None):
    """make_trainer that records every call and returns a MagicMock trainer."""

    def make(data_path, config):
        t = MagicMock()
        t.data_path = data_path
        t.config = config
        idx = len(created)
        result = MagicMock()
        result.model_path = f"ckpt-stage{idx}.soul"
        t.train.return_value = (train_results or {}).get(idx, result)
        created.append(t)
        return t

    return make


class TestRunCurriculum:
    @staticmethod
    def _curriculum(*stages, eval_text=None):
        return Curriculum(stages=tuple(stages), eval_text=eval_text)

    @staticmethod
    def _stage(name, *, epochs=1, replay=0.0, ewc=False, gate=None, on_gate_fail="continue"):
        return StageSpec(
            name=name,
            datasets=((f"data/{name}.txt", 1.0),),
            epochs=epochs,
            replay=replay,
            ewc=ewc,
            gate=gate or GateCriteria(),
            on_gate_fail=on_gate_fail,
        )

    @pytest.fixture(autouse=True)
    def _stub_gate(self, monkeypatch):
        monkeypatch.setattr(
            cur_mod,
            "evaluate_stage_gate",
            lambda trainer, stage, *, checkpoint, eval_text: (True, "", {}),
        )

    def test_single_trainer_instance_across_stages(self):
        created = []
        report = run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b"), self._stage("c")),
            make_trainer=_fake_trainer_factory(created),
        )
        assert len(created) == 1  # ONE trainer, never N
        assert len(created[0].train.call_args_list) == 3
        assert report.completed_all  # all three stages produced verdicts

    def test_swaps_data_between_stages_not_first(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        trainer = created[0]
        # stage 1 constructed with its mix; stage 2 swapped
        assert trainer.swap_stage_data.call_count == 1
        (args, kwargs) = trainer.swap_stage_data.call_args
        mix = args[0] if args else kwargs.get("data_path")
        assert dict(mix)["data/b.txt"] == pytest.approx(1.0)
        assert kwargs.get("epochs") == 1

    def test_config_epochs_follow_first_stage(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a", epochs=7), self._stage("b", epochs=2)),
            make_trainer=_fake_trainer_factory(created),
        )
        assert created[0].config.epochs == 7

    def test_report_records_every_stage_and_final_checkpoint(self):
        created = []
        report = run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        assert [r.name for r in report.stages] == ["a", "b"]
        assert report.final_checkpoint == "ckpt-stage0.soul"
        assert report.completed_all is True
        assert report.cancelled is False

    def test_gate_fail_continue_marks_provisional_and_advances(self, monkeypatch, caplog):
        monkeypatch.setattr(
            cur_mod,
            "evaluate_stage_gate",
            lambda trainer, stage, *, checkpoint, eval_text: (False, "eval_loss too high", {}),
        )
        created = []
        with caplog.at_level("WARNING"):
            report = run_curriculum(
                self._curriculum(self._stage("a"), self._stage("b")),
                make_trainer=_fake_trainer_factory(created),
            )
        assert len(created[0].train.call_args_list) == 2  # advanced anyway
        first = report.stages[0]
        assert first.passed is False
        assert first.provisional is True
        assert first.gate_reason == "eval_loss too high"
        assert report.stages[1].provisional is True
        assert any(
            "provisional" in r.message.lower() or "gate" in r.message.lower()
            for r in caplog.records
        )

    def test_gate_fail_abort_raises_and_stops(self, monkeypatch):
        monkeypatch.setattr(
            cur_mod,
            "evaluate_stage_gate",
            lambda trainer, stage, *, checkpoint, eval_text: (False, "boom", {}),
        )
        created = []
        with pytest.raises(StageGateError) as ei:
            run_curriculum(
                self._curriculum(self._stage("a", on_gate_fail="abort"), self._stage("b")),
                make_trainer=_fake_trainer_factory(created),
            )
        assert "a" in str(ei.value)
        assert len(created[0].train.call_args_list) == 1  # stage b never ran

    def test_ewc_snapshot_at_named_boundary(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a", ewc=True), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        trainer = created[0]
        trainer.save_ewc_snapshot.assert_called_once_with("a")

    def test_no_ewc_snapshot_when_stage_not_marked(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        created[0].save_ewc_snapshot.assert_not_called()

    def test_replay_mix_uses_completed_stage_paths(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b", replay=0.5)),
            make_trainer=_fake_trainer_factory(created),
        )
        (args, kwargs) = created[0].swap_stage_data.call_args
        mix = dict(args[0] if args else kwargs.get("data_path"))
        assert "data/a.txt" in mix
        assert "data/b.txt" in mix
        assert mix["data/a.txt"] == pytest.approx(0.5)

    def test_cancel_between_stages_stops(self):
        created = []
        cancel = threading.Event()

        def make(data_path, config):
            t = _fake_trainer_factory(created)(data_path, config)

            def _train(**kw):
                cancel.set()  # cancel fires during stage 1
                return t.train.return_value

            t.train.side_effect = _train
            return t

        report = run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=make,
            cancel_event=cancel,
        )
        assert report.cancelled is True
        assert report.completed_all is False
        assert len(created) == 1  # stage b never constructed

    def test_progress_callback_carries_stage_and_global_percent(self):
        created = []
        seen = []

        def make(data_path, config):
            t = _fake_trainer_factory(created)(data_path, config)

            def _train(**kw):
                cb = kw.get("on_progress")
                if cb:
                    cb({"progress_percent": 50.0, "epoch": 1, "global_step": 10})
                return t.train.return_value

            t.train.side_effect = _train
            return t

        run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=make,
            on_progress=seen.append,
        )
        assert seen, "on_progress never fired"
        info = seen[0]
        assert info["stage"] == "a"
        assert info["stage_index"] == 0
        assert info["stages_total"] == 2
        assert info["progress_percent"] == pytest.approx(25.0)  # 50% through stage 1 of 2

    def test_empty_curriculum_rejected(self):
        with pytest.raises(CurriculumError):
            run_curriculum(Curriculum(stages=()), make_trainer=_fake_trainer_factory([]))

    def test_ewc_any_stage_enables_config(self):
        created = []
        run_curriculum(
            self._curriculum(self._stage("a", ewc=True), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        assert created[0].config.use_ewc is True

    def test_default_gate_policy_is_continue(self, monkeypatch):
        # A failing gate with the schema default must advance, not raise.
        monkeypatch.setattr(
            cur_mod,
            "evaluate_stage_gate",
            lambda trainer, stage, *, checkpoint, eval_text: (False, "x", {}),
        )
        created = []
        report = run_curriculum(
            self._curriculum(self._stage("a"), self._stage("b")),
            make_trainer=_fake_trainer_factory(created),
        )
        assert report.stages[0].provisional is True
        assert len(created[0].train.call_args_list) == 2


# ── Trainer swap seam (real trainer, tiny corpora) ──────────────────────


def _tiny_config(tmp_path, **overrides):
    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=16,
        n_layer=1,
        n_head=2,
        block_size=8,
        dropout=0.0,
        batch_size=4,
        epochs=1,
        max_steps=6,
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


class TestSwapStageData:
    @pytest.fixture
    def trainer(self, tmp_path):
        p1 = tmp_path / "stage1.txt"
        p1.write_text(DATA_TEXT, encoding="utf-8")
        return SloughGPTTrainer(str(p1), config=_tiny_config(tmp_path)), tmp_path

    def test_swap_replaces_corpus_and_preserves_ids(self, trainer):
        t, tmp_path = trainer
        old_stoi = dict(t.stoi)
        old_itos = dict(t.itos)
        old_vocab = t.vocab_size

        p2 = tmp_path / "stage2.txt"
        # different text, but only chars already present in stage 1's vocab
        p2.write_text(
            "dogs sleep while foxes run across warm meadows again. " * 8, encoding="utf-8"
        )
        t.swap_stage_data(str(p2), epochs=2)

        assert t.data_path == str(p2)
        assert t.stoi == old_stoi  # ids NEVER remapped
        assert t.itos == old_itos
        assert t.vocab_size == old_vocab
        assert t.config.epochs == 2
        # stage-local state reset
        assert t.current_epoch == 0
        assert t._early_stopped is False
        assert t._patience_counter == 0

    def test_swap_tokenizes_new_chars_as_unknown_zero(self, trainer):
        t, tmp_path = trainer
        old_vocab = t.vocab_size
        p2 = tmp_path / "unicode.txt"
        p2.write_text("café résumé naïve — " * 10 + DATA_TEXT, encoding="utf-8")
        t.swap_stage_data(str(p2))
        # vocab must not grow; unseen chars map to token 0
        assert t.vocab_size == old_vocab
        assert t.data.max() < old_vocab

    def test_swap_keeps_optimizer_and_global_step(self, trainer):
        t, tmp_path = trainer
        t.train()  # run a bit so optimizer + step exist
        step_before = t.global_step
        opt_before = t.optimizer
        p2 = tmp_path / "stage2.txt"
        p2.write_text(DATA_TEXT + "more text here. " * 16, encoding="utf-8")
        t.swap_stage_data(str(p2))
        assert t.global_step == step_before
        assert t.optimizer is opt_before

    def test_swap_rejects_while_training(self, trainer):
        t, _ = trainer
        t._is_training = True
        with pytest.raises(RuntimeError):
            t.swap_stage_data("whatever.txt")
        t._is_training = False

    def test_swap_recomputes_dataset_signature(self, trainer):
        t, tmp_path = trainer
        sig_before = t.dataset_signature
        p2 = tmp_path / "stage2.txt"
        p2.write_text("different corpus entirely, still ascii only. " * 16, encoding="utf-8")
        t.swap_stage_data(str(p2))
        # different token stream → different identity (unless both empty)
        assert t.dataset_signature != sig_before or t.dataset_signature == ""

    def test_swap_missing_file_raises(self, trainer):
        t, tmp_path = trainer
        with pytest.raises(FileNotFoundError):
            t.swap_stage_data(str(tmp_path / "nope.txt"))


# ── local_api seam (route + executor target) ──────────────────────────────


class TestCurriculumConfig:
    def test_maps_legacy_trainer_defaults(self):
        """The quick-train shape must map onto real TrainerConfig fields.

        ``lr`` → ``learning_rate`` and ``vocab_size=0`` (corpus-derived, like
        passing no config at all); drift raises ``TypeError`` rather than
        silently defaulting.
        """
        from domain.shell._internal import local_api

        cfg = local_api._curriculum_config()
        assert cfg.learning_rate == local_api._TRAINER_DEFAULTS["lr"]
        assert cfg.epochs == local_api._TRAINER_DEFAULTS["epochs"]
        assert cfg.vocab_size == 0
        assert cfg.n_layer == local_api._TRAINER_DEFAULTS["n_layer"]


class TestCurriculumRoute:
    def test_requires_curriculum_path(self):
        from domain.shell._internal import local_api

        res = local_api._post_training_curriculum({}, {}, None)
        assert res["error_type"] == "BadRequest"
        assert "curriculum path" in res["error"]

    def test_fail_fast_on_bad_schema_spends_no_job_slot(self, tmp_path, monkeypatch):
        """A malformed curriculum is rejected BEFORE a job/thread is created."""
        from domain.shell._internal import local_api

        monkeypatch.setattr(local_api, "_LOCAL_JOBS", {})
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": 1, "stages": []}), encoding="utf-8")
        res = local_api._post_training_curriculum({}, {}, {"curriculum": str(bad)})
        assert res["error_type"] == "BadRequest"
        assert "stages" in res["error"]
        assert local_api._LOCAL_JOBS == {}

    def test_missing_file_is_bad_request(self, tmp_path, monkeypatch):
        from domain.shell._internal import local_api

        monkeypatch.setattr(local_api, "_LOCAL_JOBS", {})
        res = local_api._post_training_curriculum(
            {}, {}, {"curriculum": str(tmp_path / "nope.json")}
        )
        assert res["error_type"] == "BadRequest"
        assert local_api._LOCAL_JOBS == {}

    def test_starts_job_with_stage_metadata(self, tmp_path, monkeypatch):
        from domain.shell._internal import local_api

        monkeypatch.setattr(local_api, "_LOCAL_JOBS", {})
        submitted: dict = {}

        class _Exec:
            def submit(self, fn, jid):
                submitted["fn"], submitted["jid"] = fn, jid

        monkeypatch.setattr(
            "domain.training._internal.executor.get_training_executor",
            lambda: _Exec(),
        )
        monkeypatch.setattr(
            "domain.infrastructure.cancel_manager.get_cancel_manager",
            MagicMock(),
        )
        path = _write(tmp_path, {"version": 1, "stages": [_stage("one"), _stage("two")]})
        res = local_api._post_training_curriculum({}, {}, {"curriculum": path})
        assert res["status"] == "running"
        assert res["stages"] == 2
        rec = local_api._LOCAL_JOBS[res["id"]]
        assert rec["curriculum"] == path
        assert rec["stage"] == "one"  # first stage, rendered by the REPL
        assert rec["epochs"] == 2  # stage count, not trainer epochs
        assert submitted["jid"] == res["id"]
        assert submitted["fn"] is local_api._run_curriculum_training
        # summary projection: stage label now, verdict list only after run
        summary = local_api._job_summary(rec)
        assert summary["stage"] == "one"
        assert "stages" not in summary  # None filtered until completion
        assert summary["epochs"] == 2


class TestRunCurriculumJob:
    """The executor target: curriculum file → job record → verdicts."""

    def _job(self, local_api, monkeypatch, path: str) -> str:
        monkeypatch.setattr(local_api, "_LOCAL_JOBS", {})
        monkeypatch.setattr(
            "domain.infrastructure.cancel_manager.get_cancel_manager",
            MagicMock(),
        )
        jid = "job_curriculumtest"
        local_api._LOCAL_JOBS[jid] = {
            "id": jid,
            "curriculum": path,
            "status": "running",
            "progress": 0,
            "started_at": 0.0,
            "updated_at": 0.0,
            "_cancel_event": threading.Event(),
            "_pause_event": threading.Event(),
        }
        return jid

    def test_completed_run_records_stage_verdicts(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from domain.shell._internal import local_api
        from domain.training._internal.curriculum import StageResult

        path = _write(tmp_path, {"version": 1, "stages": [_stage("one"), _stage("two")]})
        jid = self._job(local_api, monkeypatch, path)
        report = SimpleNamespace(
            cancelled=False,
            final_checkpoint="checkpoints/final.soul",
            stages=[
                StageResult(
                    name="one",
                    passed=True,
                    provisional=False,
                    metrics={"max_eval_loss": 2.0},
                    checkpoint="ck1.soul",
                    gate_reason="",
                    epochs=1,
                ),
                StageResult(
                    name="two",
                    passed=False,
                    provisional=True,
                    metrics={},
                    checkpoint="ck2.soul",
                    gate_reason="max_eval_loss 9.0 > 4.0",
                    epochs=1,
                ),
            ],
        )

        def fake_run(curriculum, *, config, on_progress, cancel_event, pause_event):
            assert len(curriculum.stages) == 2  # glue loaded the real file
            on_progress(
                {
                    "progress_percent": 50,
                    "stage": "one",
                    "stage_index": 1,
                    "stages_total": 2,
                    "global_step": 7,
                }
            )
            return report

        monkeypatch.setattr(cur_mod, "run_curriculum", fake_run)
        local_api._run_curriculum_training(jid)
        rec = local_api._LOCAL_JOBS[jid]
        assert rec["status"] == "completed"
        assert rec["progress"] == 100
        assert rec["checkpoint"] == "checkpoints/final.soul"
        assert rec["global_step"] == 7  # on_progress mapping landed
        assert rec["stage"] == "one"
        assert len(rec["stages"]) == 2
        assert rec["stages"][1]["provisional"] is True
        assert "max_eval_loss" in rec["stages"][1]["gate_reason"]

    def test_stage_gate_abort_marks_job_failed(self, tmp_path, monkeypatch):
        from domain.shell._internal import local_api

        path = _write(tmp_path, {"version": 1, "stages": [_stage("one")]})
        jid = self._job(local_api, monkeypatch, path)

        def fake_run(curriculum, *, config, on_progress, cancel_event, pause_event):
            raise StageGateError("one", "max_eval_loss 9.0 > 4.0")

        monkeypatch.setattr(cur_mod, "run_curriculum", fake_run)
        local_api._run_curriculum_training(jid)
        rec = local_api._LOCAL_JOBS[jid]
        assert rec["status"] == "failed"
        assert "stage gate failed" in rec["error"]

    def test_cancelled_run_marks_job_cancelled(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from domain.shell._internal import local_api

        path = _write(tmp_path, {"version": 1, "stages": [_stage("one")]})
        jid = self._job(local_api, monkeypatch, path)

        def fake_run(curriculum, *, config, on_progress, cancel_event, pause_event):
            return SimpleNamespace(cancelled=True, final_checkpoint="", stages=[])

        monkeypatch.setattr(cur_mod, "run_curriculum", fake_run)
        local_api._run_curriculum_training(jid)
        rec = local_api._LOCAL_JOBS[jid]
        assert rec["status"] == "cancelled"


# ── Shipped example (examples/curriculum.json) ─────────────────────────────


class TestExampleCurriculum:
    """The shipped example must stay loadable and honest.

    A stale example teaches a broken recipe — this pins it to the schema and
    to files that actually exist in a fresh checkout.
    """

    ROOT = Path(__file__).resolve().parents[2]

    def _load(self):
        return load_curriculum(self.ROOT / "examples" / "curriculum.json")

    def test_loads_with_four_stages_and_marks_placeholders(self):
        example = self._load()
        assert len(example.stages) == 4
        # JSON has no comments: the example must carry its own warning
        # that the gate thresholds are placeholders to calibrate.
        assert "placeholder" in example.description.lower()

    def test_every_referenced_file_exists(self):
        example = self._load()
        for stage in example.stages:
            for path, ratio in stage.datasets:
                assert ratio > 0
                assert (self.ROOT / path).is_file(), f"{stage.name}: {path}"
        assert example.eval_text
        assert (self.ROOT / example.eval_text).is_file()

    def test_eval_text_held_out_from_training(self):
        example = self._load()
        trained = {p for s in example.stages for p, _ in s.datasets}
        assert example.eval_text not in trained

    def test_demonstrates_replay_ewc_and_gates(self):
        """The example exercises every card feature: replay %, EWC, gates."""
        example = self._load()
        assert all(not s.gate.is_empty for s in example.stages)
        assert sum(1 for s in example.stages if s.replay > 0) >= 2
        assert sum(1 for s in example.stages if s.ewc) >= 2
        names = [s.name for s in example.stages]
        assert len(set(names)) == len(names)  # stage names unique


# ── Real-trainer end-to-end (the seam fake trainers never reach) ─────────


class TestRealTrainerEndToEnd:
    """Drive the REAL ``SloughGPTTrainer`` through two stages.

    Fake trainers never call ``save_checkpoint`` — which is exactly where
    the calibration run crashed: a stage mix is a ``[(path, ratio), ...]``
    list and checkpoint naming assumed ``str`` (card 20261115_081635).
    """

    def test_two_stages_train_swap_and_checkpoint(self, tmp_path):
        a = tmp_path / "alpha.txt"
        a.write_text(DATA_TEXT, encoding="utf-8")
        b = tmp_path / "beta.txt"
        b.write_text(
            "dogs sleep while foxes run across warm meadows again. " * 16, encoding="utf-8"
        )
        cur = load_curriculum(
            _write(
                tmp_path,
                {
                    "version": 1,
                    "stages": [
                        _stage("s1", datasets=[{"path": str(a), "ratio": 1.0}]),
                        _stage(
                            "s2",
                            datasets=[{"path": str(b), "ratio": 1.0}],
                            replay=0.25,
                        ),
                    ],
                },
            )
        )
        # this test exercises staging mechanics, not data-quality gates
        cfg = _tiny_config(tmp_path, min_data_quality=0.0, max_toxicity_rate=1.0)

        report = run_curriculum(cur, config=cfg)

        assert report.completed_all
        assert all(r.passed and not r.provisional for r in report.stages)
        souls = list((tmp_path / "ckpts").glob("*.soul"))
        assert souls, "each stage must save a checkpoint through the mix path"
        # the blend must not leak into artifact names as a python repr
        assert not any(p.name.startswith("[") for p in souls)
