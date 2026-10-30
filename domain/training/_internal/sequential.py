"""Sequential multi-corpus training — stage runner with probe standings.

ONE training loop (``SloughGPTTrainer``) driven stage-by-stage over sealed
manifests. This module is the sync core; hosts (the server's training-job
runtime, ``scripts/train_sequential.py``) supply callbacks and run it on
their own thread. No asyncio here.

Continuity design:
- Every stage resumes from the shared per-job checkpoint dir, so
  ``global_step`` runs on across stage boundaries (stage budgets are
  therefore expressed as ``peek_step + steps_per_stage`` — a raw
  ``max_steps`` would trip instantly on a resumed run).
- After each stage a ``trained_on.json`` sidecar is written INTO the
  checkpoint dir (artifact-local chain: it travels with the weights) and
  probe standings accumulate via ``on_standings`` for display.
- Stage-end checkpoints are ``is_final=True`` saves, so optimizer
  momentum resets at stage boundaries (weights, step, scheduler position
  restore; momentum does not). Accepted property of the boundary — each
  stage is a fresh corpus.

Standing honesty: all stages share ONE char vocab (union of every
stage's ``train.txt``), so probe losses are comparable across stages;
per-corpus vocabs would scramble embeddings and confound forgetting.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

logger = logging.getLogger("slo.training.sequential")

TRAINED_ON_FILENAME = "trained_on.json"


def manifest_ref(manifest_path: str | Path) -> str:
    """Human-stable ``dataset_id@v<version>`` for a sealed manifest."""
    path = Path(manifest_path)
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    dataset_id = meta.get("dataset_id") or path.parent.parent.name
    version = meta.get("version") or (path.parent.name.lstrip("v") or "1")
    return f"{dataset_id}@v{version}"


class SharedCharTokenizer:
    """One char vocab shared by every stage — probe losses stay comparable.

    Unknown characters (probe-only glyphs) are dropped at encode time, so
    neither training nor probing ever sees an out-of-vocab id.
    """

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self.stoi = dict(stoi)
        self.itos = dict(itos)
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        table = self.stoi
        return [table[ch] for ch in text if ch in table]


def build_shared_tokenizer(manifests: Sequence[str]) -> SharedCharTokenizer:
    """Union char vocab from every stage's ``train.txt`` (beside its manifest)."""
    from domain.training._internal.training_handler import build_char_vocab

    texts: list[str] = []
    for manifest in manifests:
        train_txt = Path(manifest).parent / "train.txt"
        if train_txt.is_file():
            texts.append(train_txt.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"no train.txt found next to any of: {manifests}")
    stoi, itos = build_char_vocab(texts)
    return SharedCharTokenizer(stoi, itos)


def _write_trained_on(ckpt_dir: Path, entries: list[dict[str, Any]]) -> None:
    """Atomically write the chain sidecar into the checkpoint dir."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {"entries": entries, "updated_at": datetime.now(UTC).isoformat()}
    tmp = ckpt_dir / f".{TRAINED_ON_FILENAME}.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, ckpt_dir / TRAINED_ON_FILENAME)


def load_trained_on(ckpt_dir: str | Path) -> list[dict[str, Any]]:
    """Read the chain sidecar (empty list when absent/corrupt)."""
    sidecar = Path(ckpt_dir) / TRAINED_ON_FILENAME
    if not sidecar.is_file():
        return []
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("entries")
    return entries if isinstance(entries, list) else []


def run_sequential_stages(
    manifests: Sequence[str],
    *,
    trainer_kwargs: dict[str, Any],
    steps_per_stage: int | None = None,
    probe_batches: int = 5,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    on_standings: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Train stage-by-stage over sealed manifests, probing after each stage.

    Args:
        manifests: Sealed ``manifest.json`` paths, in training order.
        trainer_kwargs: Kwargs for every ``SloughGPTTrainer`` (minus
            ``data_path``/``tokenizer``, which this runner owns).
            ``checkpoint_dir`` must be one shared dir — that is the
            continuity channel between stages.
        steps_per_stage: Optimizer steps per stage; defaults to the
            trainer's configured ``epochs`` × steps/epoch.
        probe_batches: Batches per probe measurement.
        on_progress: Forwarded to each ``trainer.train`` (host's live
            metrics closure; ``global_step`` is continuous across stages).
        on_standings: Called after every stage with the full standings
            list and the trained-on chain (host mutates its job record).
        cancel_event: Cooperative cancellation (checked between stages and
            inside the training loop).
        pause_event: Cooperative pause (forwarded to the loop).

    Returns:
        ``{"standings", "trained_on", "last_trainer", "global_step"}``.
        ``last_trainer`` is ``None`` only when cancelled before stage 0.
    """
    from domain.training._internal.checkpoint_utils import normalize_raw_checkpoint
    from domain.training._internal.probes import evaluate_probe, probe_path_for
    from domain.training._internal.train_pipeline import CheckpointManager, SloughGPTTrainer

    manifests = [str(m) for m in manifests]
    if not manifests:
        raise ValueError("run_sequential_stages: no manifests given")

    shared_tok = build_shared_tokenizer(manifests)
    trainer_kwargs = dict(trainer_kwargs)
    trainer_kwargs.pop("data_path", None)
    trainer_kwargs.pop("tokenizer", None)
    ckpt_dir = Path(trainer_kwargs.get("checkpoint_dir") or "checkpoints/sequential")
    trainer_kwargs["checkpoint_dir"] = str(ckpt_dir)

    manager = CheckpointManager(str(ckpt_dir))
    standings: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = load_trained_on(ckpt_dir)
    if chain:
        logger.info("sequential: resuming chain of %d prior entries", len(chain))

    last_trainer: Any = None
    for stage, manifest in enumerate(manifests):
        if cancel_event is not None and cancel_event.is_set():
            logger.warning("sequential: cancelled before stage %d", stage)
            break

        ref = manifest_ref(manifest)
        bundle = manager.load_latest()
        step0 = int(normalize_raw_checkpoint(bundle).get("step", 0)) if bundle else 0

        trainer = SloughGPTTrainer(data_path=manifest, tokenizer=shared_tok, **trainer_kwargs)
        spe = trainer._steps_per_epoch()
        if steps_per_stage and steps_per_stage > 0:
            per_stage = steps_per_stage
        else:
            per_stage = max(1, (trainer.config.epochs or 1) * spe)
        # Stage budget is a DELTA on the restored global step (a raw max_steps
        # would trip instantly on resume); the epoch loop must outlive max_steps
        # or the outer range() exits before the budget binds.
        trainer.config.max_steps = step0 + per_stage
        trainer.config.epochs = trainer.config.max_steps + 10
        trainer.config.early_stopping_patience = 0  # stage budget must complete

        logger.info(
            "sequential: stage %d/%d %s (steps %d..%d)",
            stage + 1,
            len(manifests),
            ref,
            step0,
            trainer.config.max_steps,
            extra={"tag": "TRAIN"},
        )
        train_kwargs: dict[str, Any] = {"resume": True, "resume_checkpoint": bundle}
        if on_progress is not None:
            train_kwargs["on_progress"] = on_progress
        if cancel_event is not None:
            train_kwargs["cancel_event"] = cancel_event
        if pause_event is not None:
            train_kwargs["pause_event"] = pause_event
        trainer.train(**train_kwargs)

        last_trainer = trainer
        probes: dict[str, float | None] = {}
        for seen in manifests[: stage + 1]:
            ppath = probe_path_for(seen)
            if not ppath.is_file():
                continue
            measured = evaluate_probe(trainer, ppath, num_batches=probe_batches)
            if measured is not None:
                probes[manifest_ref(seen)] = round(float(measured["probe_loss"]), 6)
            else:
                probes[manifest_ref(seen)] = None

        entry = {
            "stage": stage,
            "dataset": ref,
            "manifest": manifest,
            "global_step": int(trainer.global_step),
            "probes": probes,
        }
        standings.append(entry)
        chain.append(
            {
                "kind": "manifest",
                "ref": ref,
                "manifest": manifest,
                "stage": stage,
                "global_step": int(trainer.global_step),
                "finished_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_trained_on(ckpt_dir, chain)
        if on_standings is not None:
            on_standings(standings, chain)

    return {
        "standings": standings,
        "trained_on": chain,
        "last_trainer": last_trainer,
        "global_step": int(last_trainer.global_step) if last_trainer is not None else 0,
    }
