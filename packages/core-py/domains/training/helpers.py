"""Training helpers — pure functions with no state dependencies."""

from __future__ import annotations

import json
import logging
import math
import re
import struct
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state import REPO_ROOT, CHECKPOINTS_DIR, TURBO_DIR, SOU_MAGIC

logger = logging.getLogger("slo.training")

_VALID_DATASET_ID = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _finite_payload(o: Any) -> Any:
    if is_dataclass(o):
        return _finite_payload(asdict(o))
    if isinstance(o, dict):
        return {k: _finite_payload(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite_payload(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def log_experiment_metric(experiment_id: str, metric: str, value: float, step: int = 0) -> None:
    try:
        from datetime import datetime, timezone
        exp_dir = REPO_ROOT / "data" / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = exp_dir / f"{experiment_id}_metrics.jsonl"
        entry = {
            "experiment_id": experiment_id,
            "metric": metric,
            "value": value,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log experiment metric %s for %s: %s", metric, experiment_id, e)


def log_experiment_param(experiment_id: str, param_name: str, value: Any) -> None:
    try:
        from datetime import datetime, timezone
        exp_dir = REPO_ROOT / "data" / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)
        params_file = exp_dir / f"{experiment_id}_params.jsonl"
        entry = {
            "experiment_id": experiment_id,
            "param": param_name,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(params_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log experiment param %s for %s: %s", param_name, experiment_id, e)


def parse_subtitle_text(text: str) -> list:
    lines = []
    srt_pattern = re.compile(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}')
    vtt_pattern = re.compile(r'\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}')

    if srt_pattern.search(text) or vtt_pattern.search(text):
        for line in text.split('\n'):
            line = line.strip()
            if srt_pattern.match(line) or vtt_pattern.match(line):
                continue
            if re.match(r'^\d+$', line):
                continue
            if line.startswith('WEBVTT'):
                continue
            if '-->' in line:
                continue
            if line and not line.startswith('['):
                lines.append(line)
    else:
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 2:
                lines.append(line)

    return lines


def resolve_dataset_path(dataset_id: str) -> str:
    if not _VALID_DATASET_ID.match(dataset_id):
        raise ValueError(f"Invalid dataset ID: {dataset_id!r}")
    for base_name in ("datasets", "data/datasets", "data"):
        ds_candidate = (REPO_ROOT / base_name / dataset_id).resolve()
        allowed_base = (REPO_ROOT / base_name).resolve()
        if not str(ds_candidate).startswith(str(allowed_base)):
            continue
        if not ds_candidate.exists():
            continue
        for name in ("corpus.jsonl", "input.txt", "train.txt", "text.txt"):
            candidate = ds_candidate / name
            if candidate.exists():
                return str(candidate)
        txt_files = list(ds_candidate.glob("*.txt"))
        if txt_files:
            return str(txt_files[0])
    return ""


def build_soul_prompt(soul_name: str) -> str:
    prompts = {
        "assistant": "You are a helpful assistant. Be clear and friendly.",
        "creative": "You are a creative thinker. Be imaginative and playful.",
        "analyst": "You are a precise analyst. Be methodical and thorough.",
        "coder": "You are an expert coder. Write clean, efficient code.",
        "teacher": "You are a patient teacher. Explain step by step.",
    }
    return prompts.get(soul_name, prompts["assistant"])


def get_soul_name(soul) -> str:
    if hasattr(soul, 'name') and soul.name:
        return soul.name
    return getattr(soul, 'soul_name', 'unknown')


def get_soul_traits(soul) -> dict:
    raw = getattr(soul, 'soul_traits', None)
    if raw:
        return raw
    if hasattr(soul, 'personality'):
        p = soul.personality
        if isinstance(p, dict):
            return p
        if hasattr(p, 'to_dict'):
            return p.to_dict()
        if hasattr(p, '__dict__'):
            return vars(p)
        return dict(p)
    return {}


def read_slo_json_header(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != SOU_MAGIC:
                return {}
            f.seek(8)
            json_len = struct.unpack("<I", f.read(4))[0]
            header = f.read(json_len)
            return json.loads(header.decode())
    except Exception as exc:
        logger.debug("Failed to parse checkpoint header %s: %s", path, exc)
        return {}


def describe_checkpoint(ckpt: dict) -> str:
    parts = []
    soul = ckpt.get("soul", "")
    loss = ckpt.get("loss")
    epochs = ckpt.get("epochs") or ckpt.get("epochs_trained")
    steps = ckpt.get("steps", 0)
    dataset = ckpt.get("training_dataset", "")
    traits = ckpt.get("traits", {})
    model_type = ckpt.get("model_type", "")

    if dataset:
        parts.append(f"Trained on {dataset}")
    elif soul and soul != "unknown":
        parts.append(f"Soul: {soul}")
    else:
        parts.append("A trained model")

    if epochs:
        parts.append(f"for {epochs} epoch{'s' if epochs != 1 else ''}")
    elif steps:
        parts.append(f"for {steps} steps")

    if loss is not None:
        if loss < 1.5:
            parts.append(f"(loss {loss:.2f} - learned well)")
        elif loss < 3.0:
            parts.append(f"(loss {loss:.2f} - moderate)")
        else:
            parts.append(f"(loss {loss:.2f} - needs more training)")

    if traits:
        trait_names = list(traits.keys())[:3]
        if trait_names:
            parts.append(f"Personality: {', '.join(trait_names)}")

    if model_type and model_type not in ("slonet", "unknown"):
        parts.append(f"[{model_type}]")

    return " ".join(parts) + "."


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy loss on (batch, vocab) logits and (batch,) targets."""
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True))
    batch_size = targets.shape[0]
    return float(-log_probs[np.arange(batch_size), targets].mean())
