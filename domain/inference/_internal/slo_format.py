"""
.soul - SloughGPT Soul Unit Format

FEATURE: soul-format — Self-contained model format (weights + personality + metadata).
DO NOT DELETE. Core infrastructure used by all model loading.

The living identity format for trained AI models. Every .soul file is self-contained:
model weights + soul profile + training metadata — fully self-contained.

Format: SOUL + version + config_len + JSON_config + [state_len + JSON state_dict]

Trademark (c) 2026 SloughGPT. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from domain.shared import repair_iso, utc_now_iso

logger = logging.getLogger("slo.inference.slo_format")

SOU_MAGIC = b"SOUL"
SOU_VERSION = 2
SOU_VERSION_V3 = 3
SOU_TRADEMARK = "SloughGPT Soul Unit (.soul) - Trademark (c) 2026 SloughGPT"


def _write_soul_v3(
    f,
    metadata_bytes: bytes,
    params: list[tuple[str, np.ndarray]],
    *,
    include_weights: bool = True,
) -> None:
    """Encode the SOUL v3 binary body onto ``f`` (magic + version + JSON + weights).

    Single source of truth for the v3 on-disk layout, shared by ``slonet.export_to_sou``
    and ``save_soul`` so the two writers cannot drift apart:

        b"SOUL" | <I 3> | <I json_len> | json_bytes | weight table

    Weight table (only when ``include_weights``): ``<I n_params>`` then per param
    ``<I len(name)> name_bytes <I ndim> <I dim...> arr.tobytes()`` (float32).
    Metadata JSON is written by the caller so each writer keeps its own schema.
    """
    f.write(SOU_MAGIC)
    f.write(struct.pack("<I", SOU_VERSION_V3))
    f.write(struct.pack("<I", len(metadata_bytes)))
    f.write(metadata_bytes)
    if not include_weights:
        return
    f.write(struct.pack("<I", len(params)))
    for key, arr in params:
        name_bytes = key.encode()
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(arr.tobytes())


def _soul_json_sanitize(obj: Any) -> Any:
    """RFC 8259–friendly structures: ``NaN`` / ``±inf`` → ``null`` (Python ``None``)."""

    if isinstance(obj, dict):
        return {k: _soul_json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_soul_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
        }


@dataclass
class ContextParams:
    context_window: int = 4096
    num_ctx: int = 4096
    num_gpu: int = 0
    num_thread: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersonalityCore:
    warmth: float = 0.5
    creativity: float = 0.5
    empathy: float = 0.5
    formality: float = 0.5
    humor: float = 0.5
    patience: float = 0.5
    confidence: float = 0.5
    curiosity: float = 0.5
    directness: float = 0.5
    optimism: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class BehavioralTraits:
    speaking_style: str = "conversational"
    reasoning_approach: str = "balanced"
    explanation_depth: str = "moderate"
    emotional_expressiveness: float = 0.5
    formality_dynamic: float = 0.5
    interruption_tolerance: float = 0.5
    follow_up_tendency: float = 0.5
    clarification_seeking: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveSignature:
    pattern_recognition: float = 0.5
    long_context_handling: float = 0.5
    abstract_reasoning: float = 0.5
    factual_precision: float = 0.5
    creative_divergence: float = 0.5
    systematic_planning: float = 0.5
    metacognitive_awareness: float = 0.5
    learning_adaptability: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class EmotionalRange:
    empathy_depth: float = 0.5
    mood_responsiveness: float = 0.5
    tone_flexibility: float = 0.5
    sentiment_awareness: float = 0.5
    distress_handling: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class SloProfile:
    name: str
    version: str = "1.0.0"
    tagline: str = ""
    description: str = ""
    created_by: str = "SloughGPT Training Pipeline"
    born_at: str = ""
    lineage: str = "nanogpt"
    base_model: str = ""
    training_dataset: str = ""
    epochs_trained: int = 0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    dataset_signature: str = ""
    personality: PersonalityCore = field(default_factory=PersonalityCore)
    behavior: BehavioralTraits = field(default_factory=BehavioralTraits)
    cognition: CognitiveSignature = field(default_factory=CognitiveSignature)
    emotion: EmotionalRange = field(default_factory=EmotionalRange)
    generation: GenerationParams = field(default_factory=GenerationParams)
    context: ContextParams = field(default_factory=ContextParams)
    system_prompt: str = ""
    sample_dialogue: list[dict[str, str]] = field(default_factory=list)
    lora_adapters: list[str] = field(default_factory=list)
    quantization: str = "none"
    acl_users: list[str] = field(default_factory=list)
    watermark_enabled: bool = False
    watermark_strength: float = 0.1
    tags: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    integrity_hash: str = ""
    tier: str = ""
    """Container tier this file was WRITTEN as — ``simple``/``canonical``/
    ``interchange``/``runtime``. Declared at save time from the suffix, never
    guessed later: see :data:`SOUL_TIER_POLICY`."""
    provenance: str = ""
    """Where the artifact came from — the ``record=`` argument to
    :func:`save_soul`. Write time is the only moment this is knowable; reading
    it back later would mean guessing, so files predating the field read
    ``""`` and stay unknown rather than being inferred."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.born_at:
            self.born_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "version": self.version,
            "tagline": self.tagline,
            "description": self.description,
            "created_by": self.created_by,
            "born_at": self.born_at,
            "lineage": self.lineage,
            "base_model": self.base_model,
            "training_dataset": self.training_dataset,
            "epochs_trained": self.epochs_trained,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "dataset_signature": self.dataset_signature,
            "personality": self.personality.to_dict(),
            "behavior": self.behavior.to_dict(),
            "cognition": self.cognition.to_dict(),
            "emotion": self.emotion.to_dict(),
            "generation": self.generation.to_dict(),
            "context": self.context.to_dict(),
            "system_prompt": self.system_prompt,
            "sample_dialogue": self.sample_dialogue,
            "lora_adapters": self.lora_adapters,
            "quantization": self.quantization,
            "acl_users": self.acl_users,
            "watermark_enabled": self.watermark_enabled,
            "watermark_strength": self.watermark_strength,
            "tags": self.tags,
            "certifications": self.certifications,
            "integrity_hash": self.integrity_hash,
            "tier": self.tier,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }
        return d

    def compute_hash(self) -> str:
        # tier and provenance are DECLARED, not derived: tier follows the
        # suffix the writer chose, provenance is a caller-supplied `record=`
        # string. Hashing either would let an argument move the file's
        # identity — identical weights saved as "training" and "export" would
        # hash apart and stop matching as the same checkpoint. Excluding them
        # also keeps every hash this field ever produced stable.
        payload = {k: v for k, v in self.to_dict().items() if k not in ("tier", "provenance")}
        data = json.dumps(
            _soul_json_sanitize(payload),
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_sou_string(self) -> str:
        lines = [
            "# SloughGPT Slo Unit",
            f"# {SOU_TRADEMARK}",
            "# This file contains the living identity of an AI model.",
            "",
            f"SOUL {self.name}",
            f"VERSION {self.version}",
            f"LINEAGE {self.lineage}",
            f"BORN {self.born_at}",
            f"CREATED_BY {self.created_by}",
            "",
        ]

        if self.tagline:
            lines.append(f"TAGLINE {self.tagline}")
            lines.append("")

        if self.description:
            lines.append(f"DESCRIPTION {self.description}")
            lines.append("")

        lines.extend(["# Identity", f"BASEMODEL {self.base_model}"])
        if self.training_dataset:
            lines.append(f"TRAINING_DATA {self.training_dataset}")
        if self.dataset_signature:
            lines.append(f"DATA_SIGNATURE {self.dataset_signature}")
        lines.append("")

        lines.extend(["# Generation Parameters", "PARAMETER"])
        for k, v in self.generation.to_dict().items():
            if k != "stop":
                lines.append(f"    {k} {v}")
        if self.generation.stop:
            for s in self.generation.stop:
                lines.append(f"    stop {s}")
        lines.append("")

        lines.extend(["# Context", "CONTEXT"])
        for k, v in self.context.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("")

        lines.append("# Personality Core (soul signature)")
        lines.append("PERSONALITY")
        for k, v in self.personality.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Behavioral Traits")
        lines.append("BEHAVIOR")
        for k, v in self.behavior.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Cognitive Signature")
        lines.append("COGNITION")
        for k, v in self.cognition.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        lines.append("# Emotional Range")
        lines.append("EMOTION")
        for k, v in self.emotion.to_dict().items():
            lines.append(f"    {k} {v}")
        lines.append("    END")
        lines.append("")

        if self.system_prompt:
            lines.append(f"SYSTEM {self.system_prompt}")
            lines.append("")

        if self.sample_dialogue:
            for msg in self.sample_dialogue:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lines.append(f"MESSAGE {role} {content}")
            lines.append("")

        if self.lora_adapters:
            lines.append("ADAPTER")
            for adapter in self.lora_adapters:
                lines.append(f"    {adapter}")
            lines.append("    END")
            lines.append("")

        if self.tags:
            lines.append("TAG " + ",".join(self.tags))
            lines.append("")

        if self.training_dataset:
            lines.append(f"METADATA epochs_trained {self.epochs_trained}")
            lines.append(f"METADATA final_train_loss {self.final_train_loss}")
            lines.append(f"METADATA final_val_loss {self.final_val_loss}")

        if self.certifications:
            for cert in self.certifications:
                lines.append(f"CERTIFICATION {cert}")

        lines.append("")
        return "\n".join(lines)


class SouParser:
    @staticmethod
    def parse(content: str) -> SloProfile:
        lines = content.strip().split("\n")

        sp = SloProfile(name="unknown")
        section = None
        current_block = {}
        dialogue = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("SOUL "):
                sp.name = line[5:].strip()
            elif line.startswith("VERSION "):
                sp.version = line[8:].strip()
            elif line.startswith("LINEAGE "):
                sp.lineage = line[8:].strip()
            elif line.startswith("BORN "):
                raw_born = line[5:].strip()
                sp.born_at = repair_iso(raw_born)  # type: ignore[assignment]
            elif line.startswith("CREATED_BY "):
                sp.created_by = line[11:].strip()
            elif line.startswith("TAGLINE "):
                sp.tagline = line[8:].strip()
            elif line.startswith("DESCRIPTION "):
                sp.description = line[12:].strip()
            elif line.startswith("BASEMODEL "):
                sp.base_model = line[10:].strip()
            elif line.startswith("TRAINING_DATA "):
                sp.training_dataset = line[14:].strip()
            elif line.startswith("DATA_SIGNATURE "):
                sp.dataset_signature = line[15:].strip()
            elif line.startswith("SYSTEM "):
                sp.system_prompt = line[7:].strip()
            elif line.startswith("PARAMETER"):
                section = "parameter"
                current_block = {}
            elif line.startswith("CONTEXT"):
                section = "context"
                current_block = {}
            elif line.startswith("PERSONALITY"):
                section = "personality"
                current_block = {}
            elif line.startswith("BEHAVIOR"):
                section = "behavior"
                current_block = {}
            elif line.startswith("COGNITION"):
                section = "cognition"
                current_block = {}
            elif line.startswith("EMOTION"):
                section = "emotion"
                current_block = {}
            elif line.startswith("MESSAGE "):
                parts = line[8:].split(" ", 1)
                if len(parts) == 2:
                    dialogue.append({"role": parts[0], "content": parts[1]})
            elif line.startswith("ADAPTER"):
                section = "adapter"
                current_block = []
            elif line == "END":
                if section == "personality":
                    sp.personality = PersonalityCore(**current_block)
                elif section == "behavior":
                    sp.behavior = BehavioralTraits(
                        **{
                            k: v
                            for k, v in current_block.items()
                            if not isinstance(v, str) or v.replace(".", "1", 1).isdigit() is False
                        }
                    )
                    for k, v in current_block.items():
                        if (
                            isinstance(v, str)
                            and not v.replace(".", "1", 1).replace("e-", "", 1).isdigit()
                        ):
                            setattr(sp.behavior, k, v)
                        else:
                            try:
                                setattr(sp.behavior, k, float(v))
                            except (ValueError, TypeError):
                                setattr(sp.behavior, k, v)
                elif section == "cognition":
                    sp.cognition = CognitiveSignature(
                        **{k: float(v) for k, v in current_block.items()}
                    )
                elif section == "emotion":
                    sp.emotion = EmotionalRange(**{k: float(v) for k, v in current_block.items()})
                elif section == "adapter":
                    sp.lora_adapters = current_block
                section = None
                current_block = {}
            elif line.startswith("TAG "):
                sp.tags = [t.strip() for t in line[4:].split(",")]
            elif line.startswith("METADATA "):
                parts = line[9:].split(" ", 1)
                if len(parts) == 2:
                    k, v = parts
                    if k == "epochs_trained":
                        sp.epochs_trained = int(v)
                    elif k in ("final_train_loss", "final_val_loss"):
                        setattr(sp, k, float(v))
                    else:
                        sp.metadata[k] = v
            elif line.startswith("CERTIFICATION "):
                sp.certifications.append(line[14:].strip())
            elif section == "parameter":
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    if k == "stop":
                        current_block.setdefault("stop", []).append(v)
                    else:
                        try:
                            current_block[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            current_block[k] = v
                sp.generation = GenerationParams(
                    **{k: v for k, v in current_block.items() if k != "stop"}
                )
                if "stop" in current_block:
                    sp.generation.stop = current_block["stop"]
            elif section == "context":
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    try:
                        current_block[k] = int(v)
                    except ValueError:
                        current_block[k] = v
                sp.context = ContextParams(**current_block)
            elif section in ("personality", "cognition", "emotion"):
                parts = line.split()
                if len(parts) == 2:
                    k, v = parts
                    try:
                        current_block[k] = float(v)
                    except ValueError:
                        current_block[k] = v
            elif section == "behavior":
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    current_block[parts[0]] = parts[1]
            elif section == "adapter":
                current_block.append(line)
            elif line.startswith("QUANTIZATION "):
                sp.quantization = line[13:].strip()

        if dialogue:
            sp.sample_dialogue = dialogue

        return sp

    @staticmethod
    def load(path: str) -> SloProfile:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return SouParser.parse(content)

    @staticmethod
    def save(sou: SloProfile, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(sou.to_sou_string())


def create_soul_profile(
    name: str,
    base_model: str = "nanogpt",
    training_dataset: str = "",
    epochs_trained: int = 0,
    final_train_loss: float = 0.0,
    final_val_loss: float = 0.0,
    personality: PersonalityCore | None = None,
    generation: GenerationParams | None = None,
    system_prompt: str = "",
    tags: list[str] | None = None,
    lineage: str = "nanogpt",
    dataset_signature: str = "",
    **kwargs,
) -> SloProfile:
    """Create a Slo Profile from training results."""
    sp = SloProfile(
        name=name,
        base_model=base_model,
        training_dataset=training_dataset,
        epochs_trained=epochs_trained,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        lineage=lineage,
        dataset_signature=dataset_signature,
        personality=personality or PersonalityCore(),
        generation=generation or GenerationParams(),
        system_prompt=system_prompt or f"You are {name}, a thoughtful AI assistant.",
        tags=tags or [],
        created_by="SloughGPT Training Pipeline",
    )
    sp.integrity_hash = sp.compute_hash()
    return sp


# ── .soul filename grammar ──────────────────────────────────────────────────
# ONE owner for the checkpoint name. Write side is strict and idempotent
# (soul_path); read side is tolerant (soul_read_candidates) because legacy
# artifacts on disk are not normalized.
#
# Format lineage:
#   .sou  the initiator — simple module structure + metadata. Same lineage as
#         .soul, so it ALIASES (collapses) instead of raising.
#   .soul the canonical extensible weight file, built on .sou inside the
#         executor: structured, richer weights.
#   .slo  the official third-party interchange — data-rich weight mapping and
#         loader logic that lets foreign models load into our engine and the
#         SloNet architecture. A DIFFERENT format: never a write target for
#         save_soul, always a read candidate.

_SOUL_SUFFIX = ".soul"
_SOU_ALIAS = ".sou"
_SLO_SUFFIX = ".slo"

# The soul family: every extension the soul reader accepts as a name for
# itself. Public because dispatch (``ModelLoader.load``) must PROJECT this
# list rather than re-declare it — a second copy is exactly how the suffixes
# came to disagree across five sites. It overlaps ``_FOREIGN_SUFFIXES`` on
# ``.slo`` on purpose: foreign is the WRITE axis (never a save target), this
# is the READ axis (always a probe candidate).
SOUL_SUFFIXES: tuple[str, ...] = (_SOUL_SUFFIX, _SOU_ALIAS, _SLO_SUFFIX)

# Container tier per suffix — the DECLARED tier axis, kept apart from the
# read axis above on purpose. SOUL_SUFFIXES answers "may the reader probe this
# name?"; this answers "what kind of artifact is it?", and the two need not
# agree: .slnc is a tier the reader never probes as a soul spelling, because
# it is mmap runtime with no soul identity at all.
#
# Tiers are declared, never inferred from content — deriving them would make
# the tier depend on parsing choices and flip the moment a reader changed.
# Read side falls back to this table ONLY when a file predates the `tier`
# field in its own sidecar; a written value always wins.
SOUL_TIER_POLICY: dict[str, str] = {
    _SOU_ALIAS: "simple",
    _SOUL_SUFFIX: "canonical",
    _SLO_SUFFIX: "interchange",
    ".slnc": "runtime",
}

# Declared provenance vocabulary — the write-side counterpart of the tier
# table. Producers NAME one of these rather than passing any string, because
# provenance is stamped into the header and sidecar at write time and is never
# corrected afterwards: a typo there becomes permanent identity metadata. The
# contract is enforced by save_soul, not by convention.
#
# Extending it means adding to this tuple — the same rule SOUL_SUFFIXES
# follows, so neither axis grows a second, private list of spellings.
SOUL_PROVENANCE_TRAINING = "training"
SOUL_PROVENANCE_EXPORT = "export"
SOUL_PROVENANCE_DISTILLATION = "distillation"

SOUL_PROVENANCE: tuple[str, ...] = (
    SOUL_PROVENANCE_TRAINING,
    SOUL_PROVENANCE_EXPORT,
    SOUL_PROVENANCE_DISTILLATION,
)

# Extensions that name some other format. Handing one to soul_path is a type
# error, not a path to guess around: we refuse rather than write soul bytes
# under a foreign name (or silently swap the extension).
# NOTE: deliberately does NOT include .gguf — export_model(fmt="all") passes
# the same output_path to both the soul and gguf exporters.
_FOREIGN_SUFFIXES = (
    ".safetensors",
    ".npz",
    ".msgpack",
    ".pickle",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".onnx",
    ".h5",
    ".npy",
    ".slo",
)


def _collapse_soul_stem(basename: str) -> str:
    """Strip the .sou initiator alias and every repeated .soul suffix.

    ``x`` -> ``x``; ``x.sou`` -> ``x``; ``x.soul.soul`` -> ``x``.
    """
    stem = basename
    while True:
        for suffix in (_SOUL_SUFFIX, _SOU_ALIAS):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            return stem


def _dedupe(candidates: list[str]) -> list[str]:
    """Preserve probe order, drop repeats (canonical and legacy often coincide)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def soul_path(path: str) -> str:
    """Canonical ``.soul`` write path for *path*.

    Only the final path component carries the grammar — directory components
    are never inspected, so ``models/v2.3/x`` is a bare stem, not extension
    ``.3/x``. Idempotent: ``soul_path(soul_path(p)) == soul_path(p)``.

    Args:
        path: destination. May be a bare stem, canonical, legacy double-appended,
            or the ``.sou`` initiator alias.

    Returns:
        ``<dir>/<stem>.soul``

    Raises:
        ValueError: *path* names a foreign format (``.slo``, ``.pt``, ...).
            Guessing an extension there would mislabel the bytes on disk.
    """
    directory, basename = os.path.split(path)
    if not basename:
        raise ValueError(f"{path!r} is not a .soul checkpoint path: empty file name")

    stem = _collapse_soul_stem(basename)
    for suffix in _FOREIGN_SUFFIXES:
        if stem.endswith(suffix):
            raise ValueError(
                f"{path!r} is not a .soul checkpoint path: {suffix} is a foreign "
                f"format; refusing to guess an extension"
            )
    return os.path.join(directory, stem + _SOUL_SUFFIX)


def soul_meta_path(path: str) -> str:
    """Sidecar path for *path*, always derived from the canonical name.

    Every writer must use this instead of ``+ ".meta.json"`` on its own
    argument, or a canonicalized checkpoint gets a sidecar under the
    pre-canonicalized name (``demo.soul`` + ``demo.soul.soul.meta.json``).
    """
    return soul_path(path) + ".meta.json"


def soul_read_candidates(path: str) -> list[str]:
    """Ordered probe list for reading *path* — never raises.

    The spelling the caller NAMED comes first, then its sibling:

    - bare stem       -> ``<stem>.soul``, ``<stem>.slo``
    - ``.soul``       -> as given (canonical), ``<given>.soul`` (legacy)
    - ``.soul.soul``  -> as given, canonical
    - ``.sou``        -> as given, ``<stem>.soul`` (a different model, never ours)
    - ``.slo``        -> as given, ``<stem>.soul`` (interchange — a real file
      in its own right, not a spelling to normalize away)

    "As given first" is not cosmetic. Two spellings can be genuinely
    DIFFERENT checkpoints: one stem in ``models/`` holds a 16:07 run at loss
    4.10 and a 16:38 run at loss 3.97. Probing the canonical name first would
    hand back the file the caller did not name, and nothing downstream —
    magic check, parse, finder globs — would catch it, because both files are
    valid SOUL containers.
    """
    directory, basename = os.path.split(path)

    if basename.endswith(_SLO_SUFFIX):
        stem = basename[: -len(_SLO_SUFFIX)]
        return _dedupe([path, os.path.join(directory, stem + _SOUL_SUFFIX)])

    if basename.endswith(_SOU_ALIAS):
        # `.sou` names a DIFFERENT model (tiny sliney-trained / GPT-y), not a
        # spelling of `.soul` — so probe as given FIRST, exactly like `.slo`.
        stem = basename[: -len(_SOU_ALIAS)]
        return _dedupe([path, os.path.join(directory, stem + _SOUL_SUFFIX)])

    canonical = os.path.join(directory, _collapse_soul_stem(basename) + _SOUL_SUFFIX)

    if basename.endswith(_SOUL_SUFFIX):
        if path != canonical:
            # Explicitly named the LEGACY double-append spelling. Honour it —
            # canonical-first would silently swap in the neighbour checkpoint.
            return _dedupe([path, canonical])
        # Canonical as given: fall back to the legacy file that may sit beside it.
        second = canonical + _SOUL_SUFFIX
    else:
        second = os.path.join(directory, _collapse_soul_stem(basename) + _SLO_SUFFIX)

    return _dedupe([canonical, second])


def is_soul_file(path: str) -> bool:
    """True if *path* begins with the soul magic. Never raises.

    The extension is a claim; the header is evidence. Dispatch falls back to
    this whenever the suffix says nothing — a soul renamed ``.bin`` still
    loads, while a ``.pt``/``.npz``/typo is refused up front by name instead
    of dying deep inside the reader with a ``struct.error``.

    Returns False for missing files, directories, and unreadable paths alike:
    absence and illegibility are both "not a soul" from a prober's view.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(len(SOU_MAGIC)) == SOU_MAGIC
    except OSError:
        return False


@dataclass(frozen=True)
class SoulVariant:
    """One concrete file competing for a name.

    Two of these under a single checkpoint name means the name does not
    identify anything — which is the whole reason classification exists.
    """

    path: str
    """Where this variant actually lives on disk, as named."""
    integrity_hash: str
    """Content hash from its own sidecar. ``""`` when there is none."""
    size: int
    mtime: float


@dataclass(frozen=True)
class SoulIdentity:
    """What a model file actually is, derived from its own bytes.

    Filenames do not identify checkpoints. ``journey_select_trained.soul`` and
    ``journey_select_trained.soul.soul`` are two DIFFERENT models — different
    bytes, different training runs — yet both sidecars declare
    ``name=sloughgpt``, ``version=1.0.0``, ``base_model=sloughgpt`` and an
    empty description. Everything that *looks* like an identity is boilerplate.

    The one field that separates them is ``integrity_hash``: content-derived,
    written at save time, and — before this type existed — never read by
    anything.
    """

    path: str
    """Canonical spelling of the input — ``soul_path`` applied."""
    resolved: str | None
    """The file a reader would actually return, or None if nothing is on disk."""
    spelling: str
    """How the input was spelled: ``canonical``, ``legacy-double``,
    ``bare``, ``alias``, ``interchange``, ``foreign``, or ``empty``."""
    format: str
    """Evidence-based container type: ``soul`` when the magic bytes match,
    ``not-soul`` when the name is soul-ish (``.soul``/``.sou``/``.slo``) but
    the header is not a SOUL container, else the extension. Empty when there
    is no file to inspect."""
    exists: bool
    size: int
    mtime: float
    integrity_hash: str
    """Content hash from the sidecar — the real unique key. ``""`` if absent."""
    born_at: str
    final_train_loss: float | None
    tier: str
    """Declared container tier (``simple``/``canonical``/``interchange``/
    ``runtime``), falling back to :data:`SOUL_TIER_POLICY` for files written
    before the field existed. ``""`` when there is nothing to classify."""
    provenance: str
    """Declared ``record=`` provenance. ``""`` means unknown — a file that
    predates the field is reported unknown, never inferred from its name."""
    siblings: tuple[str, ...]
    """Other spellings that also exist on disk beside :attr:`resolved`."""
    variants: tuple[SoulVariant, ...]
    """Every spelling that exists on disk, each with its own hash — the
    answer to "how many distinct checkpoints are competing for this name?"."""
    shadowed: bool
    """A sibling spelling exists AND holds different content — two distinct
    checkpoints sharing one stem. The sibling cannot be reached by the
    canonical name alone; it must be named in its own (legacy) spelling,
    which is exactly the guesswork a filename should not demand of anyone."""
    orphan_legacy: bool
    """Only the legacy spelling exists; the canonical name is absent."""


# A sidecar is not just identity. It also carries ``metadata.training_state``
# — the optimizer and scheduler state training resumes from, read by
# `train_pipeline` and pinned by its tests — and in a real checkpoint that
# blob runs to 40MB inside a 69MB sidecar. Parsing all of it to read a
# handful of fields costs 3.4s per file: 82s to classify one models/
# directory, and 17.9s to serve GET /training/checkpoints. Above this size,
# take the head instead.
_FULL_PARSE_LIMIT = 4 * 1024 * 1024
_HEAD_BYTES = 256 * 1024


def read_sidecar(path: str) -> dict:
    """Sidecar for an EXISTING file, looked up as that file is actually named.

    ``soul_meta_path`` canonicalizes, which is what a writer wants but is
    wrong here: a legacy ``x.soul.soul`` carries its own
    ``x.soul.soul.meta.json``, and canonicalizing first would quietly hand
    back the sidecar of the DIFFERENT checkpoint sitting at ``x.soul`` —
    making two distinct models compare as identical. As-named first, with the
    canonical spelling as fallback for legacy files written before sidecars
    tracked the name.

    Everything the checkpoint declares about itself comes back EXCEPT
    ``metadata.training_state`` — resume state that identity, loss and config
    readers never want, and one that can outweigh the model it sits beside.
    Dropping it always, rather than only when the file is large, keeps the
    result independent of file size; a caller should not have to know how big
    a sidecar is to know what it gets back.
    """
    for candidate in (path + ".meta.json", _canonical_sidecar(path)):
        if not candidate:
            continue
        loaded = _load_sidecar(candidate)
        if loaded is not None:
            metadata = loaded.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("training_state", None)
            return loaded
    return {}


def _load_sidecar(candidate: str) -> dict | None:
    """Parse *candidate*, or None when it is unreadable or not an object."""
    try:
        if os.path.getsize(candidate) > _FULL_PARSE_LIMIT:
            loaded = _read_sidecar_head(candidate)
        else:
            with open(candidate, encoding="utf-8") as handle:
                loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _read_sidecar_head(candidate: str) -> dict:
    """A sidecar without its training blob, read from a file too large to parse.

    ``metadata.training_state`` is the last key inside ``metadata`` and the
    only one that can be enormous, so everything ahead of it is ordinary
    JSON: closing the open objects at the comma before it parses ~2KB of
    sidecar for the price of a 256KB read rather than a 69MB one — measured
    3.4s down to sub-millisecond.

    Depth is not assumed — ``}}`` then ``}`` — so a layout that nests the
    blob elsewhere still closes legally. If neither yields an
    ``integrity_hash``, the layout is not the one this understands: pay for
    the full parse rather than report a real hash as absent.
    """
    with open(candidate, encoding="utf-8") as handle:
        head = handle.read(_HEAD_BYTES)
    marker = head.find('"training_state"')
    if marker != -1:
        split = head.rfind(",", 0, marker)
        if split != -1:
            for closing in ("}}", "}"):
                try:
                    loaded = json.loads(head[:split] + closing)
                except ValueError:
                    continue
                if isinstance(loaded, dict) and loaded.get("integrity_hash"):
                    return loaded
    with open(candidate, encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _canonical_sidecar(path: str) -> str | None:
    """Canonical sidecar for *path*, or None when the grammar refuses it.

    Foreign suffixes raise out of :func:`soul_path`; classification must keep
    going and report what the file is rather than die on what it isn't.
    """
    try:
        return soul_meta_path(path)
    except ValueError:
        return None


def _fingerprint(path: str, meta: dict) -> str:
    """Strongest cheap identity signal available for *path*.

    Prefers the content hash; falls back to size when a checkpoint has no
    sidecar, so a mismatch is still detectable rather than assumed away.
    """
    digest = str(meta.get("integrity_hash") or "")
    if digest:
        return f"hash:{digest}"
    try:
        return f"size:{os.path.getsize(path)}"
    except OSError:
        return "size:?"


def classify_soul(path: str) -> SoulIdentity:
    """Derive what *path* actually is — never raises.

    Probes every spelling :func:`soul_read_candidates` allows, so the caller
    learns not just what would load but what else is hiding behind that name:

    - ``shadowed``      two spellings exist and DIFFER — one stem holding two
      checkpoints. The sibling cannot be reached by the canonical name alone,
      which is how a 5.7MB checkpoint with the better loss sat invisible
      behind its own name
    - ``orphan_legacy`` only the legacy spelling exists, so the canonical
      name alone would raise ``FileNotFoundError``

    :attr:`integrity_hash` always describes the file a reader would actually
    return, not the spelling you asked for. For a shadowed pair both inputs
    therefore report the SAME hash — the distinction lives in
    :attr:`variants`, which lists every competing file with its own.

    Foreign suffixes (``.pt``, ``.gguf``, ...) are reported rather than
    refused: classifying a file is exactly the operation you reach for when
    you do not know what it is, so raising there would be self-defeating.

    Args:
        path: any spelling the grammar accepts, plus foreign ones.

    Returns:
        A :class:`SoulIdentity`. ``resolved`` is None when nothing is on disk,
        and every flag is False in that case.
    """
    try:
        canonical = soul_path(path)
    except ValueError:
        canonical = path

    basename = os.path.basename(path)
    if not basename:
        spelling = "empty"
    elif basename.endswith(_SLO_SUFFIX):
        spelling = "interchange"
    elif basename.endswith(_SOUL_SUFFIX + _SOUL_SUFFIX):
        spelling = "legacy-double"
    elif basename.endswith(_SOUL_SUFFIX):
        spelling = "canonical"
    elif basename.endswith(_SOU_ALIAS):
        spelling = "alias"
    elif os.path.basename(canonical).endswith(_SOUL_SUFFIX):
        spelling = "bare"
    else:
        spelling = "foreign"

    if spelling == "foreign":
        # Not a soul spelling at all. Running it through the grammar would
        # invent "model.safetensors.soul" candidates that cannot exist and
        # report a file that IS on disk as missing — classifying is exactly
        # the operation you reach for when you don't know what you're holding.
        candidates = [path]
    else:
        try:
            candidates = soul_read_candidates(path)
        except ValueError:
            candidates = [path]

    existing = [c for c in candidates if os.path.isfile(c)]
    resolved = existing[0] if existing else None
    siblings = tuple(existing[1:])

    metas = {candidate: read_sidecar(candidate) for candidate in existing}

    fmt = ""
    if resolved:
        if is_soul_file(resolved):
            fmt = "soul"
        elif resolved.endswith((_SOUL_SUFFIX, _SLO_SUFFIX, _SOU_ALIAS)):
            # The name claims soul and the header disagrees — a mislabeled
            # or truncated artifact, worth naming rather than rounding up.
            fmt = "not-soul"
        else:
            fmt = os.path.splitext(resolved)[1].lstrip(".").lower() or "unknown"

    chosen_meta = metas.get(resolved, {}) if resolved else {}
    shadowed = bool(resolved) and any(
        _fingerprint(sib, metas[sib]) != _fingerprint(resolved, chosen_meta) for sib in siblings
    )

    loss_raw = chosen_meta.get("final_train_loss")
    try:
        loss: float | None = float(loss_raw) if loss_raw is not None else None
    except (TypeError, ValueError):
        loss = None

    # tier: what the file DECLARES, else the suffix policy as a display
    # fallback for artifacts written before the field existed. The fallback is
    # withheld when the name claims soul but the header disagrees — that file
    # is mislabeled, and stamping it "canonical" would launder the claim.
    # provenance is never inferred: absent simply means unknown.
    tier = str(chosen_meta.get("tier") or "")
    if not tier and resolved and fmt != "not-soul":
        tier = SOUL_TIER_POLICY.get(os.path.splitext(resolved)[1].lower(), "")
    provenance = str(chosen_meta.get("provenance") or "")

    variants: list[SoulVariant] = []
    size = 0
    mtime = 0.0
    for candidate in existing:
        try:
            st = os.stat(candidate)
            vsize, vmtime = st.st_size, st.st_mtime
        except OSError:
            vsize, vmtime = 0, 0.0
        variants.append(
            SoulVariant(
                path=candidate,
                integrity_hash=str(metas[candidate].get("integrity_hash") or ""),
                size=vsize,
                mtime=vmtime,
            )
        )
        if candidate == resolved:
            size, mtime = vsize, vmtime

    return SoulIdentity(
        path=canonical,
        resolved=resolved,
        spelling=spelling,
        format=fmt,
        exists=resolved is not None,
        size=size,
        mtime=mtime,
        integrity_hash=str(chosen_meta.get("integrity_hash") or ""),
        born_at=str(chosen_meta.get("born_at") or ""),
        final_train_loss=loss,
        tier=tier,
        provenance=provenance,
        siblings=siblings,
        variants=tuple(variants),
        shadowed=shadowed,
        orphan_legacy=bool(resolved) and not os.path.isfile(canonical),
    )


def save_soul(
    model,
    output_path: str,
    soul_profile: SloProfile | None = None,
    weights_only: bool = False,
    record: str = "",
) -> str:
    """Export model to .soul format (binary: header + config + JSON weights).

    Uses pure Python/numpy binary format.

    Args:
        model: any object with a ``state_dict()`` method (PyTorch, SloNet, etc.)
        output_path: destination file path — canonicalized via :func:`soul_path`
            before anything is written, so a bare stem or legacy spelling lands
            on the one canonical name.
        soul_profile: optional SloProfile with personality/identity metadata
        weights_only: if True, skip writing weight data (header + config only)
        record: provenance — what produced this artifact. Must be one of
            :data:`SOUL_PROVENANCE` (``training`` / ``export`` /
            ``distillation``); anything else raises ``ValueError`` rather than
            being written into the file's permanent identity. Written onto the
            profile, so it reaches the header and the sidecar in one step;
            empty leaves any provenance already on *soul_profile* alone.

    Returns:
        the CANONICAL path written (for chaining) — callers must use this
        rather than re-deriving a sidecar path from their own argument

    Side effects:
        - Writes .soul binary file
        - Writes .soul.meta.json companion file with readable metadata
        - Creates parent directories if missing
    """
    output_path = soul_path(output_path)

    if soul_profile is None:
        soul_profile = SloProfile(name=Path(output_path).stem)

    if (
        not soul_profile.metadata
        and hasattr(model, "metadata")
        and isinstance(model.metadata, dict)
    ):
        soul_profile.metadata = dict(model.metadata)
    if not soul_profile.lineage and hasattr(model, "lineage"):
        soul_profile.lineage = model.lineage

    # Both fields are declared HERE because this is the only moment they are
    # knowable: soul_path() has already canonicalized output_path, so the tier
    # is simply the suffix about to be written, and provenance exists nowhere
    # but the caller's record argument. Reading either back later and guessing
    # would defeat the point of declaring them.
    tier = SOUL_TIER_POLICY.get(Path(output_path).suffix)
    if tier is not None:
        soul_profile.tier = tier
    if record:
        # Enforced here, at the boundary, because the value is stamped into
        # the header and sidecar and never rewritten: a typo would become
        # permanent identity metadata no reader could correct.
        if record not in SOUL_PROVENANCE:
            raise ValueError(
                f"{record!r} is not a declared provenance. Valid values: "
                f"{', '.join(SOUL_PROVENANCE)}. Add a new one to "
                f"SOUL_PROVENANCE rather than passing a free string."
            )
        soul_profile.provenance = record

    soul_profile.integrity_hash = soul_profile.compute_hash()
    config_json = json.dumps(
        _soul_json_sanitize(soul_profile.to_dict()),
        default=str,
        allow_nan=False,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Atomic write: write to temp file, then rename
    import tempfile

    # Write .meta.json first (small, fast — serves as sidecar for list endpoint).
    # Atomic via temp + rename so a crash never leaves a partial sidecar that
    # could be misread as matching the soul. Derived from the CANONICAL path so
    # the pair always lands together.
    meta_path = soul_meta_path(output_path)
    meta_fd, meta_tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path) or ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(meta_fd, "w", encoding="utf-8") as f:
            json.dump(
                _soul_json_sanitize(soul_profile.to_dict()),
                f,
                indent=2,
                default=str,
                allow_nan=False,
            )
        os.rename(meta_tmp_path, meta_path)
    except Exception:
        try:
            os.unlink(meta_tmp_path)
        except OSError:
            pass
        raise

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path) or ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            params: list[tuple[str, np.ndarray]] = []
            if not weights_only:
                state = model.state_dict() if hasattr(model, "state_dict") else {}
                for k, v in state.items():
                    try:
                        # SloNet Tensor: has .data attribute that is a numpy ndarray
                        if hasattr(v, "data") and isinstance(v.data, np.ndarray):
                            arr = v.data.astype(np.float32)
                        elif hasattr(v, "numpy"):
                            # PyTorch tensor
                            arr = v.cpu().numpy().astype(np.float32)
                        elif hasattr(v, "detach"):
                            # PyTorch tensor without .numpy()
                            arr = v.detach().cpu().numpy().astype(np.float32)
                        elif isinstance(v, np.ndarray):
                            arr = v.astype(np.float32)
                        elif isinstance(v, (list, tuple)):
                            arr = np.asarray(v, dtype=np.float32)
                        elif isinstance(v, dict):
                            logger.debug("Skipping non-tensor state_dict key: %s (dict value)", k)
                            continue
                        else:
                            arr = np.asarray(v, dtype=np.float32)
                        params.append((k, arr))
                    except (TypeError, ValueError) as e:
                        logger.warning("Skipping state_dict key %s: %s", k, e, extra={"tag": "INF"})
                        continue
                if len(params) == 0 and len(state) > 0:
                    logger.error(
                        "save_soul: wrote 0 params out of %d state_dict keys — "
                        "checkpoint will be unusable. Model type: %s",
                        len(state),
                        type(model).__name__,
                        extra={"tag": "INF"},
                    )

            _write_soul_v3(
                f,
                config_json.encode("utf-8"),
                params,
                include_weights=not weights_only,
            )

        os.rename(tmp_path, output_path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return output_path


def load_soul(sou_path: str):
    """Load a .soul file and return (SloProfile, state_dict).

    Reads v1/v2 (JSON weights) and v3 (binary float32) formats.

    Probes :func:`soul_read_candidates` in order, so a canonical name finds a
    legacy double-appended artifact on disk and a legacy name finds the
    canonical one. A candidate that exists but is not a SOUL container (a
    plain-text ``.slo`` profile, a truncated file) is recorded and skipped so
    the probe can advance to the next spelling.

    Args:
        sou_path: path to a .soul file — any spelling the grammar accepts

    Returns:
        (SloProfile, state_dict) where state_dict maps param names → numpy arrays.
        All versions return state_dict with flat array keys even if the original
        used state_dict-style keys internally.

    Raises:
        FileNotFoundError: no candidate exists on disk.
        ValueError: a candidate existed but none parsed as a SOUL container.

    Side effects:
        - Reads file from disk
    """
    candidates = soul_read_candidates(sou_path)
    last_err: Exception | None = None
    skipped: list[str] = []
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            result = _load_soul_file(candidate)
        except (ValueError, struct.error, UnicodeDecodeError) as exc:
            # Wrong format or corruption under this spelling — keep the reason,
            # keep probing; the next candidate may be the real checkpoint.
            last_err = exc
            skipped.append(candidate)
            continue
        if skipped:
            # We found the checkpoint, but not under the name that was asked
            # for. Silent substitution would hand back a DIFFERENT file than
            # the caller named — say so, with both paths.
            logger.warning(
                "load_soul: %s is present but not a SOUL container; loaded %s instead "
                "(candidates: %s)",
                skipped[0],
                candidate,
                ", ".join(candidates),
                extra={"tag": "INF"},
            )
        # Canonical-first picks a SPELLING, not a checkpoint. When both
        # spellings exist as genuinely different files, the loser is
        # unreachable by any input — and it may be the better model (a 5.7MB
        # checkpoint with the lower loss sat invisible behind its own name).
        # Resolve silently is not an option here: say which one won and which
        # one is now unreachable.
        if any(c != candidate and os.path.isfile(c) for c in candidates):
            identity = classify_soul(sou_path)
            if identity.shadowed:
                others = [
                    f"{v.path} (hash {v.integrity_hash})"
                    for v in identity.variants
                    if v.path != identity.resolved
                ]
                # Not an error: both are reachable, but only by knowing the
                # second spelling. Say so — silently returning one of two
                # different models is the failure mode worth logging.
                logger.warning(
                    "load_soul: %s is one of %d DIFFERENT checkpoints sharing "
                    "this name; loaded %s (hash %s). Also on disk, reachable "
                    "only by its own spelling: %s",
                    sou_path,
                    len(identity.variants),
                    identity.resolved,
                    identity.integrity_hash,
                    "; ".join(others),
                    extra={"tag": "INF"},
                )
        return result
    if last_err is not None:
        # A plain-text .slo profile can open with the four bytes "SOUL", so it
        # clears the magic check and dies later in struct.unpack. Normalize
        # every parse failure to ValueError — callers must never see a raw
        # struct.error escape from what the docstring documents as a ValueError.
        if isinstance(last_err, ValueError):
            raise last_err
        raise ValueError(
            f"Invalid .soul file: {candidates} (unparseable: {last_err})"
        ) from last_err
    raise FileNotFoundError(
        f"No .soul checkpoint found at {sou_path!r} (probed: {', '.join(candidates)})"
    )


def _load_soul_file(sou_path: str):
    """Read exactly one file as a SOUL container.

    No probing — :func:`load_soul` owns candidate order. Split out so the
    reader stays a single-file operation and the probe loop can catch a
    format mismatch here without unwinding the whole search.
    """
    import numpy as np

    with open(sou_path, "rb") as f:
        magic = f.read(4)
        if magic != SOU_MAGIC:
            raise ValueError(f"Invalid .soul file: {sou_path} (magic={magic!r})")

        version = struct.unpack("<I", f.read(4))[0]
        config_len = struct.unpack("<I", f.read(4))[0]
        config_json = f.read(config_len).decode("utf-8")

        state_dict = {}
        if version >= 3:
            # v3 binary float32 format
            num_params_raw = f.read(4)
            if len(num_params_raw) < 4:
                # weights_only files carry no weight section
                num_params = 0
            else:
                num_params = struct.unpack("<I", num_params_raw)[0]
            for _ in range(num_params):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                ndim = struct.unpack("<I", f.read(4))[0]
                dims = tuple(struct.unpack(f"<{ndim}I", f.read(4 * ndim)))
                count = 1
                for d in dims:
                    count *= d
                raw = f.read(count * 4)
                state_dict[name] = np.frombuffer(raw, dtype=np.float32).copy().reshape(dims)
        elif version >= 1:
            # v1/v2 JSON weight format
            try:
                state_len = struct.unpack("<I", f.read(4))[0]
                if state_len > 0:
                    state_json = f.read(state_len).decode("utf-8")
                    state_raw = json.loads(state_json)
                    for k, v in state_raw.items():
                        arr = np.array(v, dtype=np.float32)
                        state_dict[k] = arr
            except Exception as e:
                logger.error(
                    "v1/v2 JSON weight parse failed for %s: %s", sou_path, e, extra={"tag": "INF"}
                )
                raise ValueError(f"Corrupted .soul file weights: {sou_path} — {e}") from e

    config = json.loads(config_json)
    soul = SouParser.parse(
        f"SOUL {config.get('name', 'unknown')}\n"
        + f"VERSION {config.get('version', '1.0.0')}\n"
        + f"LINEAGE {config.get('lineage', 'nanogpt')}\n"
        + f"BORN {repair_iso(config.get('born_at', ''))}\n"
        + f"BASEMODEL {config.get('base_model', '')}\n"
        + f"DESCRIPTION {config.get('description', '')}\n"
    )
    soul.__dict__.update(config)

    # config JSON stores structured metadata as plain dicts; rehydrate the
    # dataclasses so consumers get typed attributes (e.g. soul.personality.warmth).
    _structured = {
        "personality": PersonalityCore,
        "behavior": BehavioralTraits,
        "cognition": CognitiveSignature,
        "emotion": EmotionalRange,
        "generation": GenerationParams,
        "context": ContextParams,
    }
    for _name, _cls in _structured.items():
        _raw = config.get(_name)
        if isinstance(_raw, dict):
            try:
                _built = _cls(**_raw)
            except (TypeError, ValueError):
                logger.debug("Failed to rehydrate %s on %s", _name, sou_path, extra={"tag": "INF"})
            else:
                setattr(soul, _name, _built)

    return soul, state_dict


def write_v3_sou(
    outpath: str,
    metadata: dict,
    state_dict: dict,
) -> str:
    """Write a v3 binary .soul file (raw float32 weights, 82% smaller than v2 JSON).

    v3 format:
      [4 bytes] SOU_MAGIC = b"SOUL"                                      (line_break)
      [4 bytes] version (uint32 LE, =3)
      [4 bytes] json_len (uint32 LE)
      [json_len bytes] JSON metadata (UTF-8)
      [4 bytes] num_params (uint32 LE)
      For each param i in 0..N-1:
        [4 bytes] name_len (uint32 LE)
        [name_len bytes] key name (UTF-8, e.g. "p0")
        [4 bytes] ndim (uint32 LE)
        [ndim * 4 bytes] dims (uint32 LE each)
        [prod(dims) * 4 bytes] raw float32 data (little-endian)

    Args:
        outpath: destination file path
        metadata: JSON-serializable dict (soul_name, traits, lineage, etc.)
        state_dict: ordered dict of ``{param_name: numpy_array_or_list}``

    Returns:
        outpath (for chaining)

    Side effects:
        - Writes binary file to disk
        - Creates parent directories if missing
    """
    import numpy as np

    params = {
        f"p{i}": np.asarray(v, dtype=np.float32) for i, (_, v) in enumerate(state_dict.items())
    }
    meta_bytes = json.dumps(metadata, allow_nan=False, default=str).encode()
    num_params = len(params)

    with open(outpath, "wb") as f:
        f.write(SOU_MAGIC)
        f.write(struct.pack("<I", SOU_VERSION_V3))
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<I", num_params))
        for key in sorted(params.keys(), key=lambda k: int(k[1:])):
            arr = params[key]
            name_bytes = key.encode()
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", arr.ndim))
            for dim in arr.shape:
                f.write(struct.pack("<I", dim))
            f.write(arr.tobytes())

    return outpath


def generate_sample_dialogue(
    model,
    stoi: dict[int, str],
    itos: dict[int, str],
    num_turns: int = 3,
    max_tokens: int = 50,
) -> list[dict[str, str]]:
    """Generate sample dialogue to populate the soul profile."""
    import numpy as np

    prompts = [
        ("user", "Hello! How are you today?"),
        ("user", "What's your favorite thing about helping people?"),
        ("user", "Can you tell me a short joke?"),
    ]

    dialogue = []

    for role, prompt in prompts[:num_turns]:
        idx = np.array([[stoi.get(c, 0) for c in prompt]], dtype=np.int64)
        try:
            if hasattr(model, "generate"):
                output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8)
            elif hasattr(model, "forward"):
                output = model.forward(idx)
            else:
                output = idx
            response = "".join([itos.get(int(i), "?") for i in np.asarray(output).flatten()])
            response = response[len(prompt) :].strip()
        except Exception:
            response = "[generation failed]"
        dialogue.append({"role": role, "content": prompt})
        dialogue.append({"role": "assistant", "content": response[:100]})

    return dialogue


__all__ = [
    "SloProfile",
    "PersonalityCore",
    "BehavioralTraits",
    "CognitiveSignature",
    "EmotionalRange",
    "GenerationParams",
    "ContextParams",
    "SouParser",
    "create_soul_profile",
    "save_soul",
    "load_soul",
    "write_v3_sou",
    "generate_sample_dialogue",
    "soul_path",
    "soul_meta_path",
    "soul_read_candidates",
    "is_soul_file",
    "classify_soul",
    "SoulIdentity",
    "SoulVariant",
    "SOUL_SUFFIXES",
    "SOUL_TIER_POLICY",
    "SOUL_PROVENANCE",
    "SOUL_PROVENANCE_TRAINING",
    "SOUL_PROVENANCE_EXPORT",
    "SOUL_PROVENANCE_DISTILLATION",
    "read_sidecar",
    "SOU_MAGIC",
    "SOU_VERSION",
    "SOU_VERSION_V3",
    "SOU_TRADEMARK",
]
