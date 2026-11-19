"""Format sniffers — the one seam dataset files are recognized through.

Card 20261114_151505 (pairs with ADR-002's adapter registry): a format
sniff returns a :class:`SniffResult` naming the **adapter that consumes
it** — not just a label. Image/video declare
``capabilities.unsupported`` instead of fake-success: ``IMAGE_TEXT``
detection today labels data nothing can consume, and the round-trip gate
would rather see an honest dead-end than an adapter name that resolves to
nothing.

Registration is the microkernel pattern (one entry per format, same shape
as the router manifest): a sniffer declares ``name`` + ``formats`` (its
fixture ids — the gate walks both sides) and claims paths in priority
order.

Grandfathered: conversation / instruction / audio stay in
``detect_dataset_type``'s if-chain until their adapters register
(``chat-sft``; ``tts-espeak`` after the ADR's byte-parity requirement).
The if-chain shrinks as sniffers migrate — never a big-bang rewrite.

Precedence is legacy-exact: the sniffers below only claim a jsonl record
when no *earlier* legacy elif-key is present, so ``{audio, image}`` still
detects as AUDIO_TEXT (guard-tested).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from domain.training._internal.training import DatasetType

#: Capabilities a sniff may declare — ``unsupported`` is the honest
#: dead-end for formats no registered adapter consumes (card 20261114_151505 (4)).
KNOWN_CAPABILITIES = frozenset({"unsupported"})


@dataclass(frozen=True)
class SniffResult:
    """What one format sniff resolved to: label + consuming adapter (or none)."""

    dataset_type: DatasetType
    #: ``AdapterSpec.name`` that consumes this format, or None when none does.
    adapter: str | None = None
    #: Declared capabilities — ``{"unsupported"}`` = no adapter claims it (yet).
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.capabilities - KNOWN_CAPABILITIES
        if unknown:
            raise ValueError(f"unknown capabilities: {sorted(unknown)}")
        if self.adapter is not None and "unsupported" in self.capabilities:
            raise ValueError("an unsupported result must not name an adapter")


@dataclass(frozen=True)
class FormatSniffer:
    """One registry entry: identity + claim function (microkernel shape)."""

    name: str
    #: Format ids this sniffer claims — must match the gate's fixtures 1:1.
    formats: tuple[str, ...]
    claim: Callable[[Path], SniffResult | None]

    def __call__(self, path: Path) -> SniffResult | None:
        return self.claim(path)


_SNIFFERS: dict[str, FormatSniffer] = {}


def register_sniffer(sniffer: FormatSniffer) -> FormatSniffer:
    """Admit one sniffer — fail-fast at import time, like adapters do."""
    if not isinstance(sniffer, FormatSniffer):
        raise TypeError(f"expected FormatSniffer, got {type(sniffer).__name__}")
    if not sniffer.name:
        raise ValueError("sniffer name must be non-empty")
    if sniffer.name in _SNIFFERS:
        raise ValueError(f"duplicate sniffer name {sniffer.name!r}")
    if not sniffer.formats:
        raise ValueError(f"{sniffer.name}: declares no formats")
    _SNIFFERS[sniffer.name] = sniffer
    return sniffer


def sniff_dataset(path: Path | str) -> SniffResult | None:
    """First matching sniffer wins; None = grandfathered to the if-chain.

    Directories are never claimed here — ``detect_dataset_type`` owns the
    walk-and-recurse behaviour (parity with the legacy dir branch).
    """
    p = Path(path)
    if p.is_dir():
        return None
    for sniffer in _SNIFFERS.values():
        try:
            result = sniffer(p)
        except Exception:
            # Legacy tolerance: a broken/unreadable file degrades to the
            # fallback (detect's if-chain ends at TEXT), never raises —
            # the old content checks all lived behind `except Exception`.
            continue
        if result is not None:
            return result
    return None


# ── built-in sniffers (registered at import — one entry per format) ──────

_CODE_EXTENSIONS = frozenset({".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rb", ".php"})
_CONTENT_CHECK_EXTENSIONS = frozenset({".rs", ".jsx", ".tsx", ".txt"})
_CODE_PATTERNS = ("def ", "class ", "import ", "function ", "const ", "let ", "var ")

# Legacy elif order (training.py) — earlier keys must keep winning.
_CONV_KEYS = ("messages", "conversation")
_INSTR_KEYS = ("instruction",)
_AUDIO_KEYS = ("audio", "speech", "wav")
_IMAGE_KEYS = ("image", "jpg", "png")
_VIDEO_KEYS = ("video",)


def _read_first_line(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.loads(f.readline())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _claim_code_ext(path: Path) -> SniffResult | None:
    if path.suffix.lower() in _CODE_EXTENSIONS:
        return SniffResult(dataset_type=DatasetType.CODE, adapter="text-pretrain")
    return None


def _claim_code_content(path: Path) -> SniffResult | None:
    if path.suffix.lower() not in _CONTENT_CHECK_EXTENSIONS:
        return None
    try:
        with open(path) as f:
            content = f.read(1000)
    except (OSError, UnicodeDecodeError):
        # binary .txt etc. — legacy fell through to TEXT on any read error
        return None
    matches = sum(1 for pat in _CODE_PATTERNS if pat in content)
    if matches >= 2:
        return SniffResult(dataset_type=DatasetType.CODE, adapter="text-pretrain")
    return None


def _claim_jsonl_image(path: Path) -> SniffResult | None:
    if path.suffix.lower() != ".jsonl":
        return None
    data = _read_first_line(path)
    if data is None:
        return None
    # elif precedence: no earlier key may be present (conv/instr/audio).
    if any(k in data for k in _CONV_KEYS + _INSTR_KEYS + _AUDIO_KEYS):
        return None
    if any(k in data for k in _IMAGE_KEYS):
        return SniffResult(
            dataset_type=DatasetType.IMAGE_TEXT,
            capabilities=frozenset({"unsupported"}),
        )
    return None


def _claim_jsonl_video(path: Path) -> SniffResult | None:
    if path.suffix.lower() != ".jsonl":
        return None
    data = _read_first_line(path)
    if data is None:
        return None
    # elif precedence: conv/instr/audio AND image all come before video.
    if any(k in data for k in _CONV_KEYS + _INSTR_KEYS + _AUDIO_KEYS + _IMAGE_KEYS):
        return None
    if any(k in data for k in _VIDEO_KEYS):
        return SniffResult(
            dataset_type=DatasetType.VIDEO_TEXT,
            capabilities=frozenset({"unsupported"}),
        )
    return None


def _claim_text(path: Path) -> SniffResult | None:
    # .jsonl / .json stay with the grandfathered if-chain (their conv /
    # instruction / audio branches are not migrated until their adapters
    # register); everything else with no earlier claim is plain text.
    if path.suffix.lower() in (".jsonl", ".json"):
        return None
    return SniffResult(dataset_type=DatasetType.TEXT, adapter="text-pretrain")


register_sniffer(FormatSniffer("code-ext", ("code-ext",), _claim_code_ext))
register_sniffer(FormatSniffer("code-content", ("code-content",), _claim_code_content))
register_sniffer(FormatSniffer("jsonl-image", ("jsonl-image",), _claim_jsonl_image))
register_sniffer(FormatSniffer("jsonl-video", ("jsonl-video",), _claim_jsonl_video))
register_sniffer(FormatSniffer("text", ("text",), _claim_text))

#: The registered sniffers, in claim order — what the gate walks.
SNIFFERS: tuple[FormatSniffer, ...] = tuple(_SNIFFERS.values())
