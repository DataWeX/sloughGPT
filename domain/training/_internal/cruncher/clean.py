"""Cruncher cleaning — MinHash/LSH dedup, quality filter, PII scrub.

All three stages are pure functions / small stateless classes over text, so
they compose in the streaming pipeline without holding the corpus in memory.
Dedup state (fingerprints) lives in :class:`DedupIndex`, which the pipeline
persists across runs — a record arriving now is checked against the entire
corpus history, not a batch boundary.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("slo.training.cruncher")

# ── PII scrub ──────────────────────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "PHONE",
        re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
    ),
]


def scrub_pii(text: str) -> tuple[str, int]:
    """Replace PII-shaped spans with typed tokens.

    Returns ``(scrubbed_text, replacement_count)``.  Order matters: SSN is
    scrubbed before PHONE so ``123-45-6789`` never half-matches as a phone.
    """
    count = 0
    for name, pattern in _PII_PATTERNS:
        text, n = pattern.subn(f"<{name}>", text)
        count += n
    return text, count


# ── quality filter ─────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\S+")


@dataclass
class QualityThresholds:
    """Knobs for the heuristic quality gate (embedded-scale defaults)."""

    min_chars: int = 20
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.5
    max_repetition: float = 0.4


def _repetition_rate(text: str) -> float:
    """Fraction of tokens held by the most common token (lower is better)."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 1.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words)


def _alpha_ratio(text: str) -> float:
    """Fraction of non-space characters that are alphabetic."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isalpha()) / len(chars)


def quality_ok(text: str, thresholds: QualityThresholds | None = None) -> bool:
    """Heuristic quality gate for raw text.

    Drops empty/tiny/huge records, mostly-non-alphabetic blobs (base64 dumps,
    binary pastes), and degenerate repetition (spam, tokenizer loops).  Pair
    training data should be scored with ``quality_scorer.score_pair`` upstream
    instead — this gate is for raw document text.
    """
    th = thresholds or QualityThresholds()
    if len(text) < th.min_chars or len(text) > th.max_chars:
        return False
    if _alpha_ratio(text) < th.min_alpha_ratio:
        return False
    if _repetition_rate(text) > th.max_repetition:
        return False
    return True


# ── MinHash / LSH dedup ────────────────────────────────────────────────

_SHINGLE_NGRAM = 5
_NUM_PERM = 64
_BANDS = 16  # 64 perms / 16 bands = 4 rows per band
DEFAULT_JACCARD_THRESHOLD = 0.85

# 31-bit Mersenne prime: keeps (a * x + b) < prime^2 < 2^62 inside uint64.
_PRIME = (1 << 31) - 1
_MAX_HASH = (1 << 32) - 1


def _shingle_hash(shingle: str) -> int:
    """Stable 32-bit hash (blake2b) — deterministic across runs/processes."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def text_fingerprint(text: str, num_perm: int = _NUM_PERM) -> np.ndarray:
    """MinHash signature of *text* over word n-gram shingles.

    Deterministic (fixed universal-hash coefficients): the same text always
    yields the same signature, so fingerprints stored in old snapshots stay
    comparable with newly computed ones.
    """
    words = text.lower().split()
    if not words:
        return np.full(num_perm, _MAX_HASH, dtype=np.uint64)
    if len(words) < _SHINGLE_NGRAM:
        shingles = {" ".join(words)}
    else:
        shingles = {
            " ".join(words[i : i + _SHINGLE_NGRAM]) for i in range(len(words) - _SHINGLE_NGRAM + 1)
        }
    hashes = np.array(sorted({_shingle_hash(s) for s in shingles}), dtype=np.uint64)

    # Universal hash family: h_i(x) = (a_i * x + b_i) mod prime, with
    # coefficients derived deterministically from a fixed seed.  Fully
    # vectorized: (num_perm, n_hashes) broadcast, then min over hashes.
    rng = np.random.default_rng(0xC0FFEE)
    a = rng.integers(1, _PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=num_perm, dtype=np.uint64)
    # x < 2^32 and a < 2^31 keep a*x+b < 2^63 — no uint64 overflow.
    x = hashes.astype(np.uint64)
    return np.min((a[:, None] * x[None, :] + b[:, None]) % _PRIME, axis=1)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Jaccard similarity estimate from two MinHash signatures."""
    return float(np.mean(sig_a == sig_b))


@dataclass
class DedupIndex:
    """LSH-banded near-duplicate index over MinHash fingerprints.

    The index is the corpus memory: add every kept record's signature, and
    ``is_near_duplicate`` checks new arrivals against the whole history.
    State is plain arrays so the pipeline can persist and reload it between
    runs (cross-snapshot dedup).
    """

    num_perm: int = _NUM_PERM
    bands: int = _BANDS
    threshold: float = DEFAULT_JACCARD_THRESHOLD
    _signatures: list[np.ndarray] = field(default_factory=list, repr=False)
    _buckets: dict[tuple[int, ...], list[int]] = field(default_factory=dict, repr=False)

    def _band_keys(self, sig: np.ndarray) -> list[tuple[int, ...]]:
        rows = self.num_perm // self.bands
        keys = []
        for band in range(self.bands):
            chunk = sig[band * rows : (band + 1) * rows]
            keys.append(tuple(int(v) for v in chunk))
        return keys

    def is_near_duplicate(self, sig: np.ndarray) -> bool:
        """True when *sig* is within ``threshold`` of anything already indexed."""
        if not isinstance(sig, np.ndarray):
            raise TypeError(
                f"is_near_duplicate expects a MinHash np.ndarray from "
                f"text_fingerprint(), got {type(sig).__name__}"
            )
        candidates: set[int] = set()
        for key in self._band_keys(sig):
            candidates.update(self._buckets.get(key, ()))
        for idx in candidates:
            if estimate_jaccard(sig, self._signatures[idx]) >= self.threshold:
                return True
        return False

    def add(self, sig: np.ndarray) -> int:
        """Index a signature; returns its record id."""
        idx = len(self._signatures)
        self._signatures.append(sig)
        for key in self._band_keys(sig):
            self._buckets.setdefault(key, []).append(idx)
        return idx

    def __len__(self) -> int:
        return len(self._signatures)

    def to_state(self) -> dict:
        """Serializable state for snapshot persistence."""
        return {
            "num_perm": self.num_perm,
            "bands": self.bands,
            "threshold": self.threshold,
            "signatures": [sig.tolist() for sig in self._signatures],
        }

    @classmethod
    def from_state(cls, state: dict) -> DedupIndex:
        """Rebuild an index from :meth:`to_state` output."""
        index = cls(
            num_perm=int(state.get("num_perm", _NUM_PERM)),
            bands=int(state.get("bands", _BANDS)),
            threshold=float(state.get("threshold", DEFAULT_JACCARD_THRESHOLD)),
        )
        for sig in state.get("signatures", []):
            index.add(np.array(sig, dtype=np.uint64))
        return index
