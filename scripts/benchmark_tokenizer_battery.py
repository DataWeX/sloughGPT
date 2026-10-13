#!/usr/bin/env python3
"""Tokenizer battery — one row per ALGORITHM, not per decoder.

Card 8fefc101 (native tokenizer: char + token-tree unified). The earlier
framing counted "five decoders", which was wrong by two: there are three
algorithms and five configurations of them.

  A. MorphTokenizer   external ``tokenizer.json`` loader, ``byte_level=True``
  B. SloBPE/SloUnigram   our two trainers
  C. TokenTree        ONE walker; char and tree are configurations of it
                      (char = depth-1 trie, no normalization, no ``</w>``)

Metrics
  cov   character coverage of the HARDEST case: the share of a case's
        characters present in the vocabulary. Reported first because it is the
        root cause behind F1/F2 shortfalls — a partial tokenizer drops
        out-of-vocab characters silently, so low F1 with low cov is a coverage
        limit while low F1 with full cov is a decode bug.
  F1    full round trip: decode(encode(s)) == s for every case.
  F2    streaming: encode in chunks, concatenate the ids, decode once.
  F3    determinism: encoding twice yields identical ids.
  F4    reload: serialize -> deserialize -> identical encode and decode.
  F5    round trips/sec and megacharacters/sec over a fixed payload. The
        character rate is the comparable figure; rt/s depends on payload size.

Each candidate runs under a per-phase alarm: ``SloUnigram.train`` enumerates
every segmentation of a pretoken (``_all_segmentations``), which is
exponential in pretoken length, so a single long unbroken word can hang.

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/packages/core-py" \\
      python scripts/benchmark_tokenizer_battery.py [--no-long]
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

# --------------------------------------------------------------------------
# Corpus — deliberately tiny, because that is what exposes coverage limits.
# --------------------------------------------------------------------------

CORPUS: list[str] = [
    "the quick brown fox jumps over the lazy dog",
    "hello world from the tokenizer battery",
    "abcabc abc abcabc",
    "numbers 1234567890 and more 98765",
    "CamelCase and snake_case together",
    "line with trailing space ",
    "punctuation, marks; everywhere! really?",
    "the dog barked at the cat",
    "",
]

# 12 cases. The last is a single unbroken 48-character word — the input that
# trips SloUnigram's exponential segmentation, kept deliberately.
CASES: list[str] = [
    "hello world",
    "The quick brown fox jumps over the lazy dog",
    # Title Case, deliberately next to the sentence case above: with exactly one
    # uppercase char in CORPUS, this is the case that separates case-faithful
    # formats (char, morph) from case-folding ones (tree, bpe, unigram).
    "The Quick Brown Fox Jumps Over The Lazy Dog",
    "1234567890 98765",
    "abcabc!",
    "CamelCaseMixed123",
    "line one\nline two\n",
    "punctuation, everywhere; really?",
    "  leading and trailing  ",
    "a",
    "",
    "lamavirtualassistantdesignedtoprovideinformation",
]

# 128, not 256: ``TokenTree.train`` fails at vocab_size >= 256 on this corpus
# because pugqeep's ``compress_cluster`` can emit a Point with accuracy outside
# [0,1], which ``PointLibrary.add`` rejects. Pre-existing and unrelated to the
# char/tree unification (verified against pristine HEAD) — tracked separately.
VOCAB_SIZE = 128
PHASE_TIMEOUT_S = 60


class PhaseTimeout(Exception):
    """A training or evaluation phase exceeded ``PHASE_TIMEOUT_S``."""


def _alarm(_sig, _frame):
    raise PhaseTimeout()


signal.signal(signal.SIGALRM, _alarm)


class Candidate:
    """One row: a tokenizer plus the metadata the metrics need."""

    def __init__(self, family: str, name: str, tok, charset: set[str] | None = None):
        self.family = family
        self.name = name
        self.tok = tok
        self.charset = charset or set()

    # -- helpers --------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text))

    def decode(self, ids) -> str:
        return str(self.tok.decode(list(ids)))

    def coverage(self, text: str) -> float:
        """Share of characters the vocabulary can represent at all."""
        if not text:
            return 1.0
        if not self.charset:
            return float("nan")
        return sum(1 for c in text if c in self.charset) / len(text)

    def worst_coverage(self, cases: list[str]) -> float:
        """Coverage of the hardest case — the figure that explains F1/F2.

        A mean would let a well-covered easy case hide the one that fails.
        """
        if self.charset is None:
            return 1.0  # byte-level: total by construction
        vals = [self.coverage(s) for s in cases if s]
        return min(vals) if vals else 1.0

    def reload(self):
        """Serialize/deserialize if the candidate supports it."""
        to_dict = getattr(self.tok, "to_dict", None)
        from_dict = getattr(self.tok.__class__, "from_dict", None)
        if to_dict is None or from_dict is None:
            return None
        return from_dict(to_dict())


# --------------------------------------------------------------------------
# Builders — one per algorithm/configuration
# --------------------------------------------------------------------------


def build_morph() -> Candidate:
    """External loader, byte-level: total by construction.

    Every input byte maps to a symbol in the GPT-2 byte-to-unicode alphabet,
    so a vocabulary covering all 256 symbols can never go out of vocab.
    """
    from domain.infrastructure._internal.morph_tokenizer import (
        _BYTE2UNICODE,
        MorphTokenizer,
    )

    vocab = {sym: i for i, sym in enumerate(_BYTE2UNICODE.values())}
    tok = MorphTokenizer(vocab, [], byte_level=True, eos_token_id=0)
    # Charset is every possible decoded character: byte-level is total, so the
    # honest coverage figure is 1.0 rather than a sampled approximation.
    return Candidate("A MorphTokenizer", "byte-level", tok, charset=None)


def build_bpe() -> Candidate:
    from domain.training._internal.tokenizer import SloBPE

    tok = SloBPE()
    tok.train(CORPUS, vocab_size=VOCAB_SIZE, min_frequency=1, lowercase=False)
    chars = {c for t in CORPUS for c in t}
    return Candidate("B SloBPE", f"bpe-{VOCAB_SIZE}", tok, charset=chars)


def build_unigram() -> Candidate:
    from domain.training._internal.tokenizer import SloUnigram

    tok = SloUnigram()
    tok.train(CORPUS, vocab_size=VOCAB_SIZE, lowercase=False)
    chars = {c for t in CORPUS for c in t}
    return Candidate("B SloUnigram", f"unigram-{VOCAB_SIZE}", tok, charset=chars)


def build_char() -> Candidate:
    """TokenTree, char configuration — depth-1 trie, one code point per edge."""
    from domain.inference._internal.slonet_provider import NativeTokenizer

    chars = sorted({c for t in CORPUS for c in t})
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {str(v): k for k, v in stoi.items()}
    tok = NativeTokenizer.from_char_vocab(stoi, itos)
    return Candidate("C TokenTree", "char config", tok, charset=set(chars))


def build_tree() -> Candidate:
    """TokenTree, tree configuration — trained merges, ``</w>`` word suffix."""
    from domain.inference._internal.slonet_provider import NativeTokenizer
    from domain.training._internal.token_tree import TokenTree

    tree = TokenTree().train(CORPUS, vocab_size=VOCAB_SIZE, min_frequency=1, lowercase=False)
    tok = NativeTokenizer(tree, "tree")
    chars = {c for t in CORPUS for c in t}
    return Candidate("C TokenTree", "tree config", tok, charset=chars)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def score_roundtrip(c: Candidate, cases: list[str]) -> tuple[int, int]:
    good = 0
    for s in cases:
        try:
            if c.decode(c.encode(s)) == s:
                good += 1
        except Exception:
            pass
    return good, len(cases)


def score_stream(c: Candidate, cases: list[str]) -> tuple[int, int]:
    """Encode each case in 4-char chunks; concatenated ids must decode back."""
    good = 0
    for s in cases:
        try:
            ids: list[int] = []
            for i in range(0, len(s), 4):
                ids.extend(c.encode(s[i : i + 4]))
            if c.decode(ids) == s:
                good += 1
        except Exception:
            pass
    return good, len(cases)


def score_determinism(c: Candidate, cases: list[str]) -> tuple[int, int]:
    good = 0
    for s in cases:
        try:
            if c.encode(s) == c.encode(s):
                good += 1
        except Exception:
            pass
    return good, len(cases)


def score_reload(c: Candidate, cases: list[str]) -> tuple[int, int]:
    reloaded = c.reload()
    if reloaded is None:
        return 0, 0  # not supported -> reported as n/a
    alt = Candidate(c.family, c.name, reloaded, charset=c.charset)
    good = 0
    for s in cases:
        try:
            if alt.encode(s) == c.encode(s) and alt.decode(alt.encode(s)) == c.decode(c.encode(s)):
                good += 1
        except Exception:
            pass
    return good, len(cases)


def score_speed(c: Candidate, payload: str, reps: int = 20) -> tuple[float, float]:
    """(round trips/sec, megacharacters/sec).

    rt/s alone is not comparable between runs with different payloads, so the
    character rate is reported alongside it.
    """
    c.encode(payload)  # warm caches
    signal.alarm(PHASE_TIMEOUT_S)
    try:
        st = time.perf_counter()
        for _ in range(reps):
            c.decode(c.encode(payload))
        elapsed = time.perf_counter() - st
    finally:
        signal.alarm(0)
    if elapsed <= 0:
        return float("inf"), float("inf")
    return reps / elapsed, (len(payload) * reps / elapsed) / 1e6


def fmt(n: int, d: int) -> str:
    if d == 0:
        return " n/a"
    return f"{n}/{d}"


def pct(x: float) -> str:
    return "  n/a" if x != x else f"{x * 100:4.0f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--no-long",
        action="store_true",
        help="drop the 48-char unbroken word (the SloUnigram stressor)",
    )
    args = ap.parse_args()

    cases = CASES[:-1] if args.no_long else CASES
    payload = "".join(t for t in CORPUS if t) * 5

    builders = [build_morph, build_bpe, build_unigram, build_char, build_tree]

    print(f"corpus: {len(CORPUS)} lines, vocab_size={VOCAB_SIZE}, lowercase=False")
    print(f"cases : {len(cases)}  (long-unbroken-word case: {not args.no_long})")
    print(f"phase timeout: {PHASE_TIMEOUT_S}s")
    print()

    header = (
        f"{'row':<20}{'config':<14}{'cov':>7}{'F1':>8}{'F2':>8}"
        f"{'F3':>8}{'F4':>8}{'F5 rt/s':>11}{'Mchar/s':>10}  status"
    )
    print(header)
    print("-" * len(header))

    # None marks a candidate that failed to build or score.
    results: list[object] = []
    for build in builders:
        name = build.__name__.replace("build_", "")
        try:
            signal.alarm(PHASE_TIMEOUT_S)
            try:
                c = build()
            finally:
                signal.alarm(0)
        except PhaseTimeout:
            print(f"{'':<20}{name:<14}{'':>7}TIMEOUT during build/train")
            results.append(None)
            continue
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            print(f"{'':<20}{name:<14}{'':>7}ERROR {type(exc).__name__}: {exc}")
            results.append(None)
            continue

        try:
            signal.alarm(PHASE_TIMEOUT_S)
            try:
                cov = c.worst_coverage(cases)
                f1 = score_roundtrip(c, cases)
                f2 = score_stream(c, cases)
                f3 = score_determinism(c, cases)
                f4 = score_reload(c, cases)
                rps, mcs = score_speed(c, payload)
            finally:
                signal.alarm(0)
        except PhaseTimeout:
            print(f"{c.family:<20}{c.name:<14}{'':>7}TIMEOUT during scoring")
            results.append(None)
            continue

        status = "ok" if f1[0] == f1[1] else f"coverage-limited ({f1[1] - f1[0]} fail)"
        print(
            f"{c.family:<20}{c.name:<14}{pct(cov):>7}{fmt(*f1):>8}{fmt(*f2):>8}"
            f"{fmt(*f3):>8}{fmt(*f4):>8}{rps:>11,.0f}{mcs:>10.2f}  {status}"
        )
        results.append((c, cov, f1, f2, f3, f4, rps, mcs))

    print()
    print("cov is the WORST case's coverage. F1/F2 failures with low cov are")
    print("VOCABULARY COVERAGE, not decode bugs — grow the training corpus, not")
    print("the decoder. Full cov with failing F1 means a real decode defect.")
    print("F4 'n/a' means the candidate has no to_dict/from_dict pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
