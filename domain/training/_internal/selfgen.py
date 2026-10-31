"""Self-gen flywheel — the trainer generates its own training data mid-run.

Inside the ``SloughGPTTrainer`` loop: the live model produces continuations,
the producer gates them (quality + dedup) at the corpus write boundary, the
``FeedBatchSampler`` picks them up, and ``get_batch`` mixes them into train
batches at ``TrainerConfig.selfgen_quota``. The trainer itself stays dumb
about data: generation is a producer, ingestion stays the existing feed
mechanism — no fourth training loop, no second corpus pipeline.

The quality gate lives here (producer side), never in the trainer: raw
self-samples must not enter training unfiltered, and the quota mixing with
the static dataset is the anti-collapse control (a model trained only on its
own output converges to its own distribution).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from domain.shared import find_repo_root
from domain.training._internal.quality_scorer import score_text_chunk

logger = logging.getLogger("slo.training.selfgen")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())

MIN_CONTINUATION_LEN = 8


def resolve_corpus_path(source: str = "selfgen", local_base: str | Path | None = None) -> Path:
    """Resolve the on-disk corpus path for a self-gen feed source.

    Mirrors ``TrainingFeedClient`` local resolution (``<base>/<source>/corpus.jsonl``
    with hyphen/underscore equivalence) so producer writes land exactly where
    the feed reader looks.

    Args:
        source: feed source name (default ``selfgen``).
        local_base: explicit base dir (testing); None resolves under the repo root.

    Returns:
        Absolute path of the corpus JSONL file (may not exist yet).
    """
    name = source.replace("-", "_")
    if local_base is not None:
        return Path(local_base) / name / "corpus.jsonl"
    return _REPO_ROOT / "data" / name / "corpus.jsonl"


class SelfGenProducer:
    """Gates and appends model-generated continuations to the self-gen corpus.

    Owns the write boundary: quality gate, dedup, prompt rotation, and the
    round counters. The trainer calls :meth:`next_prompts`, generates with its
    own live model, then hands each continuation to :meth:`accept`.

    Args:
        corpus_path: JSONL file the feed sampler drains.
        prompts: seed prompts to rotate through; must be non-empty.
        min_quality: minimum ``score_text_chunk`` score (0-5) to accept.
    """

    def __init__(
        self,
        corpus_path: Path,
        prompts: list[str],
        min_quality: float = 1.0,
    ) -> None:
        if not prompts:
            raise ValueError("SelfGenProducer requires at least one prompt")
        self.corpus_path = Path(corpus_path)
        self.prompts = [p for p in (str(p).strip() for p in prompts) if p]
        if not self.prompts:
            raise ValueError("SelfGenProducer prompts must be non-empty after stripping")
        self.min_quality = float(min_quality)
        self.rounds = 0
        self.written = 0
        self.rejected_quality = 0
        self.rejected_dup = 0
        self.rejected_short = 0
        self._cursor = 0
        self._seen: set[str] = self._load_seen()

    def ensure_corpus(self) -> None:
        """Create the corpus file (and parents) if missing."""
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.corpus_path.exists():
            self.corpus_path.touch()

    def _load_seen(self) -> set[str]:
        """Load sha1 digests of already-written continuations for dedup."""
        seen: set[str] = set()
        if not self.corpus_path.exists():
            return seen
        try:
            for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = rec.get("assistant_msg") or ""
                if text:
                    seen.add(_digest(text))
        except Exception as exc:  # corrupt lines must not kill the run
            logger.debug("selfgen: failed to load seen set: %s", exc)
        return seen

    @staticmethod
    def prompts_from_text(text: str, count: int, width: int = 48) -> list[str]:
        """Derive seed prompts from a training-text head.

        Args:
            text: decoded training data head.
            count: desired number of prompts.
            width: characters per prompt.

        Returns:
            Up to ``count`` prompts of ``width`` chars (each at least 8 chars).
        """
        if not text or count < 1:
            return []
        step = max(width, len(text) // count)
        out: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + width].strip()
            if len(chunk) >= 8:
                out.append(chunk)
            if len(out) >= count:
                break
        return out

    def next_prompts(self, count: int) -> list[str]:
        """Return the next rotated batch of prompts (round-robin, deduped order).

        Args:
            count: number of prompts for this round.

        Returns:
            ``count`` prompts (repeats allowed when ``count`` > len(prompts)).
        """
        out = [self.prompts[(self._cursor + i) % len(self.prompts)] for i in range(count)]
        self._cursor = (self._cursor + count) % len(self.prompts)
        return out

    def accept(self, prompt: str, continuation: str) -> bool:
        """Gate one generated continuation and append it when it passes.

        Gates in order: non-empty, minimum length, dedup (sha1), quality score.
        Appends a flat pair record the feed reader already understands.

        Args:
            prompt: the seed prompt the model continued from.
            continuation: newly generated text (prompt excluded).

        Returns:
            True when the record was written to the corpus.
        """
        text = continuation.strip()
        if len(text) < MIN_CONTINUATION_LEN:
            self.rejected_short += 1
            return False
        digest = _digest(text)
        if digest in self._seen:
            self.rejected_dup += 1
            return False
        score = score_text_chunk(text)
        if score < self.min_quality:
            self.rejected_quality += 1
            return False
        record = {
            "user_msg": prompt,
            "assistant_msg": text,
            "gen": True,
            "quality": score,
        }
        self.ensure_corpus()
        with self.corpus_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._seen.add(digest)
        self.written += 1
        return True

    def stats(self) -> dict[str, int | float | str]:
        """Concise round/acceptance counters for logs and status surfaces."""
        return {
            "rounds": self.rounds,
            "written": self.written,
            "rejected_quality": self.rejected_quality,
            "rejected_dup": self.rejected_dup,
            "rejected_short": self.rejected_short,
            "corpus": str(self.corpus_path),
        }


def _digest(text: str) -> str:
    """sha1 digest of a normalized continuation (whitespace-collapsed)."""
    normalized = " ".join(text.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()
