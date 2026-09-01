"""Data quality filter — score, gate, and deduplicate ingested content.

Instead of dredging everything, the filter sits between sources (RSS/search/URLs)
and storage (KnowledgeMemory + training buffer), passing only content that meets
configurable quality, relevance, and topic criteria.

Flow:
  Raw article → LengthGate → QualityScore → RelevanceGate → ToxicityFilter → TopicGate → Store
                    reject        score≥N       score≥N         reject        match

Config is stored in a MogDB ``data_filter_config`` collection (single document
keyed by ``_id: "config"``).
"""

from __future__ import annotations

import re


import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("slo.learner.filter")

# ─── MogDB config storage ────────────────────────────────────────────────

_db = None
_collection = None


def _get_collection(db_path: Optional[str] = None):
    """Return the ``data_filter_config`` collection, creating it on first call."""
    global _db, _collection
    if _collection is not None:
        return _collection
    from mogdb import MogDB
    if db_path is None:
        from domains.shared import find_repo_root
        repo = find_repo_root(Path(__file__).resolve())
        db_path = str(repo / "data" / "data_filter_mogdb")
    _db = MogDB(db_path)
    _collection = _db.collection("data_filter_config")
    return _collection


def set_data_filter_db(db_path: str) -> None:
    """Replace the module-level collection (for tests)."""
    global _db, _collection
    from mogdb import MogDB
    _db = MogDB(db_path)
    _collection = _db.collection("data_filter_config")


def reset_data_filter_db() -> None:
    """Clear the module-level collection reference."""
    global _db, _collection
    _db = None
    _collection = None


# ─── Default config ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Minimum content length (chars). Shorter = noise.
    "min_content_length": 200,
    # Minimum quality score (0-1). 0.3 = most things pass, 0.7 = very strict.
    "min_quality_score": 0.3,
    # Minimum relevance score (0-1). 0 = anything, 0.5 = moderate relevance.
    "min_relevance_score": 0.0,
    # Topic whitelist. If non-empty, only these topics are allowed.
    # Topics are matched as substrings (e.g. "ai" matches "artificial intelligence").
    "topic_whitelist": [],
    # Topic blacklist. Content matching these topics is rejected.
    "topic_blacklist": [
        "porn", "xxx", "adult", "nsfw", "gambling", "casino",
        "crack", "warez", "hack", "cheat", "botnet",
    ],
    # If True, whitelist acts as a hard gate (no whitelist = nothing passes).
    "whitelist_is_hard_gate": False,
    # Near-duplicate threshold (cosine similarity). 0.95 = almost identical.
    "dup_similarity_threshold": 0.95,
    # Enable filtering
    "enabled": True,
}

# ─── Quality heuristics ─────────────────────────────────────────────────────

_TOXIC_WORDS = {
    "porn", "xxx", "adult", "nsfw", "gambling", "casino",
    "crack", "warez", "hackz", "botnet", "milf", "teen",
    "viagra", "cialis", "casino", "free money", "click here",
    "buy now", "act now", "limited offer", "congratulations you won",
}


def _hashed(n: float) -> float:
    """Deterministic pseudo-random from a float, for stable sampling."""
    return ((n * 2654435761) % 2**32) / 2**32


def _score_quality(text: str) -> float:
    """Heuristic quality score 0-1. Higher = better prose."""
    if len(text) < 50:
        return 0.0

    scores = []

    # 1. Sentence count (≥3 sentences is good)
    sentences = len(re.findall(r'[.!?]+', text))
    scores.append(min(1.0, sentences / 8))

    # 2. Average word length (3-8 chars is normal prose)
    words = text.split()
    if words:
        avg_wl = sum(len(w) for w in words) / len(words)
        if 3.5 <= avg_wl <= 7.0:
            scores.append(1.0)
        elif 2.5 <= avg_wl <= 9.0:
            scores.append(0.5)
        else:
            scores.append(0.1)

    # 3. Caps ratio (<30% caps is normal)
    caps = sum(1 for c in text if c.isupper())
    caps_ratio = caps / max(1, len(text))
    if caps_ratio < 0.15:
        scores.append(1.0)
    elif caps_ratio < 0.30:
        scores.append(0.5)
    else:
        scores.append(0.0)

    # 4. Punctuation variety (.,!?;: — more variety = better writing)
    punct = sum(1 for c in text if c in '.,!?;:-')
    punct_ratio = punct / max(1, len(text))
    if 0.03 <= punct_ratio <= 0.15:
        scores.append(1.0)
    elif punct_ratio > 0:
        scores.append(0.5)
    else:
        scores.append(0.0)

    # 5. Unique word ratio (>50% unique is good)
    if words:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio > 0.5:
            scores.append(1.0)
        elif unique_ratio > 0.3:
            scores.append(0.5)
        else:
            scores.append(0.2)

    # 6. Penalize excessive short lines (listicles, nav menus)
    lines = text.split('\n')
    short_lines = sum(1 for l in lines if 0 < len(l.strip()) < 20)
    if lines and short_lines / max(1, len(lines)) > 0.5:
        scores.append(0.2)
    else:
        scores.append(1.0)

    return sum(scores) / len(scores) if scores else 0.0


def _score_relevance(text: str, whitelist: list[str]) -> float:
    """Score how well content matches the topic whitelist. 0-1."""
    if not whitelist:
        return 1.0  # no whitelist = everything is relevant
    lower = text.lower()
    matches = 0
    for topic in whitelist:
        tl = topic.lower()
        count = lower.count(tl)
        if count > 0:
            matches += count * (len(tl) / 10)  # longer matches weigh more
    score = min(1.0, matches / 3.0)  # 3 good matches = max score
    return score


def _matches_blacklist(text: str, blacklist: list[str]) -> bool:
    """Check if text contains any blacklisted terms."""
    lower = text.lower()
    for term in blacklist:
        if term in lower:
            return True
    return False


def _matches_whitelist(text: str, whitelist: list[str]) -> bool:
    """Check if text contains any whitelisted topic."""
    if not whitelist:
        return True
    lower = text.lower()
    for topic in whitelist:
        if topic.lower() in lower:
            return True
    return False


# ─── Config ─────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load filter config from MogDB."""
    col = _get_collection()
    doc = col.find_one({"_id": "config"})
    if doc is None:
        return dict(DEFAULT_CONFIG)
    try:
        cfg = dict(doc)
        cfg.pop("_id", None)
        return cfg
    except Exception as e:
        logger.warning("Failed to load filter config: %s", e, extra={"tag": "INF"})
        return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict):
    """Persist filter config to MogDB."""
    col = _get_collection()
    data = {"_id": "config", **cfg}
    existing = col.find_one({"_id": "config"})
    if existing is not None:
        col.update_one({"_id": "config"}, {"$set": cfg})
    else:
        col.insert_one(data)


# ─── Filter ─────────────────────────────────────────────────────────────────

class DataFilter:
    """Quality gate for ingested content.

    Each article passes through:
      1. Length gate (skip < min_content_length)
      2. Quality score (skip < min_quality_score)
      3. Topic gate (skip if matches blacklist / doesn't match whitelist)
      4. Relevance score (recorded for stats, doesn't block unless gated)
      5. Near-dup detection (skip if too similar to existing facts)

    Stats tracked: total_seen, passed, rejected (by reason).
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = _load_config()
        if config:
            self.config.update(config)
            _save_config(self.config)
        self.stats = {
            "total_seen": 0,
            "passed": 0,
            "rejected": 0,
            "rejected_short": 0,
            "rejected_quality": 0,
            "rejected_blacklist": 0,
            "rejected_whitelist": 0,
            "rejected_dup": 0,
        }

    def update_config(self, **kwargs):
        self.config.update(kwargs)
        _save_config(self.config)
        logger.info("Filter config updated: %s", kwargs, extra={"tag": "INF"})

    def get_config(self) -> dict:
        return dict(self.config)

    def get_stats(self) -> dict:
        return dict(self.stats)

    def filter_article(
        self,
        url: str,
        title: str,
        content: str,
        existing_facts: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """Run all filters on an article. Returns (pass: bool, reason: str).

        Args:
            url: article URL (for dup tracking)
            title: article title
            content: full article text
            existing_facts: list of existing fact contents for near-dup check

        Returns:
            (True, "") if article passes, or (False, "reason") if rejected.
        """
        if not self.config.get("enabled", True):
            self.stats["passed"] += 1
            return True, ""

        self.stats["total_seen"] += 1

        # 1. Length gate
        if len(content) < self.config.get("min_content_length", 200):
            self.stats["rejected"] += 1
            self.stats["rejected_short"] += 1
            return False, "too_short"

        # 2. Quality score
        quality = _score_quality(content)
        if quality < self.config.get("min_quality_score", 0.3):
            self.stats["rejected"] += 1
            self.stats["rejected_quality"] += 1
            return False, f"low_quality_{quality:.2f}"

        # 3. Blacklist check
        blacklist = self.config.get("topic_blacklist", [])
        if blacklist and _matches_blacklist(title + " " + content, blacklist):
            self.stats["rejected"] += 1
            self.stats["rejected_blacklist"] += 1
            return False, "blacklisted"

        # 4. Whitelist check (hard gate)
        whitelist = self.config.get("topic_whitelist", [])
        hard_gate = self.config.get("whitelist_is_hard_gate", False)
        if whitelist and hard_gate:
            if not _matches_whitelist(title + " " + content[:500], whitelist):
                self.stats["rejected"] += 1
                self.stats["rejected_whitelist"] += 1
                return False, "not_in_whitelist"

        # 5. Near-dup check (simple: long common substrings)
        if existing_facts:
            flat = content.lower()[:2000]
            for ef in existing_facts:
                ef_flat = ef.lower()[:2000]
                # Simple overlap ratio
                overlap = len(set(flat.split()) & set(ef_flat.split()))
                total = max(1, len(set(flat.split()) | set(ef_flat.split())))
                similarity = overlap / total
                if similarity > self.config.get("dup_similarity_threshold", 0.95):
                    self.stats["rejected"] += 1
                    self.stats["rejected_dup"] += 1
                    return False, "near_duplicate"

        self.stats["passed"] += 1
        return True, ""

    def filter_chunk(self, chunk: str, topic: str) -> bool:
        """Filter a single fact chunk (500 chars) by topic gate only.

        Quality/length checks already passed at article level.
        """
        if not self.config.get("enabled", True):
            return True
        # Topic blacklist
        blacklist = self.config.get("topic_blacklist", [])
        if blacklist and _matches_blacklist(chunk, blacklist):
            return False
        # Whitelist
        whitelist = self.config.get("topic_whitelist", [])
        hard_gate = self.config.get("whitelist_is_hard_gate", False)
        if whitelist and hard_gate:
            if not _matches_whitelist(topic + " " + chunk, whitelist):
                return False
        return True


# Global singleton
_filter: Optional[DataFilter] = None


def get_data_filter() -> DataFilter:
    global _filter
    if _filter is None:
        _filter = DataFilter()
    return _filter
