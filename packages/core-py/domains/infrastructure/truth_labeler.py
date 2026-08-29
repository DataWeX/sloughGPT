"""
Truth labeler — rule-based first-glance text labeling with confidence scores.

Provides lightweight, zero-download text classification using linguistic
heuristics. Each label includes a confidence score (0.0-1.0) indicating
how certain the heuristic is.

Labels:
  - factual: declarative statements of fact
  - conceptual: abstract ideas, definitions
  - procedural: step-by-step instructions
  - interrogative: questions, uncertainty
  - descriptive: observations, descriptions
  - directive: commands, requests
  - analytical: reasoning, analysis, evaluation

Usage:
    from domains.infrastructure.truth_labeler import TruthLabeler
    labeler = TruthLabeler()
    result = labeler.label("The sky is blue")
    # → {"label": "factual", "confidence": 0.8, "reason": "declarative statement"}
"""
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class LabelResult:
    """Result of rule-based text labeling."""
    label: str
    confidence: float
    reason: str
    scores: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
            "scores": self.scores,
        }


class TruthLabeler:
    """Rule-based first-glance text labeler with confidence scores.

    Uses linguistic heuristics (sentence structure, keywords, punctuation)
    to classify text into semantic meaning regions. No model download required.

    Each rule produces a score for each label. The label with the highest
    score wins, and confidence is the winning score normalized to 0.0-1.0.
    """

    def __init__(self):
        self._rules = [
            _rule_interrogative,
            _rule_directive,
            _rule_descriptive,
            _rule_analytical,
            _rule_procedural,
            _rule_conceptual,
            _rule_factual,
        ]

    def label(self, text: str) -> LabelResult:
        """Label text using rule-based heuristics.

        Args:
            text: input text string

        Returns:
            LabelResult with label, confidence, reason, and per-label scores
        """
        text = text.strip()
        if not text:
            return LabelResult(
                label="descriptive",
                confidence=0.0,
                reason="empty text",
                scores={},
            )

        # Collect scores from all rules
        scores: Dict[str, float] = {
            "factual": 0.0,
            "conceptual": 0.0,
            "procedural": 0.0,
            "interrogative": 0.0,
            "descriptive": 0.0,
            "directive": 0.0,
            "analytical": 0.0,
        }

        reasons: Dict[str, str] = {}
        for rule in self._rules:
            rule_scores, rule_reasons = rule(text)
            for label, score in rule_scores.items():
                scores[label] += score
            reasons.update(rule_reasons)

        # Find winner
        if not any(scores.values()):
            return LabelResult(
                label="factual",
                confidence=0.1,
                reason="default fallback",
                scores=scores,
            )

        winner = max(scores, key=scores.get)
        max_score = scores[winner]
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.0

        # Confidence discount: if scores are close, reduce confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            margin = sorted_scores[0] - sorted_scores[1]
            if margin < 0.2:
                confidence *= 0.7  # close call → less confident

        reason = reasons.get(winner, "heuristic match")

        return LabelResult(
            label=winner,
            confidence=min(confidence, 1.0),
            reason=reason,
            scores=scores,
        )

    def label_batch(self, texts: List[str]) -> List[LabelResult]:
        """Label a batch of texts."""
        return [self.label(t) for t in texts]


# ---------------------------------------------------------------------------
# Rules — each returns (scores_dict, reasons_dict)
# ---------------------------------------------------------------------------

def _rule_interrogative(text: str) -> tuple:
    """Questions and uncertainty."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    # Ends with question mark
    if t.endswith("?"):
        scores["interrogative"] = 0.9
        reasons["interrogative"] = "ends with ?"
        return scores, reasons

    # Question words at start
    question_words = ("what", "how", "why", "when", "where", "who", "which",
                      "is", "are", "do", "does", "can", "could", "should",
                      "would", "will", "have", "has")
    first_word = t.split()[0] if t.split() else ""
    if first_word in question_words:
        scores["interrogative"] = 0.7
        reasons["interrogative"] = f"starts with question word '{first_word}'"
        return scores, reasons

    return scores, reasons


def _rule_directive(text: str) -> tuple:
    """Commands and requests."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    directive_starts = (
        "you should", "you must", "you need to", "please", "do this",
        "run ", "create ", "make ", "build ", "write ", "delete ",
        "move ", "copy ", "install ", "update ", "set ", "configure ",
        "add ", "remove ", "fix ", "change ", "implement ", "deploy ",
    )

    for start in directive_starts:
        if t.startswith(start):
            scores["directive"] = 0.8
            reasons["directive"] = f"starts with directive '{start.strip()}'"
            return scores, reasons

    # Imperative mood: verb at start (simple heuristic)
    imperative_verbs = (
        "run", "create", "make", "build", "write", "delete", "move",
        "copy", "install", "update", "set", "configure", "add", "remove",
        "fix", "change", "implement", "deploy", "test", "check", "verify",
        "open", "close", "start", "stop", "enable", "disable", "reset",
    )
    first_word = t.split()[0] if t.split() else ""
    if first_word in imperative_verbs:
        scores["directive"] = 0.6
        reasons["directive"] = f"imperative verb '{first_word}'"
        return scores, reasons

    return scores, reasons


def _rule_descriptive(text: str) -> tuple:
    """Observations and descriptions."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    # Descriptive starters
    descriptive_starts = (
        "the ", "a ", "an ", "this ", "that ", "it ", "there ", "here ",
        "in ", "on ", "at ", "from ", "with ", "about ",
    )

    for start in descriptive_starts:
        if t.startswith(start):
            scores["descriptive"] = 0.3
            reasons["descriptive"] = f"starts with descriptive '{start.strip()}'"
            break

    # Not ending with ! or ? (statements)
    if not t.endswith(("!", "?")):
        scores["descriptive"] = scores.get("descriptive", 0.0) + 0.2
        reasons["descriptive"] = reasons.get("descriptive", "declarative statement")

    return scores, reasons


def _rule_analytical(text: str) -> tuple:
    """Reasoning and analysis."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    analytical_markers = (
        "because", "therefore", "however", "although", "since", "given that",
        "in contrast", "on the other hand", "consequently", "thus", "hence",
        "this means", "this implies", "the reason", "the cause", "the effect",
        "if .+ then", "comparing", "analyzing", "evaluation", "assessment",
    )

    for marker in analytical_markers:
        if marker in t:
            scores["analytical"] = 0.7
            reasons["analytical"] = f"contains analytical marker '{marker}'"
            return scores, reasons

    # Complex sentence structure (multiple clauses)
    if t.count(",") >= 2 or t.count(";") >= 1:
        scores["analytical"] = 0.4
        reasons["analytical"] = "complex sentence structure"

    return scores, reasons


def _rule_procedural(text: str) -> tuple:
    """Step-by-step instructions."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    # Numbered steps
    if re.search(r"\b\d+[\.\)]\s", t):
        scores["procedural"] = 0.9
        reasons["procedural"] = "contains numbered steps"
        return scores, reasons

    # Step keywords
    step_keywords = (
        "step ", "first,", "second,", "third,", "next,", "then,",
        "finally,", "after that", "before that", "followed by",
    )
    for kw in step_keywords:
        if kw in t:
            scores["procedural"] = 0.7
            reasons["procedural"] = f"contains step keyword '{kw.strip()}'"
            return scores, reasons

    return scores, reasons


def _rule_conceptual(text: str) -> tuple:
    """Abstract ideas and definitions."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    # Definition patterns
    if re.search(r"\b(is|are)\b.+\b(a|an|the)\b.+\b(that|which|who)\b", t):
        scores["conceptual"] = 0.7
        reasons["conceptual"] = "definition pattern"
        return scores, reasons

    # "X is Y" pattern (simple definition) — Y must start with an article
    if re.match(r"^[\w\s]+ is (a|an|the) [\w\s]+", t):
        scores["conceptual"] = 0.5
        reasons["conceptual"] = "X is Y definition pattern"
        return scores, reasons

    # Abstract markers
    abstract_markers = (
        "concept", "idea", "theory", "principle", "notion", "abstract",
        "definition", "meaning", "essence", "nature", "fundamental",
    )
    for marker in abstract_markers:
        if marker in t:
            scores["conceptual"] = 0.6
            reasons["conceptual"] = f"contains abstract marker '{marker}'"
            return scores, reasons

    return scores, reasons


def _rule_factual(text: str) -> tuple:
    """Factual assertions (default)."""
    scores = {}
    reasons = {}
    t = text.lower().strip()

    # Factual indicators
    factual_indicators = (
        "is ", "was ", "were ", "has ", "have ", "had ", "can ", "will ",
        "does ", "did ", "contains ", "measures ", "weighs ", "equals ",
    )

    for indicator in factual_indicators:
        if indicator in t:
            scores["factual"] = 0.4
            reasons["factual"] = f"contains factual indicator '{indicator.strip()}'"
            return scores, reasons

    # Default: declarative statement
    if not t.endswith(("?", "!")):
        scores["factual"] = 0.3
        reasons["factual"] = "declarative statement (default)"

    return scores, reasons


# Module-level singleton
_labeler: Optional[TruthLabeler] = None
_truth_labeler_lock = threading.Lock()


def get_truth_labeler() -> TruthLabeler:
    """Get or create the truth labeler singleton."""
    global _labeler
    if _labeler is None:
        with _truth_labeler_lock:
            if _labeler is None:
                _labeler = TruthLabeler()
    return _labeler


def reset_truth_labeler() -> None:
    """Reset the singleton (for testing)."""
    global _labeler
    with _truth_labeler_lock:
        _labeler = None
