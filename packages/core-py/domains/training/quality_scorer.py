"""
Training Pair Quality Scorer — Algorithmic quality assessment for (user, assistant) pairs.

Scores pairs on a 0-5 scale using 4 independent signals:
  - Length: both messages in optimal range
  - Repetition: bigram/trigram repetition rate in assistant response
  - Coherence: keyword overlap and response/question length ratio
  - Language quality: punctuation, unique words, sentence structure

Usage:
    from domains.training.quality_scorer import score_pair, score_batch
    score = score_pair("What is Python?", "Python is a programming language.")
    scores = score_batch([pair1, pair2, ...])
"""

import re
from collections import Counter
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return [w for w in re.split(r'\s+', text.lower()) if w]


def _length_score(user_msg: str, assistant_msg: str) -> float:
    """Score based on message lengths. Optimal: 50-1000 chars each."""
    u_len = len(user_msg)
    a_len = len(assistant_msg)

    def _optimal(length: float) -> float:
        if length < 10:
            return 0.0
        if length < 30:
            return 0.3
        if length < 50:
            return 0.6
        if length <= 1000:
            return 1.0
        if length <= 2000:
            return 0.7
        return 0.4

    return (_optimal(u_len) + _optimal(a_len)) / 2


def _repetition_score(assistant_msg: str) -> float:
    """Score based on bigram/trigram repetition. High repetition = low score."""
    words = _tokenize(assistant_msg)
    if len(words) < 4:
        return 0.5

    bigrams = list(zip(words, words[1:]))
    bigram_counts = Counter(bigrams)
    most_common_bigram_count = bigram_counts.most_common(1)[0][1] if bigram_counts else 1
    bigram_repetition = most_common_bigram_count / max(1, len(bigrams))

    trigrams = list(zip(words, words[1:], words[2:]))
    if trigrams:
        trigram_counts = Counter(trigrams)
        most_common_trigram_count = trigram_counts.most_common(1)[0][1]
        trigram_repetition = most_common_trigram_count / len(trigrams)
    else:  # pragma: no cover — len(words) >= 4 guarantees non-empty trigrams
        trigram_repetition = 0

    word_counts = Counter(words)
    most_common_word_count = word_counts.most_common(1)[0][1] if word_counts else 1
    word_repetition = most_common_word_count / max(1, len(words))

    rep_score = 1.0
    if bigram_repetition > 0.5:
        rep_score -= 0.4
    elif bigram_repetition > 0.3:
        rep_score -= 0.2

    if trigram_repetition > 0.3:
        rep_score -= 0.3
    elif trigram_repetition > 0.15:
        rep_score -= 0.1

    if word_repetition > 0.3:
        rep_score -= 0.2

    return max(0.0, rep_score)


def _coherence_score(user_msg: str, assistant_msg: str) -> float:
    """Score based on relevance of assistant response to user question."""
    u_words = set(_tokenize(user_msg))
    a_words = set(_tokenize(assistant_msg))

    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
        'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
        'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    }
    u_content = u_words - stop_words
    a_content = a_words - stop_words

    if u_content and a_content:
        overlap = len(u_content & a_content) / len(u_content | a_content)
    else:
        overlap = 0.0

    u_len = max(1, len(user_msg))
    a_len = max(1, len(assistant_msg))
    ratio = a_len / u_len
    if 0.5 <= ratio <= 5.0:
        ratio_score = 1.0
    elif 0.2 <= ratio <= 10.0:
        ratio_score = 0.6
    else:
        ratio_score = 0.2

    question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}
    has_question = bool(u_words & question_words)

    return min(1.0, overlap * 2 + ratio_score * 0.3 + (0.1 if has_question else 0))


def _language_quality_score(assistant_msg: str) -> float:
    """Score language quality of assistant response."""
    if len(assistant_msg) < 10:
        return 0.2

    scores = []

    sents = re.findall(r'[.!?]+', assistant_msg)
    if len(sents) >= 3:
        scores.append(1.0)
    elif len(sents) >= 1:
        scores.append(0.7)
    else:
        scores.append(0.3)

    punct_chars = set(c for c in assistant_msg if c in '.,!?;:-')
    if len(punct_chars) >= 3:
        scores.append(1.0)
    elif len(punct_chars) >= 1:
        scores.append(0.6)
    else:
        scores.append(0.2)

    words = _tokenize(assistant_msg)
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.6:
            scores.append(1.0)
        elif unique_ratio > 0.4:
            scores.append(0.7)
        else:
            scores.append(0.3)
    else:
        scores.append(0.0)

    if words:
        avg_wl = sum(len(w) for w in words) / len(words)
        if 3.5 <= avg_wl <= 7.0:
            scores.append(1.0)
        elif 2.5 <= avg_wl <= 9.0:
            scores.append(0.6)
        else:
            scores.append(0.2)
    else:
        scores.append(0.0)

    caps = sum(1 for c in assistant_msg if c.isupper())
    caps_ratio = caps / max(1, len(assistant_msg))
    if caps_ratio < 0.15:
        scores.append(1.0)
    elif caps_ratio < 0.30:
        scores.append(0.5)
    else:
        scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def score_pair(user_msg: str, assistant_msg: str) -> float:
    """
    Score a (user_msg, assistant_msg) pair on a 0-5 scale.

    Args:
        user_msg: The user's message.
        assistant_msg: The assistant's response.

    Returns:
        Quality score from 0.0 (lowest) to 5.0 (highest).
    """
    if not user_msg or not assistant_msg:
        return 0.0

    length = _length_score(user_msg, assistant_msg)
    repetition = _repetition_score(assistant_msg)
    coherence = _coherence_score(user_msg, assistant_msg)
    language = _language_quality_score(assistant_msg)

    combined = (length * 0.25 + repetition * 0.25 + coherence * 0.25 + language * 0.25)

    score = round(combined * 5, 1)
    return max(0.0, min(5.0, score))


def score_batch(pairs: List[Dict[str, str]]) -> List[float]:
    """
    Score a batch of training pairs.

    Args:
        pairs: List of dicts with 'user_msg' and 'assistant_msg' keys.

    Returns:
        List of quality scores (0-5 scale), one per pair.
    """
    return [
        score_pair(p.get("user_msg", ""), p.get("assistant_msg", ""))
        for p in pairs
    ]
