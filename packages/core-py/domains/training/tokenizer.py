"""
SloBPE — Byte-Pair Encoding Tokenizer

Pure NumPy/Python implementation matching SloNet's no-HuggingFace philosophy.

Key design:
- Trains BPE merges from raw text corpora
- Drops into the existing stoi/itos dict interface used everywhere
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
- Handles any Unicode text, lowercased by default
- Serializable to/from dict for checkpoint storage
- Efficient encode/decode with NumPy-optimised pair counting

Usage:
    tokenizer = SloBPE()
    tokenizer.train(["hello world", "hello there"], vocab_size=64)

    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)

    stoi, itos = tokenizer.stoi, tokenizer.itos  # Duck-typed interface
    checkpoint["tokenizer"] = tokenizer.to_dict()  # Save
    tokenizer = SloBPE.from_dict(checkpoint["tokenizer"])  # Load
"""

import json
import re
import os
import math
from collections import Counter
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Tuple, Set
import logging

logger = logging.getLogger("slo.tokenizer")


# GPT-2 style regex pre-tokenization.
# Compatible w/ Python 3.9 (no \p{L} Unicode property escapes).
# Uses [^\W\d_] for Unicode letters, \d+ for digits.
# Pattern order matters — greedier matches first.
#   - Contractions: 's, 'd, 'm, 't, 'll, 've, 're
#   - Words with optional leading space (Unicode letters)
#   - Digit runs with optional leading space
#   - Punctuation runs with optional leading space
#   - Trailing whitespace
_GPT2_SPLIT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)


def gpt2_pretokenize(text: str) -> List[str]:
    """Split text into pretokens using the GPT-2 regex pattern.

    Each pretoken is a word-like unit with punctuation separated,
    preserving the space-before-word convention GPT-2 uses.
    """
    return _GPT2_SPLIT.findall(text)


def default_pretokenize(text: str) -> List[str]:
    """Simple whitespace-based pre-tokenization (fallback)."""
    return text.split()


class SloBPE:
    """
    Byte-Pair Encoding tokenizer compatible with SloNet's stoi/itos interface.

    Trains merge rules from a corpus, encodes text into subword token IDs,
    and decodes IDs back into text. Drops into any code expecting
    ``stoi: Dict[str, int]`` and ``itos: Dict[int, str]``.

    Uses GPT-2-style regex pre-tokenization by default, splitting
    punctuation and numbers into separate tokens before BPE.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    WORD_SUFFIX = "</w>"

    def __init__(self, pretokenizer: str = "gpt2") -> None:
        self.vocab: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._word_suffix: str = self.WORD_SUFFIX
        self._pretokenizer: str = pretokenizer
        self._special_set: Set[str] = set(self.SPECIAL_TOKENS)
        # Analysis caches — populated by analyze_corpus()
        self._token_freqs: Counter = Counter()
        self._training_corpus_len: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi.get("<PAD>", 0)

    @property
    def unk_id(self) -> int:
        return self.stoi.get("<UNK>", 1)

    @property
    def bos_id(self) -> int:
        return self.stoi.get("<BOS>", 2)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<EOS>", 3)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: List[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
        lowercase: bool = True,
        verbose: bool = False,
    ) -> "SloBPE":
        """
        Learn BPE merge rules from a corpus of texts.

        Args:
            texts: raw text strings to learn from
            vocab_size: target vocabulary size (including special tokens + base chars)
            min_frequency: minimum pair frequency to consider a merge
            lowercase: convert text to lowercase before training
            verbose: log merge progress

        Returns:
            self (for chaining)
        """
        if not texts:
            raise ValueError("Need at least one text to train on")

        # Preprocess
        corpus = [self._normalize(t, lowercase) for t in texts]

        # Discover base character vocabulary
        chars: Counter = Counter()
        for doc in corpus:
            for word in self._pretokenize(doc, lowercase):
                for ch in word:
                    chars[ch] += 1

        # Build initial vocab: special tokens + </w> + all unique characters
        base_chars = sorted(chars.keys())
        self.vocab = list(self.SPECIAL_TOKENS) + [self._word_suffix] + base_chars.copy()
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self.merges = []
        self._special_set = set(self.SPECIAL_TOKENS)

        # Pre-tokenise: split each doc into pretokens, each pretoken into chars
        # Append </w> suffix marks word boundaries for proper decoding
        word_freqs: Counter = Counter()
        for doc in corpus:
            for word in self._pretokenize(doc, lowercase):
                word_freqs[word + self._word_suffix] += 1

        # Each "word" is represented as a list of current tokens (chars + </w>)
        word_splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            raw = word.replace(self._word_suffix, "")
            word_splits[word] = list(raw) + [self._word_suffix]

        # Merge loop
        target_size = min(vocab_size, len(self.SPECIAL_TOKENS) + 1 + len(base_chars) + (vocab_size - len(self.SPECIAL_TOKENS) - 1))
        while len(self.vocab) < vocab_size:
            # Count all adjacent pairs across all words
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                split = word_splits[word]
                if len(split) < 2:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Filter by min_frequency
            pair_counts = Counter({p: c for p, c in pair_counts.items() if c >= min_frequency})
            if not pair_counts:
                break

            # Pick most frequent pair
            best_pair, best_count = pair_counts.most_common(1)[0]
            new_token = best_pair[0] + best_pair[1]

            if verbose:
                logger.debug("Merge #%d: '%s' + '%s' -> '%s' (count=%d)",
                             len(self.vocab) - len(base_chars) - len(self.SPECIAL_TOKENS) + 1,
                             best_pair[0], best_pair[1], new_token, best_count)

            self.merges.append(best_pair)
            self.vocab.append(new_token)
            self.stoi[new_token] = len(self.vocab) - 1
            self.itos[len(self.vocab) - 1] = new_token

            # Update all word splits: replace every occurrence of best_pair with new_token
            for word in word_freqs:
                split = word_splits[word]
                if len(split) < 2:
                    continue
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = new_split

            if len(self.vocab) >= target_size:
                break

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text into a list of token IDs using learned BPE merges.

        Args:
            text: input string
            add_bos: prepend <BOS> token
            add_eos: append <EOS> token

        Returns:
            list of integer token IDs
        """
        text = self._normalize(text, lowercase=True)
        ids: List[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for word in self._pretokenize(text, lowercase=True):
            word_ids = self._encode_word(word + self._word_suffix)
            ids.extend(word_ids)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def encode_batch(
        self, texts: List[str], add_bos: bool = False, add_eos: bool = False,
        max_length: Optional[int] = None, pad: bool = False,
    ) -> List[List[int]]:
        """
        Encode multiple texts, optionally padding to uniform length.

        Args:
            texts: list of input strings
            add_bos: prepend <BOS> to each
            add_eos: append <EOS> to each
            max_length: truncate to this length
            pad: pad sequences to max_length or longest sequence

        Returns:
            list of integer ID lists
        """
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        if max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]

        if pad:
            target = max_length if max_length is not None else max(len(ids) for ids in encoded)
            for i, ids in enumerate(encoded):
                if len(ids) < target:
                    encoded[i] = ids + [self.pad_id] * (target - len(ids))

        return encoded

    def _encode_word(self, word: str) -> List[int]:
        """Encode a single word (with </w> suffix) by iteratively applying BPE merges."""
        # Split into base tokens: characters, but preserve </w> as a single token
        tokens: List[str] = []
        i = 0
        while i < len(word):
            if word[i:].startswith(self._word_suffix):
                tokens.append(self._word_suffix)
                i += len(self._word_suffix)
            else:
                tokens.append(word[i])
                i += 1
        changed = True
        while changed:
            changed = False
            if len(tokens) < 2:
                break
            # Find the lowest-index merge that applies
            best_merge_idx = None
            best_pair = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self._merge_index:
                    idx = self._merge_index[pair]
                    if best_merge_idx is None or idx < best_merge_idx:
                        best_merge_idx = idx
                        best_pair = pair
            if best_pair is not None:
                # Apply the merge
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                        changed = True
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            else:
                break

        return [self.stoi.get(t, self.unk_id) for t in tokens]

    @property
    def _merge_index(self) -> Dict[Tuple[str, str], int]:
        """Cached mapping from (left, right) pair to merge order index."""
        if not hasattr(self, "_merge_index_cache"):
            self._merge_index_cache = {pair: i for i, pair in enumerate(self.merges)}
        return self._merge_index_cache

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs back into text.

        For GPT-2 style pre-tokenization, ``</w>`` markers are removed
        (the leading-space convention in pre-tokens handles word
        separation). For whitespace pre-tokenization, ``</w>`` is
        replaced with a space.

        Args:
            ids: token ID sequence
            skip_special: omit <PAD>, <UNK>, <BOS>, <EOS> from output

        Returns:
            reconstructed text string
        """
        tokens: List[str] = []
        for i in ids:
            if i >= len(self.vocab) or i < 0:
                tokens.append("?")
                continue
            t = self.itos.get(i, "?")
            if skip_special and t in self.SPECIAL_TOKENS:
                continue
            if t == "<UNK>" and not skip_special:
                tokens.append("?")
                continue
            tokens.append(t)

        raw = "".join(tokens)
        if self._pretokenizer == "gpt2":
            # GPT-2: leading-space in pre-tokens handles word separation
            raw = raw.replace(self._word_suffix, "")
        else:
            # Whitespace: </w> marks word boundary → insert space
            raw = raw.replace(self._word_suffix, " ")
        return raw.strip()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize tokenizer state to a JSON-compatible dict."""
        return {
            "version": 2,
            "vocab": self.vocab,
            "merges": self.merges,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "pretokenizer": self._pretokenizer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SloBPE":
        """Deserialize tokenizer state from a dict."""
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = [(m[0], m[1]) for m in data["merges"]]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        ver = data.get("version", 1)
        if ver >= 2:
            tok._pretokenizer = data.get("pretokenizer", "whitespace")
        else:
            tok._pretokenizer = "whitespace"  # v1 default
        # Special token IDs for backward compat with older serialisations
        for sp in cls.SPECIAL_TOKENS:
            if sp not in tok.stoi:
                tok.stoi[sp] = len(tok.vocab)
                tok.vocab.append(sp)
                tok.itos[len(tok.vocab) - 1] = sp
        return tok

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SloBPE":
        """Load tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------

    def export_to_checkpoint(self) -> dict:
        """
        Return stoi/itos/vocab pairs for embedding in a model checkpoint.

        ``SloughGPTModel`` / ``SloTransformer`` / ``SloLSTM`` all expect
        ``checkpoint["stoi"]``, ``checkpoint["itos"]``, ``checkpoint["vocab_size"]``.
        """
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "vocab_size": self.vocab_size,
            "tokenizer_type": "slonet_bpe",
        }

    @classmethod
    def from_checkpoint(cls, bundle: dict) -> "SloBPE":
        """Rebuild tokenizer from a model checkpoint bundle."""
        stoi = bundle.get("stoi", {})
        itos_raw = bundle.get("itos", {})
        itos = {}
        for k, v in itos_raw.items():
            try:
                itos[int(k)] = v
            except (ValueError, TypeError):
                itos[len(itos)] = v

        if not stoi:
            chars = bundle.get("chars", list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?-'"))
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}

        tok = cls()
        tok.vocab = [itos[i] for i in range(len(itos))] if itos else list(stoi.keys())
        tok.stoi = stoi
        tok.itos = itos
        for sp in cls.SPECIAL_TOKENS:
            if sp not in tok.stoi:
                tok.stoi[sp] = len(tok.vocab)
                tok.vocab.append(sp)
                tok.itos[len(tok.vocab) - 1] = sp
        return tok

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def vocab_stats(self) -> dict:
        """
        Return vocabulary statistics for debugging / inspection.

        Returns:
            dict with keys: vocab_size, base_chars, merges, special_tokens, types
        """
        base_chars_count = 0
        merge_count = 0
        special_count = len(self.SPECIAL_TOKENS)
        for t in self.vocab:
            if t in self.SPECIAL_TOKENS:
                continue
            if len(t) == 1:
                base_chars_count += 1
            else:
                merge_count += 1
        return {
            "vocab_size": self.vocab_size,
            "base_chars": base_chars_count,
            "merged_subwords": merge_count,
            "special_tokens": special_count,
            "total_merges_learned": len(self.merges),
        }

    def show_merges(self, top_n: int = 20) -> None:
        """Print the first N learned merges in order."""
        logger.info("Top %d BPE merges (learned order):", top_n,
            extra={"tag": "TRAIN"},)
        logger.info("%4s  %-12s  %-12s  %-20s", "#", "Left", "Right", "Token",
            extra={"tag": "TRAIN"},)
        logger.info("-" * 52,
            extra={"tag": "TRAIN"},)
        for i, (left, right) in enumerate(self.merges[:top_n]):
            token = left + right
            logger.info("%4d  %-12s  %-12s  %-20s", i + 1, left, right, token,
                extra={"tag": "TRAIN"},)
        if len(self.merges) > top_n:
            logger.info("  ... and %d more merges", len(self.merges) - top_n,
                extra={"tag": "TRAIN"},)

    def show_vocab(self, top_n: int = 30) -> None:
        """Print the first N vocabulary entries."""
        logger.info("Vocabulary (showing %d of %d):", min(top_n, self.vocab_size), self.vocab_size,
            extra={"tag": "TRAIN"},)
        logger.info("%4s  %-20s", "ID", "Token",
            extra={"tag": "TRAIN"},)
        logger.info("-" * 26,
            extra={"tag": "TRAIN"},)
        for i, t in enumerate(self.vocab[:top_n]):
            display = t.replace("\n", "\\n").replace("\t", "\\t")
            marker = " [SPECIAL]" if t in self.SPECIAL_TOKENS else ""
            logger.info("%4d  %-20s%s", i, display, marker,
                extra={"tag": "TRAIN"},)

    # ------------------------------------------------------------------
    # Special token registry
    # ------------------------------------------------------------------

    def add_special_tokens(self, tokens: List[str]) -> int:
        """Register one or more special tokens, adding them to the vocab if new.

        Useful for downstream tasks that need custom control tokens
        (e.g. ``<|user|>``, ``<|assistant|>``, ``<|end|>``).

        Args:
            tokens: list of special token strings

        Returns:
            number of tokens actually added (already-existing ones are skipped)
        """
        added = 0
        for t in tokens:
            if t not in self.stoi:
                tid = len(self.vocab)
                self.vocab.append(t)
                self.stoi[t] = tid
                self.itos[tid] = t
                self._special_set.add(t)
                added += 1
            elif t not in self._special_set:
                self._special_set.add(t)
        return added

    def is_special(self, token: Any) -> bool:
        """Check whether a token (string or ID) is a special token.

        Args:
            token: token string or integer token ID

        Returns:
            True if the token is a registered special token
        """
        if isinstance(token, int):
            token = self.itos.get(token, "")
        return token in self._special_set

    @property
    def special_ids(self) -> List[int]:
        """Return the IDs of all registered special tokens."""
        return [self.stoi[t] for t in self._special_set if t in self.stoi]

    # ------------------------------------------------------------------
    # Pre-tokenization
    # ------------------------------------------------------------------

    def _pretokenize(self, text: str, lowercase: bool = True) -> List[str]:
        """Split text into pretokens using the configured pre-tokenizer."""
        if self._pretokenizer == "gpt2":
            return gpt2_pretokenize(text)
        return text.split()

    def show_pretokenization(self, text: str) -> Dict[str, List[str]]:
        """Visualize how text splits into pretokens before BPE encoding.

        Returns:
            dict with ``pretokens`` (list of strings) and ``segments``
            (list of dicts with ``text``, ``char_count``, ``pct``).
        """
        pretoks = self._pretokenize(text)
        total = len(text)
        segments = []
        for pt in pretoks:
            segments.append({
                "text": pt,
                "char_count": len(pt),
                "pct": round(len(pt) / total * 100, 1) if total else 0,
            })
        return {"pretokens": pretoks, "segments": segments, "count": len(pretoks)}

    # ------------------------------------------------------------------
    # Token decomposition
    # ------------------------------------------------------------------

    def decompose_token(self, token: str) -> Dict:
        """Show how a token decomposes through learned BPE merges.

        Traces the merge tree from the final token back to base characters.

        Args:
            token: a token string in the vocabulary

        Returns:
            dict with ``token``, ``id``, ``merge_path`` (list of merge steps),
            ``depth`` (number of merges), ``base_chars`` (original chars).

        Raises:
            ValueError: if token not in vocabulary
        """
        tok_id = self.stoi.get(token)
        if tok_id is None:
            # Try to find by ID
            try:
                tok_id = int(token)
                token = self.itos.get(tok_id, "")
            except (ValueError, TypeError):
                raise ValueError(f"Token {token!r} not found in vocabulary")
        if not token or token not in self.stoi:
            raise ValueError(f"Token {token!r} not found in vocabulary")

        # If token is a base character or special token, no decomposition
        if token in self._special_set or len(token) <= 1:
            return {
                "token": token,
                "id": self.stoi[token],
                "type": "special" if token in self._special_set else "base_char",
                "merge_path": [],
                "depth": 0,
                "base_chars": [token],
            }

        # Find the merge that created this token by scanning merges in reverse
        merge_path = []
        current = token
        # Build a reverse map: token -> (left, right) that formed it
        reverse_merge = {}
        for left, right in self.merges:
            child = left + right
            reverse_merge[child] = (left, right)

        # Walk backwards through merge tree
        def _trace(t: str) -> List[str]:
            if t in self._special_set or len(t) <= 1:
                return [t]
            if t in reverse_merge:
                left, right = reverse_merge[t]
                merge_path.append({"left": left, "right": right, "into": t})
                return _trace(left) + _trace(right)
            # Token without a recorded merge — likely loaded from checkpoint
            # Try to decompose character-by-character
            return list(t)

        bases = _trace(current)
        return {
            "token": token,
            "id": self.stoi[token],
            "type": "merged_subword",
            "merge_path": merge_path,
            "depth": len(merge_path),
            "base_chars": bases,
        }

    def analyze_corpus(self, texts: List[str]) -> Dict:
        """Compute token frequency and compression statistics on a corpus.

        Populates ``_token_freqs`` and ``_training_corpus_len`` for
        downstream use.

        Args:
            texts: list of text strings to analyze

        Returns:
            dict with ``total_chars``, ``total_tokens``,
            ``compression_ratio``, ``unique_tokens``,
            ``top_tokens`` (list of {token, id, count, pct}),
            ``bottom_tokens`` (rare tokens).
        """
        total_chars = sum(len(t) for t in texts)
        all_ids: List[int] = []
        tok_counts: Counter = Counter()

        for doc in texts:
            ids = self.encode(doc)
            all_ids.extend(ids)
            for tid in ids:
                tok_counts[tid] += 1

        self._token_freqs = tok_counts
        total_tokens = len(all_ids)
        unique_tokens = len(tok_counts)

        top = tok_counts.most_common(20)
        top_tokens = [
            {"id": tid, "token": self.itos.get(tid, "?"), "count": c,
             "pct": round(c / total_tokens * 100, 2) if total_tokens else 0}
            for tid, c in top
        ]

        bottom = tok_counts.most_common()[-20:] if len(tok_counts) > 20 else []
        bottom_tokens = [
            {"id": tid, "token": self.itos.get(tid, "?"), "count": c}
            for tid, c in bottom if c == min(t[1] for t in tok_counts.most_common())
        ][:10]

        return {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "compression_ratio": round(total_chars / max(total_tokens, 1), 2),
            "unique_tokens": unique_tokens,
            "vocab_utilization": round(unique_tokens / max(self.vocab_size, 1) * 100, 1),
            "top_tokens": top_tokens,
            "rare_tokens": bottom_tokens,
        }

    @classmethod
    def train_from_directory(
        cls,
        dir_path: str,
        pattern: str = "*.txt",
        vocab_size: int = 1024,
        min_frequency: int = 2,
        lowercase: bool = True,
        recursive: bool = True,
        verbose: bool = False,
        pretokenizer: str = "gpt2",
    ) -> "SloBPE":
        """Train a tokenizer on all text files in a directory.

        Args:
            dir_path: directory to scan for text files
            pattern: glob pattern for text files (default: ``*.txt``)
            vocab_size: target vocabulary size
            min_frequency: minimum pair frequency for a merge
            lowercase: lowercase text before training
            recursive: recurse into subdirectories
            verbose: log merge progress
            pretokenizer: ``"gpt2"`` or ``"whitespace"``

        Returns:
            trained SloBPE instance
        """
        base = Path(dir_path)
        if not base.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        texts = []
        for p in base.rglob(pattern) if recursive else base.glob(pattern):
            if p.is_file() and p.stat().st_size > 0:
                try:
                    texts.append(p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue

        if not texts:
            raise ValueError(f"No {pattern} files found in {dir_path}")

        tok = cls(pretokenizer=pretokenizer)
        tok.train(texts, vocab_size=vocab_size, min_frequency=min_frequency,
                  lowercase=lowercase, verbose=verbose)
        return tok

    @staticmethod
    def _normalize(text: str, lowercase: bool = True) -> str:
        """Normalize text: lowercase, collapse whitespace, strip."""
        text = text.strip()
        if lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Unigram LM Tokenizer  (SentencePiece-style)
# ══════════════════════════════════════════════════════════════════════════════


class SloUnigram:
    """
    Unigram Language Model tokenizer (SentencePiece-style).

    Learns a subword vocabulary by estimating token probabilities via EM
    and pruning low-contribution tokens.  Supports subword regularization
    (sampling multiple segmentations) during encoding.

    Drops into any code expecting ``stoi`` / ``itos`` dict interface.

    Usage::

        tok = SloUnigram()
        tok.train(["hello world", "hello there"], vocab_size=64)

        ids = tok.encode("hello world")
        text = tok.decode(ids)

        tok.encode_with_scores("hello world", nbest=4)   # subword sampling
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, pretokenizer: str = "gpt2") -> None:
        self.vocab: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._scores: Dict[int, float] = {}       # token_id → log-probability
        self._pretokenizer: str = pretokenizer
        self._special_set: Set[str] = set(self.SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi.get("<PAD>", 0)

    @property
    def unk_id(self) -> int:
        return self.stoi.get("<UNK>", 0)

    @property
    def bos_id(self) -> int:
        return self.stoi.get("<BOS>", 0)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<EOS>", 0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: List[str],
        vocab_size: int = 1024,
        lowercase: bool = True,
        seed_max_len: int = 8,
        pruning_ratio: float = 0.5,
        em_iters: int = 4,
        verbose: bool = False,
    ) -> "SloUnigram":
        """Learn a unigram subword vocabulary from a corpus.

        Args:
            texts: raw text strings to learn from
            vocab_size: target vocabulary size
            lowercase: lowercase before training
            seed_max_len: max character n-gram length for the seed vocabulary
            pruning_ratio: fraction of lowest-loss tokens to remove per iteration
            em_iters: EM iterations per pruning round
            verbose: log progress

        Returns:
            self
        """
        if not texts:
            raise ValueError("Need at least one text to train on")

        corpus = [self._normalize(t, lowercase) for t in texts]

        # ── Step 1: Build seed vocabulary ──────────────────
        # All character n-grams up to seed_max_len that appear in the corpus
        # plus common printable ASCII characters for coverage
        seen: Set[str] = set()
        for doc in corpus:
            for pretok in self._pretokenize(doc, lowercase):
                for i in range(len(pretok)):
                    for j in range(i + 1, min(i + seed_max_len + 1, len(pretok) + 1)):
                        seen.add(pretok[i:j])

        # Ensure coverage of all printable ASCII characters in seed vocab
        import string as _string_mod
        for ch in _string_mod.printable:
            seen.add(ch)

        seen = {s for s in seen if len(s.strip()) > 0 or s == " "}
        seen_list = sorted(seen)

        # Build initial vocab: special tokens + all n-grams
        self.vocab = list(self.SPECIAL_TOKENS) + seen_list
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self._special_set = set(self.SPECIAL_TOKENS)

        if verbose:
            logger.debug("Seed vocab: %d tokens (%d n-grams)", len(self.vocab), len(seen))

        # ── Step 2: Pre-tokenize — split corpus into pretokens ──
        pretok_counts: Counter = Counter()
        for doc in corpus:
            for pt in self._pretokenize(doc, lowercase):
                if pt.strip():
                    pretok_counts[pt] += 1

        pretok_list = list(pretok_counts.keys())
        # Precompute all tokenization paths for each pretoken
        # path_cache[pretoken] = list of (token_str,) segmentations
        path_cache: Dict[str, List[Tuple[str, ...]]] = {}
        for pt in pretok_list:
            path_cache[pt] = self._all_segmentations(pt)

        # ── Step 3: Iterative pruning ──────────────────────
        # Start with uniform scores, prune by loss contribution
        total_tokens = len(self.vocab)
        log_uniform = -math.log(len(self.vocab))

        for tok_id in range(total_tokens):
            self._scores[tok_id] = log_uniform

        target = min(vocab_size, total_tokens)

        while len(self.vocab) > target:
            # EM: estimate expected counts for each token
            token_counts: Counter = Counter()

            for pt in pretok_list:
                freq = pretok_counts[pt]
                paths = path_cache[pt]
                if not paths:
                    continue  # pragma: no cover (unreachable — seed vocab covers every substring)

                # Compute P(path) for all paths using current scores
                path_probs = []
                for path in paths:
                    lp = sum(self._scores[self.stoi[t]] for t in path if t in self.stoi)
                    path_probs.append((path, lp))

                # Softmax-normalize path log-probabilities
                max_lp = max(lp for _, lp in path_probs) if path_probs else 0
                exp_lps = [math.exp(lp - max_lp) for _, lp in path_probs]
                total_exp = sum(exp_lps)
                if total_exp == 0:
                    continue  # pragma: no cover (unreachable — at least one path has weight > 0)
                path_weights = [e / total_exp for e in exp_lps]

                for (path, _), weight in zip(path_probs, path_weights):
                    for t in path:
                        if t in self.stoi:
                            token_counts[self.stoi[t]] += weight * freq

            # Re-estimate scores from expected counts
            total_count = sum(token_counts.values())
            if total_count == 0:
                break

            # Re-score: log(relative frequency), floor at log_uniform
            new_scores: Dict[int, float] = {}
            for tok_id in range(len(self.vocab)):
                raw = token_counts.get(tok_id, 0)
                if raw > 0:
                    new_scores[tok_id] = math.log(raw / total_count)
                else:
                    new_scores[tok_id] = log_uniform
            self._scores = new_scores

            # Compute loss contribution of each token
            # Protect single characters and special tokens from pruning
            protected: Set[int] = self._special_token_ids.copy()
            for tid, t in enumerate(self.vocab):
                if len(t) == 1:
                    protected.add(tid)

            loss_contrib: List[Tuple[float, int]] = []  # (delta_loss, tok_id)
            for tok_id in range(len(self.vocab)):
                if tok_id in protected:
                    continue
                freq = token_counts.get(tok_id, 0)
                if freq > 0:
                    contrib = -freq * self._scores[tok_id]
                else:
                    contrib = 0  # pragma: no cover (unreachable — every in-vocab token appears in a segmentation)
                loss_contrib.append((contrib, tok_id))

            # Sort by loss contribution (ascending) and prune lowest
            loss_contrib.sort(key=lambda x: x[0])

            n_prune = max(1, int(len(loss_contrib) * pruning_ratio))
            pruned_ids = set(tid for _, tid in loss_contrib[:n_prune])
            # Protect special tokens
            pruned_ids -= self._special_token_ids

            if not pruned_ids:
                break

            # Remove pruned tokens from vocab
            old_vocab = self.vocab
            old_scores = self._scores
            new_vocab = [t for i, t in enumerate(old_vocab) if i not in pruned_ids]
            self.vocab = new_vocab
            self.stoi = {t: i for i, t in enumerate(self.vocab)}
            self.itos = {i: t for i, t in enumerate(self.vocab)}
            self._scores = {self.stoi[t]: old_scores[i]
                           for i, t in enumerate(old_vocab)
                           if i not in pruned_ids}

            if verbose:
                logger.debug("Pruned %d -> %d tokens (target %d)", n_prune, len(self.vocab), target)

            if len(self.vocab) <= target:
                break

        return self

    # ------------------------------------------------------------------
    # Encoding  (Viterbi — most likely segmentation)
    # ------------------------------------------------------------------

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text via the most likely (Viterbi) segmentation.

        Args:
            text: input string
            add_bos: prepend <BOS>
            add_eos: append <EOS>

        Returns:
            list of token IDs
        """
        text = self._normalize(text, lowercase=True)
        ids: List[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for pretok in self._pretokenize(text, True):
            if not pretok.strip():
                continue  # pragma: no cover (unreachable — pretokenize never yields empty strings)
            path = self._viterbi(pretok)
            ids.extend(self.stoi.get(t, self.unk_id) for t in path)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def encode_with_scores(
        self, text: str, nbest: int = 10, alpha: float = 0.1,
    ) -> List[Tuple[List[int], float]]:
        """Encode text, returning the n-best segmentations with scores.

        Implements subword regularization: samples from the posterior
        distribution over segmentations.  Use for training with
        regularization.

        Args:
            text: input string
            nbest: number of candidate segmentations per pretoken
            alpha: smoothing strength for sampling (0 = greedy, 1 = uniform)

        Returns:
            list of (ids, score) tuples, sorted by score descending
        """
        text = self._normalize(text, lowercase=True)
        all_candidates: List[Tuple[List[int], float]] = [([], 0.0)]

        for pretok in self._pretokenize(text, True):
            if not pretok.strip():
                continue  # pragma: no cover (unreachable — pretokenize never yields empty strings)
            paths = self._all_segmentations(pretok)
            if not paths:
                continue  # pragma: no cover (unreachable — seed vocab covers every substring)

            scored = []
            for path in paths:
                lp = sum(self._scores.get(self.stoi.get(t, 0), -10)
                        for t in path if t in self.stoi)
                ids = [self.stoi.get(t, self.unk_id) for t in path]
                scored.append((ids, lp))

            # Softmax with temperature
            max_lp = max(lp for _, lp in scored) if scored else 0
            exp_lps = [math.exp((lp - max_lp) / max(alpha, 0.01)) for _, lp in scored]
            total_exp = sum(exp_lps)
            if total_exp == 0:
                continue  # pragma: no cover (unreachable — at least one path has weight > 0)
            probs = [e / total_exp for e in exp_lps]

            # Sample or take top-nbest
            import random
            sampled = random.choices(range(len(scored)), weights=probs, k=min(nbest, len(scored)))
            chosen = [scored[i] for i in sampled]

            # Combine with existing candidates
            new_candidates = []
            for existing_ids, existing_score in all_candidates:
                for new_ids, new_score in chosen:
                    new_candidates.append((
                        existing_ids + new_ids,
                        existing_score + new_score,
                    ))

            # Keep top nbest
            new_candidates.sort(key=lambda x: -x[1])
            all_candidates = new_candidates[:nbest]

        return all_candidates or [([self.unk_id], -10.0)]  # pragma: no cover (unreachable — candidates never empty)

    def _viterbi(self, text: str) -> List[str]:
        """Find the most likely segmentation of text using Viterbi decoding.

        Dynamic programming over character positions.  ``dp[i]`` = best
        log-prob to reach position i, ``back[i]`` = (prev_pos, token).

        Returns:
            list of token strings forming the best segmentation
        """
        n = len(text)
        if n == 0:
            return []

        # Collect all valid substrings that are in the vocab
        tokens_at: Dict[int, List[Tuple[int, str, float]]] = {i: [] for i in range(n + 1)}
        for i in range(n):
            for j in range(i + 1, n + 1):
                sub = text[i:j]
                if sub in self.stoi:
                    tok_id = self.stoi[sub]
                    score = self._scores.get(tok_id, 0)
                    tokens_at[i].append((j, sub, score))

        # Viterbi DP
        neg_inf = -1e10
        dp = [neg_inf] * (n + 1)
        back: List[Optional[Tuple[int, str]]] = [None] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == neg_inf:
                continue
            for j, token, score in tokens_at[i]:
                cand = dp[i] + score
                if cand > dp[j]:
                    dp[j] = cand
                    back[j] = (i, token)

        if dp[n] == neg_inf:
            # Fallback: character-level
            return list(text)

        # Backtrack
        tokens: List[str] = []
        pos = n
        while pos > 0:
            prev, token = back[pos]
            tokens.append(token)
            pos = prev
        tokens.reverse()
        return tokens

    def _all_segmentations(self, text: str) -> List[Tuple[str, ...]]:
        """Enumerate all possible segmentations of a text string.

        Uses dynamic programming (DFS from each position).

        Returns:
            list of token-tuple segmentations
        """
        n = len(text)
        if n == 0:
            return []

        # memo[pos] = list of token-tuples from pos to end
        memo: Dict[int, List[Tuple[str, ...]]] = {}

        def _dfs(pos: int) -> List[Tuple[str, ...]]:
            if pos in memo:
                return memo[pos]
            if pos >= n:
                return [tuple()]

            results: List[Tuple[str, ...]] = []
            for end in range(pos + 1, n + 1):
                sub = text[pos:end]
                if sub in self.stoi:
                    rest = _dfs(end)
                    for r in rest:
                        results.append((sub,) + r)
            memo[pos] = results
            return results

        return _dfs(0)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: token ID sequence
            skip_special: omit special tokens from output

        Returns:
            reconstructed text
        """
        tokens: List[str] = []
        for i in ids:
            if i >= len(self.vocab) or i < 0:
                tokens.append("?")
                continue
            t = self.itos.get(i, "?")
            if skip_special and t in self.SPECIAL_TOKENS:
                continue
            tokens.append(t)

        return "".join(tokens).strip()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize tokenizer state to a JSON-compatible dict."""
        return {
            "version": 1,
            "type": "unigram",
            "vocab": self.vocab,
            "scores": {str(k): v for k, v in self._scores.items()},
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "pretokenizer": self._pretokenizer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SloUnigram":
        """Deserialize tokenizer state from a dict."""
        tok = cls()
        tok.vocab = data["vocab"]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok._scores = {int(k): v for k, v in data.get("scores", {}).items()}
        tok._pretokenizer = data.get("pretokenizer", "gpt2")
        for sp in cls.SPECIAL_TOKENS:
            if sp not in tok.stoi:
                tok.stoi[sp] = len(tok.vocab)
                tok.vocab.append(sp)
                tok.itos[len(tok.vocab) - 1] = sp
        return tok

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SloUnigram":
        """Load tokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def vocab_stats(self) -> dict:
        """Return vocabulary statistics."""
        special_count = len(self.SPECIAL_TOKENS)
        return {
            "vocab_size": self.vocab_size,
            "type": "unigram",
            "special_tokens": special_count,
            "subwords": self.vocab_size - special_count,
            "pretokenizer": self._pretokenizer,
        }

    # ------------------------------------------------------------------
    # Pre-tokenization analysis
    # ------------------------------------------------------------------

    def show_pretokenization(self, text: str) -> Dict:
        """Visualize how text splits into pretokens before encoding.

        Returns:
            dict with ``pretokens`` (list of strings) and ``segments``
            (list of dicts with ``text``, ``char_count``, ``pct``).
        """
        pretoks = self._pretokenize(text)
        total = len(text)
        segments = []
        for pt in pretoks:
            segments.append({
                "text": pt,
                "char_count": len(pt),
                "pct": round(len(pt) / total * 100, 1) if total else 0,
            })
        return {"pretokens": pretoks, "segments": segments, "count": len(pretoks)}

    # ------------------------------------------------------------------
    # Token decomposition
    # ------------------------------------------------------------------

    def decompose_token(self, token: str) -> Dict:
        """Show how a token decomposes through the Unigram vocabulary.

        For Unigram there are no merge trees — shows the base characters
        and the token's score.

        Args:
            token: a token string in the vocabulary

        Returns:
            dict with ``token``, ``id``, ``type``, ``score``,
            ``base_chars`` (character decomposition).

        Raises:
            ValueError: if token not in vocabulary
        """
        tok_id = self.stoi.get(token)
        if tok_id is None:
            try:
                tok_id = int(token)
                token = self.itos.get(tok_id, "")
            except (ValueError, TypeError):
                raise ValueError(f"Token {token!r} not found in vocabulary")
        if not token or token not in self.stoi:
            raise ValueError(f"Token {token!r} not found in vocabulary")

        token_type = "special" if token in self._special_set else \
                     "base_char" if len(token) <= 1 else "subword"

        return {
            "token": token,
            "id": self.stoi[token],
            "type": token_type,
            "score": self._scores.get(tok_id, 0),
            "merge_path": [],
            "depth": 0,
            "base_chars": list(token),
        }

    # ------------------------------------------------------------------
    # Corpus analysis
    # ------------------------------------------------------------------

    def analyze_corpus(self, texts: List[str]) -> Dict:
        """Compute token frequency and compression statistics on a corpus.

        Args:
            texts: list of text strings to analyze

        Returns:
            dict with ``total_chars``, ``total_tokens``,
            ``compression_ratio``, ``unique_tokens``,
            ``vocab_utilization``, ``top_tokens``, ``rare_tokens``.
        """
        from collections import Counter
        total_chars = sum(len(t) for t in texts)
        all_ids: List[int] = []
        tok_counts: Counter = Counter()

        for doc in texts:
            ids = self.encode(doc)
            all_ids.extend(ids)
            for tid in ids:
                tok_counts[tid] += 1

        total_tokens = len(all_ids)
        unique_tokens = len(tok_counts)

        top = tok_counts.most_common(20)
        top_tokens = [
            {"id": tid, "token": self.itos.get(tid, "?"), "count": c,
             "pct": round(c / max(total_tokens, 1) * 100, 2)}
            for tid, c in top
        ]

        bottom = tok_counts.most_common()[-20:] if len(tok_counts) > 20 else []
        bottom_tokens = [
            {"id": tid, "token": self.itos.get(tid, "?"), "count": c}
            for tid, c in bottom if c == min(t[1] for t in tok_counts.most_common())
        ][:10]

        return {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "compression_ratio": round(total_chars / max(total_tokens, 1), 2),
            "unique_tokens": unique_tokens,
            "vocab_utilization": round(unique_tokens / max(self.vocab_size, 1) * 100, 1),
            "top_tokens": top_tokens,
            "rare_tokens": bottom_tokens,
        }

    # ------------------------------------------------------------------
    # Directory training
    # ------------------------------------------------------------------

    @classmethod
    def train_from_directory(
        cls,
        dir_path: str,
        pattern: str = "*.txt",
        vocab_size: int = 1024,
        lowercase: bool = True,
        recursive: bool = True,
        verbose: bool = False,
        pretokenizer: str = "gpt2",
        **algo_kwargs,
    ) -> "SloUnigram":
        """Train a tokenizer on all text files in a directory.

        Args:
            dir_path: directory to scan for text files
            pattern: glob pattern for text files (default: ``*.txt``)
            vocab_size: target vocabulary size
            lowercase: lowercase text before training
            recursive: recurse into subdirectories
            verbose: log training progress
            pretokenizer: ``"gpt2"`` or ``"whitespace"``
            **algo_kwargs: passed to ``train()`` (seed_max_len, pruning_ratio, em_iters)

        Returns:
            trained SloUnigram instance
        """
        base = Path(dir_path)
        texts: List[str] = []
        it = base.rglob(pattern) if recursive else base.glob(pattern)
        for p in it:
            if p.is_file():
                texts.append(p.read_text(encoding="utf-8", errors="replace"))
        tok = cls(pretokenizer=pretokenizer)
        tok.train(texts, vocab_size=vocab_size, lowercase=lowercase,
                  verbose=verbose, **algo_kwargs)
        return tok

    def show_vocab(self, top_n: int = 30) -> None:
        """Print vocabulary with scores."""
        scored = sorted(
            [(tid, t, self._scores.get(tid, 0))
             for tid, t in enumerate(self.vocab)],
            key=lambda x: -x[2],
        )
        logger.info("Top %d tokens (by score):", min(top_n, len(scored)),
            extra={"tag": "TRAIN"},)
        logger.info("%4s  %-24s  %-10s", "ID", "Token", "Log-P",
            extra={"tag": "TRAIN"},)
        logger.info("-" * 42,
            extra={"tag": "TRAIN"},)
        for tid, t, lp in scored[:top_n]:
            marker = " [SPECIAL]" if t in self.SPECIAL_TOKENS else ""
            logger.info("%4d  %-24s  %.4f%s", tid, t, lp, marker,
                extra={"tag": "TRAIN"},)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pretokenize(self, text: str, lowercase: bool = True) -> List[str]:
        if self._pretokenizer == "gpt2":
            return gpt2_pretokenize(text)
        return text.split()

    @property
    def _special_token_ids(self) -> Set[int]:
        return set(self.stoi[t] for t in self._special_set if t in self.stoi)

    @staticmethod
    def _normalize(text: str, lowercase: bool = True) -> str:
        text = text.strip()
        if lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text
