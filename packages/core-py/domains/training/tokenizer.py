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
from collections import Counter
from typing import Dict, List, Optional, Tuple


class SloBPE:
    """
    Byte-Pair Encoding tokenizer compatible with SloNet's stoi/itos interface.

    Trains merge rules from a corpus, encodes text into subword token IDs,
    and decodes IDs back into text. Drops into any code expecting
    ``stoi: Dict[str, int]`` and ``itos: Dict[int, str]``.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    WORD_SUFFIX = "</w>"

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._word_suffix: str = self.WORD_SUFFIX

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
            for word in doc.split():
                for ch in word:
                    chars[ch] += 1

        # Build initial vocab: special tokens + </w> + all unique characters
        base_chars = sorted(chars.keys())
        self.vocab = list(self.SPECIAL_TOKENS) + [self._word_suffix] + base_chars.copy()
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}
        self.merges = []

        # Pre-tokenise: split each doc into words, each word into chars
        # Append </w> suffix marks word boundaries for proper decoding
        word_freqs: Counter = Counter()
        for doc in corpus:
            for word in doc.split():
                word_freqs[word + self._word_suffix] += 1

        # Each word is represented as a list of current tokens (chars + </w>)
        word_splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            raw = word.replace(self._word_suffix, "")
            word_splits[word] = list(raw) + [self._word_suffix]

        # Merge loop
        target_size = min(vocab_size, len(base_chars) + len(self.SPECIAL_TOKENS) + len(base_chars))
        while len(self.vocab) < target_size:
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
                print(f"Merge #{len(self.vocab) - len(base_chars) - len(self.SPECIAL_TOKENS) + 1}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' "
                      f"(count={best_count})")

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

        for word in text.split():
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

        ``</w>`` markers in tokens are replaced with spaces to restore
        word boundaries lost during BPE encoding.

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
        raw = raw.replace(self._word_suffix, " ")
        return raw.strip()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize tokenizer state to a JSON-compatible dict."""
        return {
            "version": 1,
            "vocab": self.vocab,
            "merges": self.merges,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SloBPE":
        """Deserialize tokenizer state from a dict."""
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = [(m[0], m[1]) for m in data["merges"]]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
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
        print(f"Top {top_n} BPE merges (learned order):")
        print(f"{'#':>4}  {'Left':<12}  {'Right':<12}  {'Token':<20}")
        print("-" * 52)
        for i, (left, right) in enumerate(self.merges[:top_n]):
            token = left + right
            print(f"{i + 1:>4}  {left:<12}  {right:<12}  {token:<20}")
        if len(self.merges) > top_n:
            print(f"  ... and {len(self.merges) - top_n} more merges")

    def show_vocab(self, top_n: int = 30) -> None:
        """Print the first N vocabulary entries."""
        print(f"Vocabulary (showing {min(top_n, self.vocab_size)} of {self.vocab_size}):")
        print(f"{'ID':>4}  {'Token':<20}")
        print("-" * 26)
        for i, t in enumerate(self.vocab[:top_n]):
            display = t.replace("\n", "\\n").replace("\t", "\\t")
            marker = " [SPECIAL]" if t in self.SPECIAL_TOKENS else ""
            print(f"{i:>4}  {display:<20}{marker}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str, lowercase: bool = True) -> str:
        """Normalize text: lowercase, collapse whitespace, strip."""
        text = text.strip()
        if lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text
