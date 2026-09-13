"""Character-level tokenizer for multimodal caption vocabulary.

Simple, deterministic tokenization that maps each character to an ID.
Produces consistent token lengths (one token per character), ideal for
small-vocabulary debugging and training.
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional
import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer with special tokens.

    Special tokens: <BOS>=0, <EOS>=1, <PAD>=2, <UNK>=3.
    Vocabulary is built from the set of unique characters in training texts,
    plus all printable ASCII characters for robustness.

    Each character maps to exactly one token ID. Encoding is:
        [BOS] + [char_ids] + [EOS]

    Decoding reverses the mapping and strips special tokens.
    """

    SPECIAL_TOKENS = ["<BOS>", "<EOS>", "<PAD>", "<UNK>"]

    def __init__(self, pad_to: Optional[int] = None):
        self.pad_to = pad_to
        self.vocab: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._built = False

    def _ensure_ascii(self) -> Set[str]:
        """Return set of all printable ASCII characters.
        Acts as a fallback base vocabulary so unseen characters at
        inference time map to <UNK> less often.
        """
        chars: Set[str] = set()
        for i in range(32, 127):  # printable ASCII
            chars.add(chr(i))
        chars.add("\n")
        chars.add("\t")
        return chars

    def build_vocab(self, texts: List[str]):
        """Build character vocabulary from training texts.

        Collects all unique characters plus printable ASCII fallback,
        then assigns IDs starting after special tokens.
        """
        chars: Set[str] = set()
        for t in texts:
            chars.update(t)

        # Merge with ASCII fallback
        chars |= self._ensure_ascii()

        # Assign IDs: special tokens first, then sorted chars
        self.vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        self.itos = {i: tok for tok, i in self.vocab.items()}
        self._built = True

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with BOS/EOS markers.

        Returns [0] + [char_id for each char] + [1], i.e.
        BOS at position 0, chars in order, EOS at end.
        """
        if not self._built:
            raise RuntimeError("Tokenizer not trained. Call build_vocab() first.")
        unk = self.vocab.get("<UNK>", 3)
        ids = [self.vocab.get("<BOS>", 0)]
        for ch in text:
            ids.append(self.vocab.get(ch, unk))
        ids.append(self.vocab.get("<EOS>", 1))
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text, stripping special tokens."""
        chars: List[str] = []
        special = {"<BOS>", "<EOS>", "<PAD>", "<UNK>"}
        for tid in token_ids:
            tok = self.itos.get(tid, "")
            if tok in special:
                continue
            chars.append(tok)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str):
        """Save tokenizer state to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chars": sorted(c for c in self.vocab if c not in set(self.SPECIAL_TOKENS)),
            "pad_to": self.pad_to,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load tokenizer state from JSON. Returns True on success."""
        if not Path(path).exists():
            return False
        with open(path) as f:
            data = json.load(f)
        self.vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        for ch in sorted(data.get("chars", [])):
            self.vocab[ch] = len(self.vocab)
        self.itos = {i: tok for tok, i in self.vocab.items()}
        self.pad_to = data.get("pad_to")
        self._built = True
        return True
