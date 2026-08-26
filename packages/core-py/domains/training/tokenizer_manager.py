"""
TokenizerManager — composable BPE/Unigram tokenizer lifecycle for the whole codebase.

All frontends (API server, CLI, TUI, Web UI) instantiate a manager and call
its methods.  The manager owns the singleton ``SloBPE`` or ``SloUnigram``
instance, handles training, caching, persistence, and integration.

Usage:
    from domains.training.tokenizer_manager import get_tokenizer_manager

    mgr = get_tokenizer_manager()
    mgr.train(["hello world", "test data"], vocab_size=512, algo="bpe")
    mgr.train(["hello world", "test data"], vocab_size=512, algo="unigram")
    ids = mgr.tokenize("hello world")
    text = mgr.detokenize(ids)
    stats = mgr.stats()
    mgr.save("/tmp/my_tokenizer.json")
    mgr.load("/tmp/my_tokenizer.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("slo.training.tokenizer_manager")


_TOKENIZER_ALGO_KEY = "_algo"


class TokenizerManager:
    """Manages the global tokenizer singleton (BPE or Unigram LM).

    All tokenizer operations (train, encode, decode, save, load) go through
    this class.  Frontends never interact with SloBPE/SloUnigram directly
    — they call ``get_tokenizer_manager()`` and use its public API.
    """

    _instance: Optional[TokenizerManager] = None

    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None
        self._algo: str = "bpe"

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "TokenizerManager":
        """Return the global singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Tokenizer lifecycle
    # ------------------------------------------------------------------

    def get_tokenizer(self) -> Any:
        """Return the current tokenizer, creating a default one if needed."""
        if self._tokenizer is None:
            from domains.training.tokenizer import SloBPE
            self._tokenizer = SloBPE()
        return self._tokenizer

    def train(
        self,
        texts: List[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
        lowercase: bool = True,
        pretokenizer: str = "gpt2",
        algo: str = "bpe",
        **algo_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the tokenizer on a corpus of texts.

        Args:
            texts: training texts
            vocab_size: target vocabulary size
            min_frequency: minimum pair frequency (BPE only)
            lowercase: lowercase text before training
            pretokenizer: ``"gpt2"`` (default) or ``"whitespace"``
            algo: ``"bpe"`` (SloBPE, default) or ``"unigram"`` (SloUnigram)
            **algo_kwargs: passed through to the algorithm's ``train()``
                (e.g. ``seed_max_len=8``, ``pruning_ratio=0.5`` for unigram)

        Returns:
            vocab_stats dict from the trained tokenizer
        """
        if algo == "unigram":
            from domains.training.tokenizer import SloUnigram
            self._tokenizer = SloUnigram(pretokenizer=pretokenizer)
            self._tokenizer.train(
                texts,
                vocab_size=vocab_size,
                lowercase=lowercase,
                **algo_kwargs,
            )
        else:
            from domains.training.tokenizer import SloBPE
            self._tokenizer = SloBPE(pretokenizer=pretokenizer)
            self._tokenizer.train(
                texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                lowercase=lowercase,
            )
        self._algo = algo
        return self._tokenizer.vocab_stats()

    def analyze_corpus(self, texts: List[str]) -> Dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, 'analyze_corpus'):
            return tok.analyze_corpus(texts)
        return {"error": "analyze_corpus not available for this tokenizer type"}

    def show_pretokenization(self, text: str) -> Dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, 'show_pretokenization'):
            return tok.show_pretokenization(text)
        return {"error": "show_pretokenization not available for this tokenizer type", "pretokens": [], "segments": [], "count": 0}

    def decompose_token(self, token: str) -> Dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, 'decompose_token'):
            return tok.decompose_token(token)
        raise ValueError("decompose_token not available for this tokenizer type")

    def train_from_directory(
        self,
        dir_path: str,
        pattern: str = "*.txt",
        vocab_size: int = 1024,
        min_frequency: int = 2,
        lowercase: bool = True,
        recursive: bool = True,
        pretokenizer: str = "gpt2",
        algo: str = "bpe",
        **algo_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the tokenizer on all text files in a directory.

        Args:
            dir_path: directory to scan for text files
            pattern: glob pattern (default: ``*.txt``)
            vocab_size: target vocabulary size
            min_frequency: minimum pair frequency (BPE only)
            lowercase: lowercase text before training
            recursive: recurse into subdirectories
            pretokenizer: ``"gpt2"`` or ``"whitespace"``
            algo: ``"bpe"`` (default) or ``"unigram"``
            **algo_kwargs: passed through to the algorithm's ``train()``

        Returns:
            vocab_stats dict
        """
        # Collect texts from directory
        root = Path(dir_path)
        texts: List[str] = []
        it = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in it:
            if p.is_file():
                texts.append(p.read_text(encoding="utf-8", errors="replace"))

        return self.train(texts, vocab_size=vocab_size, min_frequency=min_frequency,
                          lowercase=lowercase, pretokenizer=pretokenizer, algo=algo, **algo_kwargs)

    @property
    def tokenizer_type(self) -> str:
        """``"bpe"`` or ``"unigram"`` — the algorithm of the current tokenizer."""
        return self._algo

    @property
    def pretokenizer(self) -> str:
        """Current pre-tokenizer type (``"gpt2"`` or ``"whitespace"``)."""
        tok = self.get_tokenizer()
        return tok._pretokenizer if hasattr(tok, '_pretokenizer') else 'whitespace'

    # -- note: there was a dangling docstring here from an earlier edit --
    def tokenize(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.get_tokenizer().encode(text)

    def detokenize(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.get_tokenizer().decode(ids)

    def stats(self) -> Dict[str, Any]:
        """Return vocabulary statistics."""
        tok = self.get_tokenizer()
        if tok.vocab_size == 0:
            return {"vocab_size": 0, "base_chars": 0, "merged_subwords": 0, "special_tokens": 0, "total_merges": 0, "trained": False}
        s = tok.vocab_stats()
        s["trained"] = True
        s["algo"] = self._algo
        return s

    def is_trained(self) -> bool:
        return self._tokenizer is not None and self._tokenizer.vocab_size > 0

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.vocab_size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        data = self.get_tokenizer().to_dict()
        data[_TOKENIZER_ALGO_KEY] = self._algo
        return data

    def from_dict(self, data: dict) -> None:
        algo = data.get(_TOKENIZER_ALGO_KEY, "bpe")
        if algo == "unigram":
            from domains.training.tokenizer import SloUnigram
            self._tokenizer = SloUnigram.from_dict(data)
        else:
            from domains.training.tokenizer import SloBPE
            self._tokenizer = SloBPE.from_dict(data)
        self._algo = algo

    def save(self, path: str) -> None:
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def borrow_from_autotrain(self) -> bool:
        if self._tokenizer is not None and self._tokenizer.vocab_size > 0:
            return True
        try:
            from routers.auto_train import state as at_state
            at_tok = at_state.student_tokenizer
            if at_tok is not None and hasattr(at_tok, "vocab_size") and at_tok.vocab_size > 10:
                self._tokenizer = at_tok
                return True
        except Exception as e:
            logger.debug("autotrain tokenizer borrow failed: %s", e)
        return False

    def adopt(self, tokenizer: Any) -> None:
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size > 0:
            self._tokenizer = tokenizer

    def set_tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def reset(self) -> None:
        from domains.training.tokenizer import SloBPE
        self._tokenizer = SloBPE()
        self._algo = "bpe"


def get_tokenizer_manager() -> TokenizerManager:
    """Shortcut to the global TokenizerManager singleton."""
    return TokenizerManager.get_instance()
