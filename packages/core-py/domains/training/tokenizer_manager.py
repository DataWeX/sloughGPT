"""
TokenizerManager — composable BPE tokenizer lifecycle for the whole codebase.

All frontends (API server, CLI, TUI, Web UI) instantiate a manager and call
its methods.  The manager owns the singleton ``SloBPE`` instance, handles
training, caching, persistence, and integration with auto-train.

Usage:
    from domains.training.tokenizer_manager import get_tokenizer_manager

    mgr = get_tokenizer_manager()
    mgr.train(["hello world", "test data"], vocab_size=512)
    ids = mgr.tokenize("hello world")
    text = mgr.detokenize(ids)
    stats = mgr.stats()
    mgr.save("/tmp/my_tokenizer.json")
    mgr.load("/tmp/my_tokenizer.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TokenizerManager:
    """Manages the global SloBPE tokenizer singleton.

    All tokenizer operations (train, encode, decode, save, load) go through
    this class.  Frontends never interact with SloBPE directly — they call
    ``get_tokenizer_manager()`` and use its public API.
    """

    _instance: Optional[TokenizerManager] = None

    def __init__(self) -> None:
        self._tokenizer: Optional[Any] = None

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
    ) -> Dict[str, Any]:
        """Train the BPE tokenizer on a corpus of texts.

        Args:
            texts: training texts
            vocab_size: target vocabulary size
            min_frequency: minimum pair frequency for a merge
            lowercase: lowercase text before training

        Returns:
            vocab_stats dict from SloBPE.vocab_stats()
        """
        tok = self.get_tokenizer()
        tok.train(texts, vocab_size=vocab_size, min_frequency=min_frequency, lowercase=lowercase)
        return tok.vocab_stats()

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
        return s

    def is_trained(self) -> bool:
        """Check whether the tokenizer has been trained."""
        return self._tokenizer is not None and self._tokenizer.vocab_size > 0

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (0 if not trained)."""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.vocab_size

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Export the tokenizer state as a JSON-serializable dict."""
        return self.get_tokenizer().to_dict()

    def from_dict(self, data: dict) -> None:
        """Restore the tokenizer from a previously exported dict."""
        from domains.training.tokenizer import SloBPE
        self._tokenizer = SloBPE.from_dict(data)

    def save(self, path: str) -> None:
        """Save the tokenizer to disk as JSON."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load a previously saved tokenizer from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def borrow_from_autotrain(self) -> bool:
        """Try to borrow the BPE tokenizer from auto-train's state.

        Returns True if a tokenizer was found and adopted.
        """
        if self._tokenizer is not None and self._tokenizer.vocab_size > 0:
            return True
        try:
            from routers.auto_train import state as at_state
            at_tok = at_state.student_tokenizer
            if at_tok is not None and hasattr(at_tok, "vocab_size") and at_tok.vocab_size > 10:
                self._tokenizer = at_tok
                return True
        except Exception:
            pass
        return False

    def adopt(self, tokenizer: Any) -> None:
        """Adopt an already-trained SloBPE instance (e.g. from a checkpoint)."""
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size > 0:
            self._tokenizer = tokenizer

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Explicitly set the tokenizer (for injection from tests or pipelines)."""
        self._tokenizer = tokenizer

    def reset(self) -> None:
        """Reset to an untrained state."""
        from domains.training.tokenizer import SloBPE
        self._tokenizer = SloBPE()


# ------------------------------------------------------------------
# Module-level convenience accessor
# ------------------------------------------------------------------

def get_tokenizer_manager() -> TokenizerManager:
    """Shortcut to the global TokenizerManager singleton."""
    return TokenizerManager.get_instance()
