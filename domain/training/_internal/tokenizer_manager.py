"""
TokenizerManager — composable BPE/Unigram tokenizer lifecycle for the whole codebase.

All frontends (API server, CLI, TUI, Web UI) instantiate a manager and call
its methods.  The manager owns the singleton ``SloBPE`` or ``SloUnigram``
instance, handles training, caching, persistence, and integration.

Usage:
    from domain.training._internal.tokenizer_manager import get_tokenizer_manager

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
from typing import Any

logger = logging.getLogger("slo.training.tokenizer_manager")


_TOKENIZER_ALGO_KEY = "_algo"

# Special-token spellings shared by SloBPE/SloUnigram and the core TokenTree
# (both put <PAD> <UNK> <BOS> <EOS> at the front of the vocab, in that order),
# so a checkpoint view and the singleton agree on what counts as special.
_SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


class TokenizerManager:
    """Manages the global tokenizer singleton (BPE or Unigram LM).

    All tokenizer operations (train, encode, decode, save, load) go through
    this class.  Frontends never interact with SloBPE/SloUnigram directly
    — they call ``get_tokenizer_manager()`` and use its public API.
    """

    _instance: TokenizerManager | None = None

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._algo: str = "bpe"

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> TokenizerManager:
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
            from domain.training._internal.tokenizer import SloBPE

            self._tokenizer = SloBPE()
        return self._tokenizer

    def train(
        self,
        texts: list[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
        lowercase: bool = True,
        pretokenizer: str = "gpt2",
        algo: str = "bpe",
        **algo_kwargs: Any,
    ) -> dict[str, Any]:
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
            from domain.training._internal.tokenizer import SloUnigram

            self._tokenizer = SloUnigram(pretokenizer=pretokenizer)
            self._tokenizer.train(
                texts,
                vocab_size=vocab_size,
                lowercase=lowercase,
                **algo_kwargs,
            )
        else:
            from domain.training._internal.tokenizer import SloBPE

            self._tokenizer = SloBPE(pretokenizer=pretokenizer)
            self._tokenizer.train(
                texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                lowercase=lowercase,
            )
        self._algo = algo
        return self._tokenizer.vocab_stats()

    def analyze_corpus(self, texts: list[str]) -> dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, "analyze_corpus"):
            return tok.analyze_corpus(texts)
        return {"error": "analyze_corpus not available for this tokenizer type"}

    def show_pretokenization(self, text: str) -> dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, "show_pretokenization"):
            return tok.show_pretokenization(text)
        return {
            "error": "show_pretokenization not available for this tokenizer type",
            "pretokens": [],
            "segments": [],
            "count": 0,
        }

    def decompose_token(self, token: str) -> dict[str, Any]:
        tok = self.get_tokenizer()
        if hasattr(tok, "decompose_token"):
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
    ) -> dict[str, Any]:
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
        texts: list[str] = []
        it = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in it:
            if p.is_file():
                texts.append(p.read_text(encoding="utf-8", errors="replace"))

        return self.train(
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            lowercase=lowercase,
            pretokenizer=pretokenizer,
            algo=algo,
            **algo_kwargs,
        )

    @property
    def tokenizer_type(self) -> str:
        """``"bpe"`` or ``"unigram"`` — the algorithm of the current tokenizer."""
        return self._algo

    @property
    def pretokenizer(self) -> str:
        """Current pre-tokenizer type (``"gpt2"`` or ``"whitespace"``)."""
        tok = self.get_tokenizer()
        return tok._pretokenizer if hasattr(tok, "_pretokenizer") else "whitespace"

    # -- note: there was a dangling docstring here from an earlier edit --
    def tokenize(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.get_tokenizer().encode(text)

    def detokenize(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self.get_tokenizer().decode(ids)

    def stats(self) -> dict[str, Any]:
        """Return vocabulary statistics."""
        tok = self.get_tokenizer()
        if tok.vocab_size == 0:
            return {
                "vocab_size": 0,
                "base_chars": 0,
                "merged_subwords": 0,
                "special_tokens": 0,
                "total_merges": 0,
                "trained": False,
            }
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

    def from_dict(self, data: dict, origin: str | None = None) -> None:
        algo = data.get(_TOKENIZER_ALGO_KEY, "bpe")
        if algo == "unigram":
            from domain.training._internal.tokenizer import SloUnigram

            self._tokenizer = SloUnigram.from_dict(data, origin=origin)
        else:
            from domain.training._internal.tokenizer import SloBPE

            self._tokenizer = SloBPE.from_dict(data, origin=origin)
        self._algo = algo

    def save(self, path: str) -> None:
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data, origin=f"checkpoint:{path}")

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------

    def borrow_from_autotrain(self) -> bool:
        if self._tokenizer is not None and self._tokenizer.vocab_size > 0:
            return True
        try:
            from domain.training._internal.service import get_state

            at_state = get_state()
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
        from domain.training._internal.tokenizer import SloBPE

        self._tokenizer = SloBPE()
        self._algo = "bpe"

    # ------------------------------------------------------------------
    # Checkpoint-bound inspection (card 07c4e10a)
    # ------------------------------------------------------------------

    def tokenizer_for(self, checkpoint: str | None = None) -> tuple[Any, dict]:
        """The tokenizer to INSPECT — checkpoint-bound, else the singleton.

        Resolving by checkpoint IDENTITY is the whole point: the process-global
        singleton may have been trained on a toy corpus, so the inspector cannot
        answer "how does THIS model break down this text" through it. With no
        *checkpoint* the singleton is returned (source ``kind="singleton"``);
        with one, the tokenizer embedded in that ``.soul``'s sidecar is
        reconstructed — root-confined and traversal-safe via ``find_checkpoint``.

        Args:
            checkpoint: bare file name of a ``.soul`` (searched in ckpt_roots).

        Returns:
            ``(tokenizer, source)`` — *source* labels what was used
            (``kind``/``checkpoint``/``path``/``variant``/``origin``) so no
            caller can present singleton output as if it were a checkpoint's.

        Raises:
            ValueError: the named checkpoint is not found, or embeds no
                tokenizer. It must NOT fall back to the singleton — silence
                here would decorate toy output as a model's own.

        Side effects: none — the singleton is never mutated.
        """
        if checkpoint is None:
            return self.get_tokenizer(), {"kind": "singleton"}

        from domain.training._internal.checkpoints import find_checkpoint

        resolved = find_checkpoint(checkpoint)
        if resolved is None or not Path(resolved).exists():
            raise ValueError(f"checkpoint not found: {checkpoint}")
        tok = load_from_soul(resolved)
        return tok, {
            "kind": "checkpoint",
            "checkpoint": checkpoint,
            "path": str(resolved),
            "variant": getattr(tok, "variant", None),
            "origin": getattr(tok, "origin", None),
        }

    def stats_for(self, tokenizer: Any, source: dict | None = None) -> dict:
        """Vocabulary stats for ANY tokenizer (singleton or checkpoint view).

        Keeps the ``stats()`` key shape (``base_chars``/``merged_subwords``/
        ``special_tokens``/``total_merges``) so a checkpoint's stats render in
        the same slot as the singleton's, with ``kind`` naming the source.
        """
        kind = (source or {}).get("kind", "singleton")
        if hasattr(tokenizer, "vocab_stats"):  # SloBPE / SloUnigram
            stats = dict(tokenizer.vocab_stats())
            stats.setdefault("special_tokens", len(_SPECIAL_TOKENS))
            stats["trained"] = True
            stats["kind"] = kind
            return stats
        # NativeTokenizer (checkpoint view) — stats off the tree.
        tree = getattr(tokenizer, "tree", None)
        vocab = list(getattr(tree, "vocab", []) or [])
        merges = list(getattr(tree, "merges", []) or [])
        specials = sum(1 for t in vocab if t in _SPECIAL_TOKENS)
        return {
            "vocab_size": int(getattr(tokenizer, "vocab_size", len(vocab)) or len(vocab)),
            "base_chars": max(len(vocab) - len(merges) - specials, 0),
            "merged_subwords": len(merges),
            "special_tokens": specials,
            "total_merges": len(merges),
            "trained": True,
            "kind": kind,
        }

    def vocab_for(self, tokenizer: Any) -> list[str]:
        """Dense vocab list for either a SloBPE (flat ``.vocab``) or a
        checkpoint ``NativeTokenizer`` (``.tree.vocab``)."""
        tree = getattr(tokenizer, "tree", None)
        if tree is not None and hasattr(tree, "vocab"):
            return list(tree.vocab)
        return list(getattr(tokenizer, "vocab", []) or [])

    def merges_for(self, tokenizer: Any) -> list[Any]:
        """Ordered merges for either tokenizer shape (see :meth:`vocab_for`)."""
        tree = getattr(tokenizer, "tree", None)
        raw = getattr(tree, "merges", None)
        if raw is None:
            raw = getattr(tokenizer, "merges", []) or []
        return [tuple(m) if isinstance(m, (list, tuple)) else m for m in raw]

    def specials_for(self, tokenizer: Any) -> list[str]:
        """Special-token spellings, normalized across both tokenizer shapes."""
        sp = getattr(tokenizer, "SPECIAL_TOKENS", None)
        return list(sp) if sp is not None else list(_SPECIAL_TOKENS)


def load_from_soul(soul_path: Any) -> Any:
    """Reconstruct the tokenizer embedded in a ``.soul`` checkpoint.

    Reads ONLY the sidecar (``read_sidecar`` skips the large
    ``training_state`` blob, so it stays fast on 69MB checkpoints) and rebuilds
    via the ONE native-tokenizer loader, ``native_tokenizer_from_metadata`` —
    never a second decoder, which is how writer/reader drift starts.

    Args:
        soul_path: path to the ``.soul`` file (its sidecar is ``<path>.meta.json``).

    Returns:
        the reconstructed ``NativeTokenizer`` (char or tree variant).

    Raises:
        ValueError: the sidecar is absent/unreadable or embeds no tokenizer.
    """
    from domain.inference._internal.slo_format import read_sidecar
    from domain.inference._internal.slonet_provider import native_tokenizer_from_metadata

    p = Path(soul_path)
    metadata = read_sidecar(str(p)).get("metadata") or {}
    tok = native_tokenizer_from_metadata(metadata, origin=f"checkpoint:{p.name}")
    if tok is None:
        raise ValueError(f"checkpoint {p.name!r} embeds no tokenizer")
    return tok


def get_tokenizer_manager() -> TokenizerManager:
    """Shortcut to the global TokenizerManager singleton."""
    return TokenizerManager.get_instance()
