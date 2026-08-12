"""
TokenTreeManager — global in-process TokenTree holder.

Mirrors ``TokenizerManager``: the API router and runtime consumers share a
single lazily-trained :class:`TokenTree`. Default training uses a small
built-in corpus so semantic queries work immediately without a network
download; explicit training replaces the tree with one learned from the
caller's texts.

Key classes:
    - TokenTreeManager: thread-safe singleton owning one TokenTree.

Functions:
    - get_token_tree_manager(): returns the shared singleton.
"""

import json
import re
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from domains.training.token_tree import TokenTree

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SAVE_DIR = _REPO_ROOT / "data" / "token_trees"

DEFAULT_CORPUS: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox is quick",
    "the lazy dog sleeps in the sun",
    "quick brown foxes jump over sleeping dogs",
    "the sun rises over the quiet garden",
    "a gentle breeze moves the tall grass",
    "machines learn from data and patterns",
    "neural networks process language and images",
    "the model generates text from a prompt",
    "compression stores information in fewer bytes",
]


class TokenTreeManager:
    """Owns a single lazily-trained :class:`TokenTree`.

    Thread-safe: tree (re)training and reads are serialized through a lock.
    """

    _instance: Optional["TokenTreeManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize an empty manager (tree trained on first access)."""
        self._tree: Optional[TokenTree] = None
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TokenTreeManager":
        """Return the process-wide singleton (created on first call)."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_trained(
        self, vocab_size: int = 512, embed_dim: int = 16
    ) -> TokenTree:
        """Train the default corpus once, then return the live tree.

        Args:
            vocab_size: target vocabulary size for default training.
            embed_dim: embedding dimension for default training.

        Returns:
            the current :class:`TokenTree`.

        Side effects:
            - trains the default corpus on the first call.
        """
        with self._lock:
            if self._tree is None:
                self._tree = TokenTree().train(
                    list(DEFAULT_CORPUS),
                    vocab_size=vocab_size,
                    min_frequency=1,
                    embed_dim=embed_dim,
                )
            return self._tree

    def get_tree(self, vocab_size: int = 512, embed_dim: int = 16) -> TokenTree:
        """Return the live tree, lazily training the default corpus first.

        Args:
            vocab_size: vocabulary size used if default training runs.
            embed_dim: embedding dimension used if default training runs.

        Returns:
            the current :class:`TokenTree`.
        """
        return self._ensure_trained(vocab_size=vocab_size, embed_dim=embed_dim)

    def is_trained(self) -> bool:
        """Return True when a tree has been trained this session."""
        with self._lock:
            return self._tree is not None

    def train(
        self,
        texts: Sequence[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
        embed_dim: int = 16,
    ) -> TokenTree:
        """Train a fresh tree on the caller's texts and keep it as current.

        Args:
            texts: corpus documents (a single str is one document).
            vocab_size: target vocabulary size.
            min_frequency: minimum pair frequency for a merge.
            embed_dim: embedding dimension.

        Returns:
            the newly trained :class:`TokenTree`.

        Side effects:
            - replaces the manager's current tree.
        """
        tree = TokenTree().train(
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            embed_dim=embed_dim,
        )
        with self._lock:
            self._tree = tree
        return tree

    def stats(self) -> dict:
        """Return summary stats of the current tree."""
        return self.get_tree().stats()

    def top_merges(self, top_n: int = 20) -> list:
        """Return the most frequent BPE merge rules as ranked dicts.

        Args:
            top_n: maximum number of rules to return.

        Returns:
            list of ``{"rank", "left", "right", "token", "count"}``.
        """
        return self.get_tree().top_merges(top_n=top_n)

    def search_merges(self, query: str, limit: int = 20) -> list:
        """Return merge rules whose parts match a query, keeping global ranks.

        Args:
            query: case-insensitive substring matched against rule parts.
            limit: maximum number of matching rules to return.

        Returns:
            list of ``{"rank", "left", "right", "token", "count"}``.
        """
        return self.get_tree().search_merges(query=query, limit=limit)

    def vocab_entries(self, offset: int = 0, limit: int = 50) -> dict:
        """Return a paged slice of the tree vocabulary.

        Args:
            offset: number of leading entries to skip.
            limit: maximum number of entries to return (0 or negative means
                "no limit").

        Returns:
            ``{"total": int, "entries": [{"id", "token", "freq",
            "is_special", "is_merged"}]}``.
        """
        return self.get_tree().vocab_entries(offset=offset, limit=limit)

    def similar(self, token: str, top_k: int = 5) -> dict:
        """Rank nearest-neighbor tokens by generated-embedding cosine.

        Args:
            token: token id or literal string (see ``resolve_token``).
            top_k: number of neighbors to return.

        Returns:
            ``{"query": str, "neighbors": [{"id", "token", "score"}]}``.

        Raises:
            KeyError: when the token is not in the vocabulary.
        """
        tree = self.get_tree()
        token_id = tree.resolve_token(token)
        query = tree.itos.get(token_id, str(token_id))
        neighbors = [
            {
                "id": tid,
                "token": tree.itos.get(tid, "?"),
                "score": round(score, 6),
            }
            for tid, score in tree.similar(token_id, top_k=top_k)
        ]
        return {"query": query, "neighbors": neighbors}

    def embedding_info(self, token: str, top_k: int = 8) -> dict:
        """Inspect a token's generated embedding vector.

        Args:
            token: token id or literal string (see ``resolve_token``).
            top_k: number of largest-magnitude dimensions to report.

        Returns:
            ``{"token", "id", "dim", "norm", "top": [[dim, value], ...],
            "embedding_points", "compression_ratio"}``.

        Raises:
            KeyError: when the token is not in the vocabulary.
            ValueError: when embeddings are disabled (embed_dim = 0).
        """
        tree = self.get_tree()
        token_id = tree.resolve_token(token)
        vec = tree.embedding(token_id)
        if vec is None:
            raise ValueError("Token embeddings are not enabled (embed_dim = 0)")
        top_idx = np.argsort(-np.abs(vec))[: max(top_k, 1)]
        return {
            "token": tree.itos.get(token_id, str(token_id)),
            "id": token_id,
            "dim": int(vec.shape[0]),
            "norm": round(float(np.linalg.norm(vec)), 6),
            "top": [[int(i), round(float(vec[i]), 6)] for i in top_idx],
            "embedding_points": tree.embedding_points(),
            "compression_ratio": round(tree.embedding_compression_ratio(), 2),
        }

    def encode(self, text: str) -> dict:
        """Tree-walk encode text into tokens and ids.

        Args:
            text: input string.

        Returns:
            ``{"tokens": [str], "ids": [int]}``.
        """
        tree = self.get_tree()
        ids = tree.encode(text)
        return {"tokens": [tree.itos.get(i, "?") for i in ids], "ids": ids}

    def path(self, text: str) -> dict:
        """Trace the greedy longest-prefix walk the encoder takes over text.

        Replays ``tree.encode`` internals step by step: text is normalized and
        pretokenized into words, each word is padded with the word suffix, and
        ``tree.query`` consumes the longest matching token from the remaining
        suffix until the word is exhausted.

        Args:
            text: input string.

        Returns:
            ``{"steps": [{remaining, token, id, consumed}], "ids": [int]}``
            where ``remaining`` is the unconsumed suffix before the step,
            ``token``/``id`` are the matched token, and ``consumed`` is the
            number of characters the query advanced. ``ids`` matches
            ``encode(text)["ids"]``.
        """
        tree = self.get_tree()
        normalized = tree._normalize(text, lowercase=True)
        steps = []
        for word in tree._pretokenize(normalized, lowercase=True):
            s = word + tree._word_suffix
            i = 0
            while i < len(s):
                remaining = s[i:]
                token_id, advance = tree.query(remaining)
                steps.append(
                    {
                        "remaining": remaining,
                        "token": tree.itos.get(token_id, "?"),
                        "id": token_id,
                        "consumed": advance,
                    }
                )
                i += advance
        return {"steps": steps, "ids": [step["id"] for step in steps]}

    def decode(self, ids: Sequence[int]) -> dict:
        """Decode ids back to text.

        Args:
            ids: token id sequence.

        Returns:
            ``{"text": str}``.
        """
        tree = self.get_tree()
        return {"text": tree.decode(ids)}

    def lineage(self, token: str) -> dict:
        """Render a token's merge lineage down to character leaves.

        Args:
            token: token id or literal string.

        Returns:
            ``{"token": str, "leaves": [str], "tree": str}``.

        Raises:
            KeyError: when the token is not in the vocabulary.
        """
        tree = self.get_tree()
        token_id = tree.resolve_token(token)
        return {
            "token": tree.itos.get(token_id, str(token_id)),
            "leaves": tree.decompose(token_id),
            "tree": tree.show_tree(token_id),
        }

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Validate a saved-tree name against path traversal.

        Args:
            name: user-supplied tree name.

        Returns:
            the trimmed name.

        Raises:
            ValueError: when the name is empty or contains path separators or
                any character outside ``[A-Za-z0-9._-]``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Tree name must be a non-empty string")
        name = name.strip()
        if not name:
            raise ValueError("Tree name must not be blank")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError(
                "Tree name may contain only letters, digits, dots, dashes and "
                "underscores and must not start with a dot"
            )
        return name

    @staticmethod
    def _tree_info(name: str, base: Path, meta: dict) -> dict:
        """Build the metadata dict for a saved tree.

        Args:
            name: tree name.
            base: base path (without extension).
            meta: parsed ``.meta.json`` sidecar.

        Returns:
            ``{"name", "path", "vocab_size", "num_merges", "trained",
            "saved_at"}``.
        """
        return {
            "name": name,
            "path": str(base),
            "vocab_size": len(meta.get("vocab", [])),
            "num_merges": len(meta.get("merges", [])),
            "trained": bool(meta.get("trained", False)),
            "saved_at": meta.get("saved_at"),
        }

    def save(self, name: str) -> dict:
        """Persist the current tree under a named path in the save dir.

        Args:
            name: tree name (validated against path traversal).

        Returns:
            metadata dict describing the saved tree.

        Side effects:
            - writes ``data/token_trees/<name>.meta.json`` and
              ``data/token_trees/<name>.points.json``.
        """
        name = self._sanitize_name(name)
        tree = self.get_tree()
        base = _SAVE_DIR / name
        meta_path, _ = tree.save(str(base))
        meta = json.loads(meta_path.read_text())
        return self._tree_info(name, base, meta)

    def load(self, name: str) -> dict:
        """Load a saved tree and make it the manager's current tree.

        Args:
            name: tree name previously passed to :meth:`save`.

        Returns:
            metadata dict describing the loaded tree.

        Side effects:
            - replaces the manager's current tree with the loaded one.

        Raises:
            FileNotFoundError: when no saved tree with that name exists.
            ValueError: when the name is invalid.
        """
        name = self._sanitize_name(name)
        base = _SAVE_DIR / name
        meta_path = Path(str(base) + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"No saved token tree named {name!r}")
        tree = TokenTree.load(str(base))
        with self._lock:
            self._tree = tree
        return self._tree_info(name, base, json.loads(meta_path.read_text()))

    def list_saved(self) -> list:
        """List all saved trees, newest first.

        Returns:
            list of metadata dicts; corrupted sidecars are skipped.
        """
        entries = []
        if _SAVE_DIR.exists():
            for meta_path in sorted(_SAVE_DIR.glob("*.meta.json")):
                name = meta_path.name[: -len(".meta.json")]
                try:
                    meta = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                entries.append(self._tree_info(name, _SAVE_DIR / name, meta))
        entries.sort(key=lambda e: (e["saved_at"] or 0), reverse=True)
        return entries

    def delete_saved(self, name: str) -> bool:
        """Delete a saved tree's files from the save dir.

        Args:
            name: tree name previously passed to :meth:`save`.

        Returns:
            True when at least one sidecar file was removed.

        Side effects:
            - removes ``<name>.meta.json`` and ``<name>.points.json``.

        Raises:
            ValueError: when the name is invalid.
        """
        name = self._sanitize_name(name)
        base = _SAVE_DIR / name
        removed = False
        for suffix in (".meta.json", ".points.json"):
            p = Path(str(base) + suffix)
            if p.exists():
                p.unlink()
                removed = True
        return removed


def get_token_tree_manager() -> TokenTreeManager:
    """Shortcut to the global TokenTreeManager singleton.

    Returns:
        the shared :class:`TokenTreeManager`.
    """
    return TokenTreeManager.get_instance()
