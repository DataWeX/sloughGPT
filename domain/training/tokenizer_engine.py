"""TokenizerEngine — unified feature facade for tokenizer + token-tree.

Wraps TokenizerManager and TokenTreeManager so routers never import
``domain.training._internal.*`` directly (Build Order #4).

Usage:
    from domain.training.tokenizer_engine import get_tokenizer_engine

    engine = get_tokenizer_engine()
    stats = engine.stats()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slo.training.tokenizer_engine")


class TokenizerEngine:
    """Unified tokenizer + token-tree feature API."""

    def __init__(self) -> None:
        self._mgr: Any = None
        self._tree_mgr: Any = None

    @property
    def manager(self) -> Any:
        """Lazily resolve the TokenizerManager singleton."""
        if self._mgr is None:
            from domain.training._internal.tokenizer_manager import get_tokenizer_manager

            self._mgr = get_tokenizer_manager()
        return self._mgr

    @property
    def tree_manager(self) -> Any:
        """Lazily resolve the TokenTreeManager singleton."""
        if self._tree_mgr is None:
            from domain.training._internal.token_tree_manager import get_token_tree_manager

            self._tree_mgr = get_token_tree_manager()
        return self._tree_mgr

    # ── Tokenizer surface (delegates to TokenizerManager) ──

    def is_trained(self) -> bool:
        return self.manager.is_trained()

    def ensure_trained(self) -> None:
        """Borrow autotrain or train a tiny default if untrained."""
        mgr = self.manager
        if not mgr.is_trained():
            mgr.borrow_from_autotrain()
        if not mgr.is_trained():
            mgr.train(
                [
                    "the quick brown fox jumps over the lazy dog",
                    "hello world",
                    "machine learning",
                ],
                vocab_size=256,
                min_frequency=1,
                lowercase=True,
            )

    def stats(self) -> dict[str, Any]:
        return self.manager.stats()

    def pretokenize(self, text: str) -> dict[str, Any]:
        return self.manager.show_pretokenization(text)

    def decompose(self, text: str) -> dict[str, Any]:
        return self.manager.decompose_token(text)

    def analyze(self, texts: list[str]) -> dict[str, Any]:
        return self.manager.analyze_corpus(texts)

    def tokenize(self, text: str) -> list[int]:
        return self.manager.tokenize(text)

    def detokenize(self, ids: list[int]) -> str:
        return self.manager.detokenize(ids)

    def get_tokenizer(self) -> Any:
        return self.manager.get_tokenizer()

    def train(self, texts: list[str], **kwargs: Any) -> Any:
        return self.manager.train(texts, **kwargs)

    # ── Token-tree surface (delegates to TokenTreeManager) ──

    def tree_stats(self) -> dict:
        return self.tree_manager.stats()

    def tree_vocab(self, offset: int = 0, limit: int = 50) -> dict:
        return self.tree_manager.vocab_entries(offset=offset, limit=limit)

    def tree_merges(self, top_n: int = 20, query: str = "") -> list:
        if query:
            return self.tree_manager.search_merges(query=query, limit=top_n)
        return self.tree_manager.top_merges(top_n=top_n)

    def list_saved_trees(self) -> list:
        return self.tree_manager.list_saved()

    def save_tree(self, name: str) -> dict:
        return self.tree_manager.save(name)

    def load_tree(self, name: str) -> dict:
        return self.tree_manager.load(name)

    def delete_tree(self, name: str) -> bool:
        return self.tree_manager.delete_saved(name)

    def train_tree(
        self,
        texts: list[str] | None = None,
        vocab_size: int = 512,
        embed_dim: int = 16,
        min_frequency: int = 2,
    ) -> Any:
        mgr = self.tree_manager
        if texts:
            return mgr.train(
                texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                embed_dim=embed_dim,
            )
        return mgr.get_tree(vocab_size=vocab_size, embed_dim=embed_dim)

    def tree_similar(self, token: str, top_k: int = 5) -> dict:
        return self.tree_manager.similar(token, top_k=top_k)

    def tree_embedding(self, token: str, top_k: int = 8) -> dict:
        return self.tree_manager.embedding_info(token, top_k=top_k)

    def tree_encode(self, text: str) -> dict:
        return self.tree_manager.encode(text)

    def tree_path(self, text: str) -> dict:
        return self.tree_manager.path(text)

    def tree_decode(self, ids: list[int]) -> dict:
        return self.tree_manager.decode(ids)

    def tree_lineage(self, token: str) -> dict:
        return self.tree_manager.lineage(token)

    def tree_matrix(self, top_k: int = 8) -> dict:
        return self.tree_manager.matrix_summary(top_k=top_k)

    def tree_compare(self, a: str, b: str, top_n: int = 10) -> dict:
        return self.tree_manager.compare(a, b, top_n=top_n)


_engine: TokenizerEngine | None = None


def get_tokenizer_engine() -> TokenizerEngine:
    """Singleton accessor for TokenizerEngine."""
    global _engine
    if _engine is None:
        _engine = TokenizerEngine()
    return _engine


def get_tokenizer_manager() -> Any:
    """Public accessor for the TokenizerManager via TokenizerEngine.

    Routers import this (not ``domain.training._internal.tokenizer_manager``)
    so tests can patch ``routers.<name>.get_tokenizer_manager``.
    """
    return get_tokenizer_engine().manager


def get_token_tree_manager() -> Any:
    """Public accessor for the TokenTreeManager via TokenizerEngine."""
    return get_tokenizer_engine().tree_manager
