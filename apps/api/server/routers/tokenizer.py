"""
Tokenizer API Router — thin wrapper around TokenizerManager.

All business logic lives in ``packages/core-py/domains/training/tokenizer_manager.py``.
This router just exposes manager methods as HTTP endpoints.
"""

import urllib.request

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from domains.training.tokenizer_manager import get_tokenizer_manager
from schemas.common import success_response, classify_and_raise, safe_audit_log


class TokenizeRequest(BaseModel):
    text: str = Field(max_length=50000)


class DetokenizeRequest(BaseModel):
    ids: list[int] = Field(max_length=10000)


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(max_length=500)


class PretokenizeRequest(BaseModel):
    text: str = Field(max_length=50000)


class DecomposeRequest(BaseModel):
    text: str = Field(max_length=50000)


class TrainTokenizerRequest2(BaseModel):
    vocab_size: int = Field(default=512, ge=32, le=100000)
    texts: list[str] = Field(default=[], max_length=1000)


class TokenizerRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/tokenizer", tags=["tokenizer"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/stats", self.get_tokenizer_stats, methods=["GET"])
        self.router.add_api_route("/pretokenize", self.pretokenize_text, methods=["POST"])
        self.router.add_api_route("/decompose", self.decompose_token, methods=["POST"])
        self.router.add_api_route("/analyze", self.analyze_corpus, methods=["POST"])
        self.router.add_api_route("/tokenize", self.tokenize_text, methods=["POST"])
        self.router.add_api_route("/detokenize", self.detokenize_ids, methods=["POST"])
        self.router.add_api_route("/vocab", self.get_vocab, methods=["GET"])
        self.router.add_api_route("/merges", self.get_merges, methods=["GET"])
        self.router.add_api_route("/train", self.train_tokenizer, methods=["POST"])
        self.router.add_api_route("/sample", self.get_tokenization_sample, methods=["GET"])

    def _require_trained(self):
        mgr = get_tokenizer_manager()
        if not mgr.is_trained():
            mgr.borrow_from_autotrain()
        if not mgr.is_trained():
            mgr.train(
                ["the quick brown fox jumps over the lazy dog", "hello world", "machine learning"],
                vocab_size=256, min_frequency=1, lowercase=True,
            )

    async def get_tokenizer_stats(self) -> dict:
        """get_tokenizer_stats."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        stats = mgr.stats()
        return success_response(data={
            "vocab_size": stats["vocab_size"],
            "base_chars": stats.get("base_chars", 0),
            "merged_subwords": stats.get("merged_subwords", stats.get("subwords", 0)),
            "special_tokens": stats["special_tokens"],
            "total_merges": stats.get("total_merges", stats.get("total_merges_learned", 0)),
            "trained": stats.get("trained", True),
        })

    async def pretokenize_text(self, req: PretokenizeRequest) -> dict:
        """Show how text splits into pretokens before BPE encoding."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        return success_response(data=mgr.show_pretokenization(req.text))

    async def decompose_token(self, req: DecomposeRequest) -> dict:
        """Show a token's merge tree decomposition."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        try:
            return success_response(data=mgr.decompose_token(req.text))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def analyze_corpus(self, req: AnalyzeRequest) -> dict:
        """Compute token frequency and compression stats on a corpus."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        return success_response(data=mgr.analyze_corpus(req.texts))

    async def tokenize_text(self, req: TokenizeRequest) -> dict:
        """tokenize_text."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        ids = mgr.tokenize(req.text)
        tok = mgr.get_tokenizer()
        tokens = [tok.itos.get(i, "<?>") for i in ids]
        return success_response(data={"tokens": tokens, "ids": ids})

    async def detokenize_ids(self, req: DetokenizeRequest) -> dict:
        """detokenize_ids."""
        self._require_trained()
        mgr = get_tokenizer_manager()
        text = mgr.detokenize(req.ids)
        return success_response(data={"text": text})

    async def get_vocab(self, limit: int = 50, offset: int = 0) -> dict:
        """get_vocab."""
        self._require_trained()
        tok = get_tokenizer_manager().get_tokenizer()
        total = tok.vocab_size
        entries = []
        for i in range(offset, min(offset + limit, total)):
            token = tok.vocab[i]
            entries.append({"id": i, "token": token, "is_special": token in tok.SPECIAL_TOKENS})
        return success_response(data={"entries": entries, "total": total, "offset": offset, "limit": limit})

    async def get_merges(self, limit: int = 30) -> dict:
        """get_merges."""
        self._require_trained()
        tok = get_tokenizer_manager().get_tokenizer()
        merges = getattr(tok, "merges", [])
        result = []
        for i, m in enumerate(merges[:limit]):
            if isinstance(m, tuple) and len(m) == 2:
                result.append({"index": i, "left": m[0], "right": m[1], "token": m[0] + m[1]})
            else:
                result.append({"index": i, "left": str(m), "right": "", "token": str(m)})
        return success_response(data={"merges": result, "total": len(merges)})

    async def train_tokenizer(self, req: TrainTokenizerRequest2) -> dict:
        """
        Train the BPE tokenizer on provided text corpus or download default Shakespeare data.
        """
        if req.texts:
            lines = req.texts
        else:
            import asyncio
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            text = await asyncio.to_thread(urllib.request.urlopen, url)
            text = text.read().decode("utf-8")
            lines = [line.strip() for line in text.split("\n") if line.strip()][:2000]
        mgr = get_tokenizer_manager()
        mgr.train(lines, vocab_size=req.vocab_size, min_frequency=3)
        safe_audit_log("tokenizer.train", resource="bpe", vocab_size=req.vocab_size, corpus_size=len(lines))
        return success_response(data={"status": "trained", "corpus_size": len(lines), "stats": mgr.stats()})

    async def get_tokenization_sample(self) -> dict:
        """get_tokenization_sample."""
        self._require_trained()
        tok = get_tokenizer_manager().get_tokenizer()
        sample_words = [
            "the", "and", "to", "of", "a", "in", "that", "is",
            "was", "he", "for", "it", "with", "as", "his", "on",
            "hello", "world", "machine", "learning", "neural", "network",
        ]
        results = []
        for word in sample_words:
            ids = tok.encode(word)
            tokens = [tok.itos.get(i, "<?>") for i in ids]
            results.append({"word": word, "ids": ids, "tokens": tokens, "count": len(ids)})
        return success_response(data={"samples": results})


router = TokenizerRouter().router
