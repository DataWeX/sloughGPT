"""
Tokenizer API Router — thin wrapper around TokenizerManager.

All business logic lives in ``packages/core-py/domains/training/tokenizer_manager.py``.
This router just exposes manager methods as HTTP endpoints.
"""

import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from domains.training.tokenizer_manager import get_tokenizer_manager


class TrainTokenizerRequest(BaseModel):
    texts: list[str]
    vocab_size: int = 512
    min_frequency: int = 2
    lowercase: bool = True
    pretokenizer: str = "gpt2"


class TokenizeRequest(BaseModel):
    text: str


class DetokenizeRequest(BaseModel):
    ids: list[int]


class AnalyzeRequest(BaseModel):
    texts: list[str]


router = APIRouter(prefix="/tokenizer", tags=["tokenizer"])


def _require_trained():
    mgr = get_tokenizer_manager()
    if not mgr.is_trained():
        mgr.borrow_from_autotrain()
    if not mgr.is_trained():
        mgr.train(
            ["the quick brown fox jumps over the lazy dog", "hello world", "machine learning"],
            vocab_size=256, min_frequency=1, lowercase=True,
        )


@router.post("/train")
async def train_tokenizer(req: TrainTokenizerRequest):
    mgr = get_tokenizer_manager()
    stats = mgr.train(req.texts, vocab_size=req.vocab_size, min_frequency=req.min_frequency, lowercase=req.lowercase)
    return {
        "vocab_size": stats["vocab_size"],
        "base_chars": stats.get("base_chars", stats.get("subwords", 0)),
        "merged_subwords": stats.get("merged_subwords", stats.get("subwords", 0)),
        "special_tokens": stats["special_tokens"],
        "total_merges": stats.get("total_merges_learned", stats.get("total_merges", 0)),
    }


@router.get("/stats")
async def get_tokenizer_stats():
    _require_trained()
    mgr = get_tokenizer_manager()
    stats = mgr.stats()
    return {
        "vocab_size": stats["vocab_size"],
        "base_chars": stats.get("base_chars", 0),
        "merged_subwords": stats.get("merged_subwords", stats.get("subwords", 0)),
        "special_tokens": stats["special_tokens"],
        "total_merges": stats.get("total_merges", stats.get("total_merges_learned", 0)),
        "trained": stats.get("trained", True),
    }


class PretokenizeRequest(BaseModel):
    text: str


class DecomposeRequest(BaseModel):
    text: str


@router.post("/pretokenize")
async def pretokenize_text(req: PretokenizeRequest):
    """Show how text splits into pretokens before BPE encoding."""
    _require_trained()
    mgr = get_tokenizer_manager()
    return mgr.show_pretokenization(req.text)


@router.post("/decompose")
async def decompose_token(req: DecomposeRequest):
    """Show a token's merge tree decomposition."""
    _require_trained()
    mgr = get_tokenizer_manager()
    try:
        return mgr.decompose_token(req.text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/analyze")
async def analyze_corpus(req: AnalyzeRequest):
    """Compute token frequency and compression stats on a corpus."""
    _require_trained()
    mgr = get_tokenizer_manager()
    return mgr.analyze_corpus(req.texts)


@router.post("/tokenize")
async def tokenize_text(req: TokenizeRequest):
    _require_trained()
    mgr = get_tokenizer_manager()
    ids = mgr.tokenize(req.text)
    tok = mgr.get_tokenizer()
    tokens = [tok.itos.get(i, "<?>") for i in ids]
    return {"tokens": tokens, "ids": ids}


@router.post("/detokenize")
async def detokenize_ids(req: DetokenizeRequest):
    _require_trained()
    mgr = get_tokenizer_manager()
    text = mgr.detokenize(req.ids)
    return {"text": text}


@router.get("/vocab")
async def get_vocab(limit: int = 50, offset: int = 0):
    _require_trained()
    tok = get_tokenizer_manager().get_tokenizer()
    total = tok.vocab_size
    entries = []
    for i in range(offset, min(offset + limit, total)):
        token = tok.vocab[i]
        entries.append({"id": i, "token": token, "is_special": token in tok.SPECIAL_TOKENS})
    return {"entries": entries, "total": total, "offset": offset, "limit": limit}


@router.get("/merges")
async def get_merges(limit: int = 30):
    _require_trained()
    tok = get_tokenizer_manager().get_tokenizer()
    merges = getattr(tok, "merges", [])
    result = []
    for i, m in enumerate(merges[:limit]):
        if isinstance(m, tuple) and len(m) == 2:
            result.append({"index": i, "left": m[0], "right": m[1], "token": m[0] + m[1]})
        else:
            result.append({"index": i, "left": str(m), "right": "", "token": str(m)})
    return {"merges": result, "total": len(merges)}


class TrainShakespeareRequest(BaseModel):
    vocab_size: int = 512

@router.post("/train-shakespeare")
async def train_on_shakespeare(req: TrainShakespeareRequest):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    mgr = get_tokenizer_manager()
    mgr.train(lines[:2000], vocab_size=req.vocab_size, min_frequency=3)
    return {"status": "trained", "corpus_size": len(lines[:2000]), "stats": mgr.stats()}


@router.get("/sample")
async def get_tokenization_sample():
    _require_trained()
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
    return {"samples": results}
