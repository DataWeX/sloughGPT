"""
Token Tree API Router — thin wrapper around TokenTreeManager.

All business logic lives in ``packages/core-py/domains/training/token_tree_manager.py``.
This router just exposes manager methods as HTTP endpoints:

- ``GET  /token-tree/stats``  — tree summary statistics.
- ``GET  /token-tree/vocab``  — paged vocabulary entries.
- ``GET  /token-tree/merges`` — most frequent BPE merge rules.
- ``GET  /token-tree/saved``  — list saved trees.
- ``POST /token-tree/save``   — save the current tree under a name.
- ``POST /token-tree/load``   — load a saved tree as the current tree.
- ``DELETE /token-tree/saved/{name}`` — delete a saved tree.
- ``POST /token-tree/train``  — train on a corpus (or the built-in default).
- ``POST /token-tree/similar`` — nearest-neighbor tokens via generated embeddings.
- ``POST /token-tree/embedding`` — inspect a token's generated embedding vector.
- ``POST /token-tree/encode`` — tree-walk encode text to ids.
- ``POST /token-tree/path`` — trace the encoder's greedy trie walk step by step.
- ``POST /token-tree/decode`` — decode ids to text.
- ``POST /token-tree/lineage`` — merge lineage down to character leaves.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from domains.training.token_tree_manager import get_token_tree_manager
from schemas.common import success_response


class TrainTreeRequest(BaseModel):
    texts: list[str] = Field(default=[], max_length=1000)
    vocab_size: int = Field(default=512, ge=32, le=100000)
    embed_dim: int = Field(default=16, ge=0, le=4096)
    min_frequency: int = Field(default=2, ge=1, le=10000)


class SimilarRequest(BaseModel):
    token: str = Field(max_length=256)
    top_k: int = Field(default=5, ge=1, le=100)


class TokenTextRequest(BaseModel):
    text: str = Field(max_length=50000)


class TokenIdsRequest(BaseModel):
    ids: list[int] = Field(max_length=10000)


class LineageRequest(BaseModel):
    token: str = Field(max_length=256)


class EmbeddingRequest(BaseModel):
    token: str = Field(max_length=256)
    top_k: int = Field(default=8, ge=1, le=64)


class TreeNameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class TokenTreeRouter:
    """Registers the /token-tree endpoints against TokenTreeManager."""

    def __init__(self) -> None:
        """Build the router and register all routes."""
        self.router = APIRouter(prefix="/token-tree", tags=["token-tree"])
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.add_api_route("/stats", self.get_stats, methods=["GET"])
        self.router.add_api_route("/vocab", self.get_vocab, methods=["GET"])
        self.router.add_api_route("/merges", self.get_merges, methods=["GET"])
        self.router.add_api_route("/saved", self.get_saved, methods=["GET"])
        self.router.add_api_route("/save", self.save_tree, methods=["POST"])
        self.router.add_api_route("/load", self.load_tree, methods=["POST"])
        self.router.add_api_route("/saved/{name}", self.delete_saved_tree, methods=["DELETE"])
        self.router.add_api_route("/train", self.train_tree, methods=["POST"])
        self.router.add_api_route("/similar", self.similar, methods=["POST"])
        self.router.add_api_route("/embedding", self.embedding, methods=["POST"])
        self.router.add_api_route("/encode", self.encode, methods=["POST"])
        self.router.add_api_route("/path", self.path, methods=["POST"])
        self.router.add_api_route("/decode", self.decode, methods=["POST"])
        self.router.add_api_route("/lineage", self.lineage, methods=["POST"])

    def get_stats(self):
        """Return summary statistics of the current token tree."""
        return success_response(data=get_token_tree_manager().stats())

    def get_vocab(
        self,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ):
        """Return a paged slice of the current tree's vocabulary.

        Args:
            limit: maximum number of entries to return (1..500).
            offset: number of leading entries to skip.

        Returns:
            StandardResponse with ``{total, entries}`` where each entry is
            ``{"id", "token", "freq", "is_special", "is_merged"}``.
        """
        return success_response(data=get_token_tree_manager().vocab_entries(
            offset=offset, limit=limit,
        ))

    def get_merges(
        self,
        top_n: int = Query(default=20, ge=1, le=200),
        query: str = Query(default="", max_length=128),
    ):
        """Return the most frequent BPE merge rules of the current tree.

        Args:
            top_n: maximum number of rules to return (1..200).
            query: optional case-insensitive substring filter over rule parts;
                when given, returns matching rules keeping their global rank.

        Returns:
            StandardResponse with a ranked list of merge rules.
        """
        mgr = get_token_tree_manager()
        if query:
            data = mgr.search_merges(query=query, limit=top_n)
        else:
            data = mgr.top_merges(top_n=top_n)
        return success_response(data=data)

    def get_saved(self):
        """List saved token trees.

        Returns:
            StandardResponse with ``{"trees": [...]}`` where each entry is
            ``{"name", "path", "vocab_size", "num_merges", "trained",
            "saved_at"}``.
        """
        return success_response(data={"trees": get_token_tree_manager().list_saved()})

    def save_tree(self, req: TreeNameRequest):
        """Save the current tree under a name in the save directory.

        Args:
            req: tree name.

        Returns:
            StandardResponse with the saved tree's metadata.

        Raises:
            HTTPException 422: when the name is invalid.
        """
        try:
            return success_response(data=get_token_tree_manager().save(req.name))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    def load_tree(self, req: TreeNameRequest):
        """Load a saved tree and make it the current tree.

        Args:
            req: tree name.

        Returns:
            StandardResponse with the loaded tree's metadata.

        Raises:
            HTTPException 404: when no saved tree has that name.
            HTTPException 422: when the name is invalid.
        """
        try:
            return success_response(data=get_token_tree_manager().load(req.name))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    def delete_saved_tree(self, name: str):
        """Delete a saved tree's sidecar files.

        Args:
            name: tree name.

        Returns:
            StandardResponse with ``{"name", "deleted": True}``.

        Raises:
            HTTPException 404: when no saved tree has that name.
            HTTPException 422: when the name is invalid.
        """
        try:
            deleted = get_token_tree_manager().delete_saved(name)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No saved token tree named {name!r}")
        return success_response(data={"name": name, "deleted": True})

    def train_tree(self, req: TrainTreeRequest):
        """Train a token tree on the provided corpus (built-in default when empty).

        Args:
            req: corpus texts plus training hyperparameters.

        Returns:
            StandardResponse with training stats.
        """
        mgr = get_token_tree_manager()
        texts = req.texts if req.texts else None
        if texts:
            tree = mgr.train(
                texts,
                vocab_size=req.vocab_size,
                min_frequency=req.min_frequency,
                embed_dim=req.embed_dim,
            )
        else:
            tree = mgr.get_tree(
                vocab_size=req.vocab_size, embed_dim=req.embed_dim
            )
        stats = tree.stats()
        return success_response(data={
            "status": "trained",
            "vocab_size": stats["vocab_size"],
            "embedding_points": stats["embedding_points"],
            "embedding_compression_ratio": stats["embedding_compression_ratio"],
            "embed_dim": stats["embed_dim"],
        })

    def similar(self, req: SimilarRequest):
        """Return ranked nearest-neighbor tokens for a query token.

        Args:
            req: token (id or literal) and top_k.

        Returns:
            StandardResponse with query and ranked neighbors.

        Raises:
            HTTPException 404: when the token is not in the vocabulary.
        """
        try:
            data = get_token_tree_manager().similar(req.token, top_k=req.top_k)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"Token not in vocabulary: {e}")
        return success_response(data=data)

    def embedding(self, req: EmbeddingRequest):
        """Inspect a token's generated embedding vector.

        Args:
            req: token (id or literal) and top_k largest-magnitude dims.

        Returns:
            StandardResponse with ``{token, id, dim, norm, top, ...}``.

        Raises:
            HTTPException 404: when the token is not in the vocabulary.
            HTTPException 422: when embeddings are disabled.
        """
        try:
            data = get_token_tree_manager().embedding_info(
                req.token, top_k=req.top_k
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"Token not in vocabulary: {e}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return success_response(data=data)

    def encode(self, req: TokenTextRequest):
        """Encode text into token ids by walking the tree.

        Args:
            req: input text.

        Returns:
            StandardResponse with ``{tokens, ids}``.
        """
        return success_response(data=get_token_tree_manager().encode(req.text))

    def path(self, req: TokenTextRequest):
        """Trace the encoder's greedy trie walk over text.

        Args:
            req: input text.

        Returns:
            StandardResponse with ``{steps, ids}``. ``ids`` equals
            ``encode``'s ids; ``steps`` shows each query step.
        """
        return success_response(data=get_token_tree_manager().path(req.text))

    def decode(self, req: TokenIdsRequest):
        """Decode a list of token ids back to text.

        Args:
            req: token ids.

        Returns:
            StandardResponse with ``{text}``.
        """
        return success_response(data=get_token_tree_manager().decode(req.ids))

    def lineage(self, req: LineageRequest):
        """Render a token's merge lineage down to character leaves.

        Args:
            req: token (id or literal).

        Returns:
            StandardResponse with ``{token, leaves, tree}``.

        Raises:
            HTTPException 404: when the token is not in the vocabulary.
        """
        try:
            data = get_token_tree_manager().lineage(req.token)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"Token not in vocabulary: {e}")
        return success_response(data=data)


router = TokenTreeRouter().router
