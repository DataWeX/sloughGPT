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
- ``GET  /token-tree/matrix`` — embedding-matrix overview summary.
- ``POST /token-tree/compare`` — diff two saved trees (overlap + examples).
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from domains.training.token_tree_manager import get_token_tree_manager
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, success_response, safe_audit_log, classify_and_raise


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


class CompareTreesRequest(BaseModel):
    a: str = Field(min_length=1, max_length=64)
    b: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=10, ge=1, le=100)


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
        self.router.add_api_route("/matrix", self.matrix, methods=["GET"])
        self.router.add_api_route("/compare", self.compare, methods=["POST"])

    def get_stats(self) -> dict:
        """Return summary statistics of the current token tree."""
        try:
            return success_response(data=get_token_tree_manager().stats())
        except Exception as e:
            classify_and_raise(e, source="token_tree.stats")

    def get_vocab(
        self,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        """Return a paged slice of the current tree's vocabulary."""
        try:
            return success_response(data=get_token_tree_manager().vocab_entries(
                offset=offset, limit=limit,
            ))
        except Exception as e:
            classify_and_raise(e, source="token_tree.vocab")

    def get_merges(
        self,
        top_n: int = Query(default=20, ge=1, le=200),
        query: str = Query(default="", max_length=128),
    ) -> dict:
        """Return the most frequent BPE merge rules of the current tree."""
        try:
            mgr = get_token_tree_manager()
            if query:
                data = mgr.search_merges(query=query, limit=top_n)
            else:
                data = mgr.top_merges(top_n=top_n)
            return success_response(data=data)
        except Exception as e:
            classify_and_raise(e, source="token_tree.merges")

    def get_saved(self) -> dict:
        """List saved token trees."""
        try:
            return success_response(data={"trees": get_token_tree_manager().list_saved()})
        except Exception as e:
            classify_and_raise(e, source="token_tree.saved")

    def save_tree(self, req: TreeNameRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Save the current tree under a name in the save directory."""
        try:
            return success_response(data=get_token_tree_manager().save(req.name))
        except ValueError as e:
            raise_error(str(e), "E_VAL_REQUEST", status_code=422)
        except Exception as e:
            classify_and_raise(e, source="token_tree.save")

    def load_tree(self, req: TreeNameRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Load a saved tree and make it the current tree."""
        try:
            return success_response(data=get_token_tree_manager().load(req.name))
        except FileNotFoundError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        except ValueError as e:
            raise_error(str(e), "E_VAL_REQUEST", status_code=422)
        except Exception as e:
            classify_and_raise(e, source="token_tree.load")

    def delete_saved_tree(self, name: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a saved tree's sidecar files."""
        try:
            deleted = get_token_tree_manager().delete_saved(name)
            if not deleted:
                raise_error(f"No saved token tree named {name!r}", "E_NOT_FOUND", status_code=404)
            safe_audit_log("token_tree.delete", resource=name)
            return success_response(data={"name": name, "deleted": True})
        except ValueError as e:
            raise_error(str(e), "E_VAL_REQUEST", status_code=422)
        except Exception as e:
            classify_and_raise(e, source="token_tree.delete")

    def train_tree(self, req: TrainTreeRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Train a token tree on the provided corpus."""
        try:
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
        except Exception as e:
            classify_and_raise(e, source="token_tree.train")

    def similar(self, req: SimilarRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Return ranked nearest-neighbor tokens for a query token."""
        try:
            data = get_token_tree_manager().similar(req.token, top_k=req.top_k)
            return success_response(data=data)
        except KeyError as e:
            raise_error(f"Token not in vocabulary: {e}", "E_NOT_FOUND", status_code=404)
        except Exception as e:
            classify_and_raise(e, source="token_tree.similar")

    def embedding(self, req: EmbeddingRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Inspect a token's generated embedding vector."""
        try:
            data = get_token_tree_manager().embedding_info(
                req.token, top_k=req.top_k
            )
            return success_response(data=data)
        except KeyError as e:
            raise_error(f"Token not in vocabulary: {e}", "E_NOT_FOUND", status_code=404)
        except ValueError as e:
            raise_error(str(e), "E_VAL_REQUEST", status_code=422)
        except Exception as e:
            classify_and_raise(e, source="token_tree.embedding")

    def encode(self, req: TokenTextRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Encode text into token ids by walking the tree."""
        try:
            return success_response(data=get_token_tree_manager().encode(req.text))
        except Exception as e:
            classify_and_raise(e, source="token_tree.encode")

    def path(self, req: TokenTextRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Trace the encoder's greedy trie walk over text."""
        try:
            return success_response(data=get_token_tree_manager().path(req.text))
        except Exception as e:
            classify_and_raise(e, source="token_tree.path")

    def decode(self, req: TokenIdsRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Decode a list of token ids back to text."""
        try:
            return success_response(data=get_token_tree_manager().decode(req.ids))
        except Exception as e:
            classify_and_raise(e, source="token_tree.decode")

    def lineage(self, req: LineageRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Render a token's merge lineage down to character leaves."""
        try:
            data = get_token_tree_manager().lineage(req.token)
            return success_response(data=data)
        except KeyError as e:
            raise_error(f"Token not in vocabulary: {e}", "E_NOT_FOUND", status_code=404)
        except Exception as e:
            classify_and_raise(e, source="token_tree.lineage")

    def matrix(
        self,
        top_k: int = Query(default=8, ge=1, le=64),
    ) -> dict:
        """Return an embedding-matrix overview for the current tree."""
        try:
            return success_response(data=get_token_tree_manager().matrix_summary(
                top_k=top_k,
            ))
        except Exception as e:
            classify_and_raise(e, source="token_tree.matrix")

    def compare(self, req: CompareTreesRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Diff two saved trees without changing the current tree."""
        try:
            data = get_token_tree_manager().compare(req.a, req.b, top_n=req.top_k)
            return success_response(data=data)
        except FileNotFoundError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        except ValueError as e:
            raise_error(str(e), "E_BAD_REQUEST", status_code=400)
        except Exception as e:
            classify_and_raise(e, source="token_tree.compare")


router = TokenTreeRouter().router
