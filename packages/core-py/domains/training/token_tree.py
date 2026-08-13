"""
TokenTree — tree-structured BPE tokenizer backed by pugqeep Points.

The Point-Graph-Queue thesis applied to tokenization: a trained BPE merge
sequence is an implicit binary tree (every merge joins two sub-tokens into
a parent). TokenTree materializes that tree explicitly and stands it on the
pugqeep substrate, so weights and tokens share one coherent, queryable world:

  - Encoding trie:  characters and "</w>" keyed trie over every vocabulary
                    token. Encoding is a greedy longest-prefix-match tree walk
                    (the same rule production BPE tokenizers use).
  - Merge lineage:  each merged token remembers its two parents, so a token
                    can be decomposed down to its character leaves by walking
                    the merge tree.
  - Token points:   every token owns a pugqeep Point that *generates* its
                    embedding vector on demand (learned via co-occurrence
                    PPMI + SVD, then cluster-compressed). Embeddings are not
                    stored as values — they are generated, exactly like
                    ModelTree generates weights from Points.
  - Query handlers: ``query()`` / ``encode()`` resolve text to token ids by
                    descending the tree; ``encode_batch()`` fans the same
                    read-only tree out across a thread pool (multiparallel
                    processing on the tree).
  - Persistence:    PointLibrary (``<path>.points.json``) + metadata sidecar
                    (``<path>.meta.json``), mirroring the ``.sou``/``.points``
                    conventions already used by ModelTree.

Interface is SloBPE-compatible: ``stoi`` / ``itos`` / ``vocab_size`` /
``pad_id`` / ``unk_id`` / ``bos_id`` / ``eos_id`` / ``encode`` /
``decode`` / ``merges``.

Usage:
    from domains.training.token_tree import TokenTree

    tree = TokenTree()
    tree.train(["hello world", "hello there"], vocab_size=64)

    ids = tree.encode("hello world")
    text = tree.decode(ids)

    vec = tree.embedding(ids[0])          # generated from a Point
    pieces = tree.decompose(ids[0])       # walk merge lineage to leaves
    tree.save("/tmp/my_token_tree")
"""

import json
import logging
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from domains.infrastructure.pugqeep import Point, PointCompressor, PointLibrary
from domains.training.tokenizer import gpt2_pretokenize, default_pretokenize

logger = logging.getLogger("slo.token_tree")

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
WORD_SUFFIX = "</w>"
_MULTI_PIECES = sorted(SPECIAL_TOKENS + [WORD_SUFFIX], key=len, reverse=True)


@dataclass
class TrieNode:
    """A node in the encoding trie.

    Attributes:
        children: piece-keyed children (a piece is one char, or "</w>",
            or a special-token string).
        token_id: terminal token id if this node is a vocabulary token.
        freq: corpus frequency of the token ending here.
        left_id / right_id: merge lineage parents (None for base tokens).
    """
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    token_id: Optional[int] = None
    freq: int = 0
    left_id: Optional[int] = None
    right_id: Optional[int] = None


class TokenTree:
    """Tree-structured BPE tokenizer standing on the pugqeep Point substrate.

    Args:
        pretokenizer: "gpt2" or "whitespace".
    """

    def __init__(self, pretokenizer: str = "gpt2") -> None:
        self.root = TrieNode()
        self.vocab: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self._lineage: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self._freqs: Counter = Counter()
        self._word_suffix: str = WORD_SUFFIX
        self._pretokenizer: str = pretokenizer
        self._library: PointLibrary = PointLibrary(name="token_tree")
        self._embed_dim: int = 0
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Special ids / interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.stoi.get("<PAD>", 0)

    @property
    def unk_id(self) -> int:
        return self.stoi.get("<UNK>", 1)

    @property
    def bos_id(self) -> int:
        return self.stoi.get("<BOS>", 2)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<EOS>", 3)

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: Sequence[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
        lowercase: bool = True,
        embed_dim: int = 16,
        verbose: bool = False,
    ) -> "TokenTree":
        """Learn BPE merges from a corpus and materialize them as a tree.

        Args:
            texts: sequence of raw text documents (a single str is treated as
            one document).
            vocab_size: target vocabulary size (including special tokens,
                "</w>", and base characters).
            min_frequency: minimum pair frequency to consider a merge.
            lowercase: convert text to lowercase before training.
            embed_dim: dimensionality of learned token embeddings.
            verbose: log merge progress.

        Returns:
            self (for chaining).

        Side effects:
            - rebuilds the encoding trie, vocabulary, and merge lineage
            - learns and cluster-compresses one embedding Point per token
              into the internal PointLibrary
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Need at least one text to train on")
        self.root = TrieNode()
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.merges = []
        self._lineage = {}
        self._freqs = Counter()
        self._library = PointLibrary(name="token_tree")
        self._embed_dim = embed_dim
        self._trained = False

        corpus = [self._normalize(t, lowercase) for t in texts]

        chars: Counter = Counter()
        for doc in corpus:
            for word in self._pretokenize(doc, lowercase):
                for ch in word:
                    chars[ch] += 1

        base_chars = sorted(chars.keys())
        self._add_token("<PAD>", freq=0)
        self._add_token("<UNK>", freq=0)
        self._add_token("<BOS>", freq=0)
        self._add_token("<EOS>", freq=0)
        self._add_token(WORD_SUFFIX, freq=0)
        for ch in base_chars:
            self._add_token(ch, freq=chars[ch])

        word_freqs: Counter = Counter()
        for doc in corpus:
            for word in self._pretokenize(doc, lowercase):
                word_freqs[word + self._word_suffix] += 1

        word_splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            raw = word.replace(self._word_suffix, "")
            word_splits[word] = list(raw) + [self._word_suffix]

        while len(self.vocab) < vocab_size:
            pair_counts: Counter = Counter()
            for word, freq in word_freqs.items():
                split = word_splits[word]
                if len(split) < 2:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            eligible = Counter({p: c for p, c in pair_counts.items() if c >= min_frequency})
            if not eligible:
                break

            best_pair, best_count = eligible.most_common(1)[0]
            new_token = best_pair[0] + best_pair[1]

            if verbose:
                logger.debug(
                    "merge #%d: %r + %r -> %r (count=%d)",
                    len(self.merges) + 1, best_pair[0], best_pair[1],
                    new_token, best_count,
                )

            self.merges.append(best_pair)
            self._add_token(
                new_token,
                freq=best_count,
                left_id=self.stoi[best_pair[0]],
                right_id=self.stoi[best_pair[1]],
            )

            for word in word_freqs:
                split = word_splits[word]
                if len(split) < 2:
                    continue
                new_split: List[str] = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = new_split

        self._learn_embeddings(corpus, embed_dim)
        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Encoding — the tree's query handlers
    # ------------------------------------------------------------------

    def query(self, text: str) -> Tuple[int, int]:
        """Greedy longest-prefix-match query against the encoding trie.

        Args:
            text: the remaining input to match, starting at position 0.

        Returns:
            (token_id, matched_char_count). Falls back to <UNK> with a
            single-character advance when nothing in the tree matches.
        """
        node = self.root
        i = 0
        best_id: Optional[int] = None
        best_end = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "<":
                nxt: Optional[TrieNode] = None
                for piece in _MULTI_PIECES:
                    if text.startswith(piece, i) and piece in node.children:
                        nxt = node.children[piece]
                        break
                if nxt is None:
                    break
                node = nxt
                i += len(piece)
            else:
                nxt = node.children.get(ch)
                if nxt is None:
                    break
                node = nxt
                i += 1
            if node.token_id is not None:
                best_id = node.token_id
                best_end = i
        if best_id is None:
            return self.unk_id, 1
        return best_id, best_end

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text into token ids by walking the tree.

        Args:
            text: input string.
            add_bos: prepend <BOS>.
            add_eos: append <EOS>.

        Returns:
            list of integer token ids.
        """
        text = self._normalize(text, lowercase=True)
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for word in self._pretokenize(text, lowercase=True):
            s = word + self._word_suffix
            i = 0
            while i < len(s):
                token_id, advance = self.query(s[i:])
                ids.append(token_id)
                i += advance
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def trace_path(self, text: str) -> List[dict]:
        """Trace the greedy longest-prefix walk ``encode`` performs over text.

        Mirrors ``encode`` step by step: text is normalized and pretokenized
        into words, each word is padded with the word suffix, and ``query``
        consumes the longest matching token from the remaining suffix until
        every word is exhausted. Each step records the unconsumed suffix, the
        matched token id, and how many characters the query advanced.

        Args:
            text: input string.

        Returns:
            list of ``{"remaining", "id", "consumed"}`` steps. Concatenating
            the consumed prefixes reproduces ``encode(text)``: the sequence of
            ``id`` values is exactly ``encode(text)``.
        """
        normalized = self._normalize(text, lowercase=True)
        steps: List[dict] = []
        for word in self._pretokenize(normalized, lowercase=True):
            s = word + self._word_suffix
            i = 0
            while i < len(s):
                remaining = s[i:]
                token_id, advance = self.query(remaining)
                steps.append({"remaining": remaining, "id": token_id, "consumed": advance})
                i += advance
        return steps

    def encode_batch(
        self,
        texts: Sequence[str],
        max_workers: Optional[int] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """Encode many texts in parallel over the shared read-only tree.

        The trie is immutable after training, so concurrent descents are
        safe. Each worker handles a slice of the batch (multiparallel
        processing on the tree).

        Args:
            texts: input strings.
            max_workers: thread pool size (defaults to min(cpu_count, 8)).
            add_bos: prepend <BOS> to each.
            add_eos: append <EOS> to each.

        Returns:
            list of token-id lists, in input order.
        """
        if not self._trained:
            raise RuntimeError("TokenTree is not trained")
        items = list(texts)
        if not items:
            return []
        workers = max_workers or max(1, min(os.cpu_count() or 1, 8))
        workers = min(workers, len(items))
        if workers <= 1:
            return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(
                lambda t: self.encode(t, add_bos=add_bos, add_eos=add_eos),
                items,
            ))

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """Decode token ids back into text.

        Args:
            ids: token id sequence.
            skip_special: omit special-token markers (""</w>"" is always
                removed; it is a word-boundary marker, not output text).

        Returns:
            reconstructed string.
        """
        pieces: List[str] = []
        for tid in ids:
            token = self.itos.get(tid)
            if token is None:
                continue
            if token == WORD_SUFFIX:
                continue
            if token in self.stoi and token in SPECIAL_TOKENS:
                if skip_special:
                    continue
                pieces.append(token)
            else:
                pieces.append(token.replace(WORD_SUFFIX, ""))
        return "".join(pieces)

    def resolve_token(self, token: str) -> int:
        """Resolve a token id or literal token string to a vocabulary id.

        Numeric strings are treated as ids. Literal strings are matched with
        the special tokens and "</w>"-suffixed forms kept whole as written;
        otherwise a plain word tries ``word</w>``, `` word</w>``, ``word``,
        and `` word`` in that order.

        Args:
            token: integer id or literal vocabulary token.

        Returns:
            vocabulary id.

        Raises:
            KeyError: when nothing in the vocabulary matches.
        """
        try:
            return int(token)
        except ValueError:
            pass
        if token.startswith("<") or token.endswith(WORD_SUFFIX):
            candidates = [token]
        else:
            candidates = [
                token + WORD_SUFFIX,
                " " + token + WORD_SUFFIX,
                token,
                " " + token,
            ]
        for candidate in candidates:
            if candidate in self.stoi:
                return self.stoi[candidate]
        raise KeyError(token)

    # ------------------------------------------------------------------
    # Token points — embeddings are generated, not stored
    # ------------------------------------------------------------------

    def _learn_embeddings(self, corpus: List[str], embed_dim: int) -> None:
        """Learn co-occurrence embeddings and store them as cluster Points.

        Builds a token-by-token co-occurrence matrix by encoding the
        training corpus, applies PPMI weighting, projects to embed_dim via
        truncated SVD, then cluster-compresses each row into a pugqeep Point.

        Args:
            corpus: normalized training documents (encoded end-to-end so
                tokens observe real cross-word context windows).
            embed_dim: target embedding dimensionality.
        """
        vocab_n = len(self.vocab)
        if embed_dim <= 0 or vocab_n <= 4:
            return

        window = 2
        cooc = np.zeros((vocab_n, vocab_n), dtype=np.float64)
        for doc in corpus:
            ids = self.encode(doc)
            if not ids:
                continue
            for pos, tid in enumerate(ids):
                lo = max(0, pos - window)
                hi = min(len(ids), pos + window + 1)
                for j in range(lo, hi):
                    if j == pos:
                        continue
                    cooc[tid, ids[j]] += 1.0

        row_sum = cooc.sum(axis=1, keepdims=True)
        col_sum = cooc.sum(axis=0, keepdims=True)
        total = cooc.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(
                (cooc * total) / np.maximum(row_sum * col_sum, 1.0)
            )
        pmi = np.clip(pmi, 0.0, None)

        if vocab_n >= embed_dim:
            U, S, Vt = np.linalg.svd(pmi, full_matrices=False)
            emb = U[:, :embed_dim] * S[:embed_dim]
        else:
            emb = pmi
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)

        n_clusters = max(2, min(8, embed_dim))
        for tid in range(vocab_n):
            row = emb[tid].astype(np.float32)
            if len(row) < n_clusters * 2:
                point = Point(
                    identity=f"token_emb::{tid}",
                    function_type="raw",
                    params={"data_b64": _b64(row.tobytes()), "shape": list(row.shape),
                            "dtype": "float32"},
                    accuracy=1.0,
                )
            else:
                point = self._compressor().compress_cluster(
                    row, identity=f"token_emb::{tid}", n_clusters=n_clusters)
                # percentile/Lloyd refit produce float64 centroids; embeddings
                # are float32, so store them at native precision
                point.params["centroids"] = point.params["centroids"].astype(np.float32)
            self._library.add(point)

    def embedding(self, token_id: int) -> Optional[np.ndarray]:
        """Generate a token's embedding vector from its pugqeep Point.

        Args:
            token_id: vocabulary id.

        Returns:
            float32 vector of length embed_dim, or None if no point exists.
        """
        if self._embed_dim <= 0:
            return None
        point = self._library.get(f"token_emb::{token_id}")
        if point is None:
            return None
        return point.generate(self._embed_dim).astype(np.float32)

    def embedding_points(self) -> int:
        """Number of embedding Points stored in the library."""
        return len(self._library.list_all())

    def embedding_compression_ratio(self) -> float:
        """Aggregate compression ratio of all embedding Points."""
        stats = self._library.stats()
        raw = stats.get("total_raw_bytes", 0)
        comp = stats.get("total_compressed_bytes", 0)
        return raw / max(comp, 1)

    def embedding_matrix(self) -> Optional[np.ndarray]:
        """Generate the full embedding matrix from the token Points.

        Each row is produced by ``Point.generate`` (embeddings are not
        stored as values), matching the way ModelTree generates weights.

        Returns:
            float32 ``(vocab_size, embed_dim)`` matrix with L2-normalized
            rows, or None when embeddings are disabled.
        """
        if self._embed_dim <= 0:
            return None
        rows: List[np.ndarray] = []
        for tid in range(len(self.vocab)):
            v = self.embedding(tid)
            rows.append(
                v if v is not None
                else np.zeros(self._embed_dim, dtype=np.float32)
            )
        return np.stack(rows)

    def embedding_matrix_stats(self, top_n: int = 8) -> Dict[str, Any]:
        """Summarize the full embedding matrix in one shot.

        Rows are L2-normalized generated embeddings (see
        :meth:`embedding_matrix`). ``norm_*`` fields describe the row-norm
        distribution, ``dead_tokens`` counts zero-norm rows, and the
        most/least energetic lists rank live rows by norm.

        Args:
            top_n: how many most- and least-energetic tokens to return.

        Returns:
            dict with ``matrix`` ([rows, cols] or None), ``norm_min``,
            ``norm_mean``, ``norm_max``, ``dead_tokens``, ``live_tokens``,
            ``most_energetic``, ``least_energetic`` — each energy entry a
            ``[token, token_id, norm]`` triple sorted by norm.
        """
        mat = self.embedding_matrix()
        if mat is None:
            return {
                "matrix": None,
                "norm_min": 0.0,
                "norm_mean": 0.0,
                "norm_max": 0.0,
                "dead_tokens": 0,
                "live_tokens": 0,
                "most_energetic": [],
                "least_energetic": [],
            }
        norms = np.linalg.norm(mat, axis=1).astype(np.float64)
        live = np.flatnonzero(norms > 0.0)
        dead = int(len(norms) - len(live))
        k = min(top_n, len(live))

        def energy_rows(ids: np.ndarray) -> List[List[Any]]:
            return [
                [self.itos[int(tid)], int(tid), float(norms[tid])]
                for tid in ids
            ]

        top_ids = live[np.argsort(-norms[live])][:k]
        bottom_ids = live[np.argsort(norms[live])][:k]
        return {
            "matrix": [int(mat.shape[0]), int(mat.shape[1])],
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
            "dead_tokens": dead,
            "live_tokens": int(len(live)),
            "most_energetic": energy_rows(top_ids),
            "least_energetic": energy_rows(bottom_ids),
        }

    def similar(self, token_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Query the Point-generated embeddings for nearest-neighbor tokens.

        Ranks every token by cosine similarity between its generated
        embedding and the query token's — a semantic query over the same
        pugqeep substrate that generates weights.

        Args:
            token_id: vocabulary id to query around.
            top_k: number of results to return (query token excluded).

        Returns:
            list of ``(token_id, cosine_similarity)`` sorted descending.
            Empty when embeddings are disabled.
        """
        mat = self.embedding_matrix()
        if mat is None:
            return []
        scores = mat @ mat[token_id]
        results: List[Tuple[int, float]] = []
        for other in np.argsort(-scores):
            if int(other) == token_id:
                continue
            results.append((int(other), float(scores[other])))
            if len(results) >= top_k:
                break
        return results

    def _compressor(self) -> PointCompressor:
        if not hasattr(self, "_compressor_inst"):
            self._compressor_inst = PointCompressor()
        return self._compressor_inst

    # ------------------------------------------------------------------
    # Merge lineage — decompose a token down the tree to its leaves
    # ------------------------------------------------------------------

    def decompose(self, token_id: int) -> List[str]:
        """Walk a token's merge lineage down to its character leaves.

        Args:
            token_id: vocabulary id.

        Returns:
            list of base piece strings (characters, "</w>", or specials)
            that the token was merged from, in left-to-right order.
        """
        token = self.itos.get(token_id)
        if token is None:
            return []
        parents = self._lineage.get(token_id)
        if parents is None or parents[0] is None or parents[1] is None:
            return _split_pieces(token)
        left, right = parents
        return self.decompose(left) + self.decompose(right)

    # ------------------------------------------------------------------
    # Persistence — PointLibrary + metadata sidecar
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize the token tree to a JSON-safe dict.

        Embeds vocabulary, merges, lineage, frequencies, and the encoding
        trie (everything needed to reconstruct the tree without re-training).

        Returns:
            a dict suitable for ``json.dumps`` or a ``.soul`` metadata field.
        """
        return {
            "version": 1,
            "pretokenizer": self._pretokenizer,
            "word_suffix": self._word_suffix,
            "embed_dim": self._embed_dim,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "lineage": {str(k): list(v) for k, v in self._lineage.items()},
            "freqs": {str(k): int(v) for k, v in self._freqs.items()},
            "trie": _node_to_dict(self.root),
            "trained": self._trained,
        }

    @classmethod
    def from_dict(cls, meta: dict) -> "TokenTree":
        """Reconstruct a token tree from :meth:`to_dict` output.

        Args:
            meta: dict produced by :meth:`to_dict` (also readable from a
                ``.meta.json`` sidecar).

        Returns:
            a fully reconstructed TokenTree (embedding Points excluded —
            encode/decode only need the trie, vocabulary, and merges).
        """
        tree = cls(pretokenizer=meta.get("pretokenizer", "gpt2"))
        tree._word_suffix = meta.get("word_suffix", WORD_SUFFIX)
        tree._embed_dim = meta.get("embed_dim", 0)
        tree.vocab = list(meta.get("vocab", []))
        tree.stoi = {t: i for i, t in enumerate(tree.vocab)}
        tree.itos = {i: t for i, t in enumerate(tree.vocab)}
        tree.merges = [tuple(m) for m in meta.get("merges", [])]
        tree._lineage = {
            int(k): (int(v[0]), int(v[1]))
            for k, v in meta.get("lineage", {}).items()
        }
        tree._freqs = Counter({int(k): int(v) for k, v in meta.get("freqs", {}).items()})
        tree.root = _node_from_dict(meta.get("trie", {}))
        tree._trained = bool(meta.get("trained", False))
        return tree

    def save(self, path: str) -> Tuple[Path, Path]:
        """Persist the token tree.

        Writes ``<path>.meta.json`` (vocabulary, merges, trie, lineage) and
        ``<path>.points.json`` (embedding Points via PointLibrary).

        Args:
            path: base path without extension.

        Returns:
            (meta_path, points_path).

        Side effects:
            - writes two files to disk
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = self.to_dict()
        meta["saved_at"] = time.time()
        meta_path = Path(str(p) + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        points_path = self._library.save(Path(str(p) + ".points.json"))
        return meta_path, points_path

    @classmethod
    def load(cls, path: str) -> "TokenTree":
        """Load a previously saved token tree.

        Args:
            path: base path passed to :meth:`save`.

        Returns:
            a fully reconstructed TokenTree.

        Raises:
            FileNotFoundError: if the meta sidecar does not exist.
        """
        p = Path(path)
        meta_path = Path(str(p) + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Token tree metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text())

        tree = cls.from_dict(meta)
        points_path = Path(str(p) + ".points.json")
        if points_path.exists():
            tree._library = PointLibrary.load(points_path)
        return tree

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Summary statistics for the token tree."""
        return {
            "trained": self._trained,
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "num_base_tokens": self.vocab_size - len(self.merges),
            "embedding_points": self.embedding_points(),
            "embedding_compression_ratio": round(self.embedding_compression_ratio(), 2),
            "embed_dim": self._embed_dim,
            "library": self._library.stats(),
        }

    def vocab_entries(self, offset: int = 0, limit: int = 50) -> dict:
        """Return a paged slice of the vocabulary.

        Entries are in vocabulary id order (special tokens, then base
        characters, then merge tokens in merge order). Each entry reports the
        corpus frequency and whether the token is a special marker or a merge
        product, so the UI can badge the vocabulary without extra calls.

        Args:
            offset: number of leading entries to skip.
            limit: maximum number of entries to return (0 or negative means
                "no limit").

        Returns:
            ``{"total": int, "entries": [{"id", "token", "freq", "is_special",
            "is_merged"}]}``.
        """
        total = len(self.vocab)
        lo = max(0, offset)
        hi = total if limit <= 0 else min(total, offset + max(0, limit))
        entries = []
        for tid in range(lo, hi):
            token = self.vocab[tid]
            entries.append({
                "id": tid,
                "token": token,
                "freq": self._freqs.get(tid, 0),
                "is_special": token in SPECIAL_TOKENS,
                "is_merged": tid in self._lineage,
            })
        return {"total": total, "entries": entries}

    def show_merges(self, top_n: int = 20) -> None:
        """Print the most frequent merge rules (parent pair -> merged token)."""
        ranked = sorted(
            ((i, m, self._freqs.get(self.stoi.get(m[0] + m[1]), 0)) for i, m in enumerate(self.merges)),
            key=lambda x: x[2],
            reverse=True,
        )
        for _, (left, right), cnt in ranked[:top_n]:
            print(f"  {left!r} + {right!r} -> {left + right!r}  (count={cnt})")

    def _ranked_merges(self) -> List[dict]:
        """Return all merge rules sorted by corpus frequency descending.

        Returns:
            list of ``{"rank", "left", "right", "token", "count"}`` dicts
            with rank = global frequency rank (1 = most frequent). Empty
            when the tree is untrained.
        """
        if not self._trained:
            return []
        ranked = sorted(
            (
                (m, self._freqs.get(self.stoi.get(m[0] + m[1]), 0))
                for m in self.merges
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {
                "rank": i + 1,
                "left": left,
                "right": right,
                "token": left + right,
                "count": count,
            }
            for i, ((left, right), count) in enumerate(ranked)
        ]

    def top_merges(self, top_n: int = 20) -> List[dict]:
        """Return the most frequent merge rules as data.

        Args:
            top_n: maximum number of rules to return.

        Returns:
            list of ``{"rank", "left", "right", "token", "count"}`` dicts
            sorted by corpus frequency descending. Empty when untrained.
        """
        return self._ranked_merges()[:top_n]

    def search_merges(self, query: str, limit: int = 20) -> List[dict]:
        """Return merge rules whose left, right, or merged token match a query.

        Matching is case-insensitive substring search over the rule parts.
        Results keep their global frequency rank so a search hit still shows
        where the rule sits in the full ranking.

        Args:
            query: substring to match against left/right/merged token.
            limit: maximum number of matching rules to return.

        Returns:
            list of ``{"rank", "left", "right", "token", "count"}`` dicts,
            frequency-ranked. Empty when untrained or nothing matches.
        """
        if not query or not self._trained:
            return []
        q = query.lower()
        matches = [
            m
            for m in self._ranked_merges()
            if q in m["left"].lower()
            or q in m["right"].lower()
            or q in m["token"].lower()
        ]
        return matches[:limit]

    def show_tree(self, token_id: int) -> str:
        """Render a token's merge lineage as an ASCII tree."""
        return _render_lineage(self, token_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_token(self, token: str, freq: int,
                   left_id: Optional[int] = None, right_id: Optional[int] = None) -> int:
        if token in self.stoi:
            if freq:
                self._freqs[self.stoi[token]] += freq
            return self.stoi[token]
        tid = len(self.vocab)
        self.vocab.append(token)
        self.stoi[token] = tid
        self.itos[tid] = token
        self._freqs[tid] = freq
        if left_id is not None and right_id is not None:
            self._lineage[tid] = (left_id, right_id)
        node = self.root
        for piece in _split_pieces(token):
            if piece not in node.children:
                node.children[piece] = TrieNode()
            node = node.children[piece]
        node.token_id = tid
        node.freq = freq
        node.left_id = left_id
        node.right_id = right_id
        return tid

    def _pretokenize(self, text: str, lowercase: bool = True) -> List[str]:
        if self._pretokenizer == "whitespace":
            return default_pretokenize(text)
        return gpt2_pretokenize(text)

    @staticmethod
    def _normalize(text: str, lowercase: bool = True) -> str:
        if lowercase:
            return text.lower()
        return text


# ----------------------------------------------------------------------
# Piece helpers
# ----------------------------------------------------------------------

def _split_pieces(token: str) -> List[str]:
    """Split a token string into trie edge pieces.

    A piece is one character, or the multi-character "</w>" marker, or a
    special-token string. Multi-character markers are kept whole so they
    form single trie edges.
    """
    pieces: List[str] = []
    i = 0
    n = len(token)
    while i < n:
        for piece in _MULTI_PIECES:
            if token.startswith(piece, i):
                pieces.append(piece)
                i += len(piece)
                break
        else:
            pieces.append(token[i])
            i += 1
    return pieces


def _node_to_dict(node: TrieNode) -> dict:
    d: dict = {}
    if node.token_id is not None:
        d["id"] = node.token_id
    if node.left_id is not None:
        d["l"] = node.left_id
        d["r"] = node.right_id
    if node.children:
        d["c"] = {k: _node_to_dict(v) for k, v in node.children.items()}
    return d


def _node_from_dict(d: dict) -> TrieNode:
    node = TrieNode()
    node.token_id = d.get("id")
    if "l" in d:
        node.left_id = d["l"]
        node.right_id = d["r"]
    node.children = {
        k: _node_from_dict(v) for k, v in d.get("c", {}).items()
    }
    return node


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()


def _render_lineage(tree: TokenTree, token_id: int, prefix: str = "") -> str:
    token = tree.itos.get(token_id, f"#{token_id}")
    parents = tree._lineage.get(token_id)
    lines = [prefix + token]
    if parents and parents[0] is not None and parents[1] is not None:
        lines.append(_render_lineage(tree, parents[0], prefix + "  " + "├ "))
        lines.append(_render_lineage(tree, parents[1], prefix + "  " + "└ "))
    return "\n".join(lines)
