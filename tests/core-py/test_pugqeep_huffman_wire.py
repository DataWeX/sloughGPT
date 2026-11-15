"""
Tests for the CLH1 Huffman wire format — card d9028c8f (option B, user-approved).

History: ClusterCodec Huffman-encoded assignments IN MEMORY ONLY —
``compressor.py`` wrote ``huffman_data`` into params, ``serialize_params``
packed dense uint8 assignments, ``from_bytes`` never rebuilt the tree. The
compression died with the process. ``ClusterCodec.nbytes`` additionally
counted the never-persisted Huffman bitstream while OMITTING the packed
residual — the double lie proven by card 6eaa413c (vq-huffman-07): a
2,359,296-B residual reported byte-identical nbytes to the no-residual point.

Option B: persist the Huffman bitstream on the wire under a ``CLH1`` type
code (old readers fail loudly on the unknown code instead of misparsing),
materialize dense assignments on load so every ``params["assignments"]``
consumer (dedup fingerprints, ratios, lazy slices, to_dict) keeps working,
and count the residual in every nbytes branch.

Invariants pinned here:
    - huffman-carrying cluster points serialize as CLH1 and round-trip
      bit-exact, with params restored (assignments materialized + huffman
      keys present) so re-serialization is idempotent
    - legacy dense cluster blobs (no huffman params) still serialize as
      CLUS and parse byte-identically (guard)
    - the wire is smaller than the dense twin (the whole point of B)
    - nbytes counts the residual in ALL branches (huffman + quantized were
      the lying ones) and never overstates the wire size
"""

import sys
from pathlib import Path

import numpy as np

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domain.infrastructure._internal.pugqeep.compressor import PointCompressor
from domain.infrastructure._internal.pugqeep.point import Point


def _weight_like(n: int = 4096, seed: int = 0) -> np.ndarray:
    """Rank-1 weight-like data of ~n elements — low-rank VQ compresses it well.

    Size matters: this runs through Huffman encode/decode per test, and the
    decoder is a per-bit Python loop — multi-million-element data blows the
    per-test timeout (the wire itself parses fine; see the byte-exact walk).
    """
    rs = np.random.RandomState(seed)
    side = max(8, int(n**0.5))
    return (rs.randn(side, 1) @ rs.randn(1, side)).astype(np.float32).flatten()


def _compress(
    residual_threshold: float = 0.99,
    n: int = 4096,
    seed: int = 0,
) -> Point:
    return PointCompressor(
        adaptive_k=False,
        n_clusters=16,
        residual_threshold=residual_threshold,
    ).compress_cluster(_weight_like(n, seed), "hw", seed=0)


def _dense_twin(p: Point) -> Point:
    """Same point with the huffman keys stripped (the legacy on-disk shape)."""
    params = {k: v for k, v in p.params.items() if not k.startswith("huffman_")}
    return Point(
        identity=p.identity,
        function_type="cluster",
        params=params,
        residual=p.residual,
        accuracy=p.accuracy,
        dtype=p.dtype,
        shape=p.shape,
    )


class TestHuffmanWireFormat:
    """CLH1 — the bitstream persists; load restores the full param shape."""

    def test_wire_uses_clh1_type_code(self):
        p = _compress()
        assert p.params.get("huffman_data") is not None  # premise: huffman exists
        assert p.to_bytes()[:4] == b"CLH1"

    def test_roundtrip_bit_exact_and_params_restored(self):
        p = _compress()
        n = len(p.params["assignments"])
        p2 = Point.from_bytes(p.to_bytes(), identity="hw2")

        # decode through the rebuilt tree is bit-exact vs the direct decode
        np.testing.assert_array_equal(p2.generate(n), p.generate(n))

        # dense assignments materialized for every params["assignments"]
        # consumer (dedup fingerprints, ratios, to_dict, lazy slices)
        np.testing.assert_array_equal(p2.params["assignments"], p.params["assignments"])
        assert p2.params["huffman_data"] is not None
        assert p2.params["huffman_codes"] == p.params["huffman_codes"]
        assert p2.params["huffman_bits"] == p.params["huffman_bits"]
        np.testing.assert_allclose(p2.params["centroids"], p.params["centroids"])

    def test_roundtrip_with_residual_bit_exact(self):
        p = _compress(residual_threshold=1.0)  # residual always fires
        assert p.residual is not None
        n = len(p.params["assignments"])
        p2 = Point.from_bytes(p.to_bytes(), identity="hw3")
        assert p2.residual is not None
        np.testing.assert_array_equal(p2.generate(n), p.generate(n))

    def test_reserialize_is_idempotent(self):
        p = _compress(residual_threshold=1.0)
        p2 = Point.from_bytes(p.to_bytes(), identity="hw4")
        # params fully restored => second serialization reproduces the wire
        assert p2.to_bytes() == p.to_bytes()

    def test_wire_smaller_than_dense_twin(self):
        p = _compress()
        huff_len = len(p.to_bytes())
        dense_len = len(_dense_twin(p).to_bytes())
        # 4096 symbols over 16 clusters ~= 4 bits/symbol << 8 bits dense
        assert huff_len < dense_len

    def test_single_symbol_data_roundtrips(self):
        # degenerate case: every assignment identical (single-leaf tree)
        w = np.full(1024, 3.25, dtype=np.float32)
        p = PointCompressor(
            adaptive_k=False, n_clusters=4, residual_threshold=1.0
        ).compress_cluster(w, "const", seed=0)
        assert p.params.get("huffman_data") is not None
        p2 = Point.from_bytes(p.to_bytes(), identity="c2")
        np.testing.assert_allclose(p2.generate(1024), p.generate(1024))


class TestLegacyDenseCompat:
    """Guard: blobs without huffman params stay CLUS and parse unchanged."""

    def test_dense_cluster_point_still_uses_clus(self):
        centroids = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assignments = np.array([0, 1, 2, 0], dtype=np.uint8)
        p = Point(
            identity="legacy",
            function_type="cluster",
            params={"centroids": centroids, "assignments": assignments},
        )
        data = p.to_bytes()
        assert data[:4] == b"CLUS"
        p2 = Point.from_bytes(data, identity="legacy2")
        np.testing.assert_array_equal(p2.params["assignments"], assignments)
        np.testing.assert_allclose(p2.params["centroids"], centroids)
        assert "huffman_data" not in p2.params

    def test_dense_twin_wire_roundtrips(self):
        # huffman point with keys stripped = exactly what an old blob holds
        p = _dense_twin(_compress())
        p2 = Point.from_bytes(p.to_bytes(), identity="twin")
        n = len(p.params["assignments"])
        np.testing.assert_array_equal(p2.generate(n), p.generate(n))


class TestNbytesHonesty:
    """nbytes counts what is actually stored — residual in EVERY branch."""

    def test_nbytes_counts_residual_huffman_branch(self):
        p_res = _compress(residual_threshold=1.0)  # residual fires
        p_nores = _compress(residual_threshold=0.0)  # never fires
        assert p_res.residual is not None
        assert p_nores.residual is None
        # same data, same seed => same centroids/assignments/huffman; the
        # ONLY difference on the wire is the residual block
        assert p_res.nbytes() - p_nores.nbytes() == p_res.residual.nbytes

    def test_nbytes_counts_residual_quantized_branch(self):
        centroids = np.arange(8, dtype=np.uint8)
        assignments = np.array([0, 1, 2, 3, 1], dtype=np.uint8)
        params = {
            "centroids": centroids,
            "assignments": assignments,
            "centroid_quantized": True,
            "centroid_scale": 0.5,
            "centroid_zero_point": 2.0,
        }
        residual = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        p_nores = Point(identity="q", function_type="cluster", params=params)
        p_res = Point(identity="q", function_type="cluster", params=params, residual=residual)
        assert p_res.nbytes() - p_nores.nbytes() == residual.nbytes

    def test_nbytes_never_overstates_wire_size(self):
        for threshold in (0.0, 0.99, 1.0):
            p = _compress(residual_threshold=threshold)
            assert p.nbytes() <= len(p.to_bytes()), (
                f"nbytes overstates the wire at threshold={threshold}"
            )
