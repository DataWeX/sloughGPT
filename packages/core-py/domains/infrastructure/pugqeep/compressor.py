"""
PointCompressor — compresses weight tensors into Points.

Supports:
  - Vector quantization (cluster-based) with Lloyd's refinement
  - k-means++ initialization for better centroid placement
  - Huffman encoding for assignment compression
  - Adaptive k: cluster count varies per layer by weight entropy
  - Centroid int8: quantize centroids to int8 for better ratio
  - Function fitting (periodic, linear, polynomial) with residual storage
"""
from __future__ import annotations

import heapq
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .point import Point
from .config import CompressorConfig


class HuffmanTree:
    """Lossless Huffman encoding for uint8 arrays.

    Builds a prefix code from symbol frequencies, encodes to a compact
    bitstream, and decodes back to the original array.
    """

    __slots__ = ('codes', 'tree')

    class _Node:
        __slots__ = ('symbol', 'freq', 'left', 'right')
        def __init__(self, symbol=None, freq=0, left=None, right=None):
            self.symbol = symbol
            self.freq = freq
            self.left = left
            self.right = right
        def __lt__(self, other):
            return self.freq < other.freq

    @classmethod
    def build(cls, data: np.ndarray) -> "HuffmanTree":
        """Build Huffman tree from uint8 array."""
        freq = Counter(data.tolist())
        heap = [cls._Node(v, c) for v, c in freq.items()]
        heapq.heapify(heap)
        if len(heap) == 1:
            node = heapq.heappop(heap)
            # Single symbol: root has one child
            root = cls._Node(None, node.freq, node, cls._Node(node.symbol, 0))
        else:
            while len(heap) > 1:
                l = heapq.heappop(heap)
                r = heapq.heappop(heap)
                heapq.heappush(heap, cls._Node(None, l.freq + r.freq, l, r))
            root = heap[0]
        codes = {}
        def _build_codes(node, prefix=''):
            if node is None:
                return
            if node.left is None and node.right is None:
                if node.symbol is not None:
                    codes[node.symbol] = prefix or '0'
                return
            _build_codes(node.left, prefix + '0')
            _build_codes(node.right, prefix + '1')
        _build_codes(root)
        return cls(codes=codes, tree=root)

    @classmethod
    def from_dict(cls, d: dict) -> "HuffmanTree":
        """Reconstruct from serialized codes dict. Rebuilds tree from codes."""
        codes = {int(k): v for k, v in d.items()}
        tree = cls._Node()
        for symbol, code in codes.items():
            node = tree
            for bit in code[:-1]:
                if bit == '0':
                    if node.left is None:
                        node.left = cls._Node()
                    node = node.left
                else:
                    if node.right is None:
                        node.right = cls._Node()
                    node = node.right
            # Last bit creates leaf
            if code[-1] == '0':
                node.left = cls._Node(symbol=symbol)
            else:
                node.right = cls._Node(symbol=symbol)
        return cls(codes=codes, tree=tree)

    def __init__(self, codes: dict = None, tree: _Node = None):
        self.codes = codes or {}
        self.tree = tree

    def encode(self, data: np.ndarray) -> Tuple[bytes, int]:
        """Encode uint8 array to packed bitstream.

        Returns:
            (packed_bytes, total_bits)
        """
        bits = []
        for val in data:
            bits.append(self.codes[val])
        bitstring = ''.join(bits)
        total_bits = len(bitstring)
        # Pad to byte boundary
        pad = (8 - total_bits % 8) % 8
        bitstring += '0' * pad
        packed = bytearray()
        for i in range(0, len(bitstring), 8):
            packed.append(int(bitstring[i:i+8], 2))
        return bytes(packed), total_bits

    @staticmethod
    def decode(packed: bytes, total_bits: int, tree: _Node,
               n: int) -> np.ndarray:
        """Decode packed bitstream back to uint8 array."""
        if tree is None:
            return np.zeros(n, dtype=np.uint8)
        result = np.empty(n, dtype=np.uint8)
        node = tree
        bit_pos = 0
        for gi in range(n):
            while node.left is not None and node.right is not None:
                if bit_pos >= total_bits:
                    break
                byte_idx = bit_pos >> 3
                bit = (packed[byte_idx] >> (7 - (bit_pos & 7))) & 1
                bit_pos += 1
                node = node.left if bit == 0 else node.right
            result[gi] = node.symbol if node.symbol is not None else 0
            node = tree
        return result

    def tree_dict(self) -> dict:
        """Serialize codes to JSON-safe dict."""
        return {str(k): v for k, v in self.codes.items()}


class PointCompressor:
    """Compresses weight tensors into points (generator functions).

    Args:
        config: Optional CompressorConfig. If None, uses defaults.
        n_clusters: Override config n_clusters.
        lloyd_iterations: Override config lloyd_iterations.
        residual_threshold: Accuracy below this stores residual (0-1).
        adaptive_k: If True, vary cluster count per layer by weight entropy.
        quantize_centroids: If True, quantize centroids to int8 when safe.
    """

    def __init__(self, config: Optional[CompressorConfig] = None, *,
                 n_clusters: int = 16, lloyd_iterations: int = 5,
                 residual_threshold: float = 0.99,
                 adaptive_k: bool = True,
                 quantize_centroids: bool = True):
        if config is not None:
            self.n_clusters = config.n_clusters
            self.lloyd_iterations = config.lloyd_iterations
            self.gap_fill_iterations = config.gap_fill_iterations
            self.gap_fill_max_elements = config.gap_fill_max_elements
            self.method = config.method
        else:
            self.n_clusters = n_clusters
            self.lloyd_iterations = lloyd_iterations
            self.gap_fill_iterations = 4
            self.gap_fill_max_elements = 100_000
            self.method = "cluster"
        self.residual_threshold = residual_threshold
        self.adaptive_k = adaptive_k
        self.quantize_centroids = quantize_centroids

    def compress_cluster(self, weights: np.ndarray, identity: str = "unknown",
                        n_clusters: Optional[int] = None) -> Point:
        """
        Compress using vector quantization (cluster-based).

        Uses k-means++ initialization + Lloyd's refinement + Huffman encoding.
        When adaptive_k is enabled, cluster count varies by weight entropy.
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        # Input validation
        if weights.size == 0:
            raise ValueError(f"Cannot compress empty array: {identity}")
        if not np.isfinite(weights).all():
            raise ValueError(f"Array contains NaN/Inf values: {identity}")
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

        flat = weights.flatten().astype(np.float32)
        n = len(flat)

        # Adaptive k: adjust cluster count based on weight entropy
        if self.adaptive_k:
            n_clusters = self._compute_adaptive_k(flat, n_clusters)

        # Clamp n_clusters to array size
        if n_clusters > n:
            n_clusters = n

        nc = n_clusters

        # k-means++ initialization (sample-based for speed)
        centroids = np.empty(nc, dtype=np.float32)
        idx = np.random.randint(n)
        centroids[0] = flat[idx]
        for i in range(1, nc):
            sample_idx = np.random.choice(n, min(1000, n), replace=False)
            sample = flat[sample_idx]
            dists = np.min(np.abs(sample[:, None] - centroids[:i, None].T), axis=1)
            probs = dists ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(len(sample)) / len(sample)
            centroids[i] = sample[np.random.choice(len(sample), p=probs)]
        centroids.sort()

        # Lloyd's refinement with early stopping
        prev_inertia = float('inf')
        for _ in range(self.lloyd_iterations):
            assignments = np.clip(np.searchsorted(centroids, flat), 0, nc - 1).astype(np.uint8)
            sums = np.bincount(assignments, weights=flat, minlength=nc)
            counts = np.bincount(assignments, minlength=nc).astype(np.float64)
            alive = counts > 0
            centroids[alive] = (sums[alive] / counts[alive]).astype(np.float32)
            # Early stop if converged
            inertia = np.sum((flat - centroids[assignments]) ** 2)
            if abs(prev_inertia - inertia) / (prev_inertia + 1e-10) < 1e-6:
                break
            prev_inertia = inertia

        # Quantize centroids to int8 if enabled and safe
        centroid_quantized = False
        centroid_scale = None
        centroid_zero_point = None
        if self.quantize_centroids and nc >= 4:
            cmin, cmax = float(centroids.min()), float(centroids.max())
            crange = cmax - cmin
            if crange > 1e-12:
                c_scale = crange / 255.0
                c_zero_point = -cmin / c_scale
                q_centroids = np.clip(
                    np.round(centroids / c_scale + c_zero_point), 0, 255
                ).astype(np.uint8)
                recon_centroids = (q_centroids.astype(np.float32) - c_zero_point) * c_scale
                recon_via_q = recon_centroids[assignments]
                q_cos = float(np.dot(flat, recon_via_q) / (
                    np.linalg.norm(flat) * np.linalg.norm(recon_via_q) + 1e-12))
                orig_cos = float(np.dot(flat, centroids[assignments]) / (
                    np.linalg.norm(flat) * np.linalg.norm(centroids[assignments]) + 1e-12))
                if q_cos > orig_cos * 0.999:
                    centroids = q_centroids
                    centroid_scale = c_scale
                    centroid_zero_point = c_zero_point
                    centroid_quantized = True

        # Compute accuracy
        if centroid_quantized:
            reconstructed = (centroids.astype(np.float32) - centroid_zero_point) * centroid_scale
            reconstructed = reconstructed[assignments]
        else:
            reconstructed = centroids[assignments]
        mse = np.mean((flat - reconstructed) ** 2)
        var = np.var(flat)
        accuracy = 1.0 - mse / (var + 1e-8)

        # Store residual if accuracy below threshold
        residual = None
        if accuracy < self.residual_threshold:
            residual = (flat - reconstructed).astype(np.float32)

        # Huffman encode assignments (lossless compression)
        huffman = HuffmanTree.build(assignments)
        huffman_bits, total_bits = huffman.encode(assignments)

        return Point(
            identity=identity,
            function_type="cluster",
            params={
                "centroids": centroids,
                "assignments": assignments,
                "centroid_quantized": centroid_quantized,
                "centroid_scale": centroid_scale,
                "centroid_zero_point": centroid_zero_point,
                "huffman_codes": huffman.tree_dict(),
                "huffman_bits": total_bits,
                "huffman_data": huffman_bits,
            },
            residual=residual,
            accuracy=float(accuracy),
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def compress_function(self, weights: np.ndarray, identity: str = "unknown") -> Point:
        """
        Compress using function fitting (periodic, linear, polynomial).

        This works for structured weights but not random ones.
        """
        # Input validation
        if weights.size == 0:
            raise ValueError(f"Cannot compress empty array: {identity}")
        if not np.isfinite(weights).all():
            raise ValueError(f"Array contains NaN/Inf values: {identity}")

        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        var = np.var(flat)

        fits = [
            ("periodic", self._fit_periodic(flat)),
            ("linear", self._fit_linear(flat)),
            ("polynomial", self._fit_polynomial(flat)),
        ]

        best_type, (params, mse) = min(fits, key=lambda x: x[1][1])
        accuracy = 1.0 - mse / (var + 1e-8)

        if accuracy < self.residual_threshold:
            i = np.arange(n, dtype=np.float32)
            if best_type == "periodic":
                fitted = params["a"] * np.cos(i) + params["b"] * np.sin(i) + params["w"]
            elif best_type == "linear":
                fitted = params["a"] * i + params["b"]
            elif best_type == "polynomial":
                fitted = params["a"] * i**2 + params["b"] * i + params["c"]
            residual = flat - fitted
        else:
            residual = None

        return Point(
            identity=identity,
            function_type=best_type,
            params=params,
            residual=residual,
            accuracy=float(accuracy),
            dtype=str(weights.dtype),
            shape=weights.shape,
        )

    def compress_batch(self, weights_dict: Dict[str, np.ndarray],
                       method: Optional[str] = None,
                       prefix: str = "") -> Dict[str, Point]:
        """Compress multiple weight tensors in one call.

        Args:
            weights_dict: Dict mapping weight names to numpy arrays.
            method: Compression method (overrides self.method).
            prefix: Optional prefix for point identities.

        Returns:
            Dict mapping weight names to compressed Points.
        """
        results = {}
        for name, weights in weights_dict.items():
            identity = f"{prefix}{name}" if prefix else name
            results[name] = self.compress(weights, identity=identity, method=method)
        return results

    def compress(self, weights: np.ndarray, identity: str = "unknown",
                method: Optional[str] = None) -> Point:
        """Compress using specified method (defaults to self.method)."""
        if method is None:
            method = self.method
        if method == "cluster":
            return self.compress_cluster(weights, identity)
        elif method == "function":
            return self.compress_function(weights, identity)
        else:
            raise ValueError(f"Unknown method: {method}")

    def decompress(self, point: Point, n: int) -> np.ndarray:
        """Decompress a point back to weights."""
        return point.generate(n)

    def measure_compression(self, weights: np.ndarray, point: Point) -> dict:
        """Measure compression ratio and accuracy."""
        raw_size = weights.nbytes

        if point.function_type == "cluster":
            centroids = point.params["centroids"]
            assignments = point.params["assignments"]
            if point.params.get("centroid_quantized"):
                # Quantized: centroids are uint8 + 8 bytes for scale/zero_point
                compressed_bytes = centroids.nbytes + 8 + assignments.nbytes
            else:
                compressed_bytes = centroids.nbytes + assignments.nbytes
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes
        elif point.function_type == "raw":
            compressed_bytes = raw_size
        else:
            compressed_bytes = 4 + len(point.params) * 4
            if point.residual is not None:
                compressed_bytes += point.residual.nbytes

        return {
            "raw_bytes": raw_size,
            "compressed_bytes": compressed_bytes,
            "ratio": raw_size / max(compressed_bytes, 1),
            "accuracy": point.accuracy,
            "function_type": point.function_type,
        }

    def _fit_periodic(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * cos(i) + b * sin(i) + w."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([np.cos(i), np.sin(i), np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, w = result
        fitted = a * np.cos(i) + b * np.sin(i) + w
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b), "w": float(w)}, mse

    def _fit_linear(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * i + b."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b = result
        fitted = a * i + b
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b)}, mse

    def _fit_polynomial(self, flat: np.ndarray) -> Tuple[dict, float]:
        """Fit a * i^2 + b * i + c."""
        n = len(flat)
        i = np.arange(n, dtype=np.float32)
        A = np.column_stack([i**2, i, np.ones(n)])
        result, _, _, _ = np.linalg.lstsq(A, flat, rcond=None)
        a, b, c = result
        fitted = a * i**2 + b * i + c
        mse = np.mean((flat - fitted) ** 2)
        return {"a": float(a), "b": float(b), "c": float(c)}, mse

    def _compute_adaptive_k(self, flat: np.ndarray, base_k: int) -> int:
        """Choose cluster count based on weight distribution entropy."""
        hist, _ = np.histogram(flat, bins=256)
        hist = hist[hist > 0].astype(np.float64)
        probs = hist / hist.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        scale = entropy / 8.0
        k = int(base_k * (0.5 + 3.5 * scale))
        return max(4, min(k, 256))

    # ── Block quantization (Q4_K style) ──

    BLOCK_SIZE = 32

    def compress_block_q4(self, weights: np.ndarray,
                          identity: str = "unknown") -> Point:
        """Block-wise 4-bit quantization (Q4_K style)."""
        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        bs = self.BLOCK_SIZE
        pad = (bs - n % bs) % bs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        n_blocks = len(flat) // bs
        blocks = flat.reshape(n_blocks, bs)
        bmin = blocks.min(axis=1)
        bmax = blocks.max(axis=1)
        brange = np.maximum(bmax - bmin, 1e-10)
        scale = brange / 15.0
        q = ((blocks - bmin[:, None]) / scale[:, None])
        q = np.clip(np.round(q), 0, 15).astype(np.uint8)
        n_values = n_blocks * bs
        q_flat = q.ravel()
        packed = q_flat[0::2].astype(np.uint8) | (q_flat[1::2].astype(np.uint8) << 4)
        deq = q.astype(np.float32) * scale[:, None] + bmin[:, None]
        reconstructed = deq.ravel()[:n]
        mse = np.mean((flat[:n] - reconstructed) ** 2)
        var = np.var(flat[:n])
        accuracy = 1.0 - mse / (var + 1e-8)
        return Point(
            identity=identity, function_type="block_q4",
            params={"mins": bmin.astype(np.float32), "scales": scale.astype(np.float32),
                    "packed": packed, "n_elements": n, "n_blocks": n_blocks, "block_size": bs},
            accuracy=float(accuracy), dtype=str(weights.dtype), shape=weights.shape,
        )

    def decompress_block_q4(self, point: Point) -> np.ndarray:
        """Decompress block_q4 Point."""
        mins = point.params["mins"]; scales = point.params["scales"]
        packed = point.params["packed"]; n = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]; bs = point.params["block_size"]
        unpacked = np.zeros(n_blocks * bs, dtype=np.uint8)
        unpacked[0::2] = packed & 0x0F
        unpacked[1::2] = (packed >> 4) & 0x0F
        q = unpacked.reshape(n_blocks, bs).astype(np.float32)
        return (q * scales[:, None] + mins[:, None]).ravel()[:n]

    def compress_block_q8(self, weights: np.ndarray,
                          identity: str = "unknown") -> Point:
        """Block-wise 8-bit quantization."""
        flat = weights.flatten().astype(np.float32)
        n = len(flat)
        bs = self.BLOCK_SIZE
        pad = (bs - n % bs) % bs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        n_blocks = len(flat) // bs
        blocks = flat.reshape(n_blocks, bs)
        bmin = blocks.min(axis=1); bmax = blocks.max(axis=1)
        brange = np.maximum(bmax - bmin, 1e-10)
        scale = brange / 255.0
        q = ((blocks - bmin[:, None]) / scale[:, None])
        q = np.clip(np.round(q), 0, 255).astype(np.uint8)
        deq = q.astype(np.float32) * scale[:, None] + bmin[:, None]
        reconstructed = deq.ravel()[:n]
        mse = np.mean((flat[:n] - reconstructed) ** 2)
        var = np.var(flat[:n])
        accuracy = 1.0 - mse / (var + 1e-8)
        return Point(
            identity=identity, function_type="block_q8",
            params={"mins": bmin.astype(np.float32), "scales": scale.astype(np.float32),
                    "values": q, "n_elements": n, "n_blocks": n_blocks, "block_size": bs},
            accuracy=float(accuracy), dtype=str(weights.dtype), shape=weights.shape,
        )

    def decompress_block_q8(self, point: Point) -> np.ndarray:
        """Decompress block_q8 Point."""
        mins = point.params["mins"]; scales = point.params["scales"]
        values = point.params["values"]; n = point.params["n_elements"]
        n_blocks = point.params["n_blocks"]; bs = point.params["block_size"]
        q = values.reshape(n_blocks, bs).astype(np.float32)
        return (q * scales[:, None] + mins[:, None]).ravel()[:n]


