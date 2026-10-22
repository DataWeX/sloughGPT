"""True nearest-centroid assignment for 1-D VQ Lloyd loops (card 9565bd2d).

The old pattern ``np.clip(np.searchsorted(centroids, values), 0, nc - 1)``
is a **ceiling** lookup, not a nearest-neighbour lookup: any value falling
between two centroids is assigned to the higher one regardless of distance.
Inside a Lloyd loop this mis-assigns every mixed row below a cluster
midpoint (the ``token_emb::71`` finding of bdda0736: a near-zero element
landing in a far cluster), which inflates mse above the row variance and
drives accuracy negative before the [0,1] clamp.

Two paths, one semantics (nearest, ties -> LOWER centroid index):

- **Sorted fast path** — percentile init and the monotonicity of 1-D Lloyd
  updates keep most centroid arrays sorted; the sorted structure turns the
  nearest lookup into the ``searchsorted`` candidate plus its floor
  neighbour — an O(nc) neighbour check, not an O(n*nc) brute force.
- **Unsorted fallback** — empty clusters in a Lloyd iteration keep their old
  centroid while a neighbour absorbs the gap, which can leave the array
  unsorted (searchsorted on it would silently mis-assign). A per-centroid
  O(n*nc) scan (memory O(n)) keeps correctness; it is the rare degenerate
  path, not the hot one.

Callers do NOT need to guarantee sortedness — the helper detects it.
"""

from __future__ import annotations

import numpy as np


def nearest_assignments(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each value to its NEAREST centroid (ties -> lower index).

    ``values`` and ``centroids`` are 1-D float arrays; ``centroids`` must be
    non-empty (sortedness is detected, not required). Returns intp indices
    in ``[0, len(centroids))`` — the index space the Lloyd loops already
    consume via ``np.bincount``.
    """
    values = np.asarray(values)
    centroids = np.asarray(centroids)
    nc = len(centroids)
    if nc == 0:
        raise ValueError("centroids must be non-empty")

    if nc == 1:
        return np.zeros(values.shape, dtype=np.intp)

    if bool(np.all(centroids[:-1] <= centroids[1:])):
        # Sorted: candidate + floor neighbour are the only two candidates.
        idx = np.clip(np.searchsorted(centroids, values), 0, nc - 1)
        lower = np.clip(idx - 1, 0, nc - 1)
        d_cand = np.abs(values - centroids[idx])
        d_lower = np.abs(values - centroids[lower])
        # Strictly-closer floor wins; ties keep the lower index (<=).
        return np.where(d_lower <= d_cand, lower, idx).astype(np.intp)

    # Unsorted fallback: per-centroid scan, strict < keeps the FIRST
    # (lowest) index on ties — identical semantics to the fast path.
    best = np.full(values.shape, np.inf)
    best_idx = np.zeros(values.shape, dtype=np.intp)
    for i in range(nc):
        d = np.abs(values - centroids[i])
        closer = d < best
        best = np.where(closer, d, best)
        best_idx = np.where(closer, i, best_idx)
    return best_idx
