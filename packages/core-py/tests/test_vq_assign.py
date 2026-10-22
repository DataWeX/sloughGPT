"""Pins for ``vq_assign.nearest_assignments`` (card 9565bd2d).

The old Lloyd-loop pattern ``np.clip(np.searchsorted(centroids, x), 0, nc-1)``
is a *ceiling* lookup: every value between two centroids went to the higher
one regardless of distance. These pins assert the fixed semantics — true
nearest with deterministic lower-index tie-break — including the exact
degenerate-row element from the ``token_emb::71`` finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.infrastructure._internal.vq_assign import nearest_assignments


def test_nearest_not_ceiling():
    """0.4 sits between [0, 1] but is NEARER 0 — the ceiling bug gave 1."""
    out = nearest_assignments(np.array([0.4]), np.array([0.0, 1.0]))
    assert out.tolist() == [0]


def test_finding_element_assigns_to_zero_cluster():
    """The bdda0736 finding element (5.975798e-10) belongs to the ~0 cluster."""
    centroids = np.array([0.0, 0.25, 0.5, 0.75, 0.9999975], dtype=np.float32)
    out = nearest_assignments(np.array([5.975798e-10], dtype=np.float32), centroids)
    assert out.tolist() == [0]


def test_tie_prefers_lower_index():
    """Exactly half-way between two centroids -> the lower index wins."""
    out = nearest_assignments(np.array([0.5]), np.array([0.0, 1.0]))
    assert out.tolist() == [0]


def test_exact_centroid_matches_its_own_index():
    out = nearest_assignments(np.array([1.0]), np.array([0.0, 1.0, 2.0]))
    assert out.tolist() == [1]


def test_out_of_range_values_clip_to_end_clusters():
    out = nearest_assignments(np.array([-5.0, 5.0]), np.array([0.0, 1.0, 2.0]))
    assert out.tolist() == [0, 2]


def test_single_centroid():
    out = nearest_assignments(np.array([0.3, -0.2]), np.array([0.1]))
    assert out.tolist() == [0, 0]


def test_empty_centroids_raise():
    with pytest.raises(ValueError, match="non-empty"):
        nearest_assignments(np.array([1.0]), np.array([], dtype=np.float32))


def test_unsorted_centroids_use_fallback_and_match_brute_force():
    """Empty Lloyd clusters can leave centroids unsorted — must still be nearest."""
    centroids = np.array([0.5, 0.0, 0.9], dtype=np.float32)  # unsorted on purpose
    values = np.array([0.1, 0.49, 0.95], dtype=np.float32)
    got = nearest_assignments(values, centroids)
    ref = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
    np.testing.assert_array_equal(got, ref.astype(np.intp))
    assert got.tolist() == [1, 0, 2]


def test_matches_brute_force_argmin_with_lowest_index_ties():
    """Property pin: identical to argmin(|c - x|) (argmin ties -> first index)."""
    rng = np.random.RandomState(42)
    centroids = np.sort(rng.uniform(-3.0, 3.0, size=32).astype(np.float32))
    values = rng.uniform(-5.0, 5.0, size=500).astype(np.float32)
    got = nearest_assignments(values, centroids)
    ref = np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1)
    np.testing.assert_array_equal(got, ref.astype(np.intp))


def test_output_dtype_and_shape():
    values = np.linspace(-1.0, 1.0, 7).astype(np.float32)
    centroids = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out = nearest_assignments(values, centroids)
    assert out.dtype == np.intp
    assert out.shape == values.shape
    assert out.min() >= 0 and out.max() < len(centroids)
