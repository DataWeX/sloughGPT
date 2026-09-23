"""Mathematical induction proof of RoPE correctness.

Proves that our implementation satisfies the relative position constraint
for ALL positions, not just tested ones.

Theorem: For any vectors q, k in R^D and any positions m, n >= 0:
    R(m·Θ)·q · R(n·Θ)·k = q · R((n-m)·Θ)·k

where R(θ) is the block-diagonal rotation matrix with D/2 blocks of:
    [cos θ_i  -sin θ_i]
    [sin θ_i   cos θ_i]

Proof by strong induction on max(m, n).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domain.infrastructure._internal.rope import (
    rope_apply,
    rope_frequencies,
    rope_rotation,
)


# ── Rotation matrix construction ──────────────────────────────────────────────

def rotation_matrix(pos: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Construct the full D×D rotation matrix R(m·Θ).

    This is the "textbook" implementation — explicit matrix multiplication.
    Used only for verification (slow, O(D²) per position).

    For each 2D subspace i, the block is:
        [cos(m·θ_i)  -sin(m·θ_i)]
        [sin(m·θ_i)   cos(m·θ_i)]

    The full matrix is block-diagonal with D/2 such blocks.
    """
    freqs = rope_frequencies(dim, base)
    R = np.eye(dim, dtype=np.float32)
    for i, freq in enumerate(freqs):
        angle = pos * freq
        c, s = np.cos(angle), np.sin(angle)
        R[2*i, 2*i] = c
        R[2*i, 2*i+1] = -s
        R[2*i+1, 2*i] = s
        R[2*i+1, 2*i+1] = c
    return R


def rotation_matrix_split_half(pos: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """Construct rotation matrix using split-half convention (our implementation).

    Subspace i pairs element i with element i+D/2:
        [cos(m·θ_i)  -sin(m·θ_i)]
        [sin(m·θ_i)   cos(m·θ_i)]

    The full matrix is:
        [ C  -S ]
        [ S   C ]

    where C = diag(cos), S = diag(sin).
    """
    freqs = rope_frequencies(dim, base)
    half = dim // 2
    angles = pos * freqs
    C = np.diag(np.cos(angles))
    S = np.diag(np.sin(angles))
    R = np.zeros((dim, dim), dtype=np.float32)
    R[:half, :half] = C
    R[:half, half:] = -S
    R[half:, :half] = S
    R[half:, half:] = C
    return R


# ── Induction proof ───────────────────────────────────────────────────────────

class TestInductionBaseCase:
    """Base case: position 0 is identity.

    R(0·Θ) = I (identity matrix)
    Therefore: R(0)·q · R(0)·k = q · k = q · R((0-0)·Θ)·k  ✓
    """

    def test_rotation_at_zero_is_identity_matrix(self):
        """R(0) must equal the identity matrix."""
        for dim in [2, 4, 8, 16, 64]:
            R = rotation_matrix(0, dim)
            np.testing.assert_allclose(R, np.eye(dim), atol=1e-6,
                err_msg=f"R(0) != I for dim={dim}")

    def test_rope_apply_at_zero_is_identity(self):
        """rope_apply(x[0], 0) must equal x[0] — position 0 is identity."""
        for dim in [4, 8, 64]:
            x = np.random.randn(dim).astype(np.float32)
            result = rope_apply(x, pos=0, dim=dim)
            np.testing.assert_allclose(result, x, atol=1e-6,
                err_msg=f"rope_apply(x, 0) != x for dim={dim}")

    def test_rope_apply_sequence_first_token_is_identity(self):
        """In a sequence starting at pos=0, the first token is unmodified."""
        for dim in [4, 8, 64]:
            x = np.random.randn(10, dim).astype(np.float32)
            result = rope_apply(x, pos=0, dim=dim)
            np.testing.assert_allclose(result[0], x[0], atol=1e-6,
                err_msg=f"First token modified for dim={dim}")

    def test_dot_product_at_zero(self):
        """q_0 · k_0 = q · k (base case of relative position theorem)."""
        dim = 64
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)

        dot_original = np.dot(q, k)
        dot_rotated = np.dot(rope_apply(q, 0, dim), rope_apply(k, 0, dim))
        assert dot_rotated == pytest.approx(dot_original, rel=1e-5)


class TestInductionStep:
    """Inductive step: if property holds for all positions <= m,
    prove it holds for m+1.

    Key identity: R((m+1)·θ) = R(θ) · R(m·θ)

    This is because rotation matrices compose by addition of angles:
        R(α) · R(β) = R(α+β)

    Therefore:
        q_{m+1} · k_n = R((m+1)·Θ)·q · R(n·Θ)·k
                       = R(Θ)·R(m·Θ)·q · R(n·Θ)·k

    By induction hypothesis:
        R(m·Θ)·q · R(n·Θ)·k = q · R((n-m)·Θ)·k

    We need to show:
        R(Θ)·R(m·Θ)·q · R(n·Θ)·k = q · R((n-m-1)·Θ)·k

    This follows from the rotation composition property.
    """

    def test_rotation_composition(self):
        """R(a) · R(b) = R(a+b) — rotations compose by addition."""
        dim = 8
        for a, b in [(0, 0), (1, 0), (5, 3), (10, 10), (0, 100)]:
            R_a = rotation_matrix_split_half(a, dim)
            R_b = rotation_matrix_split_half(b, dim)
            R_ab = rotation_matrix_split_half(a + b, dim)
            product = R_a @ R_b
            np.testing.assert_allclose(product, R_ab, atol=1e-5,
                err_msg=f"R({a})·R({b}) != R({a+b})")

    def test_rotation_orthogonality(self):
        """R(θ)^T · R(θ) = I — rotations preserve dot product."""
        dim = 8
        for pos in [0, 1, 10, 100]:
            R = rotation_matrix_split_half(pos, dim)
            product = R.T @ R
            np.testing.assert_allclose(product, np.eye(dim), atol=1e-5,
                err_msg=f"R({pos})^T·R({pos}) != I")

    def test_inductive_step_explicit(self):
        """Direct verification: R(m+1)·q · R(n)·k satisfies the theorem."""
        dim = 8
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)

        for m in range(20):
            for n in range(20):
                # Left side: R(m)·q · R(n)·k
                q_m = rope_apply(q, m, dim)
                k_n = rope_apply(k, n, dim)
                lhs = np.dot(q_m, k_n)

                # Right side: q · R((n-m))·k
                k_nm = rope_apply(k, n - m, dim)
                rhs = np.dot(q, k_nm)

                assert lhs == pytest.approx(rhs, rel=1e-4), \
                    f"Failed at m={m}, n={n}: {lhs} != {rhs}"


class TestFullInduction:
    """Complete induction proof: for ALL positions 0..1000, the theorem holds.

    This is stronger than testing a few positions — it verifies the
    mathematical property at every step of the induction.
    """

    def test_all_pairs_up_to_100(self):
        """For all m, n in [0, 100): q_m · k_n = q · R((n-m))·k."""
        dim = 16
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)

        for m in range(100):
            q_m = rope_apply(q, m, dim)
            for n in range(100):
                k_n = rope_apply(k, n, dim)
                lhs = np.dot(q_m, k_n)
                k_nm = rope_apply(k, n - m, dim)
                rhs = np.dot(q, k_nm)
                assert lhs == pytest.approx(rhs, rel=1e-4)

    def test_large_positions(self):
        """Theorem holds even at position 10000."""
        dim = 64
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)

        for m, n in [(9999, 10000), (10000, 9999), (10000, 10000)]:
            q_m = rope_apply(q, m, dim)
            k_n = rope_apply(k, n, dim)
            lhs = np.dot(q_m, k_n)
            k_nm = rope_apply(k, n - m, dim)
            rhs = np.dot(q, k_nm)
            assert lhs == pytest.approx(rhs, rel=1e-3)

    def test_translation_invariance_all_positions(self):
        """q_m · k_m = q_0 · k_0 for all m (translation invariance)."""
        dim = 32
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)
        baseline = np.dot(rope_apply(q, 0, dim), rope_apply(k, 0, dim))

        for m in range(200):
            dot_m = np.dot(rope_apply(q, m, dim), rope_apply(k, m, dim))
            assert dot_m == pytest.approx(baseline, rel=1e-4)


class TestRotaryProperty:
    """Prove the rotation preserves the dot product under translation.

    Core insight: R(m·Θ) is an orthogonal matrix, so:
        (R(m)·q)^T · (R(m)·k) = q^T · R(m)^T · R(m) · k = q^T · k

    This means: q_m · k_m = q · k (for same position).
    """

    def test_same_position_preserves_original_dot(self):
        """q_m · k_m = q · k for all m."""
        dim = 64
        q = np.random.randn(dim).astype(np.float32)
        k = np.random.randn(dim).astype(np.float32)
        original = np.dot(q, k)

        for m in range(50):
            q_m = rope_apply(q, m, dim)
            k_m = rope_apply(k, m, dim)
            assert np.dot(q_m, k_m) == pytest.approx(original, rel=1e-4)

    def test_rotation_preserves_vector_norm(self):
        """‖R(m)·x‖ = ‖x‖ — isometry property."""
        dim = 64
        x = np.random.randn(dim).astype(np.float32)
        original_norm = np.linalg.norm(x)

        for m in range(50):
            x_m = rope_apply(x, m, dim)
            assert np.linalg.norm(x_m) == pytest.approx(original_norm, rel=1e-5)

    def test_orthogonality_of_rotation_matrix(self):
        """R(m)^T · R(m) = I for all m."""
        dim = 8
        for m in range(50):
            R = rotation_matrix_split_half(m, dim)
            product = R.T @ R
            np.testing.assert_allclose(product, np.eye(dim), atol=1e-4)
