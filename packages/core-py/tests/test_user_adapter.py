"""Tests for domains.feedback.per_user_lora — UserAdapter."""

import numpy as np
import pytest
from domains.feedback.per_user_lora import UserAdapter


class TestUserAdapter:
    def test_fields(self):
        ua = UserAdapter(
            user_id="u1",
            W_a=np.zeros((8, 64)),
            W_b=np.zeros((64, 8)),
            rank=8,
            alpha=16.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.user_id == "u1"
        assert ua.rank == 8
        assert ua.alpha == 16.0
        assert ua.feedback_count == 0

    def test_custom(self):
        ua = UserAdapter(
            user_id="u2",
            W_a=np.ones((4, 32)),
            W_b=np.ones((32, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-02",
            feedback_count=5,
        )
        assert ua.feedback_count == 5
        assert ua.rank == 4

    def test_default_feedback_count(self):
        ua = UserAdapter(
            user_id="u3",
            W_a=np.zeros((2, 16)),
            W_b=np.zeros((16, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.feedback_count == 0

    def test_wa_shape(self):
        ua = UserAdapter(
            user_id="u4",
            W_a=np.zeros((8, 64)),
            W_b=np.zeros((64, 8)),
            rank=8,
            alpha=16.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.W_a.shape == (8, 64)

    def test_wb_shape(self):
        ua = UserAdapter(
            user_id="u5",
            W_a=np.zeros((8, 64)),
            W_b=np.zeros((64, 8)),
            rank=8,
            alpha=16.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.W_b.shape == (64, 8)

    def test_wa_dtype_float32(self):
        ua = UserAdapter(
            user_id="u6",
            W_a=np.zeros((4, 32), dtype=np.float32),
            W_b=np.zeros((32, 4), dtype=np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.W_a.dtype == np.float32

    def test_wb_dtype_float32(self):
        ua = UserAdapter(
            user_id="u7",
            W_a=np.zeros((4, 32), dtype=np.float32),
            W_b=np.zeros((32, 4), dtype=np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.W_b.dtype == np.float32

    def test_rank_alpha_relationship(self):
        ua = UserAdapter(
            user_id="u8",
            W_a=np.zeros((16, 128)),
            W_b=np.zeros((128, 16)),
            rank=16,
            alpha=32.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.alpha / ua.rank == 2.0

    def test_lora_matrix_product(self):
        ua = UserAdapter(
            user_id="u9",
            W_a=np.ones((4, 8)),
            W_b=np.ones((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora_matrix = ua.W_b @ ua.W_a
        assert lora_matrix.shape == (8, 8)
        assert np.allclose(lora_matrix, 4.0)

    def test_scaled_lora_output(self):
        ua = UserAdapter(
            user_id="u10",
            W_a=np.ones((4, 8)),
            W_b=np.ones((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora_matrix = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        expected = np.ones((8, 8)) * 8.0
        assert np.allclose(lora_matrix, expected)

    def test_user_id_string(self):
        ua = UserAdapter(
            user_id="test_user_123",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert isinstance(ua.user_id, str)
        assert ua.user_id == "test_user_123"

    def test_created_at_string(self):
        ua = UserAdapter(
            user_id="u11",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-06-15T10:30:00",
            updated_at="2026-06-15T10:30:00",
        )
        assert ua.created_at == "2026-06-15T10:30:00"

    def test_updated_at_string(self):
        ua = UserAdapter(
            user_id="u12",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-06-15",
            updated_at="2026-06-16",
        )
        assert ua.updated_at == "2026-06-16"

    def test_small_rank(self):
        ua = UserAdapter(
            user_id="u13",
            W_a=np.zeros((1, 64)),
            W_b=np.zeros((64, 1)),
            rank=1,
            alpha=2.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.rank == 1
        assert ua.W_a.shape[0] == 1

    def test_large_rank(self):
        ua = UserAdapter(
            user_id="u14",
            W_a=np.zeros((64, 128)),
            W_b=np.zeros((128, 64)),
            rank=64,
            alpha=128.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.rank == 64
        assert ua.W_a.shape[0] == 64

    def test_zero_alpha(self):
        ua = UserAdapter(
            user_id="u15",
            W_a=np.ones((4, 8)),
            W_b=np.ones((8, 4)),
            rank=4,
            alpha=0.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora_matrix = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        assert np.allclose(lora_matrix, 0.0)

    def test_negative_feedback_count(self):
        ua = UserAdapter(
            user_id="u16",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
            feedback_count=-1,
        )
        assert ua.feedback_count == -1

    def test_large_feedback_count(self):
        ua = UserAdapter(
            user_id="u17",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
            feedback_count=10000,
        )
        assert ua.feedback_count == 10000

    def test_wa_nonzero_values(self):
        wa = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        ua = UserAdapter(
            user_id="u18",
            W_a=wa,
            W_b=np.zeros((2, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert np.allclose(ua.W_a, wa)

    def test_wb_nonzero_values(self):
        wb = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        ua = UserAdapter(
            user_id="u19",
            W_a=np.zeros((2, 2)),
            W_b=wb,
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert np.allclose(ua.W_b, wb)

    def test_lora_adjustment_formula(self):
        ua = UserAdapter(
            user_id="u20",
            W_a=np.random.randn(4, 16).astype(np.float32),
            W_b=np.random.randn(16, 4).astype(np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        scale = 1.0
        lora_matrix = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        assert lora_matrix.shape == (16, 16)
        adjustment = np.tanh(lora_matrix.mean(axis=0, keepdims=True)) * scale * 0.1
        assert adjustment.shape == (1, 16)
        assert np.all(np.abs(adjustment) <= 0.1)

    def test_multiple_adapters_independent(self):
        ua1 = UserAdapter(
            user_id="u21",
            W_a=np.ones((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua2 = UserAdapter(
            user_id="u22",
            W_a=np.zeros((4, 8)),
            W_b=np.ones((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert not np.allclose(ua1.W_a, ua2.W_a)
        assert not np.allclose(ua1.W_b, ua2.W_b)

    def test_dataclass_fields_equal(self):
        ua1 = UserAdapter(
            user_id="u23",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua2 = UserAdapter(
            user_id="u23",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua1.user_id == ua2.user_id
        assert ua1.rank == ua2.rank
        assert ua1.alpha == ua2.alpha
        assert ua1.created_at == ua2.created_at
        assert ua1.updated_at == ua2.updated_at
        assert ua1.feedback_count == ua2.feedback_count
        assert np.allclose(ua1.W_a, ua2.W_a)
        assert np.allclose(ua1.W_b, ua2.W_b)

    def test_dataclass_fields_inequality(self):
        ua1 = UserAdapter(
            user_id="u24",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua2 = UserAdapter(
            user_id="u25",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua1.user_id != ua2.user_id

    def test_wa_can_be_modified(self):
        ua = UserAdapter(
            user_id="u26",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua.W_a += np.ones((4, 8))
        assert np.allclose(ua.W_a, 1.0)

    def test_wb_can_be_modified(self):
        ua = UserAdapter(
            user_id="u27",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua.W_b += np.ones((8, 4)) * 2.0
        assert np.allclose(ua.W_b, 2.0)

    def test_feedback_count_mutable(self):
        ua = UserAdapter(
            user_id="u28",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua.feedback_count = 10
        assert ua.feedback_count == 10

    def test_zero_rank(self):
        ua = UserAdapter(
            user_id="u29",
            W_a=np.zeros((0, 64)),
            W_b=np.zeros((64, 0)),
            rank=0,
            alpha=0.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.rank == 0
        assert ua.W_a.shape == (0, 64)

    def test_negative_feedback(self):
        ua = UserAdapter(
            user_id="u30",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
            feedback_count=0,
        )
        ua.feedback_count -= 1
        assert ua.feedback_count == -1

    def test_timestamp_format(self):
        ua = UserAdapter(
            user_id="u31",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="1700000000.0",
            updated_at="1700001000.0",
        )
        assert ua.created_at == "1700000000.0"
        assert ua.updated_at == "1700001000.0"

    def test_lora_output_zero_when_weights_zero(self):
        ua = UserAdapter(
            user_id="u32",
            W_a=np.zeros((4, 16)),
            W_b=np.zeros((16, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora_matrix = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        assert np.allclose(lora_matrix, 0.0)

    def test_many_feedback_updates(self):
        ua = UserAdapter(
            user_id="u33",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        for _ in range(100):
            ua.feedback_count += 1
        assert ua.feedback_count == 100

    def test_user_id_empty_string(self):
        ua = UserAdapter(
            user_id="",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.user_id == ""

    def test_special_characters_in_user_id(self):
        ua = UserAdapter(
            user_id="user@domain.com/sub_123",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.user_id == "user@domain.com/sub_123"

    def test_wa_wb_independence(self):
        ua = UserAdapter(
            user_id="u34",
            W_a=np.ones((4, 8)),
            W_b=np.ones((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        original_wa = ua.W_a.copy()
        ua.W_b += 5.0
        assert np.allclose(ua.W_a, original_wa)

    def test_lora_composition(self):
        ua1 = UserAdapter(
            user_id="u35",
            W_a=np.eye(4, 8, dtype=np.float32),
            W_b=np.eye(8, 4, dtype=np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua2 = UserAdapter(
            user_id="u36",
            W_a=np.ones((4, 8), dtype=np.float32) * 0.5,
            W_b=np.ones((8, 4), dtype=np.float32) * 0.5,
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        combined_W_a = (ua1.W_a + ua2.W_a) / 2
        combined_W_b = (ua1.W_b + ua2.W_b) / 2
        assert combined_W_a.shape == (4, 8)
        assert combined_W_b.shape == (8, 4)

    def test_lora_output_rank_determines_shape(self):
        for rank in [1, 2, 4, 8, 16]:
            dim = 32
            ua = UserAdapter(
                user_id=f"rank_{rank}",
                W_a=np.zeros((rank, dim)),
                W_b=np.zeros((dim, rank)),
                rank=rank,
                alpha=float(rank * 2),
                created_at="2026-01-01",
                updated_at="2026-01-01",
            )
            lora = ua.W_b @ ua.W_a
            assert lora.shape == (dim, dim)

    def test_alpha_rank_ratio_affects_output(self):
        dim = 8
        ua1 = UserAdapter(
            user_id="u_r1",
            W_a=np.ones((4, dim)),
            W_b=np.ones((dim, 4)),
            rank=4,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua2 = UserAdapter(
            user_id="u_r2",
            W_a=np.ones((4, dim)),
            W_b=np.ones((dim, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        out1 = ua1.W_b @ ua1.W_a * (ua1.alpha / ua1.rank)
        out2 = ua2.W_b @ ua2.W_a * (ua2.alpha / ua2.rank)
        assert not np.allclose(out1, out2)
        assert np.allclose(out2, out1 * 2.0)

    def test_wa_wb_random_independence(self):
        ua = UserAdapter(
            user_id="u_ind",
            W_a=np.random.randn(4, 16).astype(np.float32),
            W_b=np.random.randn(16, 4).astype(np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        orig_a = ua.W_a.copy()
        orig_b = ua.W_b.copy()
        ua.W_a += 100.0
        ua.W_b += 100.0
        assert not np.allclose(ua.W_a, orig_a)
        assert not np.allclose(ua.W_b, orig_b)

    def test_feedback_count_starts_at_default(self):
        ua = UserAdapter(
            user_id="u_fc",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.feedback_count == 0
        ua.feedback_count += 1
        assert ua.feedback_count == 1

    def test_timestamps_are_strings(self):
        ua = UserAdapter(
            user_id="u_ts",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-02",
        )
        assert isinstance(ua.created_at, str)
        assert isinstance(ua.updated_at, str)

    def test_lora_output_with_identity_weight(self):
        dim = 8
        rank = 4
        ua = UserAdapter(
            user_id="u_id",
            W_a=np.eye(rank, dim, dtype=np.float32),
            W_b=np.eye(dim, rank, dtype=np.float32),
            rank=rank,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        expected = np.zeros((dim, dim), dtype=np.float32)
        for i in range(rank):
            expected[i, i] = 2.0
        assert np.allclose(lora, expected, atol=1e-5)

    def test_different_rank_same_dim(self):
        dim = 32
        for rank in [2, 4, 8]:
            ua = UserAdapter(
                user_id=f"r{rank}",
                W_a=np.zeros((rank, dim)),
                W_b=np.zeros((dim, rank)),
                rank=rank,
                alpha=float(rank),
                created_at="2026-01-01",
                updated_at="2026-01-01",
            )
            assert ua.W_a.shape == (rank, dim)
            assert ua.W_b.shape == (dim, rank)

    def test_lora_matrix_frobenius_norm(self):
        ua = UserAdapter(
            user_id="u_fn",
            W_a=np.random.randn(4, 16).astype(np.float32),
            W_b=np.random.randn(16, 4).astype(np.float32),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora = ua.W_b @ ua.W_a
        norm = np.linalg.norm(lora)
        assert norm > 0

    def test_dataclass_equality_with_same_values(self):
        kw = dict(
            W_a=np.ones((2, 4), dtype=np.float32),
            W_b=np.ones((4, 2), dtype=np.float32),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        ua1 = UserAdapter(user_id="same", **kw)
        ua2 = UserAdapter(user_id="same", **kw)
        assert ua1.user_id == ua2.user_id
        assert ua1.rank == ua2.rank
        assert ua1.alpha == ua2.alpha
        assert np.allclose(ua1.W_a, ua2.W_a)
        assert np.allclose(ua1.W_b, ua2.W_b)

    def test_large_alpha_small_rank(self):
        ua = UserAdapter(
            user_id="u_ls",
            W_a=np.ones((1, 128)),
            W_b=np.ones((128, 1)),
            rank=1,
            alpha=128.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        assert lora.shape == (128, 128)
        assert np.allclose(lora, 128.0)

    def test_negative_values_in_matrices(self):
        ua = UserAdapter(
            user_id="u_neg",
            W_a=np.array([[-1, 2], [3, -4]], dtype=np.float32),
            W_b=np.array([[-5, 6], [7, -8]], dtype=np.float32),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora = ua.W_b @ ua.W_a
        assert lora.shape == (2, 2)
        assert not np.allclose(lora, 0.0)

    def test_mixed_feedback_count_update(self):
        ua = UserAdapter(
            user_id="u_mix",
            W_a=np.zeros((4, 8)),
            W_b=np.zeros((8, 4)),
            rank=4,
            alpha=8.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
            feedback_count=5,
        )
        ua.feedback_count += 10
        ua.feedback_count -= 3
        assert ua.feedback_count == 12

    def test_user_id_with_unicode(self):
        ua = UserAdapter(
            user_id="user_日本語_123",
            W_a=np.zeros((2, 4)),
            W_b=np.zeros((4, 2)),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.user_id == "user_日本語_123"

    def test_wa_transpose_shape(self):
        ua = UserAdapter(
            user_id="u_t",
            W_a=np.zeros((8, 64)),
            W_b=np.zeros((64, 8)),
            rank=8,
            alpha=16.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        assert ua.W_a.T.shape == (64, 8)
        assert ua.W_b.T.shape == (8, 64)

    def test_multiple_adapters_different_ranks(self):
        adapters = []
        for rank in [1, 4, 8, 16]:
            ua = UserAdapter(
                user_id=f"u_r{rank}",
                W_a=np.zeros((rank, 32)),
                W_b=np.zeros((32, rank)),
                rank=rank,
                alpha=float(rank * 2),
                created_at="2026-01-01",
                updated_at="2026-01-01",
            )
            adapters.append(ua)
        ranks = [a.rank for a in adapters]
        assert ranks == [1, 4, 8, 16]

    def test_lora_output_zero_when_alpha_zero(self):
        ua = UserAdapter(
            user_id="u_za",
            W_a=np.random.randn(4, 8).astype(np.float32),
            W_b=np.random.randn(8, 4).astype(np.float32),
            rank=4,
            alpha=0.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        lora = ua.W_b @ ua.W_a * (ua.alpha / ua.rank)
        assert np.allclose(lora, 0.0)

    def test_wa_wb_element_wise_product_independent(self):
        ua = UserAdapter(
            user_id="u_ew",
            W_a=np.array([[1, 2], [3, 4]], dtype=np.float32),
            W_b=np.array([[5, 6], [7, 8]], dtype=np.float32),
            rank=2,
            alpha=4.0,
            created_at="2026-01-01",
            updated_at="2026-01-01",
        )
        product = ua.W_a * ua.W_b.T
        assert product.shape == (2, 2)
