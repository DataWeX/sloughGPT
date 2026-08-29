"""Tests for domains.feedback.per_user_lora — UserAdapter."""

import numpy as np
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
