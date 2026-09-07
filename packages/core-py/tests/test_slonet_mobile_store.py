"""Tests for training.mobile_training_store — MobileTrainingStore."""

from __future__ import annotations

import pytest
import tempfile
import os
from domains.training.mobile_training_store import MobileTrainingStore


# ── MobileTrainingStore ─────────────────────────────────────────────────────


class TestMobileTrainingStore:

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MobileTrainingStore(tmpdir)
            assert store._db is not None

    def test_add_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MobileTrainingStore(tmpdir)
            doc_id = store.add_pair("hello", "hi", "s1")
            assert doc_id is not None

    def test_add_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MobileTrainingStore(tmpdir)
            pairs = [
                {"user_msg": "hello", "assistant_msg": "hi", "session_id": "s1"},
                {"user_msg": "test", "assistant_msg": "data", "session_id": "s2"},
            ]
            ids = store.add_batch(pairs)
            assert len(ids) == 2
