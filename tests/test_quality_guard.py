"""
Tests for the quality‑guard rollback mechanism in the feedback workflow.

Verifies:
- Weight snapshot and restore works correctly
- Scheduled DPO triggers at the configured interval and is suppressed before it
- `get_status()` exposes the new fields (dpo, auto_dpo_interval, last_rollback)
"""
import copy
import time
import pytest
import numpy as np


@pytest.fixture
def mock_net():
    """A minimal model stub with weight‑bearing layers."""
    class Layer:
        def __init__(self, val):
            self.weight = type("W", (), {"data": np.array(val, dtype=np.float32)})()

    class Net:
        def __init__(self):
            self.layers = [Layer([[0.5, 0.3], [0.2, 0.9]]),
                           Layer([[0.1, 0.7], [0.8, 0.4]])]
            self.hidden_dim = 768

        def parameters(self):
            return [l.weight for l in self.layers if hasattr(l, "weight")]

    return Net()


@pytest.fixture
def mock_tokenizer():
    class Tok:
        pad_id = 0
        def encode(self, text):
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return Tok()


@pytest.fixture
def wf(mock_net, mock_tokenizer, tmp_path):
    from domains.feedback.workflow import FeedbackWorkflowManager, WorkflowConfig
    from domains.feedback.database import FeedbackDB

    class MockMetaManager:
        def get_weights(self):
            return {}
        def update_weights(self, *a, **kw):
            pass
        def get_stats(self):
            return {}

    class MockLoraStore:
        def aggregate_best_adapters(self, **kw):
            return {"status": "ok", "merged": 0}
        def prune_low_quality(self, **kw):
            return []
        def get_stats(self):
            return {}

    db = FeedbackDB(str(tmp_path / "test_feedback.db"))
    wfm = FeedbackWorkflowManager(
        config=WorkflowConfig(
            aggregate_interval_minutes=999,
            prune_interval_minutes=999,
            export_interval_hours=999,
            auto_dpo_interval_minutes=999,
            health_check_interval_seconds=999,
        ),
        feedback_db=db,
        meta_manager=MockMetaManager(),
        lora_store=MockLoraStore(),
    )
    wfm.set_model(mock_net, mock_tokenizer)
    return wfm


class TestSnapshotRestore:
    def test_snapshot_and_restore(self, wf, mock_net):
        snapshot = wf._snapshot_weights(mock_net)
        assert len(snapshot) > 0
        orig_weights = [l.weight.data.copy() for l in mock_net.layers]
        for l in mock_net.layers:
            l.weight.data += 1.0
        wf._restore_weights(mock_net, snapshot)
        for i, l in enumerate(mock_net.layers):
            assert np.allclose(l.weight.data, orig_weights[i], atol=1e-6)


class TestDPOScheduler:
    def test_dpo_triggers_after_interval(self, wf, monkeypatch):
        wf._last_dpo_time = 0
        called = [False]
        def fake_dpo():
            called[0] = True
        monkeypatch.setattr(wf, "_do_dpo", fake_dpo)
        wf.run_scheduled_tasks()
        assert called[0]

    def test_dpo_skips_before_interval(self, wf, monkeypatch):
        wf._last_dpo_time = time.time()
        called = [False]
        def fake_dpo():
            called[0] = True
        monkeypatch.setattr(wf, "_do_dpo", fake_dpo)
        wf.run_scheduled_tasks()
        assert not called[0]


class TestStatusFields:
    def test_get_status_contains_new_fields(self, wf):
        status = wf.get_status()
        assert "auto_dpo_interval_minutes" in status.get("config", {})
        assert "dpo" in status.get("last_runs", {})
        assert "last_rollback" in status.get("last_runs", {})
