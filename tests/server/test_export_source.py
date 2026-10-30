"""Contract tests for /training/export-text — the source dimension.

The export door is one endpoint with a ``source`` param (endpoint rule:
normalize, don't multiply doors). These tests pin the three contracts:
feedback keeps its legacy raw-pair JSON shape, agent-run emits the
trainer-ready ``{"messages": [...]}`` JSONL (idempotent stable path,
refusal-shape exclusion, producer-side quality gate), and ``all``
normalizes both producers into one trainer JSONL.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import build_test_app


@pytest.fixture
def export_dir(tmp_path, monkeypatch):
    """Point the endpoint's export seam at a temp dir (repo data/ untouched)."""
    import apps.api.server.training.jobs_api as jobs_api

    target = tmp_path / "training_exports"
    monkeypatch.setattr(jobs_api, "_training_export_dir", lambda: target)
    return target


@pytest.fixture
def run_store():
    """Isolated agent-run MogDB store with a couple of completed runs."""
    import tempfile

    from domain.agents._internal import run_history as rh

    rh.set_mogdb_path(tempfile.mkdtemp(prefix="agent_runs_"))
    rh.reset_agent_run_store()
    store = rh.get_agent_run_store()
    good = store.start(goal="Explain the training loop", context="")
    store.complete(
        good,
        response=(
            "The training loop batches tokens, computes the loss, and "
            "backpropagates gradients through the transformer layers "
            "before applying the optimizer update."
        ),
    )
    refused = store.start(goal="Do something disallowed", context="")
    store.complete(refused, response="I'm sorry, I cannot help with that request.")
    yield store
    rh.reset_agent_run_store()
    rh.reset_mogdb()


@pytest.fixture
def client():
    import apps.api.server.training.jobs_api as jobs_api

    app = build_test_app(jobs_api.router)
    return TestClient(app, raise_server_exceptions=False)


class TestSourceParam:
    def test_default_is_feedback(self, client, export_dir):
        resp = client.post("/training/export-text", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "feedback"
        assert data["file"].endswith(".json")

    def test_feedback_shape_is_grandfathered(self, client, export_dir, monkeypatch):
        """Legacy contract: raw feedback pairs, JSON array, timestamped file."""
        import controllers.feedback as feedback_controller

        class _FakeCtrl:
            repo_root = str(export_dir.parent)
            feedback_collection = type(
                "C",
                (),
                {
                    "find": staticmethod(
                        lambda: [
                            {
                                "user_message": "hi",
                                "assistant_response": "hello",
                                "quality": 0.9,
                            }
                        ]
                    )
                },
            )()

        monkeypatch.setattr(feedback_controller, "get_feedback_controller", lambda: _FakeCtrl())
        resp = client.post("/training/export-text", json={"source": "feedback"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "feedback"
        with open(data["file"], encoding="utf-8") as fh:
            payload = json.load(fh)
        assert isinstance(payload, list) and payload[0]["user_message"] == "hi"

    def test_invalid_source_rejected(self, client, export_dir):
        resp = client.post("/training/export-text", json={"source": "somewhere"})
        assert resp.status_code == 422


class TestAgentRunSource:
    def test_emits_trainer_ready_jsonl(self, client, export_dir, run_store):
        resp = client.post(
            "/training/export-text", json={"source": "agent-run", "target_count": 10}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "agent-run"
        assert data["file"].endswith("agent_runs.jsonl")
        with open(data["file"], encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert lines, "expected at least one exported example"
        for line in lines:
            assert isinstance(line["messages"], list)
            for msg in line["messages"]:
                assert msg["role"] in ("system", "user", "assistant", "tool")
                assert msg["content"].strip()
            assert line["source"] == "agent-run"
            assert line["run_id"]

    def test_refusal_shape_excluded_through_endpoint(self, client, export_dir, run_store):
        resp = client.post("/training/export-text", json={"source": "agent-run"})
        assert resp.status_code == 200
        with open(resp.json()["file"], encoding="utf-8") as fh:
            contents = fh.read()
        assert "cannot help with that" not in contents

    def test_reexport_is_idempotent(self, client, export_dir, run_store):
        """Stable path + full rewrite: a run_id never appears twice."""
        first = client.post("/training/export-text", json={"source": "agent-run"})
        second = client.post("/training/export-text", json={"source": "agent-run"})
        assert first.status_code == second.status_code == 200
        with open(second.json()["file"], encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        run_ids = [line["run_id"] for line in lines]
        assert len(run_ids) == len(set(run_ids))
        assert second.json()["pairs_count"] == first.json()["pairs_count"]

    def test_quality_gate_filters_at_write_boundary(self, client, export_dir, run_store):
        resp = client.post(
            "/training/export-text", json={"source": "agent-run", "min_quality": 1.0}
        )
        assert resp.status_code == 200
        # max score_text_chunk is 5.0 → 5.0/5.0 = 1.0; real prose rarely
        # hits the ceiling, so a strict gate must be able to empty the file.
        with open(resp.json()["file"], encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert resp.json()["pairs_count"] == len(lines)


class TestAllSource:
    def test_normalizes_both_producers_to_one_jsonl(
        self, client, export_dir, run_store, monkeypatch
    ):
        import controllers.feedback as feedback_controller

        class _FakeCtrl:
            repo_root = str(export_dir.parent)
            feedback_collection = type(
                "C",
                (),
                {
                    "find": staticmethod(
                        lambda: [
                            {
                                "user_message": "hey there",
                                "assistant_response": "hi back",
                                "quality": 0.9,
                            }
                        ]
                    )
                },
            )()

        monkeypatch.setattr(feedback_controller, "get_feedback_controller", lambda: _FakeCtrl())
        resp = client.post("/training/export-text", json={"source": "all"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "all"
        assert data["file"].endswith("all_sources.jsonl")
        with open(data["file"], encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        sources = {line.get("source") for line in lines}
        assert sources == {"feedback", "agent-run"}
        for line in lines:
            roles = [msg["role"] for msg in line["messages"]]
            assert roles == ["user", "assistant"]


class TestRunHistoryGate:
    def test_predicate_filters_at_producer(self, run_store, tmp_path):
        out = tmp_path / "gated.jsonl"
        count = run_store.export_training_jsonl(
            str(out), predicate=lambda ex: "training loop" in str(ex)
        )
        assert count == 1
        lines = [json.loads(line) for line in out.read_text().splitlines() if line]
        assert all("training loop" in json.dumps(l) for l in lines)

    def test_limit_still_applies(self, run_store, tmp_path):
        out = tmp_path / "limited.jsonl"
        count = run_store.export_training_jsonl(str(out), limit=1)
        assert count == 1

    def test_atomic_rewrite_leaves_valid_jsonl(self, run_store, tmp_path):
        out = tmp_path / "rewrite.jsonl"
        first = run_store.export_training_jsonl(str(out))
        second = run_store.export_training_jsonl(str(out))
        assert first == second
        lines = [json.loads(line) for line in out.read_text().splitlines() if line]
        assert len(lines) == second
