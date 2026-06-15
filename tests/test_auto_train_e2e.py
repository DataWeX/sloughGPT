"""
E2E test for auto-train pipeline — full HTTP round-trip.

Starts a minimal TestClient, calls /auto-train/start with a small
source_text, streams training progress to completion, then verifies
a checkpoint was saved and can be loaded.
"""
import json
import sys
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _ensure_paths():
    for _p in ('packages/core-py', 'apps/api/server'):
        _full = '/Users/mac/sloughGPT/' + _p
        if _full not in sys.path:
            sys.path.insert(0, _full)


_ensure_paths()


@pytest.fixture(scope='module')
def client():
    from routers.auto_train import router, state
    state.running = False
    state.config = {}
    state.teacher_model = None
    state.teacher_tokenizer = None
    state.student_net = None
    state.student_tokenizer = None
    state.source_lines = []

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _find_loss(events):
    """Find any loss value from events (emitted under different keys)."""
    for e in events:
        d = e.get("data") or {}
        for key in ("loss", "avg_loss", "final_loss"):
            val = d.get(key)
            if val is not None:
                return val
    return None


class TestAutoTrainE2E:
    """Full HTTP round-trip: start -> stream -> complete -> checkpoints."""

    def test_full_training_cycle(self, client):
        # 1. Start training with enough source text for batch_size=4
        source = "\n".join([
            "hello world how are you doing today",
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a subset of artificial intelligence",
            "natural language processing enables computers to understand text",
            "deep learning models require large amounts of training data",
            "transformers have revolutionized the field of NLP",
            "attention mechanisms allow models to focus on relevant parts",
            "training neural networks requires careful hyperparameter tuning",
            "the loss function measures how well the model predicts the target",
            "gradient descent is an optimization algorithm for neural networks",
            "batch processing helps stabilize training and reduce noise",
            "learning rate scheduling can improve convergence speed",
        ] * 3)
        resp = client.post("/auto-train/start", json={
            "source_text": source,
            "epochs": 2,
            "learning_rate": 0.001,
            "algo": "bpe",
            "batch_size": 4,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ready", "started")

        # 2. Stream training events to completion
        with client.stream("GET", "/auto-train/stream") as stream:
            events = []
            for chunk in stream.iter_lines():
                if chunk.startswith("data: "):
                    events.append(json.loads(chunk[6:]))
                    if events[-1].get("status") in ("complete", "error"):
                        break

        assert len(events) > 0, "No SSE events received"

        terminal = events[-1]
        if terminal["status"] == "error":
            pytest.fail(f"Training failed: {terminal.get('message', terminal)}")

        # 3. Verify phase sequence
        phase_names = [e["phase"] for e in events]
        assert "train" in phase_names or "TRAINING" in phase_names, f"No training phase in {phase_names}"

        # 4. Verify terminal status is complete
        assert terminal["status"] == "complete"
        data = terminal.get("data") or {}
        assert "checkpoint" in data, f"Missing checkpoint in terminal event: {terminal}"
        assert "final_loss" in data, f"Missing final_loss: {terminal}"

        # 5. Verify some loss value exists somewhere in the stream
        loss = _find_loss(events)
        assert loss is not None, "No loss values emitted in any event"
        assert isinstance(loss, (int, float)), f"Invalid loss type: {type(loss).__name__}"

        # 6. Verify progress values if any
        progresses = [e["data"]["progress"]
                      for e in events if e.get("data", {}).get("progress") is not None]
        for p in progresses:
            assert 0 <= p <= 100, f"Progress out of range: {p}"

        # 7. Verify checkpoint was created
        resp = client.get("/auto-train/checkpoints")
        assert resp.status_code == 200
        checkpoints = resp.json().get("checkpoints", [])
        assert len(checkpoints) > 0, "No checkpoints created"

        # 8. Verify checkpoint has expected fields
        cp = checkpoints[0]
        assert "name" in cp
        assert "loss" in cp or "epochs_trained" in cp

        # 9. Load the checkpoint
        resp = client.post(f"/auto-train/checkpoints/{cp['name']}/load")
        assert resp.status_code == 200
