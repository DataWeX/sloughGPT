"""Coverage tests for VMTrainingBridge (domains.shell.vm_training_bridge)."""

import requests

from domains.shell import vm_training_bridge
from domains.shell.vm_training_bridge import VMTrainingBridge


class _FakeResp:
    def __init__(self, status_code=200, payload=None, raise_exc=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._raise_exc = raise_exc

    def raise_for_status(self):
        if self._raise_exc:
            raise self._raise_exc

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []
        self.gets = []

    def post(self, *a, **k):
        self.posts.append((a, k))
        return self._responses.pop(0)

    def get(self, *a, **k):
        self.gets.append((a, k))
        return self._responses.pop(0)


def _bridge(responses):
    b = VMTrainingBridge()
    b._session = _FakeSession(responses)
    return b


class TestStart:
    def test_invalid_json(self):
        b = _bridge([])
        assert b.start("{not json") == -1

    def test_non_object_json(self):
        b = _bridge([])
        assert b.start("[1, 2]") == -1

    def test_success(self):
        b = _bridge([_FakeResp(200, {"job_id": "api-1"})])
        job_id = b.start('{"dataset": "shakespeare", "epochs": 5}')
        assert job_id == 1
        req = b._session.posts[0]
        assert req[0][0] == "http://localhost:8000/training/start"
        assert req[1]["json"]["dataset"] == "shakespeare"
        assert req[1]["json"]["epochs"] == 5
        assert req[1]["json"]["name"] == "vm-training"
        assert b._jobs[1] == {"api_job_id": "api-1", "status": "running"}
        assert b._next_job_id == 2

    def test_custom_fields_and_data_path(self):
        b = _bridge([_FakeResp(200, {"job_id": "api-2"})])
        b.start(
            '{"dataset": "d", "lr": 0.01, "embed_dim": 64, "batch_size": 8,'
            ' "data_path": "/tmp/x.txt", "model": "custom", "name": "my-run"}'
        )
        payload = b._session.posts[0][1]["json"]
        assert payload["learning_rate"] == 0.01
        assert payload["n_embed"] == 64
        assert payload["batch_size"] == 8
        assert payload["data_path"] == "/tmp/x.txt"
        assert payload["model"] == "custom"
        assert payload["name"] == "my-run"

    def test_api_error_returns_minus_one(self):
        b = _bridge([_FakeResp(500, raise_exc=requests.HTTPError("boom"))])
        assert b.start('{"dataset": "d"}') == -1

    def test_connection_error_returns_minus_one(self):
        b = _bridge([_FakeResp(500, raise_exc=requests.ConnectionError("down"))])
        assert b.start('{"dataset": "d"}') == -1


class TestStatus:
    def test_job_not_found(self):
        b = _bridge([])
        assert b.status(99) == {"status": "not_found", "progress": 0.0, "error": None}

    def test_cached_completed(self):
        b = _bridge([])
        b._jobs[1] = {"status": "completed", "progress": 1.0}
        out = b.status(1)
        assert out["status"] == "completed"
        assert out["progress"] == 1.0

    def test_cached_failed(self):
        b = _bridge([])
        b._jobs[1] = {"status": "failed", "error": "oops"}
        out = b.status(1)
        assert out["status"] == "failed"
        assert out["error"] == "oops"

    def test_no_api_job_id_running(self):
        b = _bridge([])
        b._jobs[1] = {"status": "running"}
        out = b.status(1)
        assert out == {"status": "running", "progress": 0.0, "error": None}

    def test_api_404(self):
        b = _bridge([_FakeResp(404)])
        b._jobs[1] = {"status": "running", "api_job_id": "gone"}
        out = b.status(1)
        assert out == {"status": "not_found", "progress": 0.0, "error": None}

    def test_poll_error_keeps_running(self):
        b = _bridge([_FakeResp(500, raise_exc=requests.ConnectionError("down"))])
        b._jobs[1] = {"status": "running", "api_job_id": "api-1"}
        out = b.status(1)
        assert out == {"status": "running", "progress": 0.0, "error": None}

    def test_transition_to_completed(self):
        b = _bridge([
            _FakeResp(200, {"status": "completed", "progress": 100, "loss": 0.5, "checkpoint": "models/c.ckpt"}),
        ])
        b._jobs[1] = {"status": "running", "api_job_id": "api-1"}
        out = b.status(1)
        assert out["status"] == "completed"
        assert out["progress"] == 1.0
        assert b._jobs[1]["status"] == "completed"

    def test_transition_to_failed(self):
        b = _bridge([
            _FakeResp(200, {"status": "failed", "error": "oom"}),
        ])
        b._jobs[1] = {"status": "running", "api_job_id": "api-1"}
        out = b.status(1)
        assert out["status"] == "failed"
        assert out["error"] == "oom"
        assert b._jobs[1]["status"] == "failed"

    def test_transition_to_cancelled(self):
        b = _bridge([
            _FakeResp(200, {"status": "cancelled", "error": "stopped"}),
        ])
        b._jobs[1] = {"status": "running", "api_job_id": "api-1"}
        out = b.status(1)
        assert out["status"] == "cancelled"

    def test_still_running_with_progress(self):
        b = _bridge([
            _FakeResp(200, {"status": "running", "progress": 50}),
        ])
        b._jobs[1] = {"status": "running", "api_job_id": "api-1"}
        out = b.status(1)
        assert out["status"] == "running"
        assert out["progress"] == 0.5

    def test_uses_configured_poll_url(self):
        b = _bridge([_FakeResp(200, {"status": "running"})])
        b._jobs[1] = {"status": "running", "api_job_id": "api-7"}
        b.status(1)
        assert b._session.gets[0][0][0] == "http://localhost:8000/training/jobs/api-7"


class TestGetResultJson:
    def test_missing_job(self):
        assert _bridge([]).get_result_json(1) is None

    def test_not_completed(self):
        b = _bridge([])
        b._jobs[1] = {"status": "running"}
        assert b.get_result_json(1) is None

    def test_completed(self):
        b = _bridge([])
        b._jobs[1] = {
            "status": "completed",
            "_result_data": {
                "status": "completed",
                "loss": 1.2,
                "eval_loss": 0.8,
                "checkpoint": "models/my-run.ckpt",
                "current_epoch": 4,
            },
        }
        out = b.get_result_json(1)
        assert '"success": true' in out
        assert '"final_loss": 1.2' in out
        assert '"eval_loss": 0.8' in out
        assert '"model_path": "models/my-run.ckpt"' in out
        assert '"checkpoint_name": "my-run.ckpt"' in out
        assert '"epochs_completed": 4' in out

    def test_completed_without_checkpoint(self):
        b = _bridge([])
        b._jobs[1] = {"status": "completed", "_result_data": {"status": "completed"}}
        out = b.get_result_json(1)
        assert '"checkpoint_name": null' in out


class TestJobTracking:
    def test_remove_present(self):
        b = _bridge([])
        b._jobs[1] = {"status": "completed"}
        assert b.remove(1) is True
        assert b._jobs == {}

    def test_remove_missing(self):
        assert _bridge([]).remove(1) is False

    def test_alive_count(self):
        b = _bridge([])
        b._jobs[1] = {"status": "running"}
        b._jobs[2] = {"status": "running"}
        b._jobs[3] = {"status": "completed"}
        assert b.alive_count() == 2


class TestSingleton:
    def test_get_bridge_singleton(self):
        vm_training_bridge._bridge = None
        a = vm_training_bridge.get_bridge()
        b = vm_training_bridge.get_bridge()
        assert a is b
        vm_training_bridge._bridge = None

    def test_bridge_recreated_after_reset(self):
        vm_training_bridge._bridge = None
        a = vm_training_bridge.get_bridge()
        vm_training_bridge._bridge = None
        b = vm_training_bridge.get_bridge()
        assert a is not b
        vm_training_bridge._bridge = None
