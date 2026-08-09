"""
Tests for the VM router endpoints.

Tests the /vm/run, /vm/builtins, and /vm/info endpoints.
"""
import sys
import os

# Ensure the server directory is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from apps.api.server.main import app
from routers.vm import router as vm_router

app.include_router(vm_router)
client = TestClient(app)


class TestVMRouter:
    """Tests for /vm/* endpoints."""

    def test_run_hello(self):
        """POST /vm/run assembles and executes hello program."""
        source = "[BITS 32]\nMOV EAX, 42\nHLT"
        resp = client.post("/vm/run", json={"source": source})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["status"] == "halted"
        assert data["steps_executed"] > 0
        assert data["exit_code"] == 42
        assert len(data["registers"]) == 8

    def test_run_with_debug(self):
        """POST /vm/run with debug includes trace."""
        source = "[BITS 32]\nNOP\nHLT"
        resp = client.post("/vm/run", json={"source": source, "debug": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["trace"] is not None
        assert len(data["trace"]) > 0

    def test_run_empty_source(self):
        """POST /vm/run with empty source."""
        resp = client.post("/vm/run", json={"source": ""})
        assert resp.status_code == 200
        data = resp.json()
        # Empty source may succeed or fail depending on assembler behavior
        assert "success" in data

    def test_run_with_keyboard_input(self):
        """POST /vm/run with keyboard input."""
        source = "[BITS 32]\nHLT"
        resp = client.post("/vm/run", json={"source": source, "keyboard_input": "abc"})
        assert resp.status_code == 200
        data = resp.json()
        # Keyboard input may fail if devices aren't available
        assert "success" in data

    def test_run_vga_output(self):
        """POST /vm/run with VGA text buffer write returns vga_cells."""
        source = """[BITS 32]
MOV BYTE [0xB8000], 0x48
MOV BYTE [0xB8001], 0x07
HLT"""
        resp = client.post("/vm/run", json={"source": source})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["vga_cells"] is not None
        assert len(data["vga_cells"]) == 80 * 25
        assert data["vga_cells"][0]["ch"] == "H"

    def test_run_max_steps_limit(self):
        """POST /vm/run respects max_steps limit."""
        source = "[BITS 32]\nJMP 0x1000"
        resp = client.post("/vm/run", json={"source": source, "max_steps": 100})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["steps_executed"] <= 100

    def test_run_admin_role(self):
        """POST /vm/run with admin role."""
        source = "[BITS 32]\nMOV EAX, 1\nHLT"
        resp = client.post("/vm/run", json={"source": source, "role": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_builtins(self):
        """GET /vm/builtins returns list of programs."""
        resp = client.get("/vm/builtins")
        assert resp.status_code == 200
        data = resp.json()
        assert "programs" in data
        assert len(data["programs"]) >= 8
        names = [p["name"] for p in data["programs"]]
        assert "hello" in names
        assert "count" in names
        assert "fib" in names
        assert "train" in names
        assert "train-status" in names

    def test_info(self):
        """GET /vm/info returns VM capabilities."""
        resp = client.get("/vm/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["isa"] == "x86-32"
        assert "EAX" in data["registers"]
        assert data["max_steps"] > 0
        assert data["default_memory"] > 0
        assert len(data["features"]) > 0

    def test_run_captures_training_job_id(self, monkeypatch):
        """SYS_TRAIN_START in assembly surfaces job_id in the run response."""

        class FakeBridge:
            def start(self, config_json):
                return 1

        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: FakeBridge())
        source = """[BITS 32]
MOV EBX, cfg
MOV EAX, 28
INT 0x80
HLT

cfg: db '{}', 0"""
        resp = client.post("/vm/run", json={"source": source, "role": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["training_job_id"] == 1

    def test_run_train_denied_for_user(self, monkeypatch):
        """USER-role SYS_TRAIN_START is denied: no job captured, EAX=-2 in registers."""
        calls = []

        class FakeBridge:
            def start(self, config_json):
                calls.append(config_json)
                return 1

        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: FakeBridge())
        source = """[BITS 32]
MOV EBX, cfg
MOV EAX, 28
INT 0x80
HLT

cfg: db '{}', 0"""
        resp = client.post("/vm/run", json={"source": source, "role": "user"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["training_job_id"] is None
        assert calls == [], "bridge must not be invoked for a USER-role training syscall"
        eax = next(r for r in data["registers"] if r["name"] == "EAX")
        assert eax["hex"] == "0xFFFFFFFE", f"expected EAX=-2 (denied), got {eax['hex']}"

    def test_run_training_job_id_null_without_syscall(self):
        """Runs without SYS_TRAIN_START leave training_job_id null."""
        source = "[BITS 32]\nMOV EAX, 42\nHLT"
        resp = client.post("/vm/run", json={"source": source, "role": "admin"})
        assert resp.status_code == 200
        assert resp.json()["training_job_id"] is None

    def test_run_captures_training_result(self, monkeypatch):
        """SYS_TRAIN_GET_RESULT bytes are surfaced as training_result in the response."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        bridge._jobs[1] = {
            "api_job_id": "abc-123",
            "status": "completed",
            "progress": 1.0,
            "_result_data": {"status": "completed", "loss": 1.5, "current_epoch": 2},
        }
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        source = """[BITS 32]
MOV EBX, 1
MOV EAX, 29
INT 0x80
MOV [0x5000], EAX
MOV EBX, 1
MOV EAX, 30
MOV ECX, 0x90000
MOV EDX, 256
INT 0x80
HLT"""
        resp = client.post("/vm/run", json={"source": source, "role": "admin"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["training_result"] is not None, "expected result JSON read back from guest memory"
        assert "final_loss" in data["training_result"]
        assert "1.5" in data["training_result"]

    def test_run_training_result_null_without_get_result(self, monkeypatch):
        """Runs without SYS_TRAIN_GET_RESULT leave training_result null."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        bridge._jobs[1] = {
            "api_job_id": "abc-123",
            "status": "completed",
            "progress": 1.0,
            "_result_data": {"status": "completed", "loss": 1.5},
        }
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        source = "[BITS 32]\nMOV EBX, 1\nMOV EAX, 29\nINT 0x80\nHLT"
        resp = client.post("/vm/run", json={"source": source, "role": "admin"})
        assert resp.status_code == 200
        assert resp.json()["training_result"] is None

    def test_training_job_status_endpoint(self, monkeypatch):
        """GET /vm/training/jobs/{id} returns bridge-tracked job status."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        bridge._jobs[7] = {
            "api_job_id": "abc-123",
            "status": "completed",
            "progress": 1.0,
            "_result_data": {"status": "completed", "loss": 1.2},
        }
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        resp = client.get("/vm/training/jobs/7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == 7
        assert data["api_job_id"] == "abc-123"
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert data["result"] is not None, "completed jobs surface result JSON"
        assert "final_loss" in data["result"]

    def test_training_job_status_result_null_when_not_completed(self, monkeypatch):
        """GET /vm/training/jobs/{id} leaves result null for running jobs."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        bridge._jobs[7] = {
            "api_job_id": "abc-123",
            "status": "running",
            "progress": 0.5,
            "_result_data": {"status": "completed", "loss": 1.2},
        }
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        resp = client.get("/vm/training/jobs/7")
        assert resp.status_code == 200
        assert resp.json()["result"] is None

    def test_training_job_status_404(self, monkeypatch):
        """GET /vm/training/jobs/{id} returns 404 for unknown job."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        resp = client.get("/vm/training/jobs/999")
        assert resp.status_code == 404

    def test_training_job_stop_endpoint(self, monkeypatch):
        """POST /vm/training/jobs/{id}/stop delegates to the bridge."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        bridge._jobs[7] = {"api_job_id": "api-7", "status": "running"}
        bridge.stop = lambda job_id: True  # noqa: E731 — avoid network in wiring test
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)

        resp = client.post("/vm/training/jobs/7/stop")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == 7
        assert resp.json()["status"] == "stopping"

    def test_training_job_stop_404(self, monkeypatch):
        """POST /vm/training/jobs/{id}/stop returns 404 for unknown job."""
        from domains.shell.vm_training_bridge import VMTrainingBridge

        bridge = VMTrainingBridge()
        monkeypatch.setattr("domains.shell.vm_training_bridge.get_bridge", lambda: bridge)
        bridge.stop = lambda job_id: False  # noqa: E731

        resp = client.post("/vm/training/jobs/999/stop")
        assert resp.status_code == 404
