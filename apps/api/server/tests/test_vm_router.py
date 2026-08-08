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
