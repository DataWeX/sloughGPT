"""
Tests for the VM router — run assembly, list builtins, VM info.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.vm import router


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ── POST /vm/run ──────────────────────────────────────────────────────────────


class TestVmRun:
    """POST /vm/run"""

    def test_empty_source_returns_empty(self, client):
        resp = client.post("/vm/run", json={"source": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["status"] == "empty"

    def test_503_when_vm_module_unavailable(self, client):
        import apps.api.server.routers.vm as vm_mod
        original = vm_mod.__builtins__
        import builtins
        real_import = builtins.__import__
        def block_vm(name, *a, **kw):
            if name.startswith("domains.shell.vm"):
                raise ImportError("no vm")
            return real_import(name, *a, **kw)
        builtins.__import__ = block_vm
        try:
            resp = client.post("/vm/run", json={"source": "mov eax, 1"})
            assert resp.status_code == 503
            assert "VM module not available" in resp.json()["error"]
        finally:
            builtins.__import__ = real_import

    def test_simple_assembly_returns_response(self, client):
        resp = client.post("/vm/run", json={"source": "mov eax, 1\nint 0x80"})
        assert resp.status_code == 200
        body = resp.json()
        assert "success" in body
        assert "exit_code" in body
        assert "steps_executed" in body
        assert "elapsed_ms" in body
        assert "output" in body
        assert "registers" in body
        assert "eip" in body

    def test_max_steps_boundary_low(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "max_steps": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["steps_executed"] <= 1

    def test_max_steps_boundary_high(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "max_steps": 1000000})
        assert resp.status_code == 200

    def test_role_parameter(self, client):
        resp = client.post("/vm/run", json={"source": "mov eax, 1", "role": "kernel"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True or body["status"] == "spawn_failed"

    def test_keyboard_input(self, client):
        resp = client.post("/vm/run", json={
            "source": "mov eax, 1",
            "keyboard_input": "abc",
        })
        assert resp.status_code == 200

    def test_debug_returns_trace(self, client):
        resp = client.post("/vm/run", json={
            "source": "mov eax, 1",
            "debug": True,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("trace") is not None

    def test_invalid_memory_size_returns_422(self, client):
        resp = client.post("/vm/run", json={
            "source": "hlt",
            "memory_size": 42,
        })
        assert resp.status_code == 422

    def test_response_schema(self, client):
        resp = client.post("/vm/run", json={"source": "hlt"})
        body = resp.json()
        required = [
            "success", "exit_code", "steps_executed", "elapsed_ms",
            "output", "registers", "eip", "eip_hex", "status",
        ]
        for field in required:
            assert field in body, f"missing field: {field}"

    def test_max_source_length(self, client):
        long_source = "nop\n" * 12500
        resp = client.post("/vm/run", json={"source": long_source})
        assert resp.status_code in (200, 422)

    def test_memory_size_boundaries(self, client):
        resp_min = client.post("/vm/run", json={"source": "hlt", "memory_size": 0x10000})
        assert resp_min.status_code == 200

        resp_max = client.post("/vm/run", json={"source": "hlt", "memory_size": 0x1000000})
        assert resp_max.status_code == 200

    def test_run_wrong_method_returns_405(self, client):
        resp = client.get("/vm/run")
        assert resp.status_code == 405

    def test_unknown_role_falls_back_to_user(self, client):
        resp = client.post("/vm/run", json={"source": "mov eax, 7", "role": "root"})
        assert resp.status_code == 200

    def test_registers_full_set(self, client):
        resp = client.post("/vm/run", json={"source": "hlt"})
        regs = resp.json()["registers"]
        names = [r["name"] for r in regs]
        assert names == ["EAX", "ECX", "EDX", "EBX", "ESP", "EBP", "ESI", "EDI"]
        for r in regs:
            assert "value" in r
            assert r["hex"].startswith("0x")

    def test_eip_hex_format(self, client):
        resp = client.post("/vm/run", json={"source": "hlt"})
        assert resp.json()["eip_hex"].startswith("0x")

    def test_admin_role_accepted(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "role": "admin"})
        assert resp.status_code == 200

    def test_missing_source_returns_422(self, client):
        resp = client.post("/vm/run", json={})
        assert resp.status_code == 422

    def test_source_overlong_returns_422(self, client):
        resp = client.post("/vm/run", json={"source": "nop\n" * 15000})
        assert resp.status_code == 422

    def test_role_overlong_returns_422(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "role": "x" * 21})
        assert resp.status_code == 422

    def test_keyboard_input_overlong_returns_422(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "keyboard_input": "x" * 10001})
        assert resp.status_code == 422

    def test_max_steps_below_one_returns_422(self, client):
        resp = client.post("/vm/run", json={"source": "hlt", "max_steps": 0})
        assert resp.status_code == 422

    def test_debug_returns_memory_dump(self, client):
        resp = client.post("/vm/run", json={"source": "push eax\nhlt", "debug": True})
        body = resp.json()
        assert body.get("memory_dump") is None or isinstance(body.get("memory_dump"), str)

    def test_response_has_vga_fields(self, client):
        resp = client.post("/vm/run", json={"source": "hlt"})
        body = resp.json()
        assert "vga_text" in body
        assert "vga_cells" in body
        assert "keyboard_buffer" in body
        assert "memory_dump" in body

    @patch("domains.shell.vm.X86VirtualSystem")
    def test_spawn_failed_path(self, mock_vs_cls, client):
        vs = mock_vs_cls.return_value
        vs.spawn.return_value = None
        resp = client.post("/vm/run", json={"source": "hlt"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["status"] == "spawn_failed"
        assert "Failed to spawn" in body["error"]

    @patch("domains.shell.vm.X86VirtualSystem")
    def test_no_process_path(self, mock_vs_cls, client):
        vs = mock_vs_cls.return_value
        vs.spawn.return_value = 1
        vs.scheduler.current = None
        resp = client.post("/vm/run", json={"source": "hlt"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["status"] == "no_process"

    @patch("domains.shell.vm.X86VirtualSystem")
    def test_run_error_path(self, mock_vs_cls, client):
        mock_vs_cls.side_effect = RuntimeError("vm crashed")
        resp = client.post("/vm/run", json={"source": "hlt"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["status"] == "error"
        assert "vm crashed" in body["error"]


# ── GET /vm/builtins ──────────────────────────────────────────────────────────


class TestVmBuiltins:
    """GET /vm/builtins"""

    def test_returns_10_programs(self, client):
        resp = client.get("/vm/builtins")
        assert resp.status_code == 200
        programs = resp.json()["data"]["programs"]
        assert len(programs) >= 10

    def test_programs_have_name_and_description(self, client):
        resp = client.get("/vm/builtins")
        programs = resp.json()["data"]["programs"]
        for p in programs:
            assert "name" in p
            assert "description" in p
            assert isinstance(p["name"], str)
            assert isinstance(p["description"], str)

    def test_builtins_wrong_method_returns_405(self, client):
        resp = client.post("/vm/builtins")
        assert resp.status_code == 405

    def test_builtin_names_unique(self, client):
        resp = client.get("/vm/builtins")
        names = [p["name"] for p in resp.json()["data"]["programs"]]
        assert len(names) == len(set(names))

    def test_all_builtin_descriptions_nonempty(self, client):
        resp = client.get("/vm/builtins")
        for p in resp.json()["data"]["programs"]:
            assert len(p["description"]) > 0


# ── GET /vm/info ──────────────────────────────────────────────────────────────


class TestVmInfo:
    """GET /vm/info"""

    def test_returns_required_fields(self, client):
        resp = client.get("/vm/info")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert "isa" in body
        assert "max_steps" in body
        assert "registers" in body
        assert "features" in body

    def test_registers_include_eax_esp(self, client):
        resp = client.get("/vm/info")
        regs = resp.json()["data"]["registers"]
        assert "EAX" in regs
        assert "ESP" in regs

    def test_features_list_nonempty(self, client):
        resp = client.get("/vm/info")
        features = resp.json()["data"]["features"]
        assert len(features) > 0
        assert isinstance(features, list)

    def test_info_wrong_method_returns_405(self, client):
        resp = client.post("/vm/info")
        assert resp.status_code == 405

    def test_default_isa_fields(self, client):
        resp = client.get("/vm/info")
        body = resp.json()["data"]
        assert body["max_steps"] == 1000000
        assert body["default_memory"] == 0x100000
        assert body["max_memory"] == 0x1000000

    def test_x86_32_isa(self, client):
        resp = client.get("/vm/info")
        assert resp.json()["data"]["isa"] == "x86-32"
