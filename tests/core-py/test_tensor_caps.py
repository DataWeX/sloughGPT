"""NPU CAPS contract — TensorDevice capability matrix (card d74e2cef, phase 1).

The CAPS is the tensor unit's machine-readable contract: what it computes,
on what dtypes, on which engine. ``tensor_device.TensorDevice`` is the ONE
NPU tensor unit with three faces — the CPU's tensor backend
(``vm._default_tensor``), the kernel NPUDevice's ``_compute``, and the
runtime DeviceManager's registered ``"tensor"`` — and every face reaches
CAPS through the same ``ioctl()``/``call()`` path.

Run:
    python -m pytest packages/core-py/tests/test_tensor_caps.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from domain.shell._internal.ioctl import IoctlCommand
from domain.shell._internal.tensor_device import TensorDevice

CAPS_KEYS = {"name", "type", "ops", "dtypes", "backend", "parity"}


class TestCapsCommand:
    """CAPS must exist as a dispatchable IoctlCommand (RED = ImportError)."""

    def test_caps_command_exists(self):
        assert IoctlCommand.CAPS.value == "CAPS"

    def test_caps_in_list_commands(self):
        caps = TensorDevice().caps()
        assert "CAPS" in caps["ops"]
        assert "CAPS" in TensorDevice().list_commands()


class TestCapsReachability:
    """Every face of the one device answers CAPS through its own path."""

    def test_ioctl_enum_dispatch(self):
        res = TensorDevice("tensor").ioctl(IoctlCommand.CAPS)
        assert res.success, res.error
        assert res.value["type"] == "tensor"

    def test_ioctl_string_dispatch(self):
        # DEV_CALL / DeviceTable pass command strings, not enum members.
        res = TensorDevice("tensor").ioctl("CAPS")
        assert res.success, res.error
        assert res.value["name"] == "tensor"

    def test_call_dispatch_is_dev_call_path(self):
        # DeviceBus.call -> device.call(method, *args) is exactly what
        # DEV_CALL executes in the VM.
        caps = TensorDevice("tensor").call("CAPS")
        assert isinstance(caps, dict)

    def test_cpu_default_tensor_backend_answers_caps(self):
        from domain.shell._internal.vm import CPU

        cpu = CPU()
        caps = cpu._tensor.caps()
        assert caps["type"] == "tensor"
        assert "MATMUL" in caps["ops"]

    def test_runtime_device_manager_answers_caps(self):
        from domain.shell._internal.devices import create_default_devices

        mgr = create_default_devices()
        dev = mgr.get("tensor")
        assert dev is not None
        caps = dev.call("CAPS")
        assert caps["type"] == "tensor"

    def test_name_is_device_name(self):
        assert TensorDevice("npu0").caps()["name"] == "npu0"


class TestCapsMatrix:
    """Shape and phase-1 truth of the matrix itself."""

    def test_stable_top_level_keys(self):
        # Phase 2 may fill backend/parity VALUES; the contract keys are fixed.
        assert set(TensorDevice().caps().keys()) == CAPS_KEYS

    def test_ops_match_list_commands_exactly(self):
        # Single source: the advertised op list is the dispatch table,
        # never a hand-maintained second list.
        caps = TensorDevice().caps()
        assert caps["ops"] == TensorDevice().list_commands()

    def test_every_advertised_op_is_dispatchable(self):
        # No advertised op may answer "command not implemented" — the
        # ops list and the _ops dispatch table cannot drift apart.
        td = TensorDevice()
        for cmd in td.list_commands():
            res = td.ioctl(cmd, *(_probe_args(cmd)))
            assert "command not implemented" not in (res.error or ""), cmd

    def test_dtypes_canonical_float64(self):
        caps = TensorDevice().caps()
        assert caps["dtypes"] == ["float64"]
        # The claim is behavioral: scalar/list inputs coerce to float64.
        out = TensorDevice().matmul([[1.0, 2.0]], [[3.0], [4.0]])
        assert out.dtype == np.float64

    def test_backend_reports_reality(self):
        # Phase-1 (numpy, nothing routed) and phase-2 (native-cpu, routed)
        # are both legal snapshots of ONE invariant: engine name,
        # availability, routed ops, and parity must never disagree.
        caps = TensorDevice().caps()
        backend = caps["backend"]
        assert backend["engine"] in ("numpy", "native-cpu")
        routed = backend["native_ops"]
        assert (backend["engine"] == "native-cpu") == (len(routed) > 0)
        assert backend["native_available"] == (len(routed) > 0)
        assert routed == sorted(routed)
        # today every routed op ships with a parity test — drift guard
        assert caps["parity"] == routed

    def test_info_contract_unchanged(self):
        # CAPS is a pure addition — info() keeps its existing shape.
        info = TensorDevice("tensor").info()
        assert info == {"name": "tensor", "type": "tensor", "commands": len(TensorDevice()._ops)}


def _probe_args(cmd: str) -> tuple:
    """Minimal args per op so dispatch reaches the handler.

    The point is only that the command RESOLVES in the dispatch table —
    handlers may still fail on bad input, but never with
    "command not implemented".
    """
    scalar = [1.0, 2.0]
    if cmd in ("MATMUL", "DOT"):
        return (scalar, scalar)
    return (scalar,)


def _has_compiler() -> bool:
    import shutil

    return shutil.which("gcc") is not None or shutil.which("clang") is not None


@pytest.mark.skipif(not _has_compiler(), reason="no C compiler — numpy stays the engine")
def test_phase2_native_backend_present():
    # RED until card d74e2cef phase 2 lands (compiler exists, engine
    # still numpy) — flips green with the routing, no skip on this box.
    backend = TensorDevice().caps()["backend"]
    assert backend["native_available"] is True
    assert backend["engine"] == "native-cpu"
    assert backend["native_ops"]
