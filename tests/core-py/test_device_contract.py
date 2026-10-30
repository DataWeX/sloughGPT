"""One device contract, four seams.

Devices used to be reachable through exactly one interface, and which one
depended on which class the device happened to inherit:

======================  ========  ======  ======  ======
base                    call()    ioctl() info()  seam
======================  ========  ======  ======  ======
``vm.Device``           yes       no      yes     DeviceBus
``DeviceDriver``        no        yes     yes     DeviceManager (fd)
``AIDevice``            no        no      no      devices.DeviceManager (path)
standalone (npu, ...)   sometimes sometimes yes    both
======================  ========  ======  ======  ======

The result was that ``vm.py``'s DEV_CALL could raise ``AttributeError`` on a
``DeviceManager`` (no ``call``), ``DeviceManager.list_devices()`` crashed with
``AttributeError`` on any registered AI device (no ``info``), and a failure
surfaced as a bare ``Exception`` from one device and a ``SyscallResult`` from
another, depending on which seam you happened to be standing on.

The contract these tests pin down has two faces of the same operation:

- ``call(method, *args)`` - name-based, returns the value, raises
  ``DeviceFault`` on failure. This is the guest/VM face.
- ``ioctl(command, *args)`` - returns a ``SyscallResult``, never raises. This
  is the kernel face.

Every device exposes both, every seam offers ``call``/``info``, and the case
dialect (lowercase names in the VM, uppercase IoctlCommand in the kernel) is
translated once at the boundary rather than relearned per device.
"""

from __future__ import annotations

import pytest

from domain.shell._internal.devices import DeviceManager as AIManager
from domain.shell._internal.devices import NullDevice, RandomDevice, create_default_devices
from domain.shell._internal.kernel_devices import DeviceDriver
from domain.shell._internal.kernel_devices import DeviceManager as KernelManager
from domain.shell._internal.kernel_devices import NullDevice as KernelNull
from domain.shell._internal.kernel_npu import NPUDevice as KernelNPU
from domain.shell._internal.kernel_syscall import SyscallResult
from domain.shell._internal.vm import Device, DeviceFault

# ── Seam 1: fd-based DeviceManager over DeviceDrivers ──────────────────────


class ExplodingDriver(DeviceDriver):
    """A driver whose ioctl raises instead of reporting - contract violation."""

    def ioctl(self, command, *args):
        raise RuntimeError("driver blew up")


def test_driver_call_unwraps_successful_ioctl():
    driver = KernelNPU("knpu")
    assert driver.call("info")["device"] == "knpu"


def test_driver_call_translates_lowercase_dialect():
    """call("info") and ioctl("INFO") are the same operation, not two spells."""
    driver = KernelNPU("knpu")
    assert driver.call("info") == driver.ioctl("INFO").value


def test_driver_call_raises_device_fault_on_failure():
    driver = KernelNPU("knpu")
    with pytest.raises(DeviceFault, match="unknown ioctl"):
        driver.call("no_such_op")


def test_driver_call_contains_a_leaking_ioctl():
    """A driver that raises out of ioctl becomes a DeviceFault, not a RuntimeError."""
    driver = ExplodingDriver("boom")
    with pytest.raises(DeviceFault, match="raised"):
        driver.call("anything")


def test_null_driver_swallows_every_command():
    """The kernel's NullDevice accepts any command and returns empty.

    It is a black hole, so success-and-empty is its contract rather than
    failure - which is why it must still be reachable through call() like
    every other driver, instead of only through ioctl().
    """
    assert KernelNull().call("anything") == b""
    assert KernelNull().call("load_model", "/x", "y") == b""


def test_ioctl_face_never_raises_on_the_same_driver():
    driver = KernelNPU("knpu")
    result = driver.ioctl("NO_SUCH_OP")
    assert isinstance(result, SyscallResult)
    assert result.success is False


def test_device_manager_call_and_ioctl_share_a_handle():
    mgr = KernelManager()
    mgr.register(KernelNPU("knpu"))
    fd = mgr.open("knpu")
    assert fd >= 0

    assert mgr.call(fd, "info")["device"] == "knpu"
    assert mgr.ioctl(fd, "INFO").success is True
    assert mgr.info(fd)["device"] == "knpu"


def test_device_manager_call_on_bad_fd_raises():
    mgr = KernelManager()
    with pytest.raises(DeviceFault, match="bad fd"):
        mgr.call(4242, "info")


def test_device_manager_info_on_bad_fd_is_empty():
    assert KernelManager().info(4242) == {}


# REMOVED (card 33773fce): test_slnc_only_rejection_is_a_fault_not_a_swallowed_result
# asserted the '.slnc is the NPU's only model format' contract from stranded
# 11c3b870f, which never landed — origin/main's KernelNPU.load accepts any
# path. Reintroduce with that port, not before.


# ── Seam 2: vm.Device / DeviceBus ──────────────────────────────────────────


class SampleDevice(Device):
    def call(self, method, *args):
        if method == "boom":
            raise DeviceFault("boom")
        if method == "explode":
            raise ValueError("not a fault")
        return {"method": method, "args": args}


def test_vm_device_ioctl_wraps_call_as_ok_result():
    result = SampleDevice().ioctl("ping", 1)
    assert isinstance(result, SyscallResult)
    assert result.success is True
    assert result.value == {"method": "ping", "args": (1,)}


def test_vm_device_ioctl_converts_a_fault():
    result = SampleDevice().ioctl("boom")
    assert result.success is False
    assert "boom" in result.error


def test_vm_device_ioctl_contains_other_exceptions():
    """ioctl reports; it does not raise - even for a bug in the device."""
    result = SampleDevice().ioctl("explode")
    assert result.success is False
    assert "not a fault" in result.error


def test_vm_device_still_raises_from_call():
    with pytest.raises(DeviceFault, match="boom"):
        SampleDevice().call("boom")


# ── Seam 3: AI path devices (/dev/llm) ─────────────────────────────────────


def test_ai_device_list_devices_renders_every_node():
    """list_devices() called .info() on devices that had none - AttributeError."""
    mgr = AIManager()
    mgr.register(NullDevice())
    mgr.register(RandomDevice())
    rendered = mgr.list_devices()
    assert "/dev/null" in rendered
    assert "/dev/random" in rendered


def test_ai_device_exposes_the_full_contract():
    dev = NullDevice()
    assert dev.info()["name"] == "null"
    assert dev.call("info") == dev.info()
    assert dev.call("write", "discarded") == ""
    assert dev.call("read") == ""


def test_ai_device_call_reports_unsupported_operations():
    with pytest.raises(DeviceFault, match="does not support"):
        NullDevice().call("quantize")


def test_ai_device_call_requires_data_for_write():
    with pytest.raises(DeviceFault, match="write requires data"):
        NullDevice().call("write")


def test_ai_device_ioctl_face():
    result = RandomDevice().ioctl("NOPE")
    assert isinstance(result, SyscallResult)
    assert result.success is False


# ── Seam 4: the path seam's own call/info, by name ─────────────────────────


def test_path_seam_call_by_name_mirrors_the_other_seams():
    mgr = AIManager()
    mgr.register(NullDevice())
    assert mgr.call("null", "info")["name"] == "null"
    assert mgr.info("null")["name"] == "null"


def test_path_seam_call_on_unknown_node_raises():
    with pytest.raises(DeviceFault, match="no such device"):
        AIManager().call("ghost", "info")


def test_path_seam_info_on_unknown_node_is_empty():
    assert AIManager().info("ghost") == {}


# ── The whole registry satisfies the contract ──────────────────────────────


def test_every_registered_device_exposes_the_contract():
    mgr = create_default_devices(get_kernel=lambda: None)
    assert mgr.names, "no devices registered"
    for name in mgr.names:
        device = mgr.get(name)
        for face in ("call", "ioctl", "info"):
            assert hasattr(device, face), f"{name} is missing {face}()"
        fd = mgr.open(name)
        assert fd >= 0, f"{name} failed to open"
        assert isinstance(mgr.info(fd), dict), f"{name} info(fd) is not a dict"
        assert mgr.close(fd) is True


def test_every_registered_device_speaks_both_faces():
    """The two conventions must not diverge per device: a failure is a
    DeviceFault from call() and a failed SyscallResult from ioctl()."""
    mgr = create_default_devices(get_kernel=lambda: None)
    for name in mgr.names:
        device = mgr.get(name)
        try:
            device.call("definitely_not_a_command")
            pytest.fail(f"{name} accepted a bogus command")
        except DeviceFault:
            pass

        result = device.ioctl("definitely_not_a_command")
        assert isinstance(result, SyscallResult), f"{name} ioctl leaked {type(result)}"
        assert result.success is False
