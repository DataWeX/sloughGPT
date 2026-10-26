"""Coverage tests for domain.shell._internal.kernel_devices.

Rewritten for the baseboard contract introduced by 96b0d69d1 ("bit-based
DeviceTable + DeviceDriver gates"): drivers expose ``ioctl`` + ``info`` only —
open/close/read/write are the table's job over integer fds, ``DeviceType``
members are bit flags, and ioctl answers with a ``SyscallResult`` instead of
raising. The previous revision of this file still asserted the pre-rewrite API
(``DeviceDriver.open``, ``read_fd``, ``write_fd``, ``ioctl_fd``, ``close_fd``,
``device_count``) and failed 24/24 on AttributeError.
"""

from __future__ import annotations

import pytest

from domain.shell._internal.kernel_devices import (
    DeviceDriver,
    DeviceHandle,
    DeviceManager,
    DeviceState,
    DeviceTable,
    DeviceType,
    NullDevice,
)
from domain.shell._internal.kernel_syscall import SyscallResult


class FakeDevice(DeviceDriver):
    """Minimal driver under the current contract: implement ioctl, nothing else."""

    def __init__(self, name="fake", device_type=DeviceType.CUSTOM):
        super().__init__(name, device_type)

    def ioctl(self, command, *args):
        return SyscallResult.ok({"cmd": command, "args": args})


# ── Enum / dataclass contracts ─────────────────────────────────────────────


def test_device_type_values_are_bit_flags():
    assert DeviceType.INFERENCE == 0b000001
    assert DeviceType.TRAINING == 0b000010
    assert DeviceType.STORAGE == 0b000100
    assert DeviceType.NETWORK == 0b001000
    assert DeviceType.DISPLAY == 0b010000
    assert DeviceType.INPUT == 0b100000
    assert DeviceType.CUSTOM == 0  # "no type"
    # masks compose — a device may be several categories at once
    assert DeviceType.INFERENCE | DeviceType.TRAINING == 0b000011


def test_device_state_values():
    assert DeviceState.CLOSED == 0
    assert DeviceState.OPEN == 1
    assert DeviceState.ERROR == 2


def test_device_handle_dataclass():
    h = DeviceHandle(fd=3, device_name="null", mode="rw", offset=2)
    assert h.fd == 3
    assert h.mode == "rw"
    assert h.offset == 2


# ── Driver (gates) ─────────────────────────────────────────────────────────


def test_device_driver_properties():
    d = FakeDevice("d1", DeviceType.INFERENCE)
    assert d.name == "d1"
    assert d.device_type == DeviceType.INFERENCE
    assert d.state == DeviceState.CLOSED


def test_device_driver_defers_io_state_to_the_table():
    # DeviceDriver: "Only ioctl. No open/close — that's DeviceTable's job."
    # The pre-baseboard driver tracked is_open/_open_count/read/write itself;
    # letting that back in would fork fd bookkeeping out of the bitmap again.
    d = FakeDevice()
    for attr in ("open", "close", "is_open", "read", "write"):
        assert not hasattr(d, attr), f"{attr} belongs to DeviceTable now"


def test_device_driver_base_ioctl_not_implemented():
    d = DeviceDriver("base")
    with pytest.raises(NotImplementedError, match="ioctl 'CMD' not implemented"):
        d.ioctl("CMD")


def test_device_driver_info():
    d = FakeDevice("d1", DeviceType.STORAGE)
    info = d.info()
    assert info["name"] == "d1"
    assert info["type"] == DeviceType.STORAGE
    assert info["state"] == "CLOSED"
    # open-count is fd-table state now, not driver state
    assert "open_count" not in info


# ── Table registration ─────────────────────────────────────────────────────


def test_table_register():
    t = DeviceTable()
    assert t.register(FakeDevice("d1")) is True
    assert t.register(FakeDevice("d1")) is False
    assert t.stats()["total_devices"] == 1
    assert len(t.list_devices()) == 1


def test_table_get():
    t = DeviceTable()
    assert t.get("missing") is None
    dev = FakeDevice("d1")
    t.register(dev)
    assert t.get("d1") is dev


def test_table_unregister():
    t = DeviceTable()
    assert t.unregister("missing") is False
    dev = FakeDevice("d1")
    t.register(dev)
    assert t.unregister("d1") is True
    assert t.get("d1") is None
    assert t.unregister("d1") is False  # already gone


def test_table_unregister_frees_open_fds():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    fd = t.open("d1")
    assert fd >= 0
    assert t.stats()["open_fds"] == 1
    assert t.unregister("d1") is True
    # a dangling fd must not dispatch into a device the table no longer manages
    assert t.stats()["open_fds"] == 0
    res = t.ioctl(fd, "CMD")
    assert res.success is False
    assert res.error == "fd not open"


def test_table_unregister_keeps_fds_alive_while_an_alias_survives():
    t = DeviceTable()
    t.register(FakeDevice("ai"))
    assert t.alias("llm", "ai") is True
    fd = t.open("llm")
    assert fd >= 0
    assert t.unregister("llm") is True
    # same driver instance is still reachable as "ai" — the fd stays valid
    assert t.stats()["open_fds"] == 1
    assert t.ioctl(fd, "CMD").success is True
    # ...until the last name goes
    assert t.unregister("ai") is True
    assert t.stats()["open_fds"] == 0


# ── Table fd operations ────────────────────────────────────────────────────


def test_table_open_and_close_fd():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    fd = t.open("d1", mode=0)
    assert isinstance(fd, int)
    assert fd >= 0
    assert t.stats()["open_fds"] == 1
    assert t.close(fd) is True
    assert t.close(fd) is False  # double close is not success
    assert t.stats()["open_fds"] == 0


def test_table_open_missing_device():
    t = DeviceTable()
    assert t.open("nope") == -1


def test_table_open_fd_exhaustion():
    t = DeviceTable(max_fds=4)
    t.register(FakeDevice("d1"))
    fds = [t.open("d1") for _ in range(4)]
    assert fds == [0, 1, 2, 3]  # bitmap hands out the first free bits
    assert t.open("d1") == -1  # bitmap full
    assert t.stats()["open_fds"] == 4


def test_table_close_fd_missing():
    t = DeviceTable()
    assert t.close(999) is False


# ── Table ioctl dispatch ───────────────────────────────────────────────────


def test_table_ioctl_dispatch():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    fd = t.open("d1")
    res = t.ioctl(fd, "FLUSH", 1, 2)
    assert res.success is True
    assert res.value == {"cmd": "FLUSH", "args": (1, 2)}
    assert t.close(fd) is True


def test_table_ioctl_bad_fd():
    t = DeviceTable()
    res = t.ioctl(999, "CMD")
    assert res.success is False
    assert res.error == "bad fd"


def test_table_ioctl_closed_fd():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    fd = t.open("d1")
    t.close(fd)
    res = t.ioctl(fd, "CMD")
    assert res.success is False
    assert res.error == "fd not open"


# ── Table info ─────────────────────────────────────────────────────────────


def test_table_list_and_stats():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    t.open("d1")
    listing = t.list_devices()
    assert len(listing) == 1
    assert listing[0]["name"] == "d1"
    stats = t.stats()
    assert stats["total_devices"] == 1
    assert stats["open_fds"] == 1
    assert stats["devices"][0]["name"] == "d1"


# ── Manager (DeviceTable wrapper) ──────────────────────────────────────────


def test_device_manager_delegates():
    m = DeviceManager()
    first = FakeDevice("d1")
    assert m.register(first) is True
    assert m.register(FakeDevice("d1")) is False
    # a rejected duplicate must not clobber the manager's registry
    assert m.get("d1") is first
    fd = m.open("d1")
    assert fd >= 0
    assert m.ioctl(fd, "CMD").success is True
    assert m.list_devices()[0]["name"] == "d1"
    assert m.stats()["total_devices"] == 1
    assert m.close(fd) is True
    assert m.unregister("d1") is True
    assert m.unregister("d1") is False


# ── Null device ────────────────────────────────────────────────────────────


def test_null_device():
    d = NullDevice()
    assert d.name == "null"
    assert d.device_type == DeviceType.CUSTOM
    res = d.ioctl("ANY")
    assert res.success is True
    assert res.value == b""  # writes discarded, reads come back empty
