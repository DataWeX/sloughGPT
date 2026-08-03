"""Coverage tests for domains/shell/kernel_devices.py."""

import pytest

from domains.shell.kernel_devices import (
    DeviceDriver,
    DeviceHandle,
    DeviceManager,
    DeviceState,
    DeviceTable,
    DeviceType,
    NullDevice,
)


class FakeDevice(DeviceDriver):
    def __init__(self, name="fake", device_type=DeviceType.CUSTOM):
        super().__init__(name, device_type)

    def read(self, offset=0, size=-1):
        return b"payload"

    def write(self, data):
        return True

    def ioctl(self, command, *args):
        return ("ok", command)


class FailingOpenDevice(DeviceDriver):
    def open(self):
        with self._lock:
            self._state = DeviceState.ERROR
            return False


def test_device_type_values():
    assert DeviceType.INFERENCE == 0
    assert DeviceType.TRAINING == 1
    assert DeviceType.STORAGE == 2
    assert DeviceType.NETWORK == 3
    assert DeviceType.DISPLAY == 4
    assert DeviceType.INPUT == 5
    assert DeviceType.CUSTOM == 6


def test_device_state_values():
    assert DeviceState.CLOSED == 0
    assert DeviceState.OPEN == 1
    assert DeviceState.ERROR == 2


def test_device_handle_dataclass():
    h = DeviceHandle(fd=3, device_name="null", mode="rw", offset=2)
    assert h.fd == 3
    assert h.mode == "rw"
    assert h.offset == 2


def test_device_driver_properties():
    d = FakeDevice("d1", DeviceType.INFERENCE)
    assert d.name == "d1"
    assert d.device_type == DeviceType.INFERENCE
    assert d.state == DeviceState.CLOSED
    assert d.is_open is False


def test_device_driver_open_close():
    d = FakeDevice()
    assert d.open() is True
    assert d.state == DeviceState.OPEN
    assert d.is_open is True
    d.close()
    assert d.state == DeviceState.CLOSED
    assert d._open_count == 0


def test_device_driver_open_count_refcount():
    d = FakeDevice()
    d.open()
    d.open()
    assert d._open_count == 2
    d.close()
    assert d._open_count == 1
    assert d.state == DeviceState.OPEN
    d.close()
    assert d.state == DeviceState.CLOSED


def test_device_driver_open_error_state_returns_false():
    d = FakeDevice()
    d._state = DeviceState.ERROR
    assert d.open() is False


def test_device_driver_base_read_not_implemented():
    d = DeviceDriver("base")
    with pytest.raises(NotImplementedError, match="read not implemented"):
        d.read(0, 10)


def test_device_driver_base_write_not_implemented():
    d = DeviceDriver("base")
    with pytest.raises(NotImplementedError, match="write not implemented"):
        d.write(b"x")


def test_device_driver_base_ioctl_not_implemented():
    d = DeviceDriver("base")
    with pytest.raises(NotImplementedError, match="ioctl 'CMD' not implemented"):
        d.ioctl("CMD")


def test_device_driver_info():
    d = FakeDevice("d1", DeviceType.STORAGE)
    d.open()
    info = d.info()
    assert info["name"] == "d1"
    assert info["type"] == "STORAGE"
    assert info["state"] == "OPEN"
    assert info["open_count"] == 1


def test_table_register():
    t = DeviceTable()
    assert t.register(FakeDevice("d1")) is True
    assert t.register(FakeDevice("d1")) is False
    assert t.device_count == 1


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


def test_table_unregister_closes_open_device():
    t = DeviceTable()
    dev = FakeDevice("d1")
    t.register(dev)
    dev.open()
    assert t.unregister("d1") is True
    assert dev.state == DeviceState.CLOSED


def test_table_open_and_close_fd():
    t = DeviceTable()
    dev = FakeDevice("d1")
    t.register(dev)
    h = t.open("d1", mode="rw")
    assert h is not None
    assert h.fd == 1
    assert h.device_name == "d1"
    assert h.mode == "rw"
    assert t.open_fd_count == 1
    assert dev.is_open is True
    assert t.close_fd(h.fd) is True
    assert t.open_fd_count == 0
    assert dev.state == DeviceState.CLOSED


def test_table_open_missing_device():
    t = DeviceTable()
    assert t.open("nope") is None


def test_table_open_failing_device():
    t = DeviceTable()
    t.register(FailingOpenDevice("f"))
    assert t.open("f") is None
    assert t.open_fd_count == 0


def test_table_close_fd_missing():
    t = DeviceTable()
    assert t.close_fd(999) is False


def test_table_read_fd():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    h = t.open("d1")
    assert t.read_fd(h.fd, offset=1, size=2) == b"payload"
    assert t.ioctl_fd(h.fd, "FLUSH", 1, 2) == ("ok", "FLUSH")
    assert t.write_fd(h.fd, b"x") is True


def test_table_read_fd_bad_fd():
    t = DeviceTable()
    with pytest.raises(ValueError, match="Bad file descriptor"):
        t.read_fd(999)


def test_table_read_fd_disconnected():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    h = t.open("d1")
    t.unregister("d1")
    with pytest.raises(ValueError, match="Device disconnected"):
        t.read_fd(h.fd)


def test_table_write_fd_bad_fd():
    t = DeviceTable()
    with pytest.raises(ValueError, match="Bad file descriptor"):
        t.write_fd(999, b"x")


def test_table_write_fd_disconnected():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    h = t.open("d1")
    t.unregister("d1")
    with pytest.raises(ValueError, match="Device disconnected"):
        t.write_fd(h.fd, b"x")


def test_table_ioctl_fd_bad_fd():
    t = DeviceTable()
    with pytest.raises(ValueError, match="Bad file descriptor"):
        t.ioctl_fd(999, "CMD")


def test_table_ioctl_fd_disconnected():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    h = t.open("d1")
    t.unregister("d1")
    with pytest.raises(ValueError, match="Device disconnected"):
        t.ioctl_fd(h.fd, "CMD")


def test_table_list_and_stats():
    t = DeviceTable()
    t.register(FakeDevice("d1"))
    h = t.open("d1")
    listing = t.list_devices()
    assert len(listing) == 1
    assert listing[0]["name"] == "d1"
    stats = t.stats()
    assert stats["total_devices"] == 1
    assert stats["open_fds"] == 1
    assert stats["devices"][0]["name"] == "d1"


def test_device_manager_delegates():
    m = DeviceManager()
    assert m.register(FakeDevice("d1")) is True
    assert m.register(FakeDevice("d1")) is False
    assert m.get("d1").name == "d1"
    h = m.open("d1", mode="rw")
    assert h is not None
    assert m.read(h.fd) == b"payload"
    assert m.write(h.fd, b"x") is True
    assert m.ioctl(h.fd, "CMD") == ("ok", "CMD")
    assert m.list_devices()[0]["name"] == "d1"
    assert m.stats()["total_devices"] == 1
    assert m.close(h.fd) is True
    assert m.unregister("d1") is True
    assert m.unregister("d1") is False


def test_null_device():
    d = NullDevice()
    assert d.name == "null"
    assert d.device_type == DeviceType.CUSTOM
    assert d.read() == b""
    assert d.write(b"anything") is True
