"""
Tests for VM device integration — DeviceBusAdapter, DeviceRegisterMap,
DEV_TABLE_*, and DEV_REG_* instructions.

Tests the bridge between standalone VM devices (tensor, npu, storage, etc.)
and the kernel's DeviceTable (fd-based) infrastructure. Covers:
- DeviceBusAdapter: register, open, close, ioctl
- DeviceRegisterMap: memory-mapped device registers
- DEV_TABLE_OPEN/CALL/CLOSE instructions
- DEV_REG_READ/WRITE instructions
- Standalone device auto-registration
- SyscallResult integration
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.vm import (
    Assembler,
    CPU,
    DeviceBus,
    DeviceBusAdapter,
    DeviceRegisterMap,
    VMRunner,
    DeviceFault,
    register_standalone_device,
    set_device_table_adapter,
    set_device_register_map,
    _STANDALONE_DEVICES,
)
from domains.shell.kernel_devices import (
    DeviceType,
    DeviceDriver,
    DeviceTable,
)
from domains.shell.kernel_syscall import SyscallResult
from domains.shell.ioctl import IoctlCommand


# =============================================================================
# Test Helpers
# =============================================================================


class FakeDriver(DeviceDriver):
    """A fake DeviceDriver for testing."""

    def __init__(self, name: str = "fake"):
        super().__init__(name, DeviceType.CUSTOM)
        self._call_count = 0

    def ioctl(self, command: str, *args) -> SyscallResult:
        self._call_count += 1
        if command == "PING":
            return SyscallResult.ok("PONG")
        elif command == "ADD":
            a, b = args[0] if len(args) > 0 else 0, args[1] if len(args) > 1 else 0
            return SyscallResult.ok(a + b)
        elif command == "INFO":
            return SyscallResult.ok({"name": self._name, "calls": self._call_count})
        return SyscallResult.fail(f"unknown command: {command}")


class FakeDevice:
    """A fake standalone device for testing DEV_OPEN auto-registration."""

    def __init__(self, name: str = "fake_standalone"):
        self._name = name
        self._calls = []

    def call(self, method, *args):
        self._calls.append((method, args))
        if method == "echo":
            return args[0] if args else None
        elif method == "add":
            return sum(args)
        return None

    def info(self):
        return {"type": "fake", "name": self._name, "calls": len(self._calls)}


# =============================================================================
# DeviceBusAdapter
# =============================================================================


class TestDeviceBusAdapter:
    """Tests for DeviceBusAdapter — bridges DeviceBus to DeviceTable."""

    @pytest.fixture
    def adapter(self):
        return DeviceBusAdapter(max_fds=64)

    @pytest.fixture
    def driver(self):
        return FakeDriver("test_dev")

    def test_register_device(self, adapter, driver):
        result = adapter.register_device(driver, DeviceType.INFERENCE)
        assert result is True
        assert adapter.get_device("test_dev") is driver

    def test_register_duplicate(self, adapter, driver):
        adapter.register_device(driver)
        result = adapter.register_device(driver)
        assert result is False

    def test_unregister_device(self, adapter, driver):
        adapter.register_device(driver)
        result = adapter.unregister_device("test_dev")
        assert result is True
        assert adapter.get_device("test_dev") is None

    def test_open_device(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        assert fd >= 0
        assert fd_is_valid(adapter, fd)

    def test_open_nonexistent(self, adapter):
        fd = adapter.open("no_such_device")
        assert fd == -1

    def test_close_device(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        assert fd >= 0
        result = adapter.close(fd)
        assert result is True

    def test_ioctl_dispatch(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        result = adapter.ioctl(fd, "PING")
        assert result.success
        assert result.value == "PONG"

    def test_ioctl_with_args(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        result = adapter.ioctl(fd, "ADD", 3, 4)
        assert result.success
        assert result.value == 7

    def test_ioctl_bad_fd(self, adapter, driver):
        adapter.register_device(driver)
        result = adapter.ioctl(999, "PING")
        assert not result.success
        assert "bad fd" in result.error

    def test_ioctl_unknown_command(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        result = adapter.ioctl(fd, "UNKNOWN_CMD")
        assert not result.success

    def test_list_devices(self, adapter, driver):
        adapter.register_device(driver)
        names = adapter.list_devices()
        assert "test_dev" in names

    def test_stats(self, adapter, driver):
        adapter.register_device(driver)
        fd = adapter.open("test_dev")
        stats = adapter.stats()
        assert stats["total_devices"] == 1
        assert stats["open_fds"] == 1


def fd_is_valid(adapter, fd):
    """Check if fd is valid in the adapter's table."""
    return adapter._table._fd_is_open(fd)


# =============================================================================
# DeviceRegisterMap
# =============================================================================


class TestDeviceRegisterMap:
    """Tests for DeviceRegisterMap — memory-mapped device registers."""

    @pytest.fixture
    def reg_map(self):
        return DeviceRegisterMap()

    def test_initial_registers_zero(self, reg_map):
        # All registers should be initialized to 0
        assert reg_map.read(0xF000) == 0
        assert reg_map.read(0xF004) == 0

    def test_read_write_basic(self, reg_map):
        reg_map.write(0xF008, 42)
        assert reg_map.read(0xF008) == 42

    def test_read_unknown_address(self, reg_map):
        assert reg_map.read(0x1234) == 0

    def test_write_unknown_address(self, reg_map):
        # Should not raise
        reg_map.write(0x1234, 99)

    def test_get_block_base(self, reg_map):
        assert reg_map.get_block_base("tensor") == 0xF000
        assert reg_map.get_block_base("npu") == 0xF100
        assert reg_map.get_block_base("storage") == 0xF200

    def test_get_block_base_unknown(self, reg_map):
        assert reg_map.get_block_base("unknown") == 0

    def test_device_status_ready(self, reg_map):
        # Register a device to set READY status
        fake_driver = FakeDriver("reg_test")
        reg_map.register_device("tensor", fake_driver)
        status = reg_map.read(0xF000 + DeviceRegisterMap.REG_STATUS)
        assert status & DeviceRegisterMap.STATUS_READY


# =============================================================================
# VMRunner with DeviceTable Integration
# =============================================================================


class TestVMRunnerDeviceIntegration:
    """Tests for VMRunner with DeviceBusAdapter and DeviceRegisterMap."""

    @pytest.fixture
    def setup(self):
        adapter = DeviceBusAdapter(max_fds=32)
        reg_map = DeviceRegisterMap()
        driver = FakeDriver("runner_dev")
        adapter.register_device(driver, DeviceType.INFERENCE)
        runner = VMRunner(
            device_table_adapter=adapter,
            register_map=reg_map,
        )
        return runner, adapter, reg_map, driver

    def test_runner_has_adapter(self, setup):
        runner, adapter, _, _ = setup
        assert runner._device_table_adapter is adapter

    def test_runner_has_register_map(self, setup):
        runner, _, reg_map, _ = setup
        assert runner._register_map is reg_map

    def test_cpu_gets_adapter(self, setup):
        runner, adapter, _, _ = setup
        source = "NOP\nHALT"
        runner.assemble_and_run(source)
        assert runner.cpu._device_table_adapter is adapter

    def test_cpu_gets_register_map(self, setup):
        runner, _, reg_map, _ = setup
        source = "NOP\nHALT"
        runner.assemble_and_run(source)
        assert runner.cpu._device_register_map is reg_map


# =============================================================================
# DEV_TABLE_* Instructions
# =============================================================================


class TestDevTableInstructions:
    """Tests for DEV_TABLE_OPEN, DEV_TABLE_CALL, DEV_TABLE_CLOSE."""

    @pytest.fixture
    def setup(self):
        adapter = DeviceBusAdapter(max_fds=32)
        driver = FakeDriver("asm_dev")
        adapter.register_device(driver, DeviceType.INFERENCE)
        runner = VMRunner(device_table_adapter=adapter)
        return runner, adapter, driver

    def test_dev_table_open(self, setup):
        runner, adapter, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "asm_dev"
        HALT
        """
        runner.assemble_and_run(source)
        fd = runner.cpu.regs[0]
        assert fd >= 0

    def test_dev_table_open_nonexistent(self, setup):
        runner, _, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "no_such_device"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] == -1

    def test_dev_table_call(self, setup):
        runner, adapter, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "asm_dev"
        DEV_TABLE_CALL R1, R0, "PING"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[1] == "PONG"

    def test_dev_table_call_with_args(self, setup):
        runner, adapter, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "asm_dev"
        DEV_TABLE_CALL R1, R0, "ADD", 10, 20
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[1] == 30

    def test_dev_table_close(self, setup):
        runner, adapter, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "asm_dev"
        DEV_TABLE_CLOSE R0
        HALT
        """
        runner.assemble_and_run(source)
        # Should not raise

    def test_dev_table_info(self, setup):
        runner, adapter, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "asm_dev"
        DEV_TABLE_INFO R1, R0
        HALT
        """
        runner.assemble_and_run(source)
        info = runner.cpu.regs[1]
        assert isinstance(info, dict)
        assert "name" in info


# =============================================================================
# DEV_REG_* Instructions
# =============================================================================


class TestDevRegInstructions:
    """Tests for DEV_REG_READ, DEV_REG_WRITE."""

    @pytest.fixture
    def setup(self):
        reg_map = DeviceRegisterMap()
        runner = VMRunner(register_map=reg_map)
        return runner, reg_map

    def test_dev_reg_read(self, setup):
        runner, reg_map = setup
        # Register a device so READY bit is set
        fake_driver = FakeDriver("reg_asm_dev")
        reg_map.register_device("tensor", fake_driver)
        source = """
        DEV_REG_READ R0, 0xF000
        HALT
        """
        runner.assemble_and_run(source)
        # Should read status register (READY bit set)
        assert runner.cpu.regs[0] & DeviceRegisterMap.STATUS_READY

    def test_dev_reg_write(self, setup):
        runner, reg_map = setup
        source = """
        DEV_REG_WRITE 0xF008, 42
        DEV_REG_READ R0, 0xF008
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] == 42

    def test_dev_reg_write_and_read_multiple(self, setup):
        runner, reg_map = setup
        source = """
        DEV_REG_WRITE 0xF008, 100
        DEV_REG_WRITE 0xF00C, 200
        DEV_REG_READ R0, 0xF008
        DEV_REG_READ R1, 0xF00C
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] == 100
        assert runner.cpu.regs[1] == 200


# =============================================================================
# DEV_OPEN with Standalone Device Auto-Registration
# =============================================================================


class TestDevOpenAutoRegistration:
    """Tests for DEV_OPEN auto-registering standalone devices."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clear standalone device registry after each test."""
        yield
        _STANDALONE_DEVICES.clear()

    def test_dev_open_auto_register(self):
        fake = FakeDevice("auto_dev")
        register_standalone_device("auto_dev", fake)

        runner = VMRunner()
        source = """
        DEV_OPEN R0, "auto_dev"
        DEV_CALL R1, R0, "echo", "hello"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] == "auto_dev"
        assert runner.cpu.regs[1] == "hello"

    def test_dev_open_auto_register_add(self):
        fake = FakeDevice("calc_dev")
        register_standalone_device("calc_dev", fake)

        runner = VMRunner()
        source = """
        DEV_OPEN R0, "calc_dev"
        DEV_CALL R1, R0, "add", 5, 3, 2
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[1] == 10

    def test_dev_open_nonexistent(self):
        runner = VMRunner()
        source = """
        DEV_OPEN R0, "ghost_device"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] == ""

    def test_dev_call_error_handling(self):
        runner = VMRunner()
        source = """
        DEV_OPEN R0, "ghost_device"
        DEV_CALL R1, R0, "anything"
        HALT
        """
        runner.assemble_and_run(source)
        # Should not crash, error logged
        assert runner.cpu.regs[1] is None


# =============================================================================
# Integration: Multiple Devices
# =============================================================================


class TestMultipleDeviceIntegration:
    """Tests for using multiple devices in a single program."""

    @pytest.fixture
    def setup(self):
        adapter = DeviceBusAdapter(max_fds=32)
        driver1 = FakeDriver("dev_a")
        driver2 = FakeDriver("dev_b")
        adapter.register_device(driver1, DeviceType.INFERENCE)
        adapter.register_device(driver2, DeviceType.STORAGE)
        runner = VMRunner(device_table_adapter=adapter)
        return runner, adapter, driver1, driver2

    def test_multiple_device_open(self, setup):
        runner, _, _, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "dev_a"
        DEV_TABLE_OPEN R1, "dev_b"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[0] >= 0
        assert runner.cpu.regs[1] >= 0
        assert runner.cpu.regs[0] != runner.cpu.regs[1]

    def test_multiple_device_call(self, setup):
        runner, _, _, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "dev_a"
        DEV_TABLE_OPEN R1, "dev_b"
        DEV_TABLE_CALL R2, R0, "PING"
        DEV_TABLE_CALL R3, R1, "PING"
        HALT
        """
        runner.assemble_and_run(source)
        assert runner.cpu.regs[2] == "PONG"
        assert runner.cpu.regs[3] == "PONG"

    def test_device_selective_close(self, setup):
        runner, adapter, _, _ = setup
        source = """
        DEV_TABLE_OPEN R0, "dev_a"
        DEV_TABLE_OPEN R1, "dev_b"
        DEV_TABLE_CLOSE R0
        HALT
        """
        runner.assemble_and_run(source)
        # dev_a fd should be closed, dev_b still open
        assert not fd_is_valid(adapter, runner.cpu.regs[0])
        assert fd_is_valid(adapter, runner.cpu.regs[1])


# =============================================================================
# ISA Documentation
# =============================================================================


class TestISADocumentation:
    """Tests that new instructions are documented in OPCODES."""

    def test_dev_table_open_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_TABLE_OPEN" in OPCODES
        assert "fd" in OPCODES["DEV_TABLE_OPEN"].lower()

    def test_dev_table_call_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_TABLE_CALL" in OPCODES

    def test_dev_table_close_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_TABLE_CLOSE" in OPCODES

    def test_dev_table_info_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_TABLE_INFO" in OPCODES

    def test_dev_reg_read_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_REG_READ" in OPCODES

    def test_dev_reg_write_documented(self):
        from domains.shell.vm import OPCODES
        assert "DEV_REG_WRITE" in OPCODES

    def test_all_new_instructions_in_opcode_table(self):
        from domains.shell.vm import _OPCODE_TABLE
        new_instructions = [
            "DEV_TABLE_OPEN", "DEV_TABLE_CALL", "DEV_TABLE_CLOSE",
            "DEV_TABLE_INFO", "DEV_REG_READ", "DEV_REG_WRITE",
        ]
        for instr in new_instructions:
            assert instr in _OPCODE_TABLE, f"{instr} not in _OPCODE_TABLE"
