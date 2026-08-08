"""
Coverage-completion tests for the VM device layer (domains/shell/vm.py).

Instantiates Memory, Device, and each device driver directly and exercises the
specific branches/error paths that were previously uncovered.  Fast — no CPU
instruction loops.
"""

import pytest

from domains.shell.vm import (
    Memory, InsFault, Device, DeviceFault,
    ConsoleDevice, FileDevice, IRQDevice,
    VGADevice, PS2KeyboardDevice, BlockDevice, SerialDevice,
    MouseDevice, CMOSDevice, DiskDevice, NICDevice, ClockDevice,
    FlatFS, DeviceBus, PageFrameAllocator,
    X86VirtualSystem, X86CPU, X86Assembler,
)


# ── Memory ────────────────────────────────────────────────────────────────


class TestMemory:
    def test_load_missing_key_raises(self):
        mem = Memory()
        with pytest.raises(InsFault, match="heap key not found"):
            mem.load("nope")

    def test_free_pops_heap_and_sizes(self):
        mem = Memory()
        mem.store("a", 1)
        mem.store("b", 2)
        mem.free("a")
        assert "a" not in mem._heap
        assert "a" not in mem._alloc_sizes
        assert mem.contains("b")

    def test_free_removes_key_from_lru(self):
        mem = Memory()
        mem.store("x", 1)
        mem.store("y", 2)
        assert mem._lru == ["x", "y"]
        mem.free("x")
        assert mem._lru == ["y"]

    def test_contains(self):
        mem = Memory()
        assert not mem.contains("a")
        mem.store("a", 1)
        assert mem.contains("a")

    def test_lru_evict_empty_returns_none(self):
        mem = Memory()
        assert mem.lru_evict() is None

    def test_lru_evict_returns_oldest(self):
        mem = Memory()
        mem.store("a", 1)
        mem.store("b", 2)
        mem.load("a")  # refresh LRU: b, a
        assert mem.lru_evict() == "b"
        assert not mem.contains("b")
        assert "b" not in mem._alloc_sizes
        assert mem.lru_evict() == "a"


# ── Device base ───────────────────────────────────────────────────────────


class TestDevice:
    def test_call_raises_device_fault(self):
        dev = Device()
        with pytest.raises(DeviceFault, match="does not support"):
            dev.call("nope")

    def test_info_returns_base_dict(self):
        assert Device().info() == {"type": "base", "methods": []}


# ── ConsoleDevice ─────────────────────────────────────────────────────────


class TestConsoleDevice:
    def test_info(self):
        console = ConsoleDevice(0)
        info = console.info()
        assert info["type"] == "console"
        assert info["port"] == 0

    def test_call_read_and_write(self):
        out = []
        console = ConsoleDevice(0, stdin_fn=lambda: "hi", stdout_fn=out.append)
        assert console.call("read") == "hi"
        assert console.call("write", "x") is None
        assert out == ["x"]

    def test_call_unknown_method_raises(self):
        console = ConsoleDevice(1)
        with pytest.raises(DeviceFault):
            console.call("bogus")


# ── FileDevice ────────────────────────────────────────────────────────────


class TestFileDevice:
    def test_init_sets_files_and_next_fd(self):
        fd = FileDevice()
        assert fd._files == {}
        assert fd._next_fd == 1

    def test_info(self):
        fd = FileDevice()
        assert fd.info() == {"type": "file", "open_files": 0}

    def test_open_write_read_close(self, tmp_path):
        fd = FileDevice()
        path = tmp_path / "f.bin"
        fh = fd.call("open", str(path), "wb")
        assert fh == 1
        assert fd._next_fd == 2
        assert fd.call("write", fh, b"hello") == 5
        fd.call("close", fh)
        assert path.read_bytes() == b"hello"
        fh2 = fd.call("open", str(path), "rb")
        assert fd.call("read", fh2, 3) == b"hel"
        fd.call("close", fh2)

    def test_open_default_mode(self, tmp_path):
        fd = FileDevice()
        path = tmp_path / "g.bin"
        path.write_bytes(b"abc")
        fh = fd.call("open", str(path))  # default mode "r"
        assert fd.call("read", fh) == "abc"
        fd.call("close", fh)

    def test_close_missing_fd_returns_true(self):
        fd = FileDevice()
        assert fd.call("close", 999) is True

    def test_read_bad_fd_raises(self):
        fd = FileDevice()
        with pytest.raises(DeviceFault, match="bad fd"):
            fd.call("read", 42)

    def test_write_bad_fd_raises(self):
        fd = FileDevice()
        with pytest.raises(DeviceFault, match="bad fd"):
            fd.call("write", 42, b"x")

    def test_listdir_and_exists(self, tmp_path):
        fd = FileDevice()
        (tmp_path / "a.txt").write_text("")
        assert "a.txt" in fd.call("listdir", str(tmp_path))
        assert fd.call("exists", str(tmp_path / "a.txt")) is True
        assert fd.call("exists", str(tmp_path / "missing.txt")) is False

    def test_call_unknown_method_raises(self):
        fd = FileDevice()
        with pytest.raises(DeviceFault):
            fd.call("bogus")


# ── IRQDevice ─────────────────────────────────────────────────────────────


class TestIRQDevice:
    def test_info(self):
        irq = IRQDevice()
        info = irq.info()
        assert info["type"] == "irq"
        assert info["ticks"] == 0

    def test_call_read_key_empty_returns_zero(self):
        irq = IRQDevice()
        assert irq.call("read_key") == 0

    def test_call_tick_returns_count(self):
        irq = IRQDevice()
        irq._tick_count = 7
        assert irq.call("tick") == 7

    def test_call_read_key_returns_key(self):
        irq = IRQDevice()
        irq.push_key(ord("Q"))
        assert irq.call("read_key") == ord("Q")

    def test_call_unknown_method_raises(self):
        irq = IRQDevice()
        with pytest.raises(DeviceFault):
            irq.call("bogus")


# ── VGADevice ─────────────────────────────────────────────────────────────


class TestVGADevice:
    def test_call_unknown_method_raises(self):
        vga = VGADevice()
        with pytest.raises(DeviceFault):
            vga.call("bogus")


# ── PS2KeyboardDevice ─────────────────────────────────────────────────────


class TestPS2KeyboardDevice:
    def test_info(self):
        kb = PS2KeyboardDevice()
        info = kb.info()
        assert info["type"] == "ps2_keyboard"
        assert info["buffered"] == 0

    def test_call_unknown_method_raises(self):
        kb = PS2KeyboardDevice()
        with pytest.raises(DeviceFault):
            kb.call("bogus")


# ── BlockDevice ───────────────────────────────────────────────────────────


class TestBlockDevice:
    def test_write_sector_out_of_range_raises(self):
        blk = BlockDevice(num_sectors=4)
        with pytest.raises(DeviceFault, match="out of range"):
            blk.write_sector(10, b"x")

    def test_read_block_short(self):
        blk = BlockDevice(num_sectors=4)
        blk.write_sector(0, b"hello world")
        assert blk.read_block(0, 5) == b"hello"

    def test_write_block(self):
        blk = BlockDevice(num_sectors=4)
        blk.write_block(0, b"xyz")
        assert bytes(blk.read_sector(0)[:3]) == b"xyz"

    def test_call_dispatch(self):
        blk = BlockDevice(num_sectors=4)
        assert blk.call("read_sector", 0)
        blk.call("write_sector", 1, b"data")
        assert blk.call("read_block", 1, 4) == b"data"
        blk.call("write_block", 2, b"blk")
        assert blk.call("read_block", 2, 3) == b"blk"

    def test_call_unknown_method_raises(self):
        blk = BlockDevice(num_sectors=4)
        with pytest.raises(DeviceFault):
            blk.call("bogus")


# ── SerialDevice ──────────────────────────────────────────────────────────


class TestSerialDevice:
    def test_read_data_empty_returns_zero(self):
        ser = SerialDevice()
        assert ser._read_data() == 0

    def test_read_lsr_rx_bit_set_when_data(self):
        ser = SerialDevice()
        ser.push_byte(0x41)
        lsr = ser._read_lsr()
        assert lsr & 0x01

    def test_call_unknown_method_raises(self):
        ser = SerialDevice()
        with pytest.raises(DeviceFault):
            ser.call("bogus")


# ── MouseDevice ───────────────────────────────────────────────────────────


class TestMouseDevice:
    def test_call_unknown_method_raises(self):
        mouse = MouseDevice()
        with pytest.raises(DeviceFault):
            mouse.call("bogus")


# ── CMOSDevice ────────────────────────────────────────────────────────────


class TestCMOSDevice:
    def _clock_at(self, hour, minute=30, second=45):
        clock = ClockDevice(freq=100)
        clock.set_time(2024, 3, 15, hour, minute, second)
        return clock

    def test_write_data_status_a_c_ignored(self):
        cmos = CMOSDevice()
        cmos._selected = CMOSDevice.REG_STATUS_A
        cmos._write_data(0xFF)
        assert cmos._cmos[CMOSDevice.REG_STATUS_A] == 0x26

    def test_write_data_status_d_preserves_vrt(self):
        cmos = CMOSDevice()
        cmos._selected = CMOSDevice.REG_STATUS_D
        cmos._write_data(0x85)
        assert cmos._cmos[CMOSDevice.REG_STATUS_D] == 0x80

    def test_write_data_low_byte(self):
        cmos = CMOSDevice()
        cmos._selected = 0x10
        cmos._write_data(0xAB)
        assert cmos._cmos[0x10] == 0xAB

    def test_refresh_rtc_no_clock(self):
        cmos = CMOSDevice()
        cmos._refresh_rtc()  # no clock -> no-op

    def test_refresh_rtc_bcd_24h(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos._refresh_rtc()
        assert cmos._cmos[CMOSDevice.REG_SECONDS] == 0x45

    def test_refresh_rtc_bcd_12h_pm(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos._cmos[CMOSDevice.REG_STATUS_B] = 0x00  # BCD + 12h
        cmos._refresh_rtc()
        assert cmos._cmos[CMOSDevice.REG_HOURS] & 0x80

    def test_refresh_rtc_bcd_12h_noon(self):
        cmos = CMOSDevice(clock=self._clock_at(12))
        cmos._cmos[CMOSDevice.REG_STATUS_B] = 0x00
        cmos._refresh_rtc()
        assert (cmos._cmos[CMOSDevice.REG_HOURS] & 0x7F) == 0x12  # bcd 12

    def test_refresh_rtc_binary_24h(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos.set_binary_mode(True)
        cmos._refresh_rtc()
        assert cmos._cmos[CMOSDevice.REG_SECONDS] == 45

    def test_refresh_rtc_binary_12h(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos.set_binary_mode(True)
        cmos._cmos[CMOSDevice.REG_STATUS_B] &= ~0x02  # 12h
        cmos._refresh_rtc()
        assert cmos._cmos[CMOSDevice.REG_HOURS] == 0x83  # h12=3 | PM bit

    def test_refresh_rtc_binary_12h_noon(self):
        cmos = CMOSDevice(clock=self._clock_at(12))
        cmos.set_binary_mode(True)
        cmos._cmos[CMOSDevice.REG_STATUS_B] &= ~0x02  # 12h
        cmos._refresh_rtc()
        assert cmos._cmos[CMOSDevice.REG_HOURS] == 0x8C  # h12=12 | PM bit

    def test_get_time_bcd_24h(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        t = cmos.get_time()
        assert t["hour"] == 15
        assert t["minute"] == 30
        assert t["second"] == 45

    def test_get_time_binary_24h(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos.set_binary_mode(True)
        t = cmos.get_time()
        assert t["hour"] == 15

    def test_get_time_bcd_12h_pm(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        cmos._cmos[CMOSDevice.REG_STATUS_B] = 0x00
        t = cmos.get_time()
        assert t["hour"] == 15  # PM converted back

    def test_get_time_bcd_12h_midnight(self):
        cmos = CMOSDevice(clock=self._clock_at(0))
        cmos._cmos[CMOSDevice.REG_STATUS_B] = 0x00
        t = cmos.get_time()
        assert t["hour"] == 0

    def test_get_unix_time_no_clock(self):
        cmos = CMOSDevice()
        assert cmos.get_unix_time() == 0

    def test_call_dispatch(self):
        cmos = CMOSDevice(clock=self._clock_at(15))
        assert isinstance(cmos.call("get_time"), dict)
        assert isinstance(cmos.call("get_unix_time"), int)
        assert cmos.call("read_cmos", 0x10) == 0
        assert cmos.call("write_cmos", 0x10, 0xAB) is True
        assert cmos.call("set_binary_mode", True) is True

    def test_call_unknown_method_raises(self):
        cmos = CMOSDevice()
        with pytest.raises(DeviceFault):
            cmos.call("bogus")


# ── DiskDevice ────────────────────────────────────────────────────────────


class TestDiskDevice:
    def test_call_dispatch(self):
        disk = DiskDevice(BlockDevice(num_sectors=64))
        assert len(disk.call("read_sectors", 0, 2)) == 2 * 512
        disk.call("write_sectors", 4, b"x" * 600)
        geo = disk.call("get_geometry")
        assert geo["total_sectors"] == 64
        assert disk.call("status") == 0x40

    def test_call_unknown_method_raises(self):
        disk = DiskDevice(BlockDevice(num_sectors=64))
        with pytest.raises(DeviceFault):
            disk.call("bogus")


# ── NICDevice ─────────────────────────────────────────────────────────────


class TestNICDevice:
    def test_call_unknown_method_raises(self):
        nic = NICDevice()
        with pytest.raises(DeviceFault):
            nic.call("bogus")


# ── FlatFS ────────────────────────────────────────────────────────────────


class TestFlatFS:
    def test_write_no_space_raises(self):
        blk = BlockDevice(num_sectors=2)
        fs = FlatFS(blk)
        with pytest.raises(DeviceFault, match="no space on disk"):
            fs.write("a.txt", b"x" * 1024)

    def test_read_missing_raises(self):
        fs = FlatFS(BlockDevice(num_sectors=16))
        with pytest.raises(DeviceFault, match="file not found"):
            fs.read("missing.txt")

    def test_size_missing_returns_zero(self):
        fs = FlatFS(BlockDevice(num_sectors=16))
        assert fs.size("missing.txt") == 0

    def test_size_existing(self):
        fs = FlatFS(BlockDevice(num_sectors=16))
        fs.write("f.txt", b"hello")
        assert fs.size("f.txt") == 512


# ── DeviceBus ─────────────────────────────────────────────────────────────


class TestDeviceBus:
    def test_open_unknown_raises(self):
        bus = DeviceBus()
        with pytest.raises(DeviceFault, match="no such device"):
            bus.open("nope")

    def test_open_known_returns_device(self):
        bus = DeviceBus()
        console = ConsoleDevice(0)
        bus.register("c", console)
        assert bus.open("c") is console

    def test_call_dispatches(self):
        bus = DeviceBus()
        console = ConsoleDevice(0, stdin_fn=lambda: "ok")
        bus.register("c", console)
        assert bus.call(bus.open("c"), "read") == "ok"

    def test_info_dispatches(self):
        bus = DeviceBus()
        console = ConsoleDevice(0)
        bus.register("c", console)
        assert bus.info(bus.open("c"))["type"] == "console"


# ── VirtualSystem ─────────────────────────────────────────────────────────


class TestVirtualSystem:
    def test_reset(self):
        from domains.shell.vm import VirtualSystem
        vs = VirtualSystem()
        vs.load_program("NOP\nHALT")
        vs.run()
        assert vs.cpu._step_count > 0
        old_cpu = vs.cpu
        vs.reset()
        assert vs.cpu is not old_cpu
        assert vs.cpu.pc == 0


# ── Smoke: x86 OS-layer classes (fast construction only) ──────────────────


class TestX86OSLayerSmoke:
    def test_x86_cpu_and_assembler(self):
        cpu = X86CPU(memory_size=65536)
        assert cpu._mem_size == 65536
        assert X86Assembler().assemble("nop") == b"\x90"

    def test_page_frame_allocator(self):
        alloc = PageFrameAllocator(total_memory=1024 * 1024)
        addr = alloc.alloc(1)
        assert addr is not None

    def test_x86_virtual_system_construction(self):
        vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
        assert vs._cpu is not None
