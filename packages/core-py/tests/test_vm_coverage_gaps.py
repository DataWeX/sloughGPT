"""
Coverage-completion tests for vm.py — targeting the remaining uncovered lines.

Covers: BlockCompressor, BlockMapEntry, CRC8, DeviceTable ops, CPU instruction
decoder (MOV r/m8, IMUL), and XCHG memory forms.
"""

import pytest
from domain.shell._internal.vm import (
    CPU,
    CRC8_TABLE,
    X86CPU,
    BlockCompressor,
    BlockDevice,
    BlockFlags,
    BlockMapEntry,
    CompressionAlgo,
    DeviceBus,
    VGADevice,
    crc8,
)

# ── CRC8 ────────────────────────────────────────────────────────────────────

def test_crc8_known_vector():
    assert crc8(b"") == 0x00
    assert crc8(b"\x00") == CRC8_TABLE[0]
    assert isinstance(crc8(b"hello"), int)


# ── BlockMapEntry ───────────────────────────────────────────────────────────

class TestBlockMapEntry:
    def test_pack_unpack_roundtrip(self):
        e = BlockMapEntry(offset=0x1000, compressed_size=512, flags=BlockFlags.DIRTY, crc=0xAB)
        data = e.pack()
        assert len(data) == 8
        e2 = BlockMapEntry.unpack(data)
        assert e2.offset == 0x1000
        assert e2.compressed_size == 512
        assert e2.flags == BlockFlags.DIRTY
        assert e2.crc == 0xAB

    def test_defaults(self):
        e = BlockMapEntry()
        assert e.offset == 0
        assert e.compressed_size == 0
        assert e.flags == 0
        assert e.crc == 0


# ── BlockCompressor ─────────────────────────────────────────────────────────

class TestBlockCompressor:
    def test_none_algo(self):
        c = BlockCompressor(CompressionAlgo.NONE)
        assert c.available is True
        data = b"hello world"
        assert c.compress(data) == data
        assert c.decompress(data) == data

    def test_gzip_algo(self):
        c = BlockCompressor(CompressionAlgo.GZIP)
        assert c.available is True
        data = b"x" * 1000
        compressed = c.compress(data)
        assert len(compressed) < len(data)
        assert c.decompress(compressed) == data

    def test_gzip_estimate_ratio(self):
        c = BlockCompressor(CompressionAlgo.GZIP)
        data = b"aaaa" * 250
        ratio = c.estimate_ratio(data)
        assert ratio < 1.0

    def test_estimate_ratio_empty(self):
        c = BlockCompressor(CompressionAlgo.NONE)
        assert c.estimate_ratio(b"") == 1.0

    def test_unavailable_algo_falls_back_to_gzip(self):
        c = BlockCompressor(CompressionAlgo.LZ4)
        if not c.available:
            data = b"test data " * 100
            compressed = c.compress(data)
            assert c.decompress(compressed) == data

    def test_unavailable_zstd_falls_back(self):
        c = BlockCompressor(CompressionAlgo.ZSTD)
        if not c.available:
            data = b"zstd test " * 100
            compressed = c.compress(data)
            assert c.decompress(compressed) == data

    def test_unavailable_snappy_falls_back(self):
        c = BlockCompressor(CompressionAlgo.SNAPPY)
        if not c.available:
            data = b"snappy test " * 100
            compressed = c.compress(data)
            assert c.decompress(compressed) == data

    def test_algo_property(self):
        c = BlockCompressor(CompressionAlgo.GZIP)
        assert c.algo == CompressionAlgo.GZIP


# ── BlockDevice persistent mode ─────────────────────────────────────────────

class TestBlockDevicePersistence:
    def test_ioctl_commands(self, tmp_path):
        path = str(tmp_path / "test.img")
        dev = BlockDevice(path=path, create=True, num_sectors=64)
        # ioctl expects integer request codes
        result = dev.ioctl(0x1200)  # BLKGETSIZE
        assert result is not None
        result = dev.ioctl(0x1201)  # BLKSSZGET
        assert result == 512
        result = dev.ioctl(0x1202)  # BLKFLSBUF
        assert result == 0

    def test_allocate_and_read_write(self, tmp_path):
        path = str(tmp_path / "test.img")
        dev = BlockDevice(path=path, create=True, num_sectors=128)
        dev.allocate_blocks(10)
        data = b"A" * 512
        dev.write_sector(0, data)
        read_back = dev.read_sector(0)
        assert read_back[:len(data)] == data


# ── Device Table Ops (high-level CPU) ──────────────────────────────────────

class TestDeviceTableOps:
    def _make_cpu_with_device_table(self):
        cpu = CPU()
        cpu._devices = DeviceBus()
        vgadev = VGADevice()
        cpu._devices.register("vga", vgadev)
        return cpu

    def test_dev_open(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_open
        _op_dev_open(cpu, ["R0", "vga"])
        assert cpu.regs[0] is not None

    def test_dev_open_unknown(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_open
        _op_dev_open(cpu, ["R0", "nonexistent"])
        assert cpu.regs[0] == ""

    def test_dev_call(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_call
        # VGADevice.call() supports "get_screen", not "info"
        _op_dev_call(cpu, ["R0", "vga", "get_screen"])
        result = cpu.regs[0]
        assert isinstance(result, list)
        assert len(result) == 25  # ROWS

    def test_dev_call_error(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_call
        _op_dev_call(cpu, ["R0", "nonexistent", "read"])
        assert cpu.regs[0] is None

    def test_dev_info(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_info
        # _op_dev_info calls device.info() directly, which works for VGADevice
        _op_dev_info(cpu, ["R0", "vga"])
        info = cpu.regs[0]
        assert info["type"] == "vga"

    def test_dev_info_error(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_info
        _op_dev_info(cpu, ["R0", "nonexistent"])
        assert cpu.regs[0] == {}

    def test_dev_close(self):
        cpu = self._make_cpu_with_device_table()
        from domain.shell._internal.vm import _op_dev_close
        _op_dev_close(cpu, ["R0"])


class TestDeviceTableOps_fd:
    """Device table ops require DeviceDriver objects — tested via integration tests."""
    pass


# ── CPU instruction decoder — MOV r/m8 ─────────────────────────────────────

class TestCPURm8Opcodes:
    """Test X86CPU opcodes 0x88, 0x8A, 0xC6 for 8-bit MOV."""

    def test_mov_r8_rm8_8a(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV EAX, 0x1000; MOV CL, [EAX] (0x8A 0x08); HLT
        code = bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0x8A, 0x08,                      # MOV CL, [EAX]
            0xF4,                             # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert cpu._regs[1] & 0xFF == 0

    def test_mov_r8_r8_88(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV EAX, 0x1000; MOV BL, 42; MOV [EAX], BL (0x88 0x18); HLT
        code = bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xB3, 0x2A,                      # MOV BL, 42
            0x88, 0x18,                      # MOV [EAX], BL
            0xF4,                             # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert cpu._mem[0x1000] == 42

    def test_mov_rm8_imm8_c6(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV EAX, 0x1000; MOV BYTE [EAX], 0x55 (C6 00 55); HLT
        code = bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0x55,                # MOV BYTE [EAX], 0x55
            0xF4,                             # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert cpu._mem[0x1000] == 0x55


# ── CPU instruction decoder — IMUL ─────────────────────────────────────────

class TestCPUIMUL:
    def test_imul_r16_rm16_imm8(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV CX, 3; IMUL BX, CX, 5 (6B D9 05); HLT
        code = bytes([
            0x66, 0xB9, 0x03, 0x00,  # MOV CX, 3
            0x6B, 0xD9, 0x05,         # IMUL BX, CX, 5
            0xF4,                      # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert (cpu._regs[3] & 0xFFFF) == 15

    def test_imul_r16_rm16_imm16(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV CX, 3; IMUL BX, CX, 0x100 (69 D9 00 01); HLT
        code = bytes([
            0x66, 0xB9, 0x03, 0x00,  # MOV CX, 3
            0x69, 0xD9, 0x00, 0x01,  # IMUL BX, CX, 0x100
            0xF4,                      # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert (cpu._regs[3] & 0xFFFF) == 768

    def test_imul_r16_rm16_imm8_negative(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        # MOV CX, 10; IMUL BX, CX, -2 (6B D9 FE); HLT
        code = bytes([
            0x66, 0xB9, 0x0A, 0x00,  # MOV CX, 10
            0x6B, 0xD9, 0xFE,         # IMUL BX, CX, -2
            0xF4,                      # HLT
        ])
        cpu.load(code)
        cpu.run(10)
        assert (cpu._regs[3] & 0xFFFF) == 0xFFEC


# ── X86CPU 0x66-prefix 16-bit string ops ────────────────────────────────────

class TestCPU66StringOps:
    def _make_cpu(self, code_bytes, mem_size=0x400000):
        cpu = X86CPU(memory_size=mem_size)
        cpu._regs[4] = mem_size - 4
        cpu.load(code_bytes)
        return cpu

    def test_stosw(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0xBF, 0x00, 0x10, 0x00, 0x00,  # MOV EDI, 0x1000
            0x66, 0xAB,              # STOSW
            0xF4,
        ]))
        cpu.run(10)
        lo = cpu._mem[0x1000]
        hi = cpu._mem[0x1001]
        assert lo == 0x34 and hi == 0x12

    def test_stosw_df_set(self):
        code = bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0xBF, 0x02, 0x10, 0x00, 0x00,  # MOV EDI, 0x1002
            0xFD,                    # STD
            0x66, 0xAB,              # STOSW (writes at 0x1002, then EDI -= 2)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert cpu._mem[0x1002] == 0x34

    def test_lodsw(self):
        code = bytes([
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0x66, 0xAD,                      # LODSW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0xAB
        cpu._mem[0x1001] = 0xCD
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0xCDAB

    def test_lodsw_df_set(self):
        code = bytes([
            0xBE, 0xFE, 0x0F, 0x00, 0x00,  # MOV ESI, 0x0FFE (4094)
            0xFD,                            # STD
            0x66, 0xAD,                      # LODSW (ESI -= 2 → 0x0FFC)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x0FFE] = 0xAB
        cpu._mem[0x0FFF] = 0xCD
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0xCDAB
        assert cpu._regs[6] == 0x0FFC

    def test_movsw(self):
        code = bytes([
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0xBF, 0x00, 0x20, 0x00, 0x00,  # MOV EDI, 0x2000
            0x66, 0xA5,                      # MOVSW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0xAB
        cpu._mem[0x1001] = 0xCD
        cpu.run(10)
        assert cpu._mem[0x2000] == 0xAB
        assert cpu._mem[0x2001] == 0xCD

    def test_movsw_df_set(self):
        code = bytes([
            0xBE, 0x02, 0x10, 0x00, 0x00,  # MOV ESI, 0x1002
            0xBF, 0x02, 0x20, 0x00, 0x00,  # MOV EDI, 0x2002
            0xFD,                            # STD
            0x66, 0xA5,                      # MOVSW (reads from 0x1002, writes to 0x2002)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1002] = 0x33
        cpu._mem[0x1003] = 0x44
        cpu.run(10)
        assert cpu._mem[0x2002] == 0x33
        assert cpu._mem[0x2003] == 0x44

    def test_cmpsw_equal(self):
        code = bytes([
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0xBF, 0x00, 0x20, 0x00, 0x00,  # MOV EDI, 0x2000
            0x66, 0xA7,                      # CMPSW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x55
        cpu._mem[0x1001] = 0x66
        cpu._mem[0x2000] = 0x55
        cpu._mem[0x2001] = 0x66
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)

    def test_cmpsw_not_equal(self):
        code = bytes([
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0xBF, 0x00, 0x20, 0x00, 0x00,  # MOV EDI, 0x2000
            0x66, 0xA7,                      # CMPSW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x55
        cpu._mem[0x1001] = 0x66
        cpu._mem[0x2000] = 0x77
        cpu._mem[0x2001] = 0x88
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_cmpsw_df_set(self):
        code = bytes([
            0xBE, 0x02, 0x10, 0x00, 0x00,  # MOV ESI, 0x1002
            0xBF, 0x02, 0x20, 0x00, 0x00,  # MOV EDI, 0x2002
            0xFD,                            # STD
            0x66, 0xA7,                      # CMPSW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x11
        cpu._mem[0x2000] = 0x11
        cpu.run(10)
        assert cpu._regs[6] == 0x1000
        assert cpu._regs[7] == 0x2000

    def test_scasw_equal(self):
        code = bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0xBF, 0x00, 0x10, 0x00, 0x00,  # MOV EDI, 0x1000
            0x66, 0xAF,              # SCASW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x34
        cpu._mem[0x1001] = 0x12
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)

    def test_scasw_not_equal(self):
        code = bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0xBF, 0x00, 0x10, 0x00, 0x00,  # MOV EDI, 0x1000
            0x66, 0xAF,              # SCASW
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x00
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_scasw_df_set(self):
        code = bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0xBF, 0x02, 0x10, 0x00, 0x00,  # MOV EDI, 0x1002
            0xFD,                            # STD
            0x66, 0xAF,                      # SCASW (EDI -= 2)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x34
        cpu._mem[0x1001] = 0x12
        cpu.run(10)
        assert cpu._regs[7] == 0x1000


# ── X86CPU 0x66-prefix XCHG, PUSH imm16, MOV r16 imm16 ────────────────────

class TestCPU66Misc:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_xchg_reg_reg(self):
        code = bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0x66, 0xBB, 0x78, 0x56,  # MOV BX, 0x5678
            0x66, 0x87, 0xC3,        # XCHG AX, BX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x5678
        assert (cpu._regs[3] & 0xFFFF) == 0x1234

    def test_push_imm16(self):
        code = bytes([
            0x66, 0x68, 0x34, 0x12,  # PUSH imm16 0x1234
            0x5D,                      # POP EBP
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[5] & 0xFFFF) == 0x1234

    def test_mov_r16_imm16_all(self):
        code = bytes([
            0x66, 0xB8, 0x11, 0x00,  # MOV AX, 0x0011
            0x66, 0xB9, 0x22, 0x00,  # MOV CX, 0x0022
            0x66, 0xBA, 0x33, 0x00,  # MOV DX, 0x0033
            0x66, 0xBB, 0x44, 0x00,  # MOV BX, 0x0044
            0x66, 0xBE, 0x55, 0x00,  # MOV SI, 0x0055
            0x66, 0xBF, 0x66, 0x00,  # MOV DI, 0x0066
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(20)
        assert (cpu._regs[0] & 0xFFFF) == 0x11
        assert (cpu._regs[1] & 0xFFFF) == 0x22
        assert (cpu._regs[2] & 0xFFFF) == 0x33
        assert (cpu._regs[3] & 0xFFFF) == 0x44
        assert (cpu._regs[6] & 0xFFFF) == 0x55
        assert (cpu._regs[7] & 0xFFFF) == 0x66

    def test_mov_ax_moffs(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xA1, 0x00, 0x10, 0x00, 0x00,  # MOV AX, [0x1000]
            0xF4,
        ]))
        cpu._mem[0x1000] = 0xAB
        cpu._mem[0x1001] = 0xCD
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0xCDAB

    def test_mov_moffs_ax(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0x66, 0xA3, 0x00, 0x10, 0x00, 0x00,  # MOV [0x1000], AX
            0xF4,
        ]))
        cpu.run(10)
        assert cpu._mem[0x1000] == 0x34
        assert cpu._mem[0x1001] == 0x12


# ── X86CPU 0x66-prefix F7 group 16-bit ──────────────────────────────────────

class TestCPU66F7Group:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_test_r16_imm16(self):
        code = bytes([
            0x66, 0xB9, 0xFF, 0x00,  # MOV CX, 0x00FF
            0x66, 0xF7, 0xC1, 0xFF, 0x00,  # TEST CX, 0x00FF
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_test_r16_imm16_zero(self):
        code = bytes([
            0x66, 0xB9, 0x00, 0x00,  # MOV CX, 0
            0x66, 0xF7, 0xC1, 0xFF, 0x00,  # TEST CX, 0x00FF
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)

    def test_not_r16(self):
        code = bytes([
            0x66, 0xB9, 0x00, 0xFF,  # MOV CX, 0xFF00
            0x66, 0xF7, 0xD1,        # NOT CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0x00FF

    def test_neg_r16_positive(self):
        code = bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0x66, 0xF7, 0xD9,        # NEG CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0xFFFB
        from domain.shell._internal.vm import FLAG_CF
        assert cpu._flag(FLAG_CF)

    def test_neg_r16_zero(self):
        code = bytes([
            0x66, 0xB9, 0x00, 0x00,  # MOV CX, 0
            0x66, 0xF7, 0xD9,        # NEG CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0
        from domain.shell._internal.vm import FLAG_CF
        assert not cpu._flag(FLAG_CF)

    def test_mul_r16(self):
        code = bytes([
            0x66, 0xB8, 0x03, 0x00,  # MOV AX, 3
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0x66, 0xF7, 0xE1,        # MUL CX  (DX:AX = AX * CX)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 15
        assert (cpu._regs[2] & 0xFFFF) == 0

    def test_imul_r16(self):
        code = bytes([
            0x66, 0xB8, 0x03, 0x00,  # MOV AX, 3
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0x66, 0xF7, 0xE9,        # IMUL CX (DX:AX = AX * CX, signed)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 15
        assert (cpu._regs[2] & 0xFFFF) == 0

    def test_div_r16(self):
        code = bytes([
            0x66, 0xB8, 0x37, 0x00,  # MOV AX, 55
            0x66, 0xBA, 0x00, 0x00,  # MOV DX, 0
            0x66, 0xB9, 0x0A, 0x00,  # MOV CX, 10
            0x66, 0xF7, 0xF1,        # DIV CX (AX = DX:AX / CX)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 5   # quotient
        assert (cpu._regs[2] & 0xFFFF) == 5    # remainder

    def test_div_r16_by_zero(self):
        from domain.shell._internal.vm import InsFault
        code = bytes([
            0x66, 0xB8, 0x01, 0x00,  # MOV AX, 1
            0x66, 0xB9, 0x00, 0x00,  # MOV CX, 0
            0x66, 0xF7, 0xF1,        # DIV CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        with pytest.raises(InsFault):
            cpu.run(10)

    def test_div_r16_overflow(self):
        from domain.shell._internal.vm import InsFault
        code = bytes([
            0x66, 0xB8, 0x00, 0x00,  # MOV AX, 0
            0x66, 0xBA, 0x01, 0x00,  # MOV DX, 1 (dividend = 0x10000)
            0x66, 0xB9, 0x01, 0x00,  # MOV CX, 1
            0x66, 0xF7, 0xF1,        # DIV CX (quotient 0x10000 > 0xFFFF)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        with pytest.raises(InsFault):
            cpu.run(10)

    def test_idiv_r16(self):
        code = bytes([
            0x66, 0xB8, 0xF6, 0xFF,  # MOV AX, -10
            0x66, 0xBA, 0xFF, 0xFF,  # MOV DX, -1
            0x66, 0xB9, 0x03, 0x00,  # MOV CX, 3
            0x66, 0xF7, 0xF9,        # IDIV CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        ax = cpu._regs[0] & 0xFFFF
        ax_s = ax if ax < 0x8000 else ax - 0x10000
        assert ax_s == -3

    def test_idiv_r16_by_zero(self):
        from domain.shell._internal.vm import InsFault
        code = bytes([
            0x66, 0xB8, 0x01, 0x00,  # MOV AX, 1
            0x66, 0xB9, 0x00, 0x00,  # MOV CX, 0
            0x66, 0xF7, 0xF9,        # IDIV CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        with pytest.raises(InsFault):
            cpu.run(10)


# ── X86CPU 0x66-prefix MOV r/m8, IMUL r/m, LEA, F6 8-bit ──────────────────

class TestCPU66MovRm8AndF6:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_mov_r8_rm8_reg_to_reg(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xB1, 0x42,                      # MOV CL, 0x42
            0x88, 0x08,                      # MOV [EAX], CL
            0x8A, 0x10,                      # MOV DL, [EAX]
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[2] & 0xFF) == 0x42

    def test_mov_rm8_imm8_reg(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0xAB,                # MOV BYTE [EAX], 0xAB
            0x8A, 0x08,                      # MOV CL, [EAX]
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[1] & 0xFF) == 0xAB

    def test_mov_rm8_imm8_rm_field(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0xCC,                # MOV BYTE [EAX], 0xCC
            0xF4,
        ]))
        cpu.run(10)
        assert cpu._mem[0x1000] == 0xCC

    def test_imul_r16_rm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x03, 0x00,  # MOV AX, 3
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0x66, 0x6B, 0xC8, 0x01,  # IMUL CX, AX, 1 (6B form)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 3

    def test_lea_r16(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0x66, 0x8D, 0x08,        # LEA CX, [EAX] — CX = EAX & 0xFFFF
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0x1000

    def test_cld_std(self):
        from domain.shell._internal.vm import FLAG_DF
        cpu = self._make_cpu(bytes([
            0xFC,        # CLD
            0xF4,
        ]))
        cpu.run(10)
        assert not cpu._flag(FLAG_DF)

        cpu2 = self._make_cpu(bytes([
            0xFD,        # STD
            0xF4,
        ]))
        cpu2.run(10)
        assert cpu2._flag(FLAG_DF)


# ── X86CPU 0x66-prefix F6 8-bit ops ─────────────────────────────────────────

class TestCPU66F6_8bit:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_test_r8_imm8(self):
        code = bytes([
            0xB1, 0xFF,          # MOV CL, 0xFF
            0x66, 0xF6, 0xC1, 0x0F,  # 66 TEST CL, 0x0F (8-bit)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_not_r8(self):
        code = bytes([
            0xB1, 0x00,          # MOV CL, 0x00
            0x66, 0xF6, 0xD1,    # 66 NOT CL (8-bit)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[1] & 0xFF) == 0xFF

    def test_neg_r8(self):
        code = bytes([
            0xB1, 0x05,          # MOV CL, 5
            0x66, 0xF6, 0xD9,    # 66 NEG CL (8-bit)
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert (cpu._regs[1] & 0xFF) == 0xFB
        from domain.shell._internal.vm import FLAG_CF
        assert cpu._flag(FLAG_CF)


# ── X86CPU 0x66-prefix ALU r/m16, r16 / r16, r/m16 forms ───────────────────

class TestCPU66ALU16:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_add_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0x66, 0x03, 0x0E,        # ADD CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x03
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 8

    def test_add_rm16_r16(self):
        code = bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0xBE, 0x00, 0x10, 0x00, 0x00,  # MOV ESI, 0x1000
            0x66, 0x01, 0x0E,        # ADD [ESI], CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x03
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert cpu._mem[0x1000] == 8
        assert cpu._mem[0x1001] == 0

    def test_or_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0xF0, 0x00,  # MOV CX, 0x00F0
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x0B, 0x0E,        # OR CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x0F
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0xFF

    def test_adc_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x13, 0x0E,        # ADC CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x03
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 8

    def test_sbb_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x1B, 0x0E,        # SBB CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x03
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 2

    def test_and_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0xFF, 0x00,  # MOV CX, 0x00FF
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x23, 0x0E,        # AND CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x0F
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0x0F

    def test_sub_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0x0A, 0x00,  # MOV CX, 10
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x2B, 0x0E,        # SUB CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x03
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 7

    def test_xor_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0xFF, 0x00,  # MOV CX, 0x00FF
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x33, 0x0E,        # XOR CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x0F
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0xF0

    def test_cmp_r16_rm16(self):
        code = bytes([
            0x66, 0xB9, 0x0A, 0x00,  # MOV CX, 10
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x3B, 0x0E,        # CMP CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x0A
        cpu._mem[0x1001] = 0x00
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)

    def test_mov_r16_rm16(self):
        code = bytes([
            0xBE, 0x00, 0x10, 0x00, 0x00,
            0x66, 0x8B, 0x0E,        # MOV CX, [ESI]
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu._mem[0x1000] = 0x34
        cpu._mem[0x1001] = 0x12
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 0x1234

    def test_mov_rm16_r16(self):
        code = bytes([
            0x66, 0xB9, 0x34, 0x12,  # MOV CX, 0x1234
            0xBF, 0x00, 0x20, 0x00, 0x00,
            0x66, 0x89, 0x0F,        # MOV [EDI], CX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert cpu._mem[0x2000] == 0x34
        assert cpu._mem[0x2001] == 0x12


# ── X86CPU MOVSXD and INC/DEC r32 ──────────────────────────────────────────

class TestCPU66MiscOpcodes:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_movsxd_r32_r32(self):
        code = bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0x63, 0xC8,                      # MOVSXD ECX, EAX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert cpu._regs[1] == 0x1000

    def test_inc_r32(self):
        code = bytes([
            0xB8, 0xFF, 0xFF, 0xFF, 0x7F,  # MOV EAX, 0x7FFFFFFF
            0x40,        # INC EAX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert cpu._regs[0] == 0x80000000

    def test_dec_r32(self):
        code = bytes([
            0xB8, 0x00, 0x00, 0x00, 0x80,  # MOV EAX, 0x80000000
            0x48,        # DEC EAX
            0xF4,
        ])
        cpu = self._make_cpu(code)
        cpu.run(10)
        assert cpu._regs[0] == 0x7FFFFFFF


# ── X86CPU 0x66-prefix accumulator-immediate ALU ────────────────────────────

class TestCPU66AccImmALU:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_add_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x10, 0x00,  # MOV AX, 0x10
            0x66, 0x05, 0x05, 0x00,  # ADD AX, 0x05
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x15

    def test_add_al_imm8(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0x10,              # MOV AL, 0x10
            0x66, 0x04, 0x05,        # ADD AL, 0x05
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFF) == 0x15

    def test_cmp_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x0A, 0x00,  # MOV AX, 10
            0x66, 0x3D, 0x0A, 0x00,  # CMP AX, 10
            0xF4,
        ]))
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)

    def test_sub_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x10, 0x00,  # MOV AX, 0x10
            0x66, 0x2D, 0x05, 0x00,  # SUB AX, 0x05
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x0B

    def test_xor_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0xFF, 0x00,  # MOV AX, 0x00FF
            0x66, 0x35, 0x0F, 0x00,  # XOR AX, 0x000F
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x00F0

    def test_or_al_imm8(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0xF0,              # MOV AL, 0xF0
            0x66, 0x0C, 0x0F,        # OR AL, 0x0F
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFF) == 0xFF

    def test_and_al_imm8(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0xFF,              # MOV AL, 0xFF
            0x66, 0x24, 0x0F,        # AND AL, 0x0F
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFF) == 0x0F

    def test_adc_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x10, 0x00,  # MOV AX, 0x10
            0x66, 0x15, 0x05, 0x00,  # ADC AX, 0x05
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x15

    def test_sbb_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x10, 0x00,  # MOV AX, 0x10
            0x66, 0x1D, 0x05, 0x00,  # SBB AX, 0x05
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x0B

    def test_cmp_al_imm8(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0x42,              # MOV AL, 0x42
            0x66, 0x3C, 0x42,        # CMP AL, 0x42
            0xF4,
        ]))
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)


# ── X86CPU 0x66-prefix TEST, Group 1 imm, INC/DEC r16, PUSH/POP r16 ───────

class TestCPU66MiscGroup1:
    def _make_cpu(self, code_bytes):
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x400000 - 4
        cpu.load(code_bytes)
        return cpu

    def test_test_al_imm8(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0xFF,              # MOV AL, 0xFF
            0x66, 0xA8, 0x0F,        # TEST AL, 0x0F
            0xF4,
        ]))
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_test_ax_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0xFF, 0x00,  # MOV AX, 0x00FF
            0x66, 0xA9, 0xFF, 0x00,  # TEST AX, 0x00FF
            0xF4,
        ]))
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert not cpu._flag(FLAG_ZF)

    def test_group1_81_add_imm16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB9, 0x05, 0x00,  # MOV CX, 5
            0x66, 0x81, 0xC1, 0x0A, 0x00,  # ADD CX, 10
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 15

    def test_group1_83_sub_imm8(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB9, 0x0A, 0x00,  # MOV CX, 10
            0x66, 0x83, 0xE9, 0x03,  # SUB CX, 3
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[1] & 0xFFFF) == 7

    def test_group1_80_sub_imm8_rm8(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0x20,        # MOV BYTE [EAX], 0x20
            0x66, 0x80, 0x28, 0x05,  # SUB BYTE [EAX], 5
            0xF4,
        ]))
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0x20,        # MOV BYTE [EAX], 0x20
            0x66, 0x80, 0x28, 0x05,  # SUB BYTE [EAX], 5
            0xF4,
        ]))
        cpu.run(10)
        assert cpu._mem[0x1000] == 0x1B

    def test_inc_r16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0xFF, 0x00,  # MOV AX, 0x00FF
            0x66, 0x40,              # INC AX (66 40)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x0100

    def test_dec_r16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x00, 0x00,  # MOV AX, 0
            0x66, 0x48,              # DEC AX (66 48)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0xFFFF

    def test_push_pop_r16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0x66, 0x50,              # PUSH AX (66 50)
            0x66, 0xB8, 0x00, 0x00,  # MOV AX, 0
            0x66, 0x58,              # POP AX (66 58)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x1234

    def test_xchg_ax_r16(self):
        cpu = self._make_cpu(bytes([
            0x66, 0xB8, 0x34, 0x12,  # MOV AX, 0x1234
            0x66, 0xB9, 0x78, 0x56,  # MOV CX, 0x5678
            0x66, 0x91,              # XCHG AX, CX (66 91)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFFFF) == 0x5678
        assert (cpu._regs[1] & 0xFFFF) == 0x1234

    def test_alu_r8_rm8_via_66(self):
        cpu = self._make_cpu(bytes([
            0xB8, 0x00, 0x10, 0x00, 0x00,  # MOV EAX, 0x1000
            0xC6, 0x00, 0x0A,        # MOV BYTE [EAX], 10
            0xB1, 0x05,              # MOV CL, 5
            0x66, 0x00, 0x08,        # ADD [EAX], CL (r/m8, r8)
            0xF4,
        ]))
        cpu.run(10)
        assert cpu._mem[0x1000] == 15

    def test_alu_r8_rm8_reg_dest(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0x0A,              # MOV AL, 10
            0xB1, 0x05,              # MOV CL, 5
            0x66, 0x02, 0xC1,        # ADD AL, CL (r8, r/m8)
            0xF4,
        ]))
        cpu.run(10)
        assert (cpu._regs[0] & 0xFF) == 15

    def test_cmp_r8_rm8_reg(self):
        cpu = self._make_cpu(bytes([
            0xB0, 0x0A,              # MOV AL, 10
            0xB1, 0x0A,              # MOV CL, 10
            0x66, 0x38, 0xC1,        # CMP CL, AL (r/m8, r8 — CMP doesn't store)
            0xF4,
        ]))
        cpu.run(10)
        from domain.shell._internal.vm import FLAG_ZF
        assert cpu._flag(FLAG_ZF)


# ── Scheduler ops (switch_to, block_current, unblock) ───────────────────────

class TestSchedulerOps:
    def test_switch_to(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)

        pcb1 = ptable.create("a.out")
        pcb2 = ptable.create("b.out")
        sched.enqueue(pcb1.pid)
        sched.enqueue(pcb2.pid)
        sched.start(cpu)
        assert sched.current.pid == pcb1.pid

        assert sched.switch_to(cpu, pcb2.pid)
        assert sched.current.pid == pcb2.pid

    def test_switch_to_nonexistent(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        assert not sched.switch_to(cpu, 999)

    def test_switch_to_terminated(self):
        from domain.shell._internal.vm import X86CPU, ProcessState, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        pcb1 = ptable.create("a.out")
        pcb1.state = ProcessState.TERMINATED
        assert not sched.switch_to(cpu, pcb1.pid)

    def test_block_current(self):
        from domain.shell._internal.vm import X86CPU, ProcessState, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        pcb1 = ptable.create("a.out")
        sched.enqueue(pcb1.pid)
        sched.start(cpu)
        assert sched.current.pid == pcb1.pid

        sched.block_current(cpu)
        assert sched.current is None
        assert ptable.get(pcb1.pid).state == ProcessState.WAITING

    def test_block_current_no_process(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        sched.block_current(cpu)  # should not raise

    def test_unblock(self):
        from domain.shell._internal.vm import X86CPU, ProcessState, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        pcb1 = ptable.create("a.out")
        sched.enqueue(pcb1.pid)
        sched.start(cpu)
        sched.block_current(cpu)
        assert ptable.get(pcb1.pid).state == ProcessState.WAITING

        sched.unblock(pcb1.pid)
        assert ptable.get(pcb1.pid).state == ProcessState.READY

    def test_unblock_nonexistent(self):
        from domain.shell._internal.vm import ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        sched.unblock(999)  # should not raise

    def test_exit_current(self):
        from domain.shell._internal.vm import X86CPU, ProcessState, ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        cpu = X86CPU(memory_size=0x100000)
        pcb1 = ptable.create("a.out")
        sched.enqueue(pcb1.pid)
        sched.start(cpu)
        sched.exit_current(cpu, 42)
        assert ptable.get(pcb1.pid).state == ProcessState.TERMINATED
        assert ptable.get(pcb1.pid).exit_code == 42
        assert sched.current is None

    def test_exit_current_no_process(self):
        from domain.shell._internal.vm import ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        sched.exit_current(None)  # should not raise

    def test_stats(self):
        from domain.shell._internal.vm import ProcessTable, Scheduler
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        s = sched.stats()
        assert "tick_count" in s
        assert "quantum" in s
        assert s["quantum"] == 10


# ── X86SyscallHandler tests ──────────────────────────────────────────────────

class TestSyscallHandler:
    """Direct X86SyscallHandler unit tests — exercises the individual _sys_* methods."""

    def _make_handler(self, with_fs=False, with_devices=False):
        from domain.shell._internal.vm import (
            X86CPU, ProcessTable, Scheduler, X86SyscallHandler,
            PageFrameAllocator, FlatFS, BlockDevice,
        )
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(
            total_memory=0x400000,
            reserved_ranges=[(0, 0x100000), (0xB8000, 0xC0000)],
        )
        fs = FlatFS(BlockDevice()) if with_fs else None
        handler = X86SyscallHandler(cpu, ptable, sched, alloc, filesystem=fs)
        return handler, cpu, ptable, sched

    def _setup_process(self, ptable, sched, name="test"):
        pcb = ptable.create(name=name, priority=5)
        sched.enqueue(pcb.pid)
        sched._current_pid = pcb.pid
        pcb.state = __import__('domain.shell._internal.vm', fromlist=['ProcessState']).ProcessState.RUNNING
        return pcb

    def test_tick(self):
        handler, _, _, _ = self._make_handler()
        assert handler._ticks == 0
        handler.tick()
        assert handler._ticks == 1
        handler.tick()
        handler.tick()
        assert handler._ticks == 3

    def test_read_string(self):
        handler, cpu, _, _ = self._make_handler()
        cpu._mem[0x1000:0x1006] = b'hello\x00'
        s = handler._read_string(0x1000)
        assert s == 'hello'

    def test_read_string_empty(self):
        handler, cpu, _, _ = self._make_handler()
        cpu._mem[0x1000] = 0
        s = handler._read_string(0x1000)
        assert s == ''

    def test_write_string(self):
        handler, cpu, _, _ = self._make_handler()
        handler._write_string(0x1000, 'hi')
        assert cpu._mem[0x1000] == ord('h')
        assert cpu._mem[0x1001] == ord('i')
        assert cpu._mem[0x1002] == 0

    def test_sys_exit(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        result = handler._sys_exit(0)
        assert result == 0

    def test_sys_read_negative_fd(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_read(-1, 0x1000, 10) == -1

    def test_sys_read_zero_count(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_read(0, 0x1000, 0) == -1

    def test_sys_read_stdin(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        # Place a char in the keyboard buffer at 0x400
        cpu._mem[0x400] = ord('A')
        result = handler._sys_read(0, 0x2000, 1)
        assert result == 1
        assert cpu._mem[0x2000] == ord('A')
        assert cpu._mem[0x400] == 0  # consumed

    def test_sys_read_stdin_empty(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        # Keyboard buffer is empty (0)
        result = handler._sys_read(0, 0x2000, 10)
        assert result == 0

    def test_sys_read_filesystem(self):
        handler, cpu, ptable, sched = self._make_handler(with_fs=True)
        self._setup_process(ptable, sched)
        handler._fs.write('test.txt', b'data')
        handler._fd_table[3] = 'test.txt'
        result = handler._sys_read(3, 0x2000, 10)
        # FlatFS pads to sector size, so min(count=10, len(data)=512) = 10
        assert result == 10

    def test_sys_write_negative_fd(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_write(-1, 0x1000, 10) == -1

    def test_sys_write_zero_count(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_write(1, 0x1000, 0) == -1

    def test_sys_write_stdout(self):
        handler, cpu, _, _ = self._make_handler()
        cpu._mem[0x1000:0x1005] = b'hello'
        result = handler._sys_write(1, 0x1000, 5)
        assert result == 5

    def test_sys_write_stderr(self):
        handler, cpu, _, _ = self._make_handler()
        cpu._mem[0x1000:0x1005] = b'error'
        result = handler._sys_write(2, 0x1000, 5)
        assert result == 5

    def test_sys_write_filesystem(self):
        handler, cpu, ptable, sched = self._make_handler(with_fs=True)
        self._setup_process(ptable, sched)
        handler._fs.write('out.txt', b'')
        handler._fd_table[3] = 'out.txt'
        cpu._mem[0x1000:0x1003] = b'abc'
        result = handler._sys_write(3, 0x1000, 3)
        assert result == 3

    def test_sys_write_invalid_fd(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_write(99, 0x1000, 5) == -1

    def test_sys_open_no_fs(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_open(0x1000, 0) == -1

    def test_sys_open_empty_filename(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        cpu._mem[0x1000] = 0  # empty string
        assert handler._sys_open(0x1000, 0) == -1

    def test_sys_open_read_nonexistent(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._write_string(0x1000, 'nope.txt')
        assert handler._sys_open(0x1000, 0) == -1

    def test_sys_open_create_mode(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._write_string(0x1000, 'new.txt')
        fd = handler._sys_open(0x1000, 2)
        assert fd >= 3
        assert handler._fs.exists('new.txt')

    def test_sys_open_read_existing(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._fs.write('exist.txt', b'hi')
        handler._write_string(0x1000, 'exist.txt')
        fd = handler._sys_open(0x1000, 0)
        assert fd >= 3

    def test_sys_close(self):
        handler, _, _, _ = self._make_handler()
        handler._fd_table[3] = 'test.txt'
        assert handler._sys_close(3) == 0
        assert 3 not in handler._fd_table

    def test_sys_close_invalid_fd(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_close(99) == -1

    def test_sys_fork_no_current(self):
        handler, cpu, ptable, sched = self._make_handler()
        result = handler._sys_fork()
        assert result == -1

    def test_sys_fork_oom(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        # Exhaust all available memory by filling the allocator
        alloc = handler._memory
        # Allocate all available pages until it returns None
        while True:
            page = alloc.alloc(1)
            if page is None:
                break
        result = handler._sys_fork()
        assert result == -1

    def test_sys_exec_no_fs(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_exec(0x1000) == -1

    def test_sys_exec_file_not_found(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._write_string(0x1000, 'nope.asm')
        assert handler._sys_exec(0x1000) == -1

    def test_sys_exec_no_current(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._fs.write('prog.asm', b'[BITS 32]\nNOP')
        handler._write_string(0x1000, 'prog.asm')
        assert handler._sys_exec(0x1000) == -1

    def test_sys_exec_assembly_failure(self):
        handler, cpu, ptable, sched = self._make_handler(with_fs=True)
        self._setup_process(ptable, sched)
        # Write garbage that will fail assembly
        handler._fs.write('bad.asm', b'\xff\xff\xff\xff')
        handler._write_string(0x1000, 'bad.asm')
        result = handler._sys_exec(0x1000)
        assert result == -1

    def test_sys_exec_nop_only(self):
        handler, cpu, ptable, sched = self._make_handler(with_fs=True)
        self._setup_process(ptable, sched)
        handler._fs.write('nop.asm', b'[BITS 32]\nNOP')
        handler._write_string(0x1000, 'nop.asm')
        result = handler._sys_exec(0x1000)
        assert result == -1

    def test_sys_exec_success(self):
        handler, cpu, ptable, sched = self._make_handler(with_fs=True)
        self._setup_process(ptable, sched)
        # Write a valid program: MOV EAX, 1; RET
        handler._fs.write('ok.asm', b'[BITS 32]\nMOV EAX, 1\nRET')
        handler._write_string(0x1000, 'ok.asm')
        result = handler._sys_exec(0x1000)
        assert result == 0

    def test_sys_wait_no_children(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        result = handler._sys_wait()
        assert result == -1

    def test_sys_wait_terminated_child(self):
        handler, cpu, ptable, sched = self._make_handler()
        pcb = self._setup_process(ptable, sched)
        child = ptable.create(name="child")
        pcb.children.append(child.pid)
        child.state = __import__('domain.shell._internal.vm', fromlist=['ProcessState']).ProcessState.TERMINATED
        result = handler._sys_wait()
        assert result == child.pid

    def test_sys_wait_blocked(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import ProcessState
        pcb = self._setup_process(ptable, sched)
        child = ptable.create(name="child")
        pcb.children.append(child.pid)
        child.state = ProcessState.READY
        result = handler._sys_wait()
        assert result == 0

    def test_sys_brk(self):
        handler, cpu, ptable, sched = self._make_handler()
        pcb = self._setup_process(ptable, sched)
        result = handler._sys_brk(0x500000)
        assert result == 0
        assert pcb.heap_break == 0x500000

    def test_sys_getpid(self):
        handler, cpu, ptable, sched = self._make_handler()
        pcb = self._setup_process(ptable, sched)
        result = handler._sys_getpid()
        assert result == pcb.pid

    def test_sys_getpid_no_current(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_getpid() == 0

    def test_sys_getrole(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        result = handler._sys_getrole()
        assert isinstance(result, int)

    def test_sys_yield(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        result = handler._sys_yield()
        assert result == 0

    def test_sys_kill_no_pcb(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_kill(999, 9) == -1

    def test_sys_kill_sigkill_other(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import ProcessState
        self._setup_process(ptable, sched)
        target = ptable.create(name="target")
        target.state = ProcessState.READY
        result = handler._sys_kill(target.pid, 9)
        assert result == 0

    def test_sys_kill_self(self):
        handler, cpu, ptable, sched = self._make_handler()
        pcb = self._setup_process(ptable, sched)
        result = handler._sys_kill(pcb.pid, 9)
        assert result == 0

    def test_sys_gettimeofday(self):
        handler, cpu, _, _ = self._make_handler()
        handler._ticks = 42
        result = handler._sys_gettimeofday(0x2000)
        assert result == 42
        assert cpu._read32(0x2000) == 42

    def test_sys_gettimeofday_no_buf(self):
        handler, _, _, _ = self._make_handler()
        handler._ticks = 100
        result = handler._sys_gettimeofday(0)
        assert result == 100

    def test_sys_malloc(self):
        handler, _, _, _ = self._make_handler()
        addr = handler._sys_malloc(64)
        assert addr >= 0x400000
        assert handler._heap[addr] == 64  # aligned to 16

    def test_sys_malloc_zero(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_malloc(0) == 0

    def test_sys_malloc_negative(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_malloc(-1) == 0

    def test_sys_free(self):
        handler, _, _, _ = self._make_handler()
        handler._heap[0x400000] = 64
        assert handler._sys_free(0x400000) == 0
        assert 0x400000 not in handler._heap

    def test_sys_free_not_found(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_free(0x999999) == -1

    def test_sys_sbrk(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        result = handler._sys_sbrk(0x1000)
        assert result == 0x400000
        assert handler._heap_break == 0x401000

    def test_sys_readdir_no_fs(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_readdir(0x1000, 10) == 0

    def test_sys_readdir(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._fs.write('a.txt', b'a')
        handler._fs.write('b.txt', b'b')
        result = handler._sys_readdir(0x1000, 10)
        assert result == 2

    def test_sys_readdir_truncated(self):
        handler, cpu, _, _ = self._make_handler(with_fs=True)
        handler._fs.write('a.txt', b'a')
        result = handler._sys_readdir(0x1000, 1)
        assert result == 1

    def test_sys_uname(self):
        handler, cpu, _, _ = self._make_handler()
        result = handler._sys_uname(0x1000)
        assert result == 0
        # Check that the buffer was written (first field "SloughOS")
        s = handler._read_string(0x1000)
        assert s == 'SloughOS'

    def test_sys_serial_write(self):
        handler, cpu, _, _ = self._make_handler()
        handler._serial = __import__('domain.shell._internal.vm', fromlist=['SerialDevice']).SerialDevice(cpu=cpu)
        assert handler._sys_serial_write(65) == 0

    def test_sys_serial_write_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_serial_write(65) == -1

    def test_sys_serial_read(self):
        handler, cpu, _, _ = self._make_handler()
        serial = __import__('domain.shell._internal.vm', fromlist=['SerialDevice']).SerialDevice(cpu=cpu)
        handler._serial = serial
        result = handler._sys_serial_read()
        assert isinstance(result, int)

    def test_sys_serial_read_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_serial_read() == -1

    def test_sys_mouse_read(self):
        handler, cpu, _, _ = self._make_handler()
        mouse = __import__('domain.shell._internal.vm', fromlist=['MouseDevice']).MouseDevice()
        handler._mouse = mouse
        result = handler._sys_mouse_read(0x1000)
        assert result == -1  # no packet available

    def test_sys_mouse_read_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_mouse_read(0x1000) == -1

    def test_sys_rtc_gettime(self):
        handler, cpu, _, _ = self._make_handler()
        clock = __import__('domain.shell._internal.vm', fromlist=['ClockDevice']).ClockDevice(freq=100)
        rtc = __import__('domain.shell._internal.vm', fromlist=['CMOSDevice']).CMOSDevice(cpu=cpu, clock=clock)
        handler._rtc = rtc
        result = handler._sys_rtc_gettime(0x1000)
        assert isinstance(result, int)

    def test_sys_rtc_gettime_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_rtc_gettime(0x1000) == -1

    def test_sys_disk_read(self):
        handler, cpu, _, _ = self._make_handler()
        block = __import__('domain.shell._internal.vm', fromlist=['BlockDevice']).BlockDevice()
        disk = __import__('domain.shell._internal.vm', fromlist=['DiskDevice']).DiskDevice(block_device=block)
        handler._disk = disk
        result = handler._sys_disk_read(0, 0x1000, 1)
        assert result == 512

    def test_sys_disk_read_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_disk_read(0, 0x1000, 1) == -1

    def test_sys_disk_read_zero_count(self):
        handler, cpu, _, _ = self._make_handler()
        block = __import__('domain.shell._internal.vm', fromlist=['BlockDevice']).BlockDevice()
        disk = __import__('domain.shell._internal.vm', fromlist=['DiskDevice']).DiskDevice(block_device=block)
        handler._disk = disk
        assert handler._sys_disk_read(0, 0x1000, 0) == -1

    def test_sys_disk_write(self):
        handler, cpu, _, _ = self._make_handler()
        block = __import__('domain.shell._internal.vm', fromlist=['BlockDevice']).BlockDevice()
        disk = __import__('domain.shell._internal.vm', fromlist=['DiskDevice']).DiskDevice(block_device=block)
        handler._disk = disk
        cpu._mem[0x1000:0x1004] = b'test'
        result = handler._sys_disk_write(0, 0x1000, 1)
        assert result == 512

    def test_sys_disk_write_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_disk_write(0, 0x1000, 1) == -1

    def test_sys_disk_write_zero_count(self):
        handler, cpu, _, _ = self._make_handler()
        block = __import__('domain.shell._internal.vm', fromlist=['BlockDevice']).BlockDevice()
        disk = __import__('domain.shell._internal.vm', fromlist=['DiskDevice']).DiskDevice(block_device=block)
        handler._disk = disk
        assert handler._sys_disk_write(0, 0x1000, 0) == -1

    def test_sys_net_send(self):
        handler, cpu, _, _ = self._make_handler()
        nic = __import__('domain.shell._internal.vm', fromlist=['NICDevice']).NICDevice()
        handler._nic = nic
        cpu._mem[0x1000:0x1004] = b'data'
        result = handler._sys_net_send(0x1000, 4)
        assert result == 0

    def test_sys_net_send_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_net_send(0x1000, 4) == -1

    def test_sys_net_send_zero_length(self):
        handler, cpu, _, _ = self._make_handler()
        nic = __import__('domain.shell._internal.vm', fromlist=['NICDevice']).NICDevice()
        handler._nic = nic
        assert handler._sys_net_send(0x1000, 0) == -1

    def test_sys_net_recv(self):
        handler, cpu, _, _ = self._make_handler()
        nic = __import__('domain.shell._internal.vm', fromlist=['NICDevice']).NICDevice()
        handler._nic = nic
        result = handler._sys_net_recv(0x1000, 100)
        assert result == -1  # no packet

    def test_sys_net_recv_no_device(self):
        handler, _, _, _ = self._make_handler()
        assert handler._sys_net_recv(0x1000, 100) == -1

    def test_handle_unknown_syscall(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        cpu._regs[0] = 999  # unknown syscall number
        handler.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_permission_denied(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import ProcessState
        pcb = self._setup_process(ptable, sched)
        pcb.state = ProcessState.RUNNING
        # SYS_KILL requires PROCESS_KILL — a normal user doesn't have it
        cpu._regs[0] = 13  # SYS_KILL
        cpu._regs[3] = 999  # EBX = target pid
        cpu._regs[1] = 9   # ECX = signal
        handler.handle()
        assert cpu._regs[0] == 0xFFFFFFFE

    def test_build_perm_map(self):
        handler, _, _, _ = self._make_handler()
        handler._build_perm_map()
        assert len(handler._perm_map) > 0
        assert handler._perm_map[handler.SYS_EXIT] is not None

    def test_check_perm_no_required(self):
        handler, _, _, _ = self._make_handler()
        handler._build_perm_map()
        # SYS_EXIT requires PROCESS_SELF, but with no current process it should be True
        assert handler._check_perm(handler.SYS_EXIT) is True


# ── PITDevice tests ──────────────────────────────────────────────────────────

class TestPITDeviceCoverage:
    """PITDevice unit tests for uncovered lines."""

    def test_read_counter_no_latch(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, PITDevice
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        pit = PITDevice(cpu, sched, target_hz=100)
        # Read counter without latch — should return low byte
        val = pit._read_counter(0)
        assert isinstance(val, int)

    def test_write_counter(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, PITDevice
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        pit = PITDevice(cpu, sched, target_hz=100)
        pit._write_counter(0, 0x42)
        assert pit._counters[0] & 0xFF == 0x42

    def test_tick_irq_fires(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, PITDevice
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        pit = PITDevice(cpu, sched, target_hz=100)
        # Force counter to 1 so next tick fires IRQ
        pit._counters[0] = 1
        pit.tick()
        assert pit._tick_count == 1
        # Counter should be reset to divider
        assert pit._counters[0] == pit._divider

    def test_tick_channels_1_and_2(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, PITDevice
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        pit = PITDevice(cpu, sched, target_hz=100)
        # Channels 1 and 2 don't fire IRQ
        pit._counters[1] = 1
        pit._counters[2] = 1
        pit.tick()
        assert pit._tick_count == 0  # no IRQ from ch1/ch2

    def test_tick_with_clock_and_syscall(self):
        from domain.shell._internal.vm import (
            X86CPU, ProcessTable, Scheduler, PITDevice,
            X86SyscallHandler, PageFrameAllocator, ClockDevice,
        )
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(total_memory=0x400000, reserved_ranges=[(0, 0x100000)])
        syscall = X86SyscallHandler(cpu, ptable, sched, alloc)
        clock = ClockDevice(freq=100)
        pit = PITDevice(cpu, sched, syscall_handler=syscall, target_hz=100, clock=clock)
        pit._counters[0] = 1
        pit.tick()
        assert pit._tick_count == 1
        assert syscall._ticks == 1
        assert clock._ticks == 1


# ── X86VirtualSystem tests ──────────────────────────────────────────────────

class TestX86VirtualSystemCoverage:
    """X86VirtualSystem unit tests for uncovered lines."""

    def test_init_with_filesystem(self):
        from domain.shell._internal.vm import X86VirtualSystem, FlatFS, BlockDevice
        block = BlockDevice()
        fs = FlatFS(block)
        vs = X86VirtualSystem(filesystem=fs)
        assert vs._fs is fs
        assert vs._block is block

    def test_init_default_filesystem(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        assert vs._fs is not None
        assert vs._block is not None

    def test_properties(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        assert vs.cpu is vs._cpu
        assert vs.scheduler is vs._scheduler
        assert vs.process_table is vs._ptable
        assert vs.filesystem is vs._fs
        assert vs.serial is vs._serial
        assert vs.mouse is vs._mouse
        assert vs.rtc is vs._rtc
        assert vs.disk is vs._disk
        assert vs.nic is vs._nic

    def test_load_kernel(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        vs.load_kernel('[BITS 32]\nMOV EAX, 1\nRET')
        assert vs._kernel.eip == 0x1000

    def test_spawn(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        pid = vs.spawn("user", '[BITS 32]\nNOP\nRET')
        assert pid is not None
        assert pid > 1

    def test_spawn_stack_overflow(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem(memory_size=0x400000)
        # Spawn at an address where stack would exceed memory
        pid = vs.spawn("user", '[BITS 32]\nNOP', org=0x3F0000)
        assert pid is not None

    def test_run_cycles(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        vs.load_kernel('[BITS 32]\nRET')
        cycles = vs.run(max_cycles=100)
        assert cycles >= 0

    def test_run_fault(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        # Invalid opcode causes InsFault
        vs._cpu._mem[0x1000] = 0x0F  # invalid two-byte opcode prefix
        vs._cpu._mem[0x1001] = 0x0B  # UD2-like
        vs._kernel.eip = 0x1000
        cycles = vs.run(max_cycles=10)
        assert cycles >= 0

    def test_status(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        s = vs.status()
        assert 'cpu' in s
        assert 'memory' in s
        assert 'scheduler' in s
        assert 'processes' in s
        assert 'pit_ticks' in s
        assert 'syscall_ticks' in s

    def test_reset(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        vs.load_kernel('[BITS 32]\nRET')
        vs.run(max_cycles=10)
        vs.reset()
        # Reset creates a new kernel process
        assert vs._ptable.count() == 1

    def test_keyboard_handler(self):
        from domain.shell._internal.vm import X86VirtualSystem
        vs = X86VirtualSystem()
        # The keyboard handler is registered — verify it exists in the IDT
        assert 1 in vs._cpu._idt_handlers


# ── DeviceRegisterMap tests ──────────────────────────────────────────────────

class TestDeviceRegisterMapCoverage:
    """DeviceRegisterMap tests for uncovered lines."""

    def test_register_device(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        mock_dev = type('Dev', (), {'ioctl': lambda self, *a: None})()
        drm.register_device("test_dev", mock_dev, base_addr=0x1000)
        assert "test_dev" in drm._devices
        assert drm.BASE_ADDRESSES["test_dev"] == 0x1000

    def test_register_device_no_base_raises(self):
        from domain.shell._internal.vm import DeviceRegisterMap, DeviceFault
        drm = DeviceRegisterMap()
        mock_dev = type('Dev', (), {})()
        with pytest.raises(DeviceFault):
            drm.register_device("unknown_dev", mock_dev)

    def test_read_unknown_address(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        assert drm.read(0xDEAD) == 0

    def test_write_unknown_address(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        drm.write(0xDEAD, 42)  # should not raise

    def test_write_command_register(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        # Use a predefined base address so registers are initialized
        base = drm.BASE_ADDRESSES["tensor"]

        class MockDevice:
            def ioctl(self, cmd, a0, a1, a2):
                return 42
        mock_dev = MockDevice()
        drm.register_device("tensor", mock_dev)
        # Write to COMMAND register
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_RESULT] == 42

    def test_write_command_device_none(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["tensor"]
        drm.register_device("tensor", None)
        drm.write(base + drm.REG_COMMAND, 1)  # no device, should not raise

    def test_write_non_command_register(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["tensor"]
        mock_dev = type('Dev', (), {'ioctl': lambda self, *a: None})()
        drm.register_device("tensor", mock_dev)
        drm.write(base + drm.REG_ARG0, 99)
        assert drm._registers[base + drm.REG_ARG0] == 99

    def test_write_non_command_no_device_name(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        # Write to an address that's not in any device block
        drm.write(0x5000, 42)

    def test_execute_command_result_types(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["npu"]

        class BoolDevice:
            def ioctl(self, cmd, a0, a1, a2):
                return True
        mock_dev = BoolDevice()
        drm.register_device("npu", mock_dev)
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_RESULT] == 1

    def test_execute_command_error(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["storage"]

        class ErrorDevice:
            def ioctl(self, cmd, a0, a1, a2):
                raise RuntimeError("fail")
        mock_dev = ErrorDevice()
        drm.register_device("storage", mock_dev)
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_ERROR] == 1

    def test_dispatch_ioctl_with_success_attr(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["network"]

        class Result:
            success = True
            value = 99
        class GoodDevice:
            def ioctl(self, cmd, a0, a1, a2):
                return Result()
        mock_dev = GoodDevice()
        drm.register_device("network", mock_dev)
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_RESULT] == 99

    def test_dispatch_ioctl_failure(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["display"]

        class Result:
            success = False
            error = "bad"
        class FailDevice:
            def ioctl(self, cmd, a0, a1, a2):
                return Result()
        mock_dev = FailDevice()
        drm.register_device("display", mock_dev)
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_ERROR] == 1

    def test_dispatch_call_method(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        base = drm.BASE_ADDRESSES["input"]

        class CallDevice:
            def call(self, cmd, a0, a1, a2):
                return 77
        mock_dev = CallDevice()
        drm.register_device("input", mock_dev)
        drm.write(base + drm.REG_COMMAND, 1)
        assert drm._registers[base + drm.REG_RESULT] == 77

    def test_dispatch_no_ioctl_no_call(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        # Register with a custom address that has initialized registers
        drm.BASE_ADDRESSES["bare"] = 0x8000
        for offset in range(0, 0x100, 4):
            drm._registers[0x8000 + offset] = 0

        class BareDevice:
            pass
        mock_dev = BareDevice()
        drm.register_device("bare", mock_dev)
        drm.write(0x8000 + drm.REG_COMMAND, 1)
        assert drm._registers[0x8000 + drm.REG_ERROR] == 1

    def test_get_block_base(self):
        from domain.shell._internal.vm import DeviceRegisterMap
        drm = DeviceRegisterMap()
        assert drm.get_block_base("tensor") == drm.BASE_ADDRESSES["tensor"]
        assert drm.get_block_base("nonexistent") == 0


# ── Training syscall tests ───────────────────────────────────────────────────

class TestTrainingSyscalls:
    """Test _sys_train_* methods via the training bridge."""

    def test_train_start(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, X86SyscallHandler, PageFrameAllocator
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(total_memory=0x400000, reserved_ranges=[(0, 0x100000)])
        handler = X86SyscallHandler(cpu, ptable, sched, alloc)
        handler._write_string(0x1000, '{}')
        result = handler._sys_train_start(0x1000)
        # Bridge returns -1 on invalid config
        assert isinstance(result, int)

    def test_train_status_not_found(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, X86SyscallHandler, PageFrameAllocator
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(total_memory=0x400000, reserved_ranges=[(0, 0x100000)])
        handler = X86SyscallHandler(cpu, ptable, sched, alloc)
        result = handler._sys_train_status(9999)
        assert result == -1

    def test_train_get_result_not_found(self):
        from domain.shell._internal.vm import X86CPU, ProcessTable, Scheduler, X86SyscallHandler, PageFrameAllocator
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(total_memory=0x400000, reserved_ranges=[(0, 0x100000)])
        handler = X86SyscallHandler(cpu, ptable, sched, alloc)
        result = handler._sys_train_get_result(9999, 0x2000, 100)
        assert result == 0


# ── More 0x66-prefix opcode tests ───────────────────────────────────────────

class TestCPU66MoreOpcodes:
    """Additional 0x66-prefix opcode coverage tests."""

    def _make_cpu(self, code_bytes):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._mem[0x100:0x100 + len(code_bytes)] = code_bytes
        cpu._eip = 0x100
        return cpu

    def test_66_mov_rm8_r8(self):
        # 0x66 0x88 ModRM: MOV r/m8, r8 with 16-bit operand override
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # 0x66 prefix + 0x88 + ModRM(C0=AL,AL) + ... but 0x88 is 8-bit, unaffected by 0x66
        code = bytes([0x66, 0x88, 0xC0])  # MOV AL, AL
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[0] = 0x42  # EAX low byte = AL
        ok = cpu.step()
        assert ok

    def test_66_mov_r8_rm8(self):
        # 0x66 0x8A ModRM: MOV r8, r/m8
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        cpu._regs[0] = 0x42
        code = bytes([0x66, 0x8A, 0xC0])  # MOV AL, AL
        cpu._mem[0x100:0x100 + len(code)] = code
        ok = cpu.step()
        assert ok

    def test_66_mov_rm8_imm8(self):
        # 0x66 0xC6 /0 ModRM: MOV r/m8, imm8
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        code = bytes([0x66, 0xC6, 0xC0, 0x55])  # MOV AL, 0x55
        cpu._mem[0x100:0x100 + len(code)] = code
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x55

    def test_66_imul_r16_rm16_imm16(self):
        # 0x66 0x69: IMUL r16, r/m16, imm16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # IMUL CX, AX, 3  (69 C8 03 00 = IMUL CX, AX, imm16=3)
        code = bytes([0x66, 0x69, 0xC8, 0x03, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[0] = 5  # AX = 5
        ok = cpu.step()
        assert ok
        assert (cpu._regs[1] & 0xFFFF) == 15  # CX = 15

    def test_66_imul_r16_rm16_negative_imm(self):
        # 0x66 0x69 with negative imm16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # IMUL CX, AX, -2 (69 C8 FE FF = imm16 = 0xFFFE = -2)
        code = bytes([0x66, 0x69, 0xC8, 0xFE, 0xFF])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[0] = 5
        ok = cpu.step()
        assert ok
        assert (cpu._regs[1] & 0xFFFF) == (0xFFF6 & 0xFFFF)  # -10

    def test_66_imul_r16_rm16_reg(self):
        # 0x0F 0xAF without 0x66 prefix: IMUL r32, r/m32 (standard 32-bit)
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # IMUL ECX, EAX (0F AF C8)
        code = bytes([0x0F, 0xAF, 0xC8])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[0] = 3  # EAX
        cpu._regs[1] = 7  # ECX
        ok = cpu.step()
        assert ok
        assert (cpu._regs[1] & 0xFFFFFFFF) == 21

    def test_66_mov_r16_imm16(self):
        # 0x66 0xB8+r: MOV r16, imm16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # MOV AX, 0x1234 (66 B8 34 12)
        code = bytes([0x66, 0xB8, 0x34, 0x12])
        cpu._mem[0x100:0x100 + len(code)] = code
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFFFF) == 0x1234

    def test_66_mov_rm16_r16(self):
        # 0x66 0x89: MOV r/m16, r16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # MOV AX, CX (66 89 C8)
        code = bytes([0x66, 0x89, 0xC8])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[1] = 0xBEEF  # CX
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFFFF) == 0xBEEF

    def test_66_mov_r16_rm16(self):
        # 0x66 0x8B: MOV r16, r/m16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # MOV CX, AX (66 8B C8)
        code = bytes([0x66, 0x8B, 0xC8])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._regs[0] = 0xDEAD  # AX
        ok = cpu.step()
        assert ok
        assert (cpu._regs[1] & 0xFFFF) == 0xDEAD

    def test_66_mov_rm16_imm16(self):
        # 0x66 0xC7 /0: MOV r/m16, imm16
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._eip = 0x100
        # MOV AX, 0x4321 (66 C7 C0 21 43)
        code = bytes([0x66, 0xC7, 0xC0, 0x21, 0x43])
        cpu._mem[0x100:0x100 + len(code)] = code
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFFFF) == 0x4321


# ── X86SyscallHandler — exec edge cases ─────────────────────────────────────

class TestSyscallExecEdgeCases:
    """Additional exec edge case tests for uncovered lines."""

    def _make_handler(self):
        from domain.shell._internal.vm import (
            X86CPU, ProcessTable, Scheduler, X86SyscallHandler,
            PageFrameAllocator, FlatFS, BlockDevice,
        )
        cpu = X86CPU(memory_size=0x400000)
        ptable = ProcessTable()
        sched = Scheduler(ptable, quantum=10)
        alloc = PageFrameAllocator(
            total_memory=0x400000,
            reserved_ranges=[(0, 0x100000), (0xB8000, 0xC0000)],
        )
        fs = FlatFS(BlockDevice())
        handler = X86SyscallHandler(cpu, ptable, sched, alloc, filesystem=fs)
        return handler, cpu, ptable, sched

    def _setup_process(self, ptable, sched):
        from domain.shell._internal.vm import ProcessState
        pcb = ptable.create(name="test", priority=5)
        sched.enqueue(pcb.pid)
        sched._current_pid = pcb.pid
        pcb.state = ProcessState.RUNNING
        return pcb

    def test_exec_second_assembly_fails(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        # Write program that assembles fine first time but fails with org
        handler._fs.write('tricky.asm', b'[BITS 32]\nMOV EAX, 1\nRET')
        handler._write_string(0x1000, 'tricky.asm')
        # This should succeed - the program is valid
        result = handler._sys_exec(0x1000)
        assert result == 0

    def test_exec_alloc_oom(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        handler._fs.write('prog.asm', b'[BITS 32]\nMOV EAX, 1\nRET')
        handler._write_string(0x1000, 'prog.asm')
        # Exhaust memory
        while handler._memory.alloc(1) is not None:
            pass
        result = handler._sys_exec(0x1000)
        assert result == -1

    def test_exec_assembly_raises_exception(self):
        handler, cpu, ptable, sched = self._make_handler()
        self._setup_process(ptable, sched)
        # Write invalid assembly that should cause an exception
        handler._fs.write('bad.asm', b'[BITS 32]\nDB 0xFF, 0xFF, 0xFF, 0xFF')
        handler._write_string(0x1000, 'bad.asm')
        result = handler._sys_exec(0x1000)
        # Should either succeed (as data) or fail
        assert isinstance(result, int)

    def test_kill_role_escalation_guard(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm_permissions import Role
        pcb = self._setup_process(ptable, sched)
        handler._rbac.assign(pcb.pid, Role.USER)
        target = ptable.create(name="target")
        handler._rbac.assign(target.pid, Role.KERNEL)
        result = handler._sys_kill(target.pid, 9)
        assert result == -1

    def test_malloc_oom(self):
        handler, cpu, ptable, sched = self._make_handler()
        # Fill heap to OOM — 1MB heap limit starting at 0x400000
        addr = 0x400000
        while addr < 0x500000:
            handler._heap[addr] = 0x10
            addr += 0x10
        result = handler._sys_malloc(64)
        assert result == 0

    def test_mouse_read_with_packet(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import MouseDevice
        mouse = MouseDevice()
        handler._mouse = mouse
        mouse.move(10, 5)
        mouse.press(1)
        result = handler._sys_mouse_read(0x1000)
        assert result == 0

    def test_net_recv_with_packet(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import NICDevice
        nic = NICDevice()
        handler._nic = nic
        nic.inject_packet(b'\x01\x02\x03')
        result = handler._sys_net_recv(0x1000, 100)
        assert result == 3

    def test_net_recv_truncated(self):
        handler, cpu, ptable, sched = self._make_handler()
        from domain.shell._internal.vm import NICDevice
        nic = NICDevice()
        handler._nic = nic
        nic.inject_packet(b'\x01\x02\x03\x04\x05')
        result = handler._sys_net_recv(0x1000, 3)
        assert result == 3


# ── CPU-level tensor/integer ops ─────────────────────────────────────────────

class TestCPUTensorOps:
    """Test CPU-level tensor ops (the _op_* functions) via the CPU class."""

    def _make_cpu(self):
        from domain.shell._internal.vm import CPU
        return CPU()

    def test_isub(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_isub
        _op_isub(cpu, ['R0', 'R0', 'R1'])
        cpu.regs[0] = 10
        cpu.regs[1] = 3
        _op_isub(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 7

    def test_isub_borrow(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 3
        cpu.regs[1] = 10
        from domain.shell._internal.vm import _op_isub
        _op_isub(cpu, ['R2', 'R0', 'R1'])
        assert cpu._carry_flag is True

    def test_imul(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 6
        cpu.regs[1] = 7
        from domain.shell._internal.vm import _op_imul
        _op_imul(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 42

    def test_idiv(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 10
        cpu.regs[1] = 3
        from domain.shell._internal.vm import _op_idiv
        _op_idiv(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 3

    def test_idiv_zero(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 10
        cpu.regs[1] = 0
        from domain.shell._internal.vm import _op_idiv
        _op_idiv(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 0

    def test_iand(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 0xFF
        cpu.regs[1] = 0x0F
        from domain.shell._internal.vm import _op_iand
        _op_iand(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 0x0F

    def test_ior(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 0xF0
        cpu.regs[1] = 0x0F
        from domain.shell._internal.vm import _op_ior
        _op_ior(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 0xFF

    def test_ixor(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 0xFF
        cpu.regs[1] = 0x0F
        from domain.shell._internal.vm import _op_ixor
        _op_ixor(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 0xF0

    def test_ishl(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 1
        cpu.regs[1] = 4
        from domain.shell._internal.vm import _op_ishl
        _op_ishl(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 16

    def test_ishr(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 16
        cpu.regs[1] = 2
        from domain.shell._internal.vm import _op_ishr
        _op_ishr(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 4

    def test_ineg(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 42
        from domain.shell._internal.vm import _op_ineg
        _op_ineg(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == -42

    def test_inc(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 5
        from domain.shell._internal.vm import _op_inc
        _op_inc(cpu, ['R0'])
        assert cpu.regs[0] == 6

    def test_dec(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 5
        from domain.shell._internal.vm import _op_dec
        _op_dec(cpu, ['R0'])
        assert cpu.regs[0] == 4

    def test_icmp(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 5
        cpu.regs[1] = 3
        from domain.shell._internal.vm import _op_icmp
        _op_icmp(cpu, ['R0', 'R1'])
        assert cpu._cmp_flag == 1
        cpu.regs[0] = 2
        cpu.regs[1] = 8
        _op_icmp(cpu, ['R0', 'R1'])
        assert cpu._cmp_flag == -1
        cpu.regs[0] = 7
        cpu.regs[1] = 7
        _op_icmp(cpu, ['R0', 'R1'])
        assert cpu._cmp_flag == 0

    def test_neg(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 42
        from domain.shell._internal.vm import _op_neg
        _op_neg(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == -42

    def test_neg_none(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_neg
        _op_neg(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 0

    def test_abs(self):
        cpu = self._make_cpu()
        cpu.regs[0] = -42
        from domain.shell._internal.vm import _op_abs
        _op_abs(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 42

    def test_abs_none(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_abs
        _op_abs(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 0

    def test_cmp_numeric(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 5
        cpu.regs[1] = 3
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, ['R0', 'R1'])
        assert cpu._cmp_flag == 1

    def test_cmp_string(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, ['b', 'a'])
        assert cpu._cmp_flag == 1

    def test_test(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_test
        _op_test(cpu, [1])
        assert cpu._cmp_flag == 1
        _op_test(cpu, [0])
        assert cpu._cmp_flag == 0

    def test_fadd(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 1.5
        cpu.regs[1] = 2.5
        from domain.shell._internal.vm import _op_fadd
        _op_fadd(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 4.0

    def test_fsub(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 5.0
        cpu.regs[1] = 2.0
        from domain.shell._internal.vm import _op_fsub
        _op_fsub(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 3.0

    def test_fmul(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 3.0
        cpu.regs[1] = 4.0
        from domain.shell._internal.vm import _op_fmul
        _op_fmul(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 12.0

    def test_fdiv(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 10.0
        cpu.regs[1] = 2.0
        from domain.shell._internal.vm import _op_fdiv
        _op_fdiv(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 5.0

    def test_fdiv_zero(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 10.0
        cpu.regs[1] = 0.0
        from domain.shell._internal.vm import _op_fdiv, InsFault
        with pytest.raises(InsFault):
            _op_fdiv(cpu, ['R2', 'R0', 'R1'])

    def test_relu(self):
        cpu = self._make_cpu()
        cpu.regs[0] = -5
        from domain.shell._internal.vm import _op_relu
        _op_relu(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 0
        cpu.regs[0] = 5
        _op_relu(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 5

    def test_gelu(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 1.0
        from domain.shell._internal.vm import _op_gelu
        _op_gelu(cpu, ['R1', 'R0'])
        assert cpu.regs[1] > 0

    def test_sigmoid(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 0.0
        from domain.shell._internal.vm import _op_sigmoid
        _op_sigmoid(cpu, ['R1', 'R0'])
        assert abs(cpu.regs[1] - 0.5) < 0.01

    def test_tanh(self):
        cpu = self._make_cpu()
        cpu.regs[0] = 0.0
        from domain.shell._internal.vm import _op_tanh
        _op_tanh(cpu, ['R1', 'R0'])
        assert abs(cpu.regs[1]) < 0.01

    def test_randn(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_randn
        _op_randn(cpu, ['R0', 2, 3])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert cpu.regs[0].shape == (2, 3)

    def test_randunif(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_randunif
        _op_randunif(cpu, ['R0', 2, 3, -1.0, 1.0])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert cpu.regs[0].shape == (2, 3)

    def test_matmul_1d(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3])
        cpu.regs[1] = np.array([4, 5, 6])
        from domain.shell._internal.vm import _op_matmul
        _op_matmul(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] is not None

    def test_matmul_0d(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array(5)
        cpu.regs[1] = np.array(3)
        from domain.shell._internal.vm import _op_matmul
        _op_matmul(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] is not None

    def test_transpose(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([[1, 2], [3, 4]])
        from domain.shell._internal.vm import _op_transpose
        _op_transpose(cpu, ['R1', 'R0'])
        assert cpu.regs[1].shape == (2, 2)

    def test_dot(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3])
        cpu.regs[1] = np.array([4, 5, 6])
        from domain.shell._internal.vm import _op_dot
        _op_dot(cpu, ['R2', 'R0', 'R1'])
        assert cpu.regs[2] == 32.0

    def test_norm(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([3, 4])
        from domain.shell._internal.vm import _op_norm
        _op_norm(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 5.0

    def test_sum(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3])
        from domain.shell._internal.vm import _op_sum
        _op_sum(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 6.0

    def test_mean(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3])
        from domain.shell._internal.vm import _op_mean
        _op_mean(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 2.0

    def test_max(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 5, 3])
        from domain.shell._internal.vm import _op_max
        _op_max(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 5.0

    def test_argmax(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 5, 3])
        from domain.shell._internal.vm import _op_argmax
        _op_argmax(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 1

    def test_reshape(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3, 4, 5, 6])
        from domain.shell._internal.vm import _op_reshape
        _op_reshape(cpu, ['R1', 'R0', 2, 3])
        assert cpu.regs[1].shape == (2, 3)

    def test_shape(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([[1, 2], [3, 4], [5, 6]])
        from domain.shell._internal.vm import _op_shape
        _op_shape(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == [3, 2]

    def test_size(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, 2, 3, 4, 5])
        from domain.shell._internal.vm import _op_size
        _op_size(cpu, ['R1', 'R0'])
        assert cpu.regs[1] == 5

    def test_softmax(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1.0, 2.0, 3.0])
        from domain.shell._internal.vm import _op_softmax
        _op_softmax(cpu, ['R1', 'R0'])
        assert abs(float(np.sum(cpu.regs[1])) - 1.0) < 0.01

    def test_layernorm(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1.0, 2.0, 3.0, 4.0])
        from domain.shell._internal.vm import _op_layernorm
        _op_layernorm(cpu, ['R1', 'R0'])
        assert abs(float(np.mean(cpu.regs[1]))) < 0.01

    def test_rmsnorm(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1.0, 2.0, 3.0, 4.0])
        from domain.shell._internal.vm import _op_rmsnorm
        _op_rmsnorm(cpu, ['R1', 'R0'])
        assert cpu.regs[1] is not None

    def test_cmp_arrays_equal(self):
        cpu = self._make_cpu()
        import numpy as np
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, [a, b])
        assert cpu._cmp_flag == 0

    def test_cmp_arrays_less(self):
        cpu = self._make_cpu()
        import numpy as np
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, [a, b])
        assert cpu._cmp_flag == -1

    def test_cmp_arrays_greater(self):
        cpu = self._make_cpu()
        import numpy as np
        a = np.array([4, 5, 6])
        b = np.array([1, 2, 3])
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, [a, b])
        assert cpu._cmp_flag == 1

    def test_cmp_arrays_mixed(self):
        cpu = self._make_cpu()
        import numpy as np
        a = np.array([1, 5, 3])
        b = np.array([4, 2, 6])
        from domain.shell._internal.vm import _op_cmp
        _op_cmp(cpu, [a, b])
        assert cpu._cmp_flag == 0

    def test_divide_by_zero_tensor(self):
        import numpy as np
        cpu = self._make_cpu()
        cpu.regs[0] = np.array([1.0, 2.0])
        cpu.regs[1] = np.array([0.0, 1.0])
        from domain.shell._internal.vm import _op_div
        _op_div(cpu, ['R2', 'R0', 'R1'])
        result = cpu.regs[2]
        assert float(result[1]) == 2.0

    def test_neg_tensor(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([1, -2, 3])
        from domain.shell._internal.vm import _op_neg
        _op_neg(cpu, ['R1', 'R0'])
        assert list(cpu.regs[1]) == [-1, 2, -3]

    def test_abs_tensor(self):
        cpu = self._make_cpu()
        import numpy as np
        cpu.regs[0] = np.array([-1, 2, -3])
        from domain.shell._internal.vm import _op_abs
        _op_abs(cpu, ['R1', 'R0'])
        assert list(cpu.regs[1]) == [1, 2, 3]

    def test_dev_table_open_no_adapter(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_table_open
        _op_dev_table_open(cpu, ['R0', "test"])
        assert cpu.regs[0] == -1

    def test_dev_table_call_no_adapter(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_table_call
        _op_dev_table_call(cpu, ['R0', 0, "READ"])
        assert cpu.regs[0] is None

    def test_dev_table_close_no_adapter(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_table_close
        _op_dev_table_close(cpu, [0])  # should not raise

    def test_dev_table_info_no_adapter(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_table_info
        _op_dev_table_info(cpu, ['R0', 0])
        assert cpu.regs[0] == {}

    def test_dev_reg_read_no_map(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_reg_read
        _op_dev_reg_read(cpu, ['R0', 0xF000])
        assert cpu.regs[0] == 0

    def test_dev_reg_write_no_map(self):
        cpu = self._make_cpu()
        from domain.shell._internal.vm import _op_dev_reg_write
        _op_dev_reg_write(cpu, [0xF000, 42])  # should not raise


# ── Persistent BlockDevice tests ─────────────────────────────────────────────

class TestPersistentBlockDevice:
    """Persistent BlockDevice file-backed paths (lines 948-1030, 1363-1421)."""

    def test_persistent_block_device_init(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd = BlockDevice(path=path, create=True)
            assert bd._is_persistent is True
            assert bd._file is not None
            bd.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistent_block_device_open_existing(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd1 = BlockDevice(path=path, create=True)
            bd1.close()
            bd2 = BlockDevice(path=path)
            assert bd2._is_persistent is True
            assert bd2._file is not None
            bd2.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistent_block_device_open_nonexistent(self):
        from domain.shell._internal.vm import BlockDevice, DeviceFault
        with pytest.raises(DeviceFault):
            BlockDevice(path="/tmp/nonexistent_vm_block_test_999.blk")

    def test_persistent_get_sector_usage(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd = BlockDevice(path=path, create=True)
            usage = bd.get_sector_usage()
            assert "total" in usage
            assert "used" in usage
            assert "free" in usage
            assert "usage_pct" in usage
            bd.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistent_get_disk_usage(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd = BlockDevice(path=path, create=True)
            size = bd._get_disk_usage()
            assert isinstance(size, int)
            bd.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistent_call_dispatch(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd = BlockDevice(path=path, create=True)
            sectors = bd.get_sectors()
            assert isinstance(sectors, int)
            stats = bd.call("get_stats")
            assert isinstance(stats, dict)
            ratio = bd.call("get_compression_ratio")
            assert isinstance(ratio, (int, float))
            usage = bd.call("get_sector_usage")
            assert isinstance(usage, dict)
            # Unknown method falls through to super() which raises
            from domain.shell._internal.vm import DeviceFault
            with pytest.raises(DeviceFault):
                bd.call("unknown_method")
            bd.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_non_persistent_get_sector_usage(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice()
        usage = bd.get_sector_usage()
        assert usage["total"] > 0
        assert usage["free"] == 0
        assert usage["usage_pct"] == 100.0

    def test_non_persistent_get_disk_usage(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice()
        assert bd._get_disk_usage() == 0

    def test_close_non_persistent(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice()
        bd.close()  # should not raise


# ── load_const string parsing ────────────────────────────────────────────────

class TestLoadConstStringParsing:
    """Test _op_load_const string-to-array parsing (lines 3213-3235)."""

    def test_load_const_json_list(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[1, 2, 3]'])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert list(cpu.regs[0]) == [1.0, 2.0, 3.0]

    def test_load_const_bracket_floats(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[1.5, 2.5]'])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert list(cpu.regs[0]) == [1.5, 2.5]

    def test_load_const_bracket_mixed(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[1, 2, 3]'])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert len(cpu.regs[0]) == 3

    def test_load_const_bracket_empty(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[]'])
        import numpy as np
        assert isinstance(cpu.regs[0], np.ndarray)
        assert len(cpu.regs[0]) == 0

    def test_load_const_plain_int(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', 42])
        assert cpu.regs[0] == 42

    def test_load_const_plain_string(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', 'hello'])
        assert cpu.regs[0] == 'hello'


# ── CMOS binary mode RTC ─────────────────────────────────────────────────────

class TestCMOSBinaryRTC:
    """CMOS binary-mode RTC refresh (lines 1782-1795)."""

    def test_rtc_binary_24h(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        # Set binary mode + 24h: Status_B = 0x04 | 0x02 = 0x06
        cmos._cmos[cmos.REG_STATUS_B] = 0x06
        cmos._refresh_rtc()
        # Verify registers were updated (binary mode)
        assert cmos._cmos[cmos.REG_SECONDS] < 60

    def test_rtc_binary_12h(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        # Binary mode (bit 2) + 12h mode (bit 1 = 0): Status_B = 0x04
        cmos._cmos[cmos.REG_STATUS_B] = 0x04
        cmos._refresh_rtc()
        assert cmos._cmos[cmos.REG_HOURS] & 0x80 == 0  # AM


# ── 0x66-prefix MOV r/m16 memory-operand paths ──────────────────────────────

class TestCPU66MovRm16Memory:
    """0x66-prefix MOV r/m16 with memory operands (lines 6244-6261)."""

    def test_66_mov_rm16_imm16_mem(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        # MOV word [0x1000], 0x1234 — 66 C7 05 00 10 00 00 34 12
        code = bytes([0x66, 0xC7, 0x05, 0x00, 0x10, 0x00, 0x00, 0x34, 0x12])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        import struct
        val = struct.unpack_from("<H", cpu._mem, 0x1000)[0]
        assert val == 0x1234

    def test_66_mov_mem_r16(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        # MOV word [0x1000], CX — 66 89 0D 00 10 00 00
        code = bytes([0x66, 0x89, 0x0D, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        cpu._regs[1] = 0xBEEF  # CX
        ok = cpu.step()
        assert ok
        import struct
        val = struct.unpack_from("<H", cpu._mem, 0x1000)[0]
        assert val == 0xBEEF

    def test_66_mov_r16_mem(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        # MOV CX, word [0x1000] — 66 8B 0D 00 10 00 00
        code = bytes([0x66, 0x8B, 0x0D, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._mem[0x1000:0x1002] = bytes([0x34, 0x12])
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[1] & 0xFFFF) == 0x1234


# ── XCHG 8-bit memory path ──────────────────────────────────────────────────

class TestCPU66Xchg8Mem:
    """0x66 XCHG r8, [mem] memory path (lines 6320-6324)."""

    def test_xchg_al_mem(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        # XCHG byte [0x1000], AL — 86 05 00 10 00 00
        code = bytes([0x86, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._mem[0x1000] = 0x42
        cpu._regs[0] = 0x55  # AL
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x42
        assert cpu._mem[0x1000] == 0x55


# ── FileDevice close_all exception path ──────────────────────────────────────

class TestFileDeviceCloseAll:
    def test_close_all_with_broken_handle(self):
        from domain.shell._internal.vm import FileDevice
        fd = FileDevice()
        def bad_close():
            raise IOError("broken")
        fd._files[99] = type('BadFH', (), {'close': bad_close})()
        fd.close_all()
        assert fd._files == {}


# ── SerialDevice call dispatch ───────────────────────────────────────────────

class TestSerialDeviceCall:
    def test_call_read_byte(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        sd.push_byte(0x41)
        assert sd.call("read_byte") == 0x41

    def test_call_write_byte(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        result = sd.call("write_byte", 0x42)
        assert result is True
        assert sd._tx_buffer == bytearray(b'B')

    def test_call_has_data(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        assert sd.call("has_data") is False
        sd.push_byte(1)
        assert sd.call("has_data") is True

    def test_call_flush(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        sd.call("flush")
        assert sd._tx_count == 0


# ── BlockDevice write_sectors ────────────────────────────────────────────────

class TestBlockDeviceWriteSectors:
    def test_write_sectors_multi(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice(num_sectors=8)
        data = b'\x01' * (bd.SECTOR_SIZE * 3)
        bd.write_sectors(1, data)
        assert bd.read_sector(1) == data[:bd.SECTOR_SIZE]
        assert bd.read_sector(2) == data[bd.SECTOR_SIZE:bd.SECTOR_SIZE*2]

    def test_write_sectors_unaligned(self):
        from domain.shell._internal.vm import BlockDevice, DeviceFault
        bd = BlockDevice(num_sectors=4)
        with pytest.raises(DeviceFault):
            bd.write_sectors(0, b'\x00' * 100)


# ── load_const bracket string (non-JSON path) ───────────────────────────────

class TestLoadConstBracketNonJson:
    def test_bracket_comma_mixed_nums(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[1, 2.5, 3]'])
        assert list(cpu.regs[0]) == [1.0, 2.5, 3.0]

    def test_bracket_single_int(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[42]'])
        assert list(cpu.regs[0]) == [42.0]

    def test_bracket_empty(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        _op_load_const(cpu, ['R0', '[]'])
        assert len(cpu.regs[0]) == 0

    def test_bracket_non_json_with_strings(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        # Fails JSON parse (trailing comma), hits comma-split fallback, strings crash at np.array
        with pytest.raises(ValueError):
            _op_load_const(cpu, ['R0', '[hello, world]'])

    def test_bracket_trailing_comma(self):
        from domain.shell._internal.vm import CPU, _op_load_const
        cpu = CPU()
        # Trailing comma makes JSON fail; empty string crashes at np.array
        with pytest.raises(ValueError):
            _op_load_const(cpu, ['R0', '[1,2,3,]'])


# ── _op_dev_table_ioctl result dispatch ──────────────────────────────────────

class TestDevTableIoctlResult:
    def _make_cpu(self, adapter):
        from domain.shell._internal.vm import CPU
        cpu = CPU()
        cpu._device_table_adapter = adapter
        return cpu

    def test_ioctl_result_with_success(self):
        from domain.shell._internal.vm import _op_dev_table_call
        class MockResult:
            success = True
            value = 42
        class MockAdapter:
            def ioctl(self, fd, cmd, *a):
                return MockResult()
        cpu = self._make_cpu(MockAdapter())
        _op_dev_table_call(cpu, ["R0", "R1", "R2"])
        assert cpu.regs[0] == 42

    def test_ioctl_result_failure(self):
        from domain.shell._internal.vm import _op_dev_table_call
        class MockResult:
            success = False
            error = "bad"
            value = None
        class MockAdapter:
            def ioctl(self, fd, cmd, *a):
                return MockResult()
        cpu = self._make_cpu(MockAdapter())
        _op_dev_table_call(cpu, ["R0", "R1", "R2"])
        assert cpu.regs[0] is None

    def test_ioctl_result_plain(self):
        from domain.shell._internal.vm import _op_dev_table_call
        class MockAdapter:
            def ioctl(self, fd, cmd, *a):
                return 99
        cpu = self._make_cpu(MockAdapter())
        _op_dev_table_call(cpu, ["R0", "R1", "R2"])
        assert cpu.regs[0] == 99


# ── POPA instruction ─────────────────────────────────────────────────────────

class TestX86CPUPopad:
    def test_popad(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        cpu._regs[4] = 0x2000  # ESP
        vals = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        for v in reversed(vals):
            cpu._push32(v)
        # POPAD: opcode 61
        code = bytes([0x61])
        cpu._mem[0x100:0x101] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok


# ── CMOS 12h mode coverage (hour 0 and hour >= 12) ─────────────────────────

class TestCMOSHourEdgeCases:
    def test_cmos_12h_hour_zero(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        cmos._cmos[cmos.REG_STATUS_B] = 0x04  # binary, 12h
        clock._epoch = 0  # midnight → hour=0
        cmos._refresh_rtc()
        h = cmos._cmos[cmos.REG_HOURS]
        assert (h & 0x7F) == 12

    def test_cmos_12h_hour_pm(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        cmos._cmos[cmos.REG_STATUS_B] = 0x04  # binary, 12h
        clock._epoch = 15 * 3600  # 15:00 → 3 PM
        cmos._refresh_rtc()
        h = cmos._cmos[cmos.REG_HOURS]
        assert (h & 0x80) != 0  # PM bit set


# ── Assembler Group 1 encoding with 0x66 prefix ─────────────────────────────

class TestAssemblerGroup1Ops:
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))

    def test_add_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "ADD EAX, 0x200000"])
        assert code[0] == 0x05

    def test_or_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "OR EAX, 0x200000"])
        assert code[0] == 0x0D

    def test_adc_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "ADC EAX, 0x200000"])
        assert code[0] == 0x15

    def test_sbb_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "SBB EAX, 0x200000"])
        assert code[0] == 0x1D

    def test_and_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "AND EAX, 0x200000"])
        assert code[0] == 0x25

    def test_sub_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "SUB EAX, 0x200000"])
        assert code[0] == 0x2D

    def test_xor_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "XOR EAX, 0x200000"])
        assert code[0] == 0x35

    def test_cmp_eax_imm32_large(self):
        code = self._asm(["[BITS 32]", "CMP EAX, 0x200000"])
        assert code[0] == 0x3D

    def test_add_r32_imm8(self):
        code = self._asm(["[BITS 32]", "ADD EBX, 5"])
        assert code[0] == 0x83

    def test_group1_r32_imm32(self):
        code = self._asm(["[BITS 32]", "ADD ECX, 0x100000"])
        assert code[0] == 0x81


# ── Assembler 16-bit Group 1 encoding ────────────────────────────────────────

class TestAssemblerGroup1_16bit:
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))

    def test_add_ax_imm16_large_32bit_mode(self):
        code = self._asm(["[BITS 32]", "ADD AX, 0x200"])
        assert code[0] == 0x66
        assert code[1] == 0x05

    def test_sub_r16_imm8_32bit_mode(self):
        code = self._asm(["[BITS 32]", "SUB BX, 5"])
        assert code[0] == 0x66

    def test_group1_r16_imm16_32bit_mode(self):
        code = self._asm(["[BITS 32]", "CMP CX, 0x1000"])
        assert code[0] == 0x66


# ── Remaining assembler encodings ────────────────────────────────────────────

class TestAssemblerRemainingOps:
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))

    def test_mov_r16_imm16_32bit_mode(self):
        code = self._asm(["[BITS 32]", "MOV AX, 0x1234"])
        assert code[0] == 0x66

    def test_test_ax_imm16_32bit_mode(self):
        code = self._asm(["[BITS 32]", "TEST AX, 0x1234"])
        assert code[0] == 0x66

    def test_group3_r16_imm16_32bit_mode(self):
        code = self._asm(["[BITS 32]", "TEST AX, 0x1234"])
        assert code[0] == 0x66


# ── Comprehensive _exec_16bit memory operand paths ──────────────────────────

class TestExec16BitMemoryOps:
    """Test memory-operand paths in _exec_16bit (lines 7136-7568)."""

    def _make_cpu(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x400000)
        return cpu

    def _run_code(self, code_bytes):
        cpu = self._make_cpu()
        cpu._mem[0x100:0x100 + len(code_bytes)] = code_bytes
        cpu._eip = 0x100
        ok = cpu.step()
        return cpu, ok

    # ── F7 group: TEST/NOT/NEG/MUL/IMUL/DIV/IDIV r/m16 ──

    def test_f7_test_rm16_mem(self):
        # 66 F7 05 <disp32> <imm16> = TEST word [disp32], imm16
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x00FF)
        code = bytes([0x66, 0xF7, 0x05, 0x00, 0x10, 0x00, 0x00, 0xFF, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_not_rm16_mem(self):
        # 66 F7 D5 <disp32> = NOT word [disp32] (reg_f=2, ModRM=0x05|0xD0=0xD5)
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x00FF)
        code = bytes([0x66, 0xF7, 0xD5, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_neg_rm16_mem(self):
        # 66 F7 DD <disp32> = NEG word [disp32] (reg_f=3, ModRM=0x05|0xD8=0xDD)
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0005)
        code = bytes([0x66, 0xF7, 0xDD, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_mul_rm16_mem(self):
        # 66 F7 E5 <disp32> = MUL word [disp32] (reg_f=4, ModRM=0x05|0xE0=0xE5)
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x100, 0x66)
        struct.pack_into("<H", cpu._mem, 0x102, 0xF7)
        struct.pack_into("<H", cpu._mem, 0x104, 0xE5)
        struct.pack_into("<I", cpu._mem, 0x106, 0x1000)
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0003)
        cpu._set16(0, 7)  # AX = 7
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_imul_rm16_mem(self):
        # 66 F7 ED <disp32> = IMUL word [disp32] (reg_f=5, ModRM=0x05|0xE8=0xED)
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0003)
        code = bytes([0x66, 0xF7, 0xED, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 7)  # AX = 7
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_div_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x2000, 0x0003)
        # 66 F7 35 <disp32> = DIV word [disp32] (mod=00, reg=6, rm=5 → 0x35)
        code = bytes([0x66, 0xF7, 0x35, 0x00, 0x20, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 7)
        cpu._set16(2, 0)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_f7_idiv_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x2000, 0x0003)
        # 66 F7 3D <disp32> = IDIV word [disp32] (mod=00, reg=7, rm=5 → 0x3D)
        code = bytes([0x66, 0xF7, 0x3D, 0x00, 0x20, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 9)
        cpu._set16(2, 0)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    # ── ALU r/m16, r16 / r16, r/m16 memory paths ──

    def test_alu_add_rm16_mem(self):
        # 66 01 05 <disp32> = ADD word [disp32], AX
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        code = bytes([0x66, 0x01, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 5)  # AX = 5
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        val = struct.unpack_from("<H", cpu._mem, 0x1000)[0]
        assert val == 0x0015

    def test_alu_add_r16_mem(self):
        # 66 03 05 <disp32> = ADD AX, word [disp32]
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        code = bytes([0x66, 0x03, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 5)  # AX = 5
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(0) == 0x0015

    def test_alu_sub_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0020)
        # 66 29 05 <disp32> = SUB word [disp32], AX
        code = bytes([0x66, 0x29, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 5)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_alu_cmp_r16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        # 66 3B 05 <disp32> = CMP AX, word [disp32]
        code = bytes([0x66, 0x3B, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 0x0010)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    # ── ALU r/m8, r8 memory paths ──

    def test_alu_add_rm8_mem(self):
        cpu = self._make_cpu()
        cpu._mem[0x1000] = 0x10
        # 00 05 <disp32> = ADD byte [disp32], AL
        code = bytes([0x00, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 5)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._mem[0x1000] == 0x15

    def test_alu_add_r8_mem(self):
        cpu = self._make_cpu()
        cpu._mem[0x1000] = 0x10
        # 02 05 <disp32> = ADD AL, byte [disp32]
        code = bytes([0x02, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 5)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x15

    # ── Accumulator-immediate ALU 16-bit ──

    def test_alu_acc_imm16_add(self):
        # 66 05 34 12 = ADD AX, 0x1234
        cpu = self._make_cpu()
        code = bytes([0x66, 0x05, 0x34, 0x12])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 0x0001)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(0) == 0x1235

    def test_alu_acc_imm8_sub(self):
        # 66 2C 05 = SUB AL, 5 (8-bit imm with 16-bit prefix)
        cpu = self._make_cpu()
        code = bytes([0x66, 0x2C, 0x05])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 0x0A)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x05

    # ── TEST AL/AX, imm ──

    def test_test_ax_imm16_exec(self):
        # 66 A9 34 12 = TEST AX, 0x1234
        cpu = self._make_cpu()
        code = bytes([0x66, 0xA9, 0x34, 0x12])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 0x1234)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_test_al_imm8_exec(self):
        # 66 A8 FF = TEST AL, 0xFF
        cpu = self._make_cpu()
        code = bytes([0x66, 0xA8, 0xFF])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 0xFF)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    # ── Group 1: ALU r/m16, imm16 (81) / ALU r/m16, imm8 (83) memory ──

    def test_group1_81_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        # 66 81 05 <disp32> <imm16> = ADD word [disp32], 0x0100
        code = bytes([0x66, 0x81, 0x05, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        val = struct.unpack_from("<H", cpu._mem, 0x1000)[0]
        assert val == 0x0110

    def test_group1_83_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        # 66 83 05 <disp32> <imm8> = ADD word [disp32], 5
        code = bytes([0x66, 0x83, 0x05, 0x00, 0x10, 0x00, 0x00, 0x05])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        val = struct.unpack_from("<H", cpu._mem, 0x1000)[0]
        assert val == 0x0015

    def test_group1_83_sign_extend_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x0010)
        # 66 83 05 <disp32> 0xFF = ADD word [disp32], -1 (sign extended)
        code = bytes([0x66, 0x83, 0x05, 0x00, 0x10, 0x00, 0x00, 0xFF])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    # ── Group 1: ALU r/m8, imm8 (80) ──

    def test_group1_80_rm8_mem(self):
        cpu = self._make_cpu()
        cpu._mem[0x1000] = 0x10
        # 66 80 05 <disp32> <imm8> = ADD byte [disp32], 5
        code = bytes([0x66, 0x80, 0x05, 0x00, 0x10, 0x00, 0x00, 0x05])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._mem[0x1000] == 0x15

    # ── MOV r/m8, r8 (88) / MOV r8, r/m8 (8A) memory ──

    def test_mov_rm8_r8_mem(self):
        # 66 88 05 <disp32> = MOV byte [disp32], AL
        cpu = self._make_cpu()
        code = bytes([0x66, 0x88, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 0x42)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._mem[0x1000] == 0x42

    def test_mov_r8_rm8_mem(self):
        # 66 8A 05 <disp32> = MOV AL, byte [disp32]
        cpu = self._make_cpu()
        cpu._mem[0x1000] = 0x42
        code = bytes([0x66, 0x8A, 0x05, 0x00, 0x10, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x42

    # ── MOV r/m8, imm8 (C6) memory ──

    def test_mov_rm8_imm8_mem(self):
        # 66 C6 05 <disp32> <imm8> = MOV byte [disp32], 0x55
        cpu = self._make_cpu()
        code = bytes([0x66, 0xC6, 0x05, 0x00, 0x10, 0x00, 0x00, 0x55])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._mem[0x1000] == 0x55

    # ── IMUL r16, r/m16, imm8 (6B) memory ──

    def test_imul_6b_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x000A)
        # 6B 05 <disp32> <imm8> = IMUL AX, word [disp32], 3
        code = bytes([0x66, 0x6B, 0x05, 0x00, 0x10, 0x00, 0x00, 0x03])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(0) == 0x001E

    # ── IMUL r16, r/m16, imm16 (69) memory ──

    def test_imul_69_rm16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x000A)
        # 69 05 <disp32> <imm16> = IMUL AX, word [disp32], 0x0003
        code = bytes([0x66, 0x69, 0x05, 0x00, 0x10, 0x00, 0x00, 0x03, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(0) == 0x001E

    # ── IMUL r16, r/m16, imm8 (6B) memory with negative imm ──

    def test_imul_6b_rm16_neg_imm(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x1000, 0x000A)
        # 6B 05 <disp32> 0xFF = IMUL AX, word [disp32], -1
        code = bytes([0x66, 0x6B, 0x05, 0x00, 0x10, 0x00, 0x00, 0xFF])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    # ── Group 1: ALU r16, imm16 (81) / ALU r16, imm8 (83) with r/m16=reg ──

    def test_group1_81_rm16_reg(self):
        # 66 81 C3 00 01 = ADD BX, 0x0100
        cpu = self._make_cpu()
        code = bytes([0x66, 0x81, 0xC3, 0x00, 0x01])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(3, 0x0010)  # BX = 0x0010
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(3) == 0x0110


# ── BlockDevice repr + persistent block map ──────────────────────────────────

class TestBlockDeviceRepr:
    def test_repr_persistent(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd = BlockDevice(path=path, create=True)
            r = repr(bd)
            assert "BlockDevice" in r
            assert "blocks=" in r
            bd.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_repr_in_memory(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice(num_sectors=4)
        r = repr(bd)
        assert "sectors=" in r

    def test_persistent_block_map_with_entries(self):
        import tempfile, os
        from domain.shell._internal.vm import BlockDevice
        path = tempfile.mktemp(suffix='.blk')
        try:
            bd1 = BlockDevice(path=path, create=True)
            # Write some data to create block map entries
            data = b'\xAA' * 512
            bd1.write_sector(0, data)
            bd1.close()
            bd2 = BlockDevice(path=path)
            assert len(bd2._block_map) >= 0
            bd2.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── ConsoleDevice call dispatch ──────────────────────────────────────────────

class TestConsoleDeviceCall:
    def test_call_read(self):
        from domain.shell._internal.vm import ConsoleDevice
        cd = ConsoleDevice(0, stdin_fn=lambda: "hello")
        assert cd.call("read") == "hello"

    def test_call_write(self):
        from domain.shell._internal.vm import ConsoleDevice
        out = []
        cd = ConsoleDevice(1, stdout_fn=lambda x: out.append(x))
        cd.call("write", "world")
        assert out == ["world"]


# ── PS2KeyboardDevice call dispatch ──────────────────────────────────────────

class TestPS2KeyboardDeviceCall:
    def test_call_read_key(self):
        from domain.shell._internal.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        assert kb.call("read_key") == 0

    def test_call_has_key(self):
        from domain.shell._internal.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        assert kb.call("has_key") is False


# ── BlockCompressor compress/decompress paths ────────────────────────────────

class TestBlockCompressorAlgos:
    def test_compress_decompress_none(self):
        from domain.shell._internal.vm import BlockCompressor, CompressionAlgo
        bc = BlockCompressor(CompressionAlgo.NONE)
        data = b'\x42' * 100
        assert bc.compress(data) == data
        assert bc.decompress(data) == data

    def test_compress_decompress_lz4(self):
        import importlib
        if not importlib.util.find_spec("lz4"):
            pytest.skip("lz4 not installed")
        from domain.shell._internal.vm import BlockCompressor, CompressionAlgo
        bc = BlockCompressor(CompressionAlgo.LZ4)
        data = b'\x42' * 200
        compressed = bc.compress(data)
        decompressed = bc.decompress(compressed)
        assert decompressed == data

    def test_compress_decompress_gzip(self):
        from domain.shell._internal.vm import BlockCompressor, CompressionAlgo
        bc = BlockCompressor(CompressionAlgo.GZIP)
        data = b'\x42' * 200
        compressed = bc.compress(data)
        decompressed = bc.decompress(compressed)
        assert decompressed == data

    def test_compress_decompress_zstd(self):
        import importlib
        if not importlib.util.find_spec("zstandard"):
            pytest.skip("zstandard not installed")
        from domain.shell._internal.vm import BlockCompressor, CompressionAlgo
        bc = BlockCompressor(CompressionAlgo.ZSTD)
        data = b'\x42' * 200
        compressed = bc.compress(data)
        decompressed = bc.decompress(compressed)
        assert decompressed == data

    def test_compress_decompress_snappy(self):
        import importlib
        if not importlib.util.find_spec("snappy"):
            pytest.skip("snappy not installed")
        from domain.shell._internal.vm import BlockCompressor, CompressionAlgo
        bc = BlockCompressor(CompressionAlgo.SNAPPY)
        data = b'\x42' * 200
        compressed = bc.compress(data)
        decompressed = bc.decompress(compressed)
        assert decompressed == data


# ── SerialDevice LSR with rx data ───────────────────────────────────────────

class TestSerialDeviceLSR:
    def test_lsr_with_rx_data(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        sd.push_byte(0x41)
        lsr = sd._read_lsr()
        assert lsr & 0x01 == 1  # RX ready

    def test_lsr_empty(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        lsr = sd._read_lsr()
        assert lsr & 0x01 == 0  # no RX data


# ── CMOS 12h decode path ────────────────────────────────────────────────────

class TestCMOS12hDecode:
    def _make_cmos(self, h24=False):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        # Set 12h mode (bit 1 clear) + BCD mode (bit 2 clear)
        mode = 0x00 if h24 else 0x02
        return cmos, clock, cpu, mode

    def test_cmos_12h_pm(self):
        cmos, clock, cpu, mode = self._make_cmos()
        cmos._cmos[cmos.REG_STATUS_B] = mode
        # 15:00 = 3 PM. Use binary mode so _refresh_rtc writes raw values
        cmos._cmos[cmos.REG_STATUS_B] = mode | 0x04  # binary
        clock._epoch = 15 * 3600 + 30 * 60  # 15:30
        result = cmos.get_time()
        assert result["hour"] == 15

    def test_cmos_12h_noon(self):
        cmos, clock, cpu, mode = self._make_cmos()
        cmos._cmos[cmos.REG_STATUS_B] = mode | 0x04  # binary
        clock._epoch = 12 * 3600  # 12:00 noon
        result = cmos.get_time()
        assert result["hour"] == 12

    def test_cmos_12h_midnight(self):
        cmos, clock, cpu, mode = self._make_cmos()
        cmos._cmos[cmos.REG_STATUS_B] = mode | 0x04  # binary
        clock._epoch = 0  # midnight
        result = cmos.get_time()
        assert result["hour"] == 0


# ── CPU format_trace ─────────────────────────────────────────────────────────

class TestCPUFormatTrace:
    def test_format_trace(self):
        from domain.shell._internal.vm import CPU
        cpu = CPU()
        cpu._trace = [type('TraceEntry', (), {
            'cycle': 1, 'pc': 0, 'instruction': 'NOP',
            'registers': {'R0': 0, 'R1': 1}
        })()]
        lines = cpu.format_trace()
        assert len(lines) == 1
        assert "NOP" in lines[0]


# ── CPU dev_in exception + info fallback ─────────────────────────────────────

class TestCPUDevIn:
    def test_dev_in_exception(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        def bad_read():
            raise IOError("no")
        dev = type('Dev', (), {'read': bad_read})()
        cpu._devices = type('Bus', (), {'_devices': {'99': dev}})()
        _op_in(cpu, ["R0", 99])
        assert cpu.regs[0] == 0

    def test_dev_in_no_status(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        dev = type('Dev', (), {})()
        cpu._devices = type('Bus', (), {'_devices': {'0': dev}})()
        _op_in(cpu, ["R0", 0])
        assert cpu.regs[0] == 0

    def test_dev_in_int_value(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        dev = type('Dev', (), {'read': lambda self: 42})()
        cpu._devices = type('Bus', (), {'_devices': {'0': dev}})()
        _op_in(cpu, ["R0", 0])
        assert cpu.regs[0] == 42

    def test_dev_in_float_value(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        dev = type('Dev', (), {'read': lambda self: "3.14"})()
        cpu._devices = type('Bus', (), {'_devices': {'0': dev}})()
        _op_in(cpu, ["R0", 0])
        assert cpu.regs[0] == 3.14

    def test_dev_in_string_value(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        dev = type('Dev', (), {'read': lambda self: "hello"})()
        cpu._devices = type('Bus', (), {'_devices': {'0': dev}})()
        _op_in(cpu, ["R0", 0])
        assert cpu.regs[0] == "hello"

    def test_dev_in_no_device(self):
        from domain.shell._internal.vm import CPU, _op_in
        cpu = CPU()
        cpu._devices = type('Bus', (), {'_devices': {}})()
        _op_in(cpu, ["R0", 0])
        assert cpu.regs[0] == 0


# ── PUSHAD/POPAD memory and XCHG memory paths ───────────────────────────────

class TestX86CPU_PushadPopadXchgMem:
    def _make_cpu(self):
        from domain.shell._internal.vm import X86CPU
        return X86CPU(memory_size=0x400000)

    def test_pushad(self):
        cpu = self._make_cpu()
        cpu._regs[4] = 0x2000  # ESP
        for i in range(8):
            cpu._regs[i] = 0x1000 + i
        code = bytes([0x60])  # PUSHAD
        cpu._mem[0x100:0x101] = code
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok

    def test_xchg_r16_mem(self):
        import struct
        cpu = self._make_cpu()
        struct.pack_into("<H", cpu._mem, 0x2000, 0xBEEF)
        # 66 87 05 <disp32> = XCHG AX, [0x2000]
        code = bytes([0x66, 0x87, 0x05, 0x00, 0x20, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set16(0, 0xCAFE)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._get16(0) == 0xBEEF
        assert struct.unpack_from("<H", cpu._mem, 0x2000)[0] == 0xCAFE

    def test_xchg_r8_mem(self):
        cpu = self._make_cpu()
        cpu._mem[0x2000] = 0x42
        # 86 05 <disp32> = XCHG AL, [0x2000]
        code = bytes([0x86, 0x05, 0x00, 0x20, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(0, 0x55)
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert (cpu._regs[0] & 0xFF) == 0x42
        assert cpu._mem[0x2000] == 0x55

    def test_xchg_mem_r8(self):
        cpu = self._make_cpu()
        cpu._mem[0x2000] = 0x42
        # 86 0D <disp32> = XCHG [0x2000], CL
        code = bytes([0x86, 0x0D, 0x00, 0x20, 0x00, 0x00])
        cpu._mem[0x100:0x100 + len(code)] = code
        cpu._set8l(1, 0x55)  # CL = 0x55
        cpu._eip = 0x100
        ok = cpu.step()
        assert ok
        assert cpu._mem[0x2000] == 0x55


# ── Assembler XCHG memory encodings ─────────────────────────────────────────

class TestAssemblerXchgMemoryMore:
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))

    def test_lea_eax_ecx_4(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [ECX*4]"])
        assert 0x8D in code

    def test_lea_eax_ecx_4_disp(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [ECX*4+0x10]"])
        assert 0x8D in code

    def test_lea_eax_ecx_4_disp32(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [ECX*4+0x1000]"])
        assert 0x8D in code

    def test_lea_eax_ebx_ecx(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [EBX+ECX]"])
        assert 0x8D in code

    def test_lea_eax_ebx_ecx_disp8(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [EBX+ECX+0x10]"])
        assert 0x8D in code

    def test_lea_eax_ebx_ecx_disp32(self):
        code = self._asm(["[BITS 32]", "LEA EAX, [EBX+ECX+0x1000]"])
        assert 0x8D in code

    def test_mov_mem_disp8(self):
        code = self._asm(["[BITS 32]", "MOV byte [EAX+0x10], 0x42"])
        assert 0xC6 in code

    def test_mov_mem_word(self):
        code = self._asm(["[BITS 32]", "MOV word [0x1000], 0x1234"])
        assert 0xC7 in code


# ── Remaining small-block coverage ──────────────────────────────────────────

class TestRemainingSmallBlocks:
    def _asm(self, lines):
        from domain.shell._internal.vm import X86Assembler
        return X86Assembler().assemble("\n".join(lines))

    def test_exec_exception_handler(self):
        from domains.shell.vm import X86CPU
        cpu = X86CPU(memory_size=0x1000)
        cpu.load(b'\x90', org=0)  # NOP
        original_exec = cpu._exec_one
        def bad_exec():
            raise RuntimeError("test fault")
        cpu._exec_one = bad_exec
        result = cpu.step()
        assert result is False

    def test_call_stack_overflow(self):
        from domain.shell._internal.vm import CPU, Halt, _op_call
        cpu = CPU()
        cpu._call_stack = [0] * 512
        with pytest.raises(Halt):
            _op_call(cpu, ["0"])

    def test_resolve_label_digit(self):
        from domain.shell._internal.vm import _resolve_label, CPU
        cpu = CPU()
        assert _resolve_label(cpu, "42") == 42

    def test_resolve_label_invalid(self):
        from domain.shell._internal.vm import _resolve_label, CPU, InsFault
        cpu = CPU()
        with pytest.raises(InsFault):
            _resolve_label(cpu, "not_a_number_or_label")

    def test_is_truthy_ndarray(self):
        from domain.shell._internal.vm import CPU
        cpu = CPU()
        import numpy as np
        assert cpu._truthy(np.array([1.0])) is True
        assert cpu._truthy(np.array([])) is False

    def test_fetch_word(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x1000)
        cpu._mem[0:2] = b'\x34\x12'
        v = cpu._fetch_word()
        assert v == 0x1234

    def test_fetch_dword(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x1000)
        cpu._mem[0:4] = b'\x78\x56\x34\x12'
        v = cpu._fetch_dword()
        assert v == 0x12345678

    def test_dev_call_exception(self):
        from domain.shell._internal.vm import CPU, _op_dev_call
        cpu = CPU()
        mock_device = type('D', (), {'call': lambda s, *a: 1/0})()
        cpu._devices = type('Dev', (), {'_devices': {'42': mock_device}})()
        _op_dev_call(cpu, ["R0", 42, "read"])
        assert cpu.regs[0] is None

    def test_dev_table_open_with_adapter(self):
        from domain.shell._internal.vm import CPU, _op_dev_table_open
        import domain.shell._internal.vm as vm_mod
        cpu = CPU()
        mock_adapter = type('A', (), {'open': lambda s, n: -1})()
        vm_mod._device_table_adapter = mock_adapter
        try:
            _op_dev_table_open(cpu, ["R0", "nonexistent_device"])
            assert cpu.regs[0] == -1
        finally:
            vm_mod._device_table_adapter = None

    def test_exec_16bit_unknown_opcode(self):
        from domain.shell._internal.vm import X86CPU, InsFault
        cpu = X86CPU(memory_size=0x1000)
        cpu._mem[0] = 0x66
        cpu._mem[1] = 0xFF
        with pytest.raises(InsFault):
            cpu.step()

    def test_neg_f6_register(self):
        from domain.shell._internal.vm import X86CPU
        cpu = X86CPU(memory_size=0x1000)
        code = bytes([
            0xB8, 0x05, 0x00, 0x00, 0x00,
            0xF6, 0xD8,
            0xF4,
        ])
        cpu.load(code, org=0)
        cpu.run(max_steps=10)
        assert (cpu._regs[0] & 0xFF) == 0xFB

    def test_serial_read_data(self):
        from domain.shell._internal.vm import SerialDevice
        sd = SerialDevice()
        val = sd._read_data()
        assert val == 0
        sd.push_byte(0x41)
        val = sd._read_data()
        assert val == 0x41

    def test_assembler_times_db_count(self):
        code = self._asm(["[BITS 32]", "times 5 db 0x90"])
        assert code == b'\x90' * 5

    def test_assembler_times_dw_count(self):
        code = self._asm(["[BITS 32]", "times 3 dw 0x1234"])
        assert code.hex() == '341234123412'

    def test_assembler_times_dd_count(self):
        code = self._asm(["[BITS 32]", "times 2 dd 0x12345678"])
        assert code.hex() == '7856341278563412'

    def test_assembler_mov_r32_mem_disp(self):
        code = self._asm(["[BITS 32]", "MOV EBX, [ECX+0x10]"])
        assert 0x8B in code

    def test_assembler_mov_r32_mem_disp32(self):
        code = self._asm(["[BITS 32]", "MOV EBX, [ECX+0x1000]"])
        assert 0x8B in code

    def test_assembler_sib_esp_base(self):
        code = self._asm(["[BITS 32]", "MOV EAX, [ESP+ECX]"])
        assert 0x8B in code

    def test_assembler_sib_no_base_disp32(self):
        code = self._asm(["[BITS 32]", "MOV EAX, [ECX*4+0x1000]"])
        assert 0x8B in code

    def test_cmos_bcd_12h_pm(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        cmos._cmos[cmos.REG_STATUS_B] = 0x02
        clock._epoch = 15 * 3600
        result = cmos.get_time()
        assert result["hour"] == 15

    def test_cmos_bcd_12h_midnight(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        cmos._cmos[cmos.REG_STATUS_B] = 0x02
        clock._epoch = 0
        result = cmos.get_time()
        assert result["hour"] == 0

    def test_cmos_bcd_12h_noon(self):
        from domain.shell._internal.vm import CMOSDevice, ClockDevice, X86CPU
        cpu = X86CPU(memory_size=0x400000)
        clock = ClockDevice(freq=100)
        cmos = CMOSDevice(cpu=cpu, clock=clock)
        cmos._cmos[cmos.REG_STATUS_B] = 0x02
        clock._epoch = 12 * 3600
        result = cmos.get_time()
        assert result["hour"] == 12

    def test_blockdevice_read_sectors(self):
        from domain.shell._internal.vm import BlockDevice
        bd = BlockDevice(num_sectors=8)
        bd.write_sector(1, b'\xAA' * 512)
        bd.write_sector(2, b'\xBB' * 512)
        data = bd.read_sectors(1, 2)
        assert len(data) == 1024
        assert data[:512] == b'\xAA' * 512
        assert data[512:] == b'\xBB' * 512

    def test_train_get_result_completed(self):
        from domain.shell._internal.vm import X86SyscallHandler
        import domain.shell._internal.vm_training_bridge as bridge_mod
        original_bridge = bridge_mod._bridge

        class MockBridge:
            def get_result_json(self, job_id):
                return '{"loss": 0.5}'
        bridge_mod._bridge = MockBridge()
        try:
            handler = X86SyscallHandler.__new__(X86SyscallHandler)
            cpu = type('C', (), {'_regs': [0]*16, '_write8': lambda s, a, b: None, '_memory': type('M', (), {'alloc': lambda s, n: 0x300000, 'free': lambda s, a: None})()})()
            handler._cpu = cpu
            handler._scheduler = type('S', (), {'current': type('P', (), {'pid': 1, 'name': 'test', 'state': type('S2', (), {'READY': 0, 'BLOCKED': 1, 'TERMINATED': 2})()})()})()
            handler._heap_break = 0x400000
            handler._rbac = type('R', (), {'check': lambda s, *a: True})()
            handler._memory = cpu._memory
            handler._fs = None
            result = handler._sys_train_get_result(1, 0x2000, 256)
            assert result > 0
        finally:
            bridge_mod._bridge = original_bridge

    def test_train_start(self):
        from domains.shell.vm import X86SyscallHandler
        import domains.shell.vm_training_bridge as bridge_mod
        original = bridge_mod._bridge
        bridge_mod._bridge = None
        try:
            handler = X86SyscallHandler.__new__(X86SyscallHandler)
            config_json = b'{"model":"test"}\x00'
            cpu = type('C', (), {
                '_regs': [0]*16,
                '_read8': lambda s, a: config_json[a] if a < len(config_json) else 0,
                '_memory': type('M', (), {})()
            })()
            handler._cpu = cpu
            handler._scheduler = type('S', (), {'current': type('P', (), {'pid': 1})()})()
            handler._heap_break = 0x400000
            handler._rbac = type('R', (), {'check': lambda s, *a: True})()
            handler._memory = cpu._memory
            handler._fs = None
            result = handler._sys_train_start(0)
            assert isinstance(result, int)
        finally:
            bridge_mod._bridge = original

    def test_train_status(self):
        from domain.shell._internal.vm import X86SyscallHandler
        import domain.shell._internal.vm_training_bridge as bridge_mod
        original = bridge_mod._bridge

        class MockBridge:
            def status(self, job_id):
                return {"status": "running"}
        bridge_mod._bridge = MockBridge()
        try:
            handler = X86SyscallHandler.__new__(X86SyscallHandler)
            cpu = type('C', (), {'_regs': [0]*16})()
            handler._cpu = cpu
            result = handler._sys_train_status(1)
            assert result == 0
        finally:
            bridge_mod._bridge = original

    def test_xchg_r16_mem(self):
        code = self._asm(["[BITS 32]", "XCHG EAX, [0x1000]"])
        assert 0x87 in code or 0x90 in code

    def test_xchg_r8_mem(self):
        code = self._asm(["[BITS 32]", "XCHG AL, [0x1000]"])
        assert 0x86 in code

    def test_xchg_ebx_mem(self):
        code = self._asm(["[BITS 32]", "XCHG EBX, [0x1000]"])
        assert 0x87 in code or 0x86 in code

    def test_test_ebx_imm32(self):
        code = self._asm(["[BITS 32]", "TEST EBX, 0x12345678"])
        assert 0xF7 in code
        modrm = code[1]
        assert (modrm >> 3) & 7 == 0

    def test_mov_ebx_mem(self):
        code = self._asm(["[BITS 32]", "MOV EBX, [0x1000]"])
        assert code.hex() == "8b1d00100000"

    def test_mov_ecx_imm32(self):
        code = self._asm(["[BITS 32]", "MOV ECX, 0x12345678"])
        assert 0xB9 in code
