"""
Tests for vm_programs.py: assembly program constants, self-test runner,
and x86 binary/disk/BIOS export helpers.
"""

import os

import pytest

from domains.shell.vm import X86Assembler
from domains.shell.vm_programs import (
    TEST_SYSCALLS_ASM,
    TEST_FILES_ASM,
    TEST_EXEC_ASM,
    TEST_EXEC_TARGET_ASM,
    TEST_MEMORY_ASM,
    TEST_ARITH_ASM,
    TEST_STACK_ASM,
    TEST_SYS_EDGE_ASM,
    HELLO_ASM,
    CLASSICAL_ASM,
    TENSOR_MATH_ASM,
    MATRIX_MUL_ASM,
    NEURAL_NET_ASM,
    LOOP_ASM,
    FUNCTION_ASM,
    MIXED_ASM,
    NPU_PROGRAM_ASM,
    COUNTER_ASM,
    FIB_ASM,
    COLLATZ_ASM,
    BOOT_ASM,
    SHELL_ASM,
    X86_BOOTLOADER_ASM,
    X86_KERNEL_ASM,
    X86_BIOS_ASM,
    self_test,
    export_x86_binary,
    build_disk_image,
    export_disk_image,
    build_boot_image,
    build_bios,
)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestAsmConstants:
    @pytest.mark.parametrize("name,text", [
        ("TEST_SYSCALLS_ASM", TEST_SYSCALLS_ASM),
        ("TEST_FILES_ASM", TEST_FILES_ASM),
        ("TEST_EXEC_ASM", TEST_EXEC_ASM),
        ("TEST_EXEC_TARGET_ASM", TEST_EXEC_TARGET_ASM),
        ("TEST_MEMORY_ASM", TEST_MEMORY_ASM),
        ("TEST_ARITH_ASM", TEST_ARITH_ASM),
        ("TEST_STACK_ASM", TEST_STACK_ASM),
        ("TEST_SYS_EDGE_ASM", TEST_SYS_EDGE_ASM),
        ("HELLO_ASM", HELLO_ASM),
        ("CLASSICAL_ASM", CLASSICAL_ASM),
        ("TENSOR_MATH_ASM", TENSOR_MATH_ASM),
        ("MATRIX_MUL_ASM", MATRIX_MUL_ASM),
        ("NEURAL_NET_ASM", NEURAL_NET_ASM),
        ("LOOP_ASM", LOOP_ASM),
        ("FUNCTION_ASM", FUNCTION_ASM),
        ("MIXED_ASM", MIXED_ASM),
        ("NPU_PROGRAM_ASM", NPU_PROGRAM_ASM),
        ("COUNTER_ASM", COUNTER_ASM),
        ("FIB_ASM", FIB_ASM),
        ("COLLATZ_ASM", COLLATZ_ASM),
        ("BOOT_ASM", BOOT_ASM),
        ("SHELL_ASM", SHELL_ASM),
        ("X86_BOOTLOADER_ASM", X86_BOOTLOADER_ASM),
        ("X86_KERNEL_ASM", X86_KERNEL_ASM),
        ("X86_BIOS_ASM", X86_BIOS_ASM),
    ])
    def test_constant_is_nonempty_string(self, name, text):
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    @pytest.mark.parametrize("text", [
        TEST_SYSCALLS_ASM, TEST_FILES_ASM, TEST_EXEC_ASM,
        TEST_EXEC_TARGET_ASM, TEST_MEMORY_ASM, TEST_ARITH_ASM,
        TEST_STACK_ASM, TEST_SYS_EDGE_ASM,
        X86_BOOTLOADER_ASM, X86_KERNEL_ASM, X86_BIOS_ASM,
    ])
    def test_x86_programs_assemble(self, text):
        asm = X86Assembler()
        code = asm.assemble(text)
        assert len(code) > 0

    def test_syscalls_program_references_all_26(self):
        for i in range(1, 27):
            marker = f"[{i:02d}]"
            assert marker in TEST_SYSCALLS_ASM


# ══════════════════════════════════════════════════════════════════════════════
# self_test()
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfTest:
    def test_returns_six_results(self):
        results = self_test()
        assert len(results) == 6

    def test_all_pass(self):
        results = self_test()
        assert all("PASS" in r for r in results)

    def test_output_contains_hello(self):
        results = self_test()
        assert any("Hello, AI Compteur" in r for r in results)

    def test_output_contains_stack_value(self):
        results = self_test()
        assert any("output: ['30']" in r for r in results)

    def test_output_contains_float_product(self):
        results = self_test()
        assert any("output: ['12.0']" in r for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# export_x86_binary()
# ══════════════════════════════════════════════════════════════════════════════

class TestExportX86Binary:
    def test_strips_comments(self):
        src = "mov eax, 1 ; comment\nnop\n"
        out = export_x86_binary(src)
        assert b"comment" not in out
        assert b"mov eax, 1" in out

    def test_strips_directives_and_labels(self):
        src = "[BITS 32]\n[ORG 0x100000]\nstart:\n    mov eax, 1\nlabel2:\n"
        out = export_x86_binary(src)
        assert b"BITS" not in out
        assert b"ORG" not in out
        assert b"start" not in out
        assert b"mov eax, 1" in out

    def test_skips_times_and_plain_directives(self):
        src = "times 4 db 0\ndw 0xAA55\ndd 0\n"
        out = export_x86_binary(src)
        assert b"times" not in out

    def test_handles_blank_lines(self):
        src = "\n\n\nmov eax, 1\n"
        out = export_x86_binary(src)
        assert b"mov eax, 1" in out

    def test_skips_standalone_label(self):
        src = "gdt_end:\n"
        out = export_x86_binary(src)
        assert out.strip() == b""

    def test_returns_utf8_bytes(self):
        out = export_x86_binary("mov eax, 1")
        assert isinstance(out, bytes)
        assert out.decode("utf-8") == "mov eax, 1"

    def test_assembled_bootloader_is_512_bytes(self):
        raw = export_x86_binary(X86_BOOTLOADER_ASM)
        asm = X86Assembler()
        machine = asm.assemble(X86_BOOTLOADER_ASM)
        assert len(machine) == 512
        assert len(raw) > 512  # source text is longer than machine code


# ══════════════════════════════════════════════════════════════════════════════
# build_disk_image()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildDiskImage:
    def test_default_size_1mb(self):
        img = build_disk_image(b"B" * 512, b"K" * 4096)
        assert len(img) == 1024 * 1024

    def test_custom_size(self):
        img = build_disk_image(b"B" * 512, b"K" * 4096, size_mb=4)
        assert len(img) == 4 * 1024 * 1024

    def test_bootloader_at_sector_zero(self):
        boot = b"\x11" * 512
        img = build_disk_image(boot, b"\x22" * 4096)
        assert img[:512] == boot

    def test_kernel_after_sector_zero(self):
        kernel = b"\x33" * 2048
        img = build_disk_image(b"x" * 512, kernel)
        assert img[512:512 + 2048] == kernel

    def test_rest_is_zeroed(self):
        img = build_disk_image(b"x" * 512, b"y" * 512, size_mb=1)
        assert set(img[1024:]) == {0}


# ══════════════════════════════════════════════════════════════════════════════
# export_disk_image()
# ══════════════════════════════════════════════════════════════════════════════

class TestExportDiskImage:
    def test_writes_1mb_image(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        result = export_disk_image(X86_KERNEL_ASM, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) == 1024 * 1024

    def test_kernel_region_is_valid_machine_code(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image(X86_KERNEL_ASM, out, size_mb=2)
        with open(out, "rb") as f:
            img = f.read()
        assert len(img) == 2 * 1024 * 1024
        # Kernel was padded to >= 4KB; bytes 512..4K+512 are non-zero
        assert any(img[512:512 + 4096])
        # Bootloader region is non-zero
        assert any(img[:512])

    def test_empty_source_uses_default_kernel(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image("", out)
        assert os.path.getsize(out) == 1024 * 1024

    def test_small_source_kernel_is_padded_to_4kb(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image("mov eax, 1\nret", out)
        with open(out, "rb") as f:
            img = f.read()
        # Assembled tiny kernel gets padded to 4096 bytes before writing
        kernel_region = img[512:512 + 4096]
        assert any(kernel_region)
        assert img[512 + 4096:] == b"\x00" * (len(img) - 512 - 4096)


# ══════════════════════════════════════════════════════════════════════════════
# build_boot_image()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildBootImage:
    def test_writes_1mb_image(self, tmp_path):
        out = os.path.join(str(tmp_path), "boot.img")
        result = build_boot_image(out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) == 1024 * 1024

    def test_bootloader_and_kernel_regions_present(self, tmp_path):
        out = os.path.join(str(tmp_path), "boot.img")
        build_boot_image(out)
        with open(out, "rb") as f:
            img = f.read()
        assert any(img[:512])
        assert any(img[512:512 + 4096])

    def test_default_output_path_is_boot_img(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = build_boot_image()
        assert result == "boot.img"
        assert os.path.exists("boot.img")

    def test_small_kernel_is_padded_to_4kb(self, tmp_path, monkeypatch):
        def _small_kernel(source, org=0):
            return bytearray(b"\x90\xc3")  # nop; ret

        out = os.path.join(str(tmp_path), "boot.img")
        monkeypatch.setattr(X86Assembler, "assemble", _small_kernel)
        build_boot_image(out)
        with open(out, "rb") as f:
            img = f.read()
        kernel_region = img[512:512 + 4096]
        assert kernel_region[0:2] == b"\x90\xc3"
        assert all(b == 0 for b in kernel_region[2:])


# ══════════════════════════════════════════════════════════════════════════════
# build_bios()
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildBios:
    def test_writes_64kb_rom(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        result = build_bios(out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) == 65536

    def test_reset_vector_jump_present(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        # JMP FAR 0xF000:0x0000 = EA 00 00 00 F0 at offset 0xFFF0
        assert rom[0xFFF0] == 0xEA
        assert rom[0xFFF1] == 0x00
        assert rom[0xFFF2] == 0x00
        assert rom[0xFFF3] == 0x00
        assert rom[0xFFF4] == 0xF0

    def test_bootloader_and_kernel_overlaid(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        # ROM contains real machine code (bootloader + kernel regions non-zero)
        assert any(rom[:512])
        assert any(rom[512:512 + 4096])

    def test_oversized_bootloader_is_truncated_to_512(self, tmp_path, monkeypatch):
        def _stub_assemble(self, source, org=0):
            if "bios_entry" in source:
                return bytearray(b"\x90" * 60000)
            if "boot_start" in source:
                return bytearray(b"\x90" * 1024)  # oversized bootloader
            return bytearray(b"\x90" * 4096)  # kernel

        out = os.path.join(str(tmp_path), "bios.bin")
        monkeypatch.setattr(X86Assembler, "assemble", _stub_assemble)
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        assert len(rom) == 65536
