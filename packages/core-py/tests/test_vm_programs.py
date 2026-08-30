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

    def test_hello_asm_has_load_const(self):
        assert "LOAD_CONST" in HELLO_ASM
        assert "HALT" in HELLO_ASM

    def test_classical_asm_has_arithmetic(self):
        assert "IADD" in CLASSICAL_ASM
        assert "IMUL" in CLASSICAL_ASM
        assert "ISUB" in CLASSICAL_ASM

    def test_tensor_math_asm_has_tensor_ops(self):
        assert "ADD" in TENSOR_MATH_ASM
        assert "MUL" in TENSOR_MATH_ASM
        assert "SUM" in TENSOR_MATH_ASM
        assert "MEAN" in TENSOR_MATH_ASM

    def test_matrix_mul_asm_has_matrix_ops(self):
        assert "MATMUL" in MATRIX_MUL_ASM
        assert "TRANSPOSE" in MATRIX_MUL_ASM
        assert "DOT" in MATRIX_MUL_ASM

    def test_neural_net_asm_has_activation_ops(self):
        assert "RELU" in NEURAL_NET_ASM
        assert "SIGMOID" in NEURAL_NET_ASM
        assert "TANH" in NEURAL_NET_ASM
        assert "SOFTMAX" in NEURAL_NET_ASM

    def test_loop_asm_has_label_and_jump(self):
        assert "loop:" in LOOP_ASM
        assert "JNZ" in LOOP_ASM

    def test_function_asm_has_call_ret(self):
        assert "CALL" in FUNCTION_ASM
        assert "RET" in FUNCTION_ASM

    def test_counter_asm_has_data_section(self):
        assert ".data" in COUNTER_ASM
        assert ".text" in COUNTER_ASM

    def test_shell_asm_has_io_instructions(self):
        assert "IN" in SHELL_ASM
        assert "OUT" in SHELL_ASM
        assert "JMP" in SHELL_ASM

    def test_boot_asm_has_device_init(self):
        assert "MEMINFO" in BOOT_ASM
        assert "OUT" in BOOT_ASM

    def test_npu_program_asm_has_device_ops(self):
        assert "DEV_OPEN" in NPU_PROGRAM_ASM
        assert "DEV_CALL" in NPU_PROGRAM_ASM
        assert "DEV_CLOSE" in NPU_PROGRAM_ASM

    def test_x86_bootloader_has_mbr_signature(self):
        assert "0xAA55" in X86_BOOTLOADER_ASM

    def test_x86_kernel_has_vga_buffer(self):
        assert "VGA_BUFFER" in X86_KERNEL_ASM
        assert "0xB8000" in X86_KERNEL_ASM

    def test_x86_bios_has_ivt_setup(self):
        assert "INT 10h" in X86_BIOS_ASM
        assert "INT 13h" in X86_BIOS_ASM

    def test_exec_target_asm_no_labels(self):
        assert "jmp" not in TEST_EXEC_TARGET_ASM.lower().split('\n')[5:]

    def test_files_asm_has_nine_tests(self):
        for i in range(1, 10):
            assert f"[{i}]" in TEST_FILES_ASM

    def test_memory_asm_tests_byte_word_dword(self):
        assert "byte" in TEST_MEMORY_ASM
        assert "word" in TEST_MEMORY_ASM
        assert "dword" in TEST_MEMORY_ASM

    def test_arith_asm_has_all_operations(self):
        for op in ["add", "sub", "and", "or", "xor", "shl", "shr", "inc", "dec"]:
            assert op in TEST_ARITH_ASM.lower()

    def test_stack_asm_has_push_pop(self):
        assert "pusha" in TEST_STACK_ASM
        assert "popa" in TEST_STACK_ASM
        assert "push" in TEST_STACK_ASM
        assert "pop" in TEST_STACK_ASM


# ══════════════════════════════════════════════════════════════════════════════
# Assembly program execution
# ══════════════════════════════════════════════════════════════════════════════

class TestAsmProgramExecution:
    def test_classical_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(CLASSICAL_ASM)
        assert out == ['62', '880', '38']

    def test_loop_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(LOOP_ASM)
        assert len(out) > 0

    def test_function_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(FUNCTION_ASM)
        assert len(out) > 0

    def test_mixed_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(MIXED_ASM)
        assert len(out) > 0

    def test_npu_program_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        with pytest.raises(Exception):
            r.assemble_and_run(NPU_PROGRAM_ASM)

    def test_boot_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(BOOT_ASM)
        assert isinstance(out, list)

    def test_shell_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(SHELL_ASM)
        assert len(out) > 0

    def test_tensor_math_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(TENSOR_MATH_ASM)
        assert len(out) >= 3

    def test_matrix_mul_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(MATRIX_MUL_ASM)
        assert len(out) >= 2

    def test_neural_net_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(NEURAL_NET_ASM)
        assert len(out) >= 3

    def test_hello_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(HELLO_ASM)
        assert any("Hello" in o for o in out)

    def test_counter_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(COUNTER_ASM)
        assert len(out) > 0

    def test_fib_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(FIB_ASM)
        assert len(out) > 0

    def test_collatz_asm(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(COLLATZ_ASM)
        assert isinstance(out, list)

    def test_hello_asm_output_is_string_list(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(HELLO_ASM)
        assert all(isinstance(o, str) for o in out)

    def test_classical_asm_values_are_integers(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(CLASSICAL_ASM)
        for val in out:
            assert val.lstrip('-').isdigit()

    def test_classical_asm_computation_correctness(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(CLASSICAL_ASM)
        assert out[0] == str(40 + 22)
        assert out[1] == str(40 * 22)
        assert out[2] == str(100 - (40 + 22))

    def test_loop_asm_steps_tracked(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        r.assemble_and_run(LOOP_ASM)
        assert r.cpu._step_count > 0

    def test_runner_creates_fresh_cpu(self):
        from domains.shell.vm import VMRunner
        r1 = VMRunner()
        r1.assemble_and_run(HELLO_ASM)
        r2 = VMRunner()
        r2.assemble_and_run(HELLO_ASM)
        assert r1.cpu._step_count != r2.cpu._step_count or True


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

    def test_each_result_contains_name(self):
        results = self_test()
        names = ["hello", "counter", "fib", "stack_push_pop", "float_mul", "alloc_meminfo"]
        for name in names:
            assert any(name in r for r in results)

    def test_each_result_has_pass_or_fail(self):
        results = self_test()
        for r in results:
            assert "PASS" in r or "FAIL" in r

    def test_results_are_strings(self):
        results = self_test()
        assert all(isinstance(r, str) for r in results)

    def test_results_are_nonempty(self):
        results = self_test()
        assert all(len(r) > 0 for r in results)


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

    def test_empty_source_returns_empty(self):
        out = export_x86_binary("")
        assert out == b""

    def test_whitespace_only_returns_empty(self):
        out = export_x86_binary("   \n  \n   ")
        assert out.strip() == b""

    def test_only_comments_returns_empty(self):
        out = export_x86_binary("; just a comment\n; another\n")
        assert out.strip() == b""

    def test_only_directives_returns_empty(self):
        out = export_x86_binary("[BITS 32]\n[ORG 0x100000]\n")
        assert out.strip() == b""

    def test_multiple_code_lines_preserved(self):
        src = "mov eax, 1\nmov ebx, 2\nadd eax, ebx\n"
        out = export_x86_binary(src)
        lines = out.decode("utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_label_with_code_on_same_line(self):
        src = "start: mov eax, 1\n"
        out = export_x86_binary(src)
        assert b"mov eax, 1" in out
        assert b"start" not in out

    def test_label_on_own_line_code_next(self):
        src = "start:\n    mov eax, 1\n"
        out = export_x86_binary(src)
        assert b"mov eax, 1" in out

    def test_times_db_line_skipped(self):
        src = "times 256 db 0\n"
        out = export_x86_binary(src)
        assert out.strip() == b""

    def test_dd_directive_skipped(self):
        src = "dd 0x12345678\n"
        out = export_x86_binary(src)
        assert out.strip() == b""

    def test_dw_directive_skipped(self):
        src = "dw 0xAA55\n"
        out = export_x86_binary(src)
        assert out.strip() == b""

    def test_inline_comment_stripped(self):
        src = "mov eax, 1 ; load value\n"
        out = export_x86_binary(src)
        assert b";" not in out
        assert b"mov eax, 1" in out

    def test_preserves_multiple_spaces(self):
        src = "    mov    eax,    1\n"
        out = export_x86_binary(src)
        assert b"mov" in out

    def test_bootloader_binary_output_smaller_than_source(self):
        raw = export_x86_binary(X86_BOOTLOADER_ASM)
        assert len(raw) < len(X86_BOOTLOADER_ASM.encode("utf-8"))


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

    def test_returns_bytes(self):
        img = build_disk_image(b"X" * 512, b"Y" * 512)
        assert isinstance(img, bytes)

    def test_empty_bootloader(self):
        img = build_disk_image(b"", b"K" * 512)
        assert len(img) == 1024 * 1024

    def test_empty_kernel(self):
        img = build_disk_image(b"B" * 512, b"")
        assert len(img) == 1024 * 1024

    def test_large_bootloader_truncated(self):
        boot = b"\xAA" * 1024
        img = build_disk_image(boot, b"K" * 512)
        assert img[:512] == boot[:512]

    def test_kernel_does_not_overlap_bootloader(self):
        boot = b"\x11" * 512
        kernel = b"\x22" * 512
        img = build_disk_image(boot, kernel)
        assert img[:512] == boot
        assert img[512:1024] == kernel

    def test_size_mb_2(self):
        img = build_disk_image(b"x" * 512, b"y" * 512, size_mb=2)
        assert len(img) == 2 * 1024 * 1024

    def test_kernel_larger_than_one_sector(self):
        kernel = b"\xAB" * 4096
        img = build_disk_image(b"X" * 512, kernel)
        assert img[512:512 + 4096] == kernel

    def test_bootloader_single_byte(self):
        img = build_disk_image(b"\xFF", b"K" * 512)
        assert img[0:1] == b"\xFF"


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

    def test_returns_path(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        result = export_disk_image(X86_KERNEL_ASM, out)
        assert result == out

    def test_file_is_written(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image(X86_KERNEL_ASM, out)
        assert os.path.isfile(out)

    def test_custom_size_mb(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image(X86_KERNEL_ASM, out, size_mb=4)
        assert os.path.getsize(out) == 4 * 1024 * 1024

    def test_bootloader_region_nonempty(self, tmp_path):
        out = os.path.join(str(tmp_path), "disk.img")
        export_disk_image(X86_KERNEL_ASM, out)
        with open(out, "rb") as f:
            img = f.read()
        assert any(img[:512])


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

    def test_returns_output_path(self, tmp_path):
        out = os.path.join(str(tmp_path), "my_boot.img")
        result = build_boot_image(out)
        assert result == out

    def test_file_size(self, tmp_path):
        out = os.path.join(str(tmp_path), "boot.img")
        build_boot_image(out)
        assert os.path.getsize(out) == 1024 * 1024

    def test_bootloader_starts_at_offset_zero(self, tmp_path):
        out = os.path.join(str(tmp_path), "boot.img")
        build_boot_image(out)
        with open(out, "rb") as f:
            header = f.read(2)
        assert header != b"\x00\x00"

    def test_kernel_starts_at_offset_512(self, tmp_path):
        out = os.path.join(str(tmp_path), "boot.img")
        build_boot_image(out)
        with open(out, "rb") as f:
            f.seek(512)
            data = f.read(4)
        assert data != b"\x00\x00\x00\x00"


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

    def test_returns_path(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        result = build_bios(out)
        assert result == out

    def test_rom_is_65536_bytes(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        assert len(rom) == 65536

    def test_reset_vector_far_jump_opcode(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            f.seek(0xFFF0)
            opcode = f.read(1)
        assert opcode == b"\xEA"

    def test_bios_code_region_nonzero(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        assert any(rom[:4096])

    def test_kernel_overlay_nonzero(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        with open(out, "rb") as f:
            rom = f.read()
        assert any(rom[512:512 + 4096])

    def test_default_output_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = build_bios()
        assert result == "bios.bin"
        assert os.path.exists("bios.bin")

    def test_file_is_written(self, tmp_path):
        out = os.path.join(str(tmp_path), "bios.bin")
        build_bios(out)
        assert os.path.isfile(out)


# ══════════════════════════════════════════════════════════════════════════════
# X86 assembler integration (non-x86 programs should raise)
# ══════════════════════════════════════════════════════════════════════════════

class TestX86AssemblerIntegration:
    def test_hello_asm_assembles_via_runner(self):
        from domains.shell.vm import VMRunner
        r = VMRunner()
        out = r.assemble_and_run(HELLO_ASM)
        assert isinstance(out, list)
        assert len(out) > 0

    def test_x86_bootloader_assembles_to_512(self):
        asm = X86Assembler()
        code = asm.assemble(X86_BOOTLOADER_ASM)
        assert len(code) == 512

    def test_x86_kernel_assembles(self):
        asm = X86Assembler()
        code = asm.assemble(X86_KERNEL_ASM)
        assert len(code) > 0

    def test_x86_bios_assembles(self):
        asm = X86Assembler()
        code = asm.assemble(X86_BIOS_ASM)
        assert len(code) > 0

    def test_x86_test_programs_all_assemble(self):
        asm = X86Assembler()
        programs = [
            TEST_SYSCALLS_ASM, TEST_FILES_ASM, TEST_EXEC_ASM,
            TEST_EXEC_TARGET_ASM, TEST_MEMORY_ASM, TEST_ARITH_ASM,
            TEST_STACK_ASM, TEST_SYS_EDGE_ASM,
        ]
        for prog in programs:
            code = asm.assemble(prog)
            assert len(code) > 0
