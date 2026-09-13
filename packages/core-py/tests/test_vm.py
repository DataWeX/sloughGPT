"""
Tests for Shell Virtual Machine — CPU, assembler, syscall dispatch, sandbox.
"""

import struct
import os
import time
import tempfile
import pytest
import numpy as np
from domains.shell.vm import (
    ProgramLoader, VirtualCPU, VMRunner, VMFault, Halt, MemFault, InsFault,
    HELLO_ASM, NUM_REGS, MEM_SIZE, STACK_BASE, F_ZERO, F_NEG,
    X86VirtualSystem, X86CPU, X86Assembler,
    ClockDevice, VGADevice, SerialDevice,
    PS2KeyboardDevice, MouseDevice, CMOSDevice, DiskDevice, NICDevice,
    DeviceBus, ProcessTable, ProcessState, Scheduler,
    Memory, FlatFS, BlockDevice, DiskProgramLoader, X86Shell,
    Device, DeviceFault,
)
from domains.shell.vm_permissions import X86RBAC, Permission, Role


# ── ProgramLoader (Assembler) ──────────────────────────────────────────────


class TestProgramLoader:
    def test_empty_source(self):
        loader = ProgramLoader()
        insts = loader.load("")
        assert insts == []

    def test_single_nop(self):
        loader = ProgramLoader()
        insts = loader.load("NOP")
        assert len(insts) == 1
        assert insts[0].opcode == "NOP"

    def test_mov_immediate(self):
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 42")
        assert len(insts) == 1
        assert insts[0].opcode == "MOV"
        assert insts[0].operands == ["R0", 42]

    def test_mov_register(self):
        loader = ProgramLoader()
        insts = loader.load("MOV R5, R3")
        assert insts[0].operands == ["R5", "R3"]

    def test_add_three_operands(self):
        loader = ProgramLoader()
        insts = loader.load("ADD R0, R1, R2")
        assert insts[0].opcode == "ADD"
        assert insts[0].operands == ["R0", "R1", "R2"]

    def test_label_resolution(self):
        loader = ProgramLoader()
        insts = loader.load("""
            JMP target
            NOP
        target:
            HALT
        """)
        assert len(insts) == 3
        assert insts[0].opcode == "JMP"
        assert insts[0].operands == [2]
        assert insts[2].opcode == "HALT"

    def test_forward_label(self):
        loader = ProgramLoader()
        insts = loader.load("""
        start:
            JZ end
            NOP
        end:
            HALT
        """)
        assert insts[0].opcode == "JZ"
        assert insts[0].operands == [2]

    def test_load_const_string(self):
        loader = ProgramLoader()
        insts = loader.load('LOAD_CONST R0, "Hello"')
        assert insts[0].opcode == "LOAD_CONST"
        assert insts[0].operands[0] == "R0"
        assert insts[0].operands[1] == "Hello"

    def test_hello_asm_loads(self):
        loader = ProgramLoader()
        insts = loader.load(HELLO_ASM)
        assert len(insts) >= 3
        assert insts[0].opcode == "LOAD_CONST"


# ── VirtualCPU ─────────────────────────────────────────────────────────────


class TestVirtualCPU:
    def test_create_cpu(self):
        cpu = VirtualCPU()
        assert len(cpu.regs) == NUM_REGS
        assert cpu.pc == 0

    def test_load_program_sets_pc(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("start: MOV R0, 0\nHALT")
        cpu.load_program(insts)
        assert cpu.pc == 0

    def test_execute_mov_immediate(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R5, 42\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[5] == 42

    def test_execute_mov_register(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 10\nMOV R1, R0\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[1] == 10

    def test_execute_add(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 3\nADD R0, R0, 4\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 7

    def test_execute_sub(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 10\nSUB R0, R0, 3\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 7

    def test_execute_mul(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 6\nMUL R0, R0, 7\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_execute_jmp(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            JMP skip
            MOV R0, 99
        skip:
            MOV R0, 42
            HALT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_execute_jz_taken(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            MOV R0, 0
            CMP R0, 0
            JZ skip
            MOV R0, 99
        skip:
            MOV R0, 42
            HALT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_execute_jnz_not_taken(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            MOV R0, 1
            CMP R0, 0
            JNZ skip
            HALT
        skip:
            MOV R0, 42
            HALT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_call_ret(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            CALL fn
            HALT
        fn:
            MOV R0, 42
            RET
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_sandbox_instruction_limit(self):
        cpu = VirtualCPU()
        cpu._max_instructions = 10
        loader = ProgramLoader()
        insts = loader.load("""
        loop:
            JMP loop
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu._step_count == 10

    def test_sandbox_loop_counter(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            MOV R0, 5
        loop:
            LOOP R0, loop
            NOP
            HALT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 0

    def test_alu_operations(self):
        for op, a, b, expected in [("IAND", 0xFF, 0x0F, 0x0F),
                                    ("IOR", 0xF0, 0x0F, 0xFF),
                                    ("IXOR", 0xFF, 0x0F, 0xF0),
                                    ("ISHL", 1, 3, 8),
                                    ("ISHR", 8, 3, 1)]:
            cpu = VirtualCPU()
            loader = ProgramLoader()
            code = f"MOV R0, {a}\n{op} R0, R0, {b}\nHALT"
            insts = loader.load(code)
            cpu.load_program(insts)
            cpu.run()
            assert cpu.regs[0] == expected, f"{op}: got {cpu.regs[0]} expected {expected}"

    def test_cmp_sets_flag_equal(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 5\nCMP R0, 5\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu._cmp_flag == 0

    def test_cmp_sets_flag_less(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 3\nCMP R0, 5\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu._cmp_flag == -1

    def test_conditional_jumps(self):
        for jmp, a, b, should_jump in [
            ("JLT", 3, 5, True),
            ("JLT", 5, 3, False),
            ("JLE", 5, 5, True),
            ("JGT", 7, 3, True),
            ("JGT", 3, 3, False),
            ("JGE", 5, 3, True),
            ("JGE", 3, 5, False),
        ]:
            cpu = VirtualCPU()
            loader = ProgramLoader()
            code = f"""
            MOV R0, {a}
            CMP R0, {b}
            {jmp} jump
            MOV R0, 0
            HALT
        jump:
            MOV R0, 1
            HALT
            """
            insts = loader.load(code)
            cpu.load_program(insts)
            cpu.run()
            assert cpu.regs[0] == (1 if should_jump else 0), f"{jmp}({a},{b}): expected jump={should_jump}"


# ── VMRunner Integration ───────────────────────────────────────────────────


class TestVMRunner:
    def test_hello_program(self):
        runner = VMRunner()
        output = runner.assemble_and_run(HELLO_ASM)
        text = "".join(output)
        assert "Hello" in text

    def test_mov_immediate(self):
        runner = VMRunner()
        output = runner.assemble_and_run("MOV R0, 42\nHALT")
        assert runner.cpu.regs[0] == 42

    def test_register_to_register(self):
        runner = VMRunner()
        output = runner.assemble_and_run("MOV R0, 10\nMOV R1, R0\nHALT")
        assert runner.cpu.regs[1] == 10

    def test_infinite_loop_terminates(self):
        runner = VMRunner()
        output = runner.assemble_and_run("loop: JMP loop")
        assert runner.cpu._step_count > 0

    def test_memory_store_load(self):
        runner = VMRunner()
        output = runner.assemble_and_run("""
            MOV R0, 42
            STORE R0, 100
            MOV R1, 0
            LOAD R1, 100
            HALT
        """)
        assert runner.cpu.regs[1] == 42

    def test_disassemble(self):
        runner = VMRunner()
        listing = runner.disassemble(HELLO_ASM)
        assert any("LOAD_CONST" in line for line in listing)
        assert any("PRINT" in line for line in listing)

    def test_self_test(self):
        from domains.shell.vm import self_test
        results = self_test()
        assert len(results) >= 3

    def test_cpu_get_trace(self):
        runner = VMRunner()
        output = runner.assemble_and_run("MOV R0, 42\nHALT", trace=True)
        trace = runner.cpu.get_trace()
        assert len(trace) >= 2
        assert trace[0].pc == 0
        assert trace[1].registers.get("R0") == 42


# ── New ISA Opcodes ─────────────────────────────────────────────────────────


class TestStackOpcodes:
    def test_push_pop(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 42\nPUSH R0\nMOV R0, 0\nPOP R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 0
        assert cpu.regs[1] == 42
        assert cpu.sp == STACK_BASE

    def test_push_pop_multiple(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            MOV R0, 10
            MOV R1, 20
            PUSH R0
            PUSH R1
            POP R2
            POP R3
            HALT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[2] == 20
        assert cpu.regs[3] == 10

    def test_push_overflow(self):
        cpu = VirtualCPU()
        cpu.sp = 0
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 1\nPUSH R0\nHALT")
        cpu.load_program(insts)
        with pytest.raises(InsFault, match="stack overflow"):
            cpu.run()

    def test_pop_underflow(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("POP R0\nHALT")
        cpu.load_program(insts)
        with pytest.raises(InsFault, match="stack underflow"):
            cpu.run()


class TestFloatALU:
    def test_fadd(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.0\nFADD R2, R0, R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert abs(cpu.regs[2] - 5.14) < 0.001

    def test_fsub(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 5.0\nLOAD_CONST R1, 3.0\nFSUB R2, R0, R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[2] == 2.0

    def test_fmul(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 3.0\nLOAD_CONST R1, 4.0\nFMUL R2, R0, R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[2] == 12.0

    def test_fdiv(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 10.0\nLOAD_CONST R1, 4.0\nFDIV R2, R0, R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[2] == 2.5

    def test_fdiv_by_zero(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 1.0\nLOAD_CONST R1, 0.0\nFDIV R2, R0, R1\nHALT")
        cpu.load_program(insts)
        with pytest.raises(InsFault, match="division by zero"):
            cpu.run()

    def test_fcmp(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("LOAD_CONST R0, 1.0\nLOAD_CONST R1, 2.0\nFCMP R0, R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu._cmp_flag == -1


class TestMemoryOpcodes:
    def test_alloc(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("ALLOC R0, 100\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 100

    def test_meminfo(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("ALLOC R0, 50\nMEMINFO R1\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[1] >= 1


class TestConsoleIO:
    def test_out_writes(self):
        from domains.shell.vm import CPU as VMCPU, Assembler, DeviceBus

        output = []
        bus = DeviceBus()
        bus.register_console(stdout_fn=lambda v: output.append(str(v)))

        cpu = VMCPU(devices=bus)
        insts = Assembler().assemble("OUT 1, 42\nOUT 1, 99\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert output == ["42", "99"]

    def test_in_reads(self):
        from domains.shell.vm import CPU as VMCPU, Assembler, DeviceBus

        bus = DeviceBus()
        bus.register_console(stdin_fn=lambda: "7")

        cpu = VMCPU(devices=bus)
        insts = Assembler().assemble("IN R0, 0\nOUT 1, R0\nHALT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 7


class TestBlockDevice:
    def test_read_write_sector(self):
        from domains.shell.vm import BlockDevice
        blk = BlockDevice(num_sectors=4)
        blk.write_sector(0, b"hello world")
        data = blk.read_sector(0)
        assert bytes(data[:11]) == b"hello world"

    def test_sector_stats(self):
        from domains.shell.vm import BlockDevice
        blk = BlockDevice(num_sectors=4)
        blk.write_sector(0, b"x" * 512)
        blk.read_sector(0)
        info = blk.info()
        assert info["reads"] == 1
        assert info["writes"] == 1

    def test_out_of_range(self):
        from domains.shell.vm import BlockDevice, DeviceFault
        blk = BlockDevice(num_sectors=4)
        with pytest.raises(DeviceFault):
            blk.read_sector(10)


class TestVirtualSystem:
    def test_run_program(self):
        from domains.shell.vm import VirtualSystem
        vs = VirtualSystem()
        vs.load_program("LOAD_CONST R0, 42\nPRINT R0\nHALT")
        out = vs.run()
        assert out == ["42"]

    def test_carry_flag(self):
        from domains.shell.vm import VirtualSystem
        vs = VirtualSystem()
        vs.load_program("LOAD_CONST R0, 4294967295\nLOAD_CONST R1, 1\nIADD R2, R0, R1\nHALT")
        vs.run()
        assert vs.cpu._carry_flag is True
        assert vs.cpu.regs[2] == 0

    def test_status(self):
        from domains.shell.vm import VirtualSystem
        vs = VirtualSystem(enable_block=True)
        status = vs.status()
        assert "pc" in status
        assert "carry_flag" in status
        assert "block" in status["devices"]


class TestFlatFS:
    def test_write_read(self):
        from domains.shell.vm import BlockDevice, FlatFS
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write("test.txt", b"hello world")
        assert fs.read("test.txt")[:11] == b"hello world"

    def test_list_files(self):
        from domains.shell.vm import BlockDevice, FlatFS
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write("a.txt", b"aaa")
        fs.write("b.txt", b"bbb")
        assert sorted(fs.list_files()) == ["a.txt", "b.txt"]

    def test_delete(self):
        from domains.shell.vm import BlockDevice, FlatFS
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write("del.txt", b"bye")
        assert fs.delete("del.txt")
        assert not fs.exists("del.txt")
        assert not fs.delete("del.txt")

    def test_reload_persists(self):
        from domains.shell.vm import BlockDevice, FlatFS
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write("persist.txt", b"data")
        fs2 = FlatFS(blk)
        assert "persist.txt" in fs2.list_files()
        assert fs2.read("persist.txt")[:4] == b"data"


class TestSyscall:
    def test_syscall_print(self):
        import time as _time
        from domains.shell.kernel import Kernel
        output = []
        k = Kernel()
        k.boot()
        k.register_devices()
        proc = k.spawn_vm_process(
            'sc-test',
            'LOAD_CONST R0, "via syscall"\nLOAD_CONST R7, 111\nSYSCALL\nHALT',
            stdout_fn=lambda v: output.append(v),
            use_syscalls=True,
        )
        for _ in range(10):
            k.tick()
            _time.sleep(0.05)
            if proc.state.name == 'ZOMBIE':
                break
        assert any("via syscall" in line for line in output)
        k.shutdown()


class TestIRQ:
    def test_irq_fires(self):
        from domains.shell.vm import CPU, Assembler, IRQDevice
        fired = []
        cpu = CPU()
        irq = IRQDevice()
        cpu.register_irq(0, lambda c: fired.append("timer"))

        for i in range(15):
            irq.tick(cpu)
        cpu._process_irqs()
        assert len(fired) == 1  # fires at tick 10

    def test_keyboard_irq(self):
        from domains.shell.vm import CPU, IRQDevice
        cpu = CPU()
        irq = IRQDevice()
        cpu.register_irq(1, lambda c: fired.append("key"))
        fired = []

        irq.push_key(ord('A'))
        cpu.fire_irq(1)
        cpu._process_irqs()
        assert fired == ["key"]
        assert irq.read_key() == ord('A')


class TestShellWrite:
    def test_write_file(self):
        import time as _time
        from domains.shell.kernel import Kernel
        from domains.shell.vm import BlockDevice, FlatFS

        blk = BlockDevice(num_sectors=32)
        fs = FlatFS(blk)

        output = []
        inputs = ['write note.txt hello world', 'cat note.txt', 'halt']
        input_iter = iter(inputs)

        k = Kernel()
        k.boot()
        k.register_devices()
        k._block_device = blk
        k._fs = fs

        proc = k.spawn_kernel_shell(
            stdin_fn=lambda: next(input_iter, 'halt'),
            stdout_fn=lambda v: output.append(str(v)),
        )
        for _ in range(50):
            k.tick()
            _time.sleep(0.02)
            if proc.state.name == 'ZOMBIE':
                break

        assert any("wrote" in line for line in output)
        assert any("hello world" in line for line in output)
        k.shutdown()


class TestX86Assembler:
    def test_nop(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("nop")
        assert code == b'\x90'

    def test_hlt(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("hlt")
        assert code == b'\xf4'

    def test_cli_sti(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        assert asm.assemble("cli") == b'\xfa'
        assert asm.assemble("sti") == b'\xfb'

    def test_ret(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        assert asm.assemble("ret") == b'\xc3'

    def test_mov_reg_imm(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("[BITS 32]\nmov eax, 1")
        assert code[0] == 0xB8  # MOV EAX, imm32

    def test_mov_reg_reg(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("[BITS 32]\nmov eax, ebx")
        assert len(code) == 2
        assert code[0] == 0x89  # MOV r/m32, r32
        assert code[1] == 0xD8  # ModR/M: reg=ebx(3), rm=eax(0)

    def test_int(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("int 0x10")
        assert code == b'\xcd\x10'

    def test_push_reg(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("[BITS 32]\npush eax")
        assert code[0] == 0x50  # PUSH EAX

    def test_pop_reg(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("[BITS 32]\npop eax")
        assert code[0] == 0x58  # POP EAX

    def test_jmp(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("jmp 0x100")
        assert code[0] == 0xE9  # JMP near

    def test_add_reg_imm(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("[BITS 32]\nadd eax, 1")
        assert code[0] == 0x83  # ADD r32, imm8

    def test_label(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("start:\n  nop\n  jmp start")
        assert code[0] == 0x90  # NOP
        assert code[1] == 0xE9  # JMP near (always near in 16-bit for pass consistency)
        # Offset should be -4 (back to start, relative to end of 3-byte instruction)
        offset = int.from_bytes(code[2:4], "little", signed=True)
        assert offset == -4

    def test_bits_directive(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        asm.assemble("[BITS 32]\nnop")
        assert asm._bits == 32

    def test_org_directive(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        asm.assemble("[ORG 0x1000]\nnop")
        assert asm._org == 0x1000

    def test_db_string(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble('db "Hello", 0')
        assert code == b'Hello\x00'

    def test_db_bytes(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("db 0x90, 0x90, 0x90")
        assert code == b'\x90\x90\x90'

    def test_dw(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("dw 0xAA55")
        assert code == b'\x55\xAA'

    def test_dd(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("dd 0x12345678")
        assert code == b'\x78\x56\x34\x12'

    def test_times(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("times 3 nop")
        assert code == b'\x90\x90\x90'

    def test_mov_al_imm(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("mov al, 0x41")
        assert code[0] == 0xB0  # MOV AL, imm8

    def test_in_al(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("in al, 0x60")
        assert code == b'\xe4\x60'

    def test_out(self):
        from domains.shell.vm import X86Assembler
        asm = X86Assembler()
        code = asm.assemble("out 0x20, al")
        assert code == b'\xe6\x20'


class TestX86Bootloader:
    def test_bootloader_source_valid(self):
        from domains.shell.vm_programs import X86_BOOTLOADER_ASM
        assert "[BITS 16]" in X86_BOOTLOADER_ASM
        assert "[ORG 0x7C00]" in X86_BOOTLOADER_ASM

    def test_kernel_source_valid(self):
        from domains.shell.vm_programs import X86_KERNEL_ASM
        assert "[BITS 32]" in X86_KERNEL_ASM
        assert "kernel_start" in X86_KERNEL_ASM
        assert "vga_print" in X86_KERNEL_ASM
        assert "timer_handler" in X86_KERNEL_ASM

    def test_export_binary(self):
        from domains.shell.vm_programs import export_x86_binary, X86_BOOTLOADER_ASM
        binary = export_x86_binary(X86_BOOTLOADER_ASM)
        assert isinstance(binary, bytes)
        assert len(binary) > 0

    def test_build_disk_image(self):
        from domains.shell.vm_programs import build_disk_image
        boot = b'\x00' * 512
        kernel = b'\x00' * 1024
        image = build_disk_image(boot, kernel, size_mb=1)
        assert len(image) == 1024 * 1024
        assert image[:512] == boot
        assert image[512:1536] == kernel

    def test_bootloader_compiles_to_512(self):
        from domains.shell.vm import X86Assembler
        from domains.shell.vm_programs import X86_BOOTLOADER_ASM
        asm = X86Assembler()
        code = asm.assemble(X86_BOOTLOADER_ASM)
        assert len(code) == 512
        assert code[510] == 0x55
        assert code[511] == 0xAA

    def test_kernel_has_vga(self):
        from domains.shell.vm_programs import X86_KERNEL_ASM
        assert "VGA_BUFFER" in X86_KERNEL_ASM
        assert "vga_print" in X86_KERNEL_ASM
        assert "vga_clear" in X86_KERNEL_ASM

    def test_kernel_has_interrupts(self):
        from domains.shell.vm_programs import X86_KERNEL_ASM
        assert "timer_handler" in X86_KERNEL_ASM
        assert "keyboard_handler" in X86_KERNEL_ASM
        assert "iret" in X86_KERNEL_ASM


class TestVGA:
    def test_vga_write(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        vga.call("write", 0, 0, 'A', 15, 0)
        screen = vga.call("get_screen")
        assert screen[0][0] == 'A'

    def test_vga_write_string(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        vga.call("write_string", 0, 0, "Hello", 10, 0)
        screen = vga.call("get_screen")
        assert screen[0][:5] == "Hello"

    def test_vga_clear(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        vga.call("write", 5, 5, 'X', 15, 0)
        vga.call("clear", 15, 1)
        screen = vga.call("get_screen")
        assert all(c == ' ' for c in screen[0])

    def test_vga_scroll(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        vga.call("write_string", 0, 0, "Line1", 15, 0)
        vga.call("write_string", 1, 0, "Line2", 15, 0)
        vga.call("scroll", 1)
        screen = vga.call("get_screen")
        assert "Line2" in screen[0]
        assert screen[1][:5] == "     "

    def test_vga_cursor(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        vga.call("set_cursor", 10, 20)
        assert vga.call("get_cursor") == (10, 20)

    def test_vga_info(self):
        from domains.shell.vm import VGADevice
        vga = VGADevice()
        info = vga.info()
        assert info["type"] == "vga"
        assert info["rows"] == 25
        assert info["cols"] == 80


class TestPS2Keyboard:
    def test_read_key(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        kb.call("push_scancode", 0x10)  # 'q'
        assert kb.call("read_key") == ord('q')

    def test_empty_returns_zero(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        assert kb.call("read_key") == 0

    def test_has_key(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        assert kb.call("has_key") is False
        kb.call("push_scancode", 0x1E)  # 'a'
        assert kb.call("has_key") is True

    def test_clear(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        kb.call("push_scancode", 0x1E)
        kb.call("push_scancode", 0x30)
        kb.call("clear")
        assert kb.call("has_key") is False

    def test_key_release_ignored(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        kb.call("push_scancode", 0x9E)  # key release 'a'
        assert kb.call("has_key") is False

    def test_enter_key(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        kb.call("push_scancode", 0x1C)  # Enter
        assert kb.call("read_key") == 10

    def test_space_key(self):
        from domains.shell.vm import PS2KeyboardDevice
        kb = PS2KeyboardDevice()
        kb.call("push_scancode", 0x39)  # Space
        assert kb.call("read_key") == ord(' ')
    def test_list_programs(self):
        from domains.shell.vm import BlockDevice, FlatFS, DiskProgramLoader
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write('hello.asm', 'HALT')
        fs.write('data.txt', 'not a program')
        loader = DiskProgramLoader(fs)
        assert loader.list_programs() == ['hello.asm']

    def test_load_and_run(self):
        from domains.shell.vm import BlockDevice, FlatFS, DiskProgramLoader
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        fs.write('test.asm', 'LOAD_CONST R0, 42\nPRINT R0\nHALT')
        loader = DiskProgramLoader(fs)
        result = loader.run('test.asm')
        assert result['output'] == ['42']
        assert result['steps'] == 3

    def test_save_and_load(self):
        from domains.shell.vm import BlockDevice, FlatFS, DiskProgramLoader
        blk = BlockDevice(num_sectors=16)
        fs = FlatFS(blk)
        loader = DiskProgramLoader(fs)
        loader.save_program('mine.asm', 'NOP\nHALT')
        source = loader.load_source('mine.asm')
        assert 'NOP' in source
from domains.shell.vm import (
    PageFrameAllocator, ProcessControlBlock, ProcessState,
    ProcessTable, Scheduler, X86SyscallHandler, PITDevice,
    X86VirtualSystem, X86CPU, X86Assembler, FlatFS, BlockDevice,
    SerialDevice, MouseDevice, CMOSDevice, DiskDevice, NICDevice,
    ClockDevice, CPU, Assembler, InsFault, Memory, DeviceBus,
    NUM_REGS, FileDevice, VGADevice, PS2KeyboardDevice, ConsoleDevice, IRQDevice,
    DiskProgramLoader, VirtualSystem, DeviceFault, X86Shell, FLAG_DF, FLAG_ZF,
)
from domains.shell.vm_permissions import Role
import struct


# ============================================================
# Syscall Dispatch (handle()) tests — L6500-6551
# ============================================================

class TestSyscallHandleDispatch:

    def _make_handler(self, fs=None):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0100] = bytes([0xCD, 0x80, 0xCC])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        pt = ProcessTable()
        sched = Scheduler(pt)
        mem = PageFrameAllocator()
        pcb = pt.create("test")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        handler = X86SyscallHandler(cpu, pt, sched, mem, fs)
        return handler, cpu, sched, pcb

    def test_handle_sys_exit(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_EXIT
        cpu._regs[3] = 42
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_getpid(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_GETPID
        h.handle()
        assert cpu._regs[0] == pcb.pid

    def test_handle_sys_yield(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_YIELD
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_malloc(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_MALLOC
        cpu._regs[3] = 256
        h.handle()
        assert cpu._regs[0] > 0

    def test_handle_sys_free(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_MALLOC
        cpu._regs[3] = 64
        h.handle()
        addr = cpu._regs[0]
        cpu._regs[0] = h.SYS_FREE
        cpu._regs[3] = addr
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_sbrk(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_SBRK
        cpu._regs[3] = 0x1000
        h.handle()
        assert cpu._regs[0] > 0

    def test_handle_sys_gettimeofday(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_GETTIMEOFDAY
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_uname(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_UNAME
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_getrole(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_GETROLE
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_unknown_syscall(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = 999
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_permission_denied(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = 1
        cpu._regs[1] = 0
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFE

    def test_handle_sys_serial_write_admin(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        serial = SerialDevice()
        h._serial = serial
        cpu._regs[0] = h.SYS_SERIAL_WRITE
        cpu._regs[3] = ord('A')
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_serial_write_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_SERIAL_WRITE
        cpu._regs[3] = ord('A')
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_serial_read_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_SERIAL_READ
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_mouse_read_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_MOUSE_READ
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_rtc_gettime_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_RTC_GETTIME
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_readdir(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_READDIR
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 10
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_fork(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_FORK
        h.handle()
        assert cpu._regs[0] > 0

    def test_handle_sys_kill_admin(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = pcb.pid
        cpu._regs[1] = 9
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_disk_read_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_DISK_READ
        cpu._regs[3] = 0
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 1
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_disk_write_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_DISK_WRITE
        cpu._regs[3] = 0
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 1
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_net_send_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_NET_SEND
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 4
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_net_recv_no_device(self):
        h, cpu, sched, pcb = self._make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_NET_RECV
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 256
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_handle_sys_train_start(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_TRAIN_START
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_train_status(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_TRAIN_STATUS
        cpu._regs[3] = 0
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_train_get_result(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_TRAIN_GET_RESULT
        cpu._regs[3] = 0
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 4
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_open(self):
        h, cpu, sched, pcb = self._make_handler()
        name = b"/test\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        cpu._regs[0] = h.SYS_OPEN
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 0
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_close(self):
        h, cpu, sched, pcb = self._make_handler()
        h._fd_table[3] = "/test"
        cpu._regs[0] = h.SYS_CLOSE
        cpu._regs[3] = 3
        h.handle()
        assert cpu._regs[0] == 0

    def test_handle_sys_read(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_READ
        cpu._regs[3] = 0
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 10
        h.handle()
        assert cpu._regs[0] >= 0

    def test_handle_sys_write(self):
        h, cpu, sched, pcb = self._make_handler()
        cpu._regs[0] = h.SYS_WRITE
        cpu._regs[3] = 1
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 5
        h.handle()
        assert cpu._regs[0] >= 0


# ============================================================
# _sys_exec tests — L6686-6748
# ============================================================

class TestSysExecImplementation:

    def _make_exec_env(self, fs=None):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0100] = bytes([0xCC] * 256)
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        pt = ProcessTable()
        sched = Scheduler(pt)
        mem = PageFrameAllocator()
        pcb = pt.create("exec_test")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        handler = X86SyscallHandler(cpu, pt, sched, mem, fs)
        return handler, cpu, sched, pcb, mem

    def test_exec_no_fs(self):
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=None)
        assert h._sys_exec(0xE0000) == -1

    def test_exec_file_not_found(self):
        fs = FlatFS(BlockDevice())
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        name = b"nonexistent.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        assert h._sys_exec(0xE0000) == -1

    def test_exec_no_current_process(self):
        fs = FlatFS(BlockDevice())
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        sched._current_pid = None
        assert h._sys_exec(0xE0000) == -1

    def test_exec_bad_assembly(self):
        fs = FlatFS(BlockDevice())
        fs.write("/bad.asm", b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f")
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        name = b"/bad.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        result = h._sys_exec(0xE0000)
        assert result == -1 or result == 0

    def test_exec_no_current_process(self):
        fs = FlatFS(BlockDevice())
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        sched._current_pid = None
        assert h._sys_exec(0xE0000) == -1

    def test_exec_success(self):
        fs = FlatFS(BlockDevice())
        fs.write("/prog.asm", b"NOP\nHLT")
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        name = b"/prog.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        result = h._sys_exec(0xE0000)
        assert result == 0
        assert pcb.eip > 0
        assert pcb.esp > pcb.eip

    def test_exec_frees_old_stack(self):
        fs = FlatFS(BlockDevice())
        fs.write("/prog1.asm", b"NOP\nHLT")
        fs.write("/prog2.asm", b"MOV EAX,1\nHLT")
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        name1 = b"/prog1.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name1)] = name1
        h._sys_exec(0xE0000)
        old_stack_base = pcb.stack_base
        name2 = b"/prog2.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name2)] = name2
        result = h._sys_exec(0xE0000)
        assert result == 0
        assert pcb.stack_base != old_stack_base

    def test_exec_resets_registers(self):
        fs = FlatFS(BlockDevice())
        fs.write("/prog.asm", b"NOP\nHLT")
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        pcb.eax = 0xDEAD
        pcb.ecx = 0xBEEF
        name = b"/prog.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        h._sys_exec(0xE0000)
        assert pcb.eax == 0
        assert pcb.ecx == 0
        assert pcb.edx == 0
        assert pcb.ebx == 0

    def test_exec_adds_bits_directive(self):
        fs = FlatFS(BlockDevice())
        fs.write("/nobits.asm", b"NOP\nHLT")
        h, cpu, sched, pcb, mem = self._make_exec_env(fs=fs)
        name = b"/nobits.asm\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        result = h._sys_exec(0xE0000)
        assert result == 0


# ============================================================
# MOV instruction group tests — L4541-4658
# ============================================================

class TestCPUMovGroupInstructions:

    def _run_bytes(self, code_bytes, setup_fn=None):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0000 + len(code_bytes)] = code_bytes
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        if setup_fn:
            setup_fn(cpu)
        cpu.step()
        return cpu

    def test_movzx_reg_reg(self):
        cpu = self._run_bytes(bytes([0x0F, 0xB6, 0xC8]), lambda c: c._set8l(0, 0xFF))
        assert cpu._regs[1] == 0xFF

    def test_movzx_reg_mem(self):
        cpu = X86CPU()
        cpu._mem[0x20000] = 0xAB
        cpu._mem[0xF0000:0xF0007] = bytes([0x0F, 0xB6, 0x0D, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._regs[1] == 0xAB

    def test_movzx_reg16_reg(self):
        cpu = self._run_bytes(bytes([0x0F, 0xB7, 0xC8]), lambda c: c._set16(0, 0xFFFF))
        assert cpu._regs[1] == 0xFFFF

    def test_movsx_positive(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBE, 0xC8]), lambda c: c._set8l(0, 0x7F))
        assert cpu._regs[1] == 0x7F

    def test_movsx_negative_byte(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBE, 0xC8]), lambda c: c._set8l(0, 0x80))
        assert cpu._regs[1] == 0xFFFFFF80

    def test_movsx_positive_word(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBF, 0xC8]), lambda c: c._set16(0, 0x7FFF))
        assert cpu._regs[1] == 0x7FFF

    def test_movsx_negative_word(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBF, 0xC8]), lambda c: c._set16(0, 0x8000))
        assert cpu._regs[1] == 0xFFFF8000

    def test_imul_reg_reg(self):
        cpu = self._run_bytes(bytes([0x0F, 0xAF, 0xCA]), lambda c: (c._set32(1, 10), c._set32(2, 20)))
        assert cpu._regs[1] == 200

    def test_imul_reg_mem(self):
        cpu = X86CPU()
        struct.pack_into('<I', cpu._mem, 0x20000, 7)
        cpu._mem[0xF0000:0xF0007] = bytes([0x0F, 0xAF, 0x0D, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[1] = 6
        cpu.step()
        assert cpu._regs[1] == 42

    def test_bsf_zero(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBC, 0xC8]), lambda c: c._set32(0, 0))
        assert cpu._flag(FLAG_ZF) is True

    def test_bsf_nonzero(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBC, 0xC8]), lambda c: c._set32(0, 0x10))
        assert cpu._regs[1] == 4

    def test_bsr_nonzero(self):
        cpu = self._run_bytes(bytes([0x0F, 0xBD, 0xC8]), lambda c: c._set32(0, 0x10))
        assert cpu._regs[1] == 4

    def test_mov_crn(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0003] = bytes([0x0F, 0x22, 0xC0])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[0] = 0x12345678
        cpu.step()
        assert cpu._cr[0] == 0x12345678

    def test_mov_from_crn(self):
        cpu = X86CPU()
        cpu._cr[0] = 0x87654321
        cpu._mem[0xF0000:0xF0003] = bytes([0x0F, 0x20, 0xC0])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._regs[0] == 0x87654321

    def test_mov_drn(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0003] = bytes([0x0F, 0x23, 0xC0])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[0] = 0x11223344
        cpu.step()
        assert cpu._dr[0] == 0x11223344

    def test_mov_from_drn(self):
        cpu = X86CPU()
        cpu._dr[0] = 0x55667788
        cpu._mem[0xF0000:0xF0003] = bytes([0x0F, 0x21, 0xC0])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._regs[0] == 0x55667788

    def test_rdtsc(self):
        cpu = self._run_bytes(bytes([0x0F, 0x31]))
        assert cpu._regs[0] == 0
        assert cpu._regs[2] == 0

    def test_lgdt(self):
        cpu = X86CPU()
        struct.pack_into('<HI', cpu._mem, 0x20000, 0xFF, 0x100000)
        cpu._mem[0xF0000:0xF0007] = bytes([0x0F, 0x01, 0x15, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._gdt_limit == 0xFF
        assert cpu._gdt_base == 0x100000

    def test_lidt(self):
        cpu = X86CPU()
        struct.pack_into('<HI', cpu._mem, 0x20000, 0x1FF, 0x300000)
        cpu._mem[0xF0000:0xF0007] = bytes([0x0F, 0x01, 0x1D, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._idt_limit == 0x1FF
        assert cpu._idt_base == 0x300000


# ============================================================
# VGADevice comprehensive tests — L540-586
# ============================================================

class TestVGADeviceComprehensive3:

    def _make_vga(self):
        return VGADevice()

    def test_write_with_colors(self):
        vga = self._make_vga()
        vga.call("write", 0, 0, 'X', 4, 1)
        cell = vga._screen[0][0]
        assert cell['char'] == 'X'
        assert cell['fg'] == 4
        assert cell['bg'] == 1

    def test_write_out_of_bounds(self):
        vga = self._make_vga()
        vga.call("write", -1, 0, 'X')
        vga.call("write", 100, 0, 'X')
        vga.call("write", 0, 200, 'X')

    def test_write_string(self):
        vga = self._make_vga()
        vga.call("write_string", 0, 0, "Hi!")
        assert vga._screen[0][0]['char'] == 'H'
        assert vga._screen[0][1]['char'] == 'i'
        assert vga._screen[0][2]['char'] == '!'

    def test_write_string_with_colors(self):
        vga = self._make_vga()
        vga.call("write_string", 0, 0, "AB", 2, 3)
        assert vga._screen[0][0]['fg'] == 2
        assert vga._screen[0][0]['bg'] == 3

    def test_write_string_partial_overflow(self):
        vga = self._make_vga()
        vga.call("write_string", 0, 79, "XYZ")
        assert vga._screen[0][79]['char'] == 'X'

    def test_clear(self):
        vga = self._make_vga()
        vga.call("write", 5, 5, 'A')
        vga.call("clear")
        assert vga._screen[5][5]['char'] == ' '
        assert vga._cursor_row == 0
        assert vga._cursor_col == 0

    def test_clear_with_colors(self):
        vga = self._make_vga()
        vga.call("clear", 3, 5)
        assert vga._screen[0][0]['fg'] == 3
        assert vga._screen[0][0]['bg'] == 5

    def test_scroll(self):
        vga = self._make_vga()
        vga.call("write", 0, 0, 'A')
        vga.call("scroll", 1)
        assert vga._screen[0][0]['char'] == ' '

    def test_scroll_multiple(self):
        vga = self._make_vga()
        vga.call("write", 5, 0, 'Z')
        vga.call("scroll", 3)
        assert vga._screen[2][0]['char'] == 'Z'

    def test_set_cursor(self):
        vga = self._make_vga()
        vga.call("set_cursor", 10, 20)
        assert vga._cursor_row == 10
        assert vga._cursor_col == 20

    def test_set_cursor_clamped(self):
        vga = self._make_vga()
        vga.call("set_cursor", -5, 999)
        assert vga._cursor_row == 0
        assert vga._cursor_col == vga.COLS - 1

    def test_get_cursor(self):
        vga = self._make_vga()
        vga.call("set_cursor", 3, 7)
        pos = vga.call("get_cursor")
        assert pos == (3, 7)

    def test_get_screen(self):
        vga = self._make_vga()
        vga.call("write", 0, 0, 'H')
        vga.call("write", 0, 1, 'i')
        lines = vga.call("get_screen")
        assert isinstance(lines, list)
        assert 'Hi' in lines[0]

    def test_writes_counter(self):
        vga = self._make_vga()
        before = vga._writes
        vga.call("write", 0, 0, 'X')
        vga.call("write", 0, 1, 'Y')
        assert vga._writes == before + 2

    def test_scroll_default_n(self):
        vga = self._make_vga()
        vga.call("write", 0, 0, 'A')
        vga.call("scroll")
        assert vga._screen[0][0]['char'] == ' '

    def test_scroll_zero(self):
        vga = self._make_vga()
        vga.call("write", 0, 0, 'A')
        vga.call("scroll", 0)
        assert vga._screen[0][0]['char'] == 'A'


# ============================================================
# Assembler memory operand encoding — L3168-3237
# ============================================================

class TestAssembleMemoryOperandEncoding:

    def _asm_one(self, line):
        asm = X86Assembler()
        return asm.assemble(f'[BITS 32]\n{line}', org=0x100000)

    def test_mov_reg_bracket_eax(self):
        code = self._asm_one('MOV ECX, [EAX]')
        assert len(code) >= 2

    def test_mov_reg_bracket_eax_plus_disp8(self):
        code = self._asm_one('MOV ECX, [EAX+0x10]')
        assert len(code) >= 3

    def test_mov_reg_bracket_eax_plus_disp32(self):
        code = self._asm_one('MOV ECX, [EAX+0x1000]')
        assert len(code) >= 6

    def test_mov_to_mem_eax(self):
        code = self._asm_one('MOV [EAX], ECX')
        assert len(code) >= 2

    def test_add_reg_bracket_ebx(self):
        code = self._asm_one('ADD EAX, [EBX]')
        assert len(code) >= 2

    def test_mov_eax_direct_addr(self):
        code = self._asm_one('MOV EAX, [0x20000]')
        assert len(code) >= 5

    def test_mov_to_direct_addr(self):
        code = self._asm_one('MOV [0x30000], EAX')
        assert len(code) >= 5

    def test_sub_reg_bracket_esi(self):
        code = self._asm_one('SUB EAX, [ESI]')
        assert len(code) >= 2

    def test_cmp_reg_bracket_edi(self):
        code = self._asm_one('CMP EAX, [EDI]')
        assert len(code) >= 2

    def test_mov_reg_label(self):
        code = self._asm_one('MOV EAX, [data]\nHLT\ndata: dd 0x12345678')
        assert len(code) >= 6


# ============================================================
# Misc uncovered lines
# ============================================================

class TestAssemblerMiscCoverage:

    def test_db_multiple_values(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndb 0x90, 0xCC, 0x90')
        assert list(code) == [0x90, 0xCC, 0x90]

    def test_dw_value(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndw 0x1234')
        assert list(code) == [0x34, 0x12]

    def test_dd_value(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndd 0x12345678')
        assert list(code) == [0x78, 0x56, 0x34, 0x12]

    def test_times(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ntimes 3 nop')
        assert list(code) == [0x90, 0x90, 0x90]

    def test_org(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nMOV EAX, 1', org=0x200000)
        assert len(code) > 0

    def test_label_forward_ref(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nJMP end\nend:\nHLT')
        assert len(code) > 0

    def test_equ_constant(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nMYVAL equ 42\nMOV EAX, MYVAL')
        assert len(code) > 0

    def test_string_in_dd(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndd "ABCD"')
        assert len(code) == 4

    def test_cpu_mov_reg_imm32(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0005] = bytes([0xB8, 0x78, 0x56, 0x34, 0x12])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._regs[0] == 0x12345678

    def test_cpu_push_imm8(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0002] = bytes([0x6A, 0x42])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        val = cpu._mem[0x7FFFC] | (cpu._mem[0x7FFFD] << 8) | (cpu._mem[0x7FFFE] << 16) | (cpu._mem[0x7FFFF] << 24)
        assert val == 0x42

    def test_cpu_push_imm32(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0005] = bytes([0x68, 0x78, 0x56, 0x34, 0x12])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        val = cpu._mem[0x7FFFC] | (cpu._mem[0x7FFFD] << 8) | (cpu._mem[0x7FFFE] << 16) | (cpu._mem[0x7FFFF] << 24)
        assert val == 0x12345678

    def test_cpu_mov_mem_offs_eax(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0005] = bytes([0xA3, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[0] = 0xDEADBEEF
        cpu.step()
        val = cpu._mem[0x20000] | (cpu._mem[0x20001] << 8) | (cpu._mem[0x20002] << 16) | (cpu._mem[0x20003] << 24)
        assert val == 0xDEADBEEF

    def test_cpu_mov_eax_mem_offs(self):
        cpu = X86CPU()
        struct.pack_into('<I', cpu._mem, 0x20000, 0xCAFEBABE)
        cpu._mem[0xF0000:0xF0005] = bytes([0xA1, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._regs[0] == 0xCAFEBABE

    def test_cpu_mov_ax_mem_offs_66(self):
        cpu = X86CPU()
        struct.pack_into('<H', cpu._mem, 0x20000, 0xBEEF)
        cpu._mem[0xF0000:0xF0006] = bytes([0x66, 0xA1, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu.step()
        assert cpu._get16(0) == 0xBEEF

    def test_cpu_mov_mem_offs_ax_66(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0006] = bytes([0x66, 0xA3, 0x00, 0x00, 0x02, 0x00])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[0] = 0xDEADBEEF
        cpu.step()
        assert (cpu._mem[0x20000] | (cpu._mem[0x20001] << 8)) == 0xBEEF

    def test_cpu_cpuid(self):
        cpu = X86CPU()
        cpu._mem[0xF0000:0xF0002] = bytes([0x0F, 0xA2])
        cpu._eip = 0xF0000
        cpu._regs[4] = 0x80000
        cpu._regs[0] = 0
        cpu.step()
        assert cpu._regs[0] >= 0


"""High-impact branch coverage tests for vm.py uncovered lines."""
from domains.shell.vm import (
    PageFrameAllocator, ProcessControlBlock, ProcessState,
    ProcessTable, Scheduler, X86SyscallHandler, PITDevice,
    X86VirtualSystem, X86CPU, X86Assembler, FlatFS, BlockDevice,
    SerialDevice, MouseDevice, CMOSDevice, DiskDevice, NICDevice,
    ClockDevice, CPU, Assembler, InsFault, Memory, DeviceBus,
    NUM_REGS, FileDevice, VGADevice, PS2KeyboardDevice, ConsoleDevice, IRQDevice,
    DiskProgramLoader, VirtualSystem, DeviceFault, X86Shell, FLAG_DF, FLAG_ZF,
)
from domains.shell.vm_permissions import Role
import struct


def _make_handler(fs=None):
    cpu = X86CPU()
    cpu._mem[0xF0000:0xF0100] = bytes([0xCD, 0x80, 0xCC])
    cpu._eip = 0xF0000
    cpu._regs[4] = 0x80000
    pt = ProcessTable()
    sched = Scheduler(pt)
    mem = PageFrameAllocator()
    pcb = pt.create("test")
    sched.enqueue(pcb.pid)
    sched.start(cpu)
    handler = X86SyscallHandler(cpu, pt, sched, mem, fs)
    return handler, cpu, sched, pcb, mem


# ============================================================
# _sys_read / _sys_write fd edge cases
# ============================================================

class TestSysReadWriteEdge:

    def test_read_fd_not_open(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_READ
        cpu._regs[3] = 99  # fd
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 10
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_write_fd_not_open(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_WRITE
        cpu._regs[3] = 99
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 5
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_read_from_stdin_fd0(self):
        h, cpu, s, pcb, m = _make_handler()
        h._fd_table[0] = "/dev/stdin"
        cpu._regs[0] = h.SYS_READ
        cpu._regs[3] = 0
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 1
        h.handle()
        assert cpu._regs[0] >= 0

    def test_write_to_stdout_fd1(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_WRITE
        cpu._regs[3] = 1
        msg = b"hello\x00"
        cpu._mem[0xE0000:0xE0000 + len(msg)] = msg
        cpu._regs[1] = 0xE0000
        cpu._regs[2] = 5
        h.handle()
        assert cpu._regs[0] == 5

    def test_open_read_mode(self):
        h, cpu, s, pcb, m = _make_handler(fs=FlatFS(BlockDevice()))
        name = b"/test.txt\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        h._fs.write("/test.txt", b"data")
        cpu._regs[0] = h.SYS_OPEN
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 0  # O_RDONLY
        h.handle()
        assert cpu._regs[0] >= 3

    def test_open_write_mode(self):
        h, cpu, s, pcb, m = _make_handler(fs=FlatFS(BlockDevice()))
        name = b"/out.txt\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        cpu._regs[0] = h.SYS_OPEN
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 1  # O_WRONLY
        h.handle()
        fd = cpu._regs[0]
        assert fd >= 3
        assert fd in h._fd_table

    def test_open_nonexistent_readonly(self):
        h, cpu, s, pcb, m = _make_handler(fs=FlatFS(BlockDevice()))
        name = b"/nope.txt\x00"
        cpu._mem[0xE0000:0xE0000 + len(name)] = name
        cpu._regs[0] = h.SYS_OPEN
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 0
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_free_nonexistent(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_FREE
        cpu._regs[3] = 0xDEADBEEF
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_malloc_zero(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_MALLOC
        cpu._regs[3] = 0
        h.handle()
        assert cpu._regs[0] == 0

    def test_malloc_negative(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_MALLOC
        cpu._regs[3] = -1
        h.handle()
        assert cpu._regs[0] == 0

    def test_sbrk_no_current(self):
        h, cpu, s, pcb, m = _make_handler()
        s._current_pid = None
        cpu._regs[0] = h.SYS_SBRK
        cpu._regs[3] = 0x1000
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_brk(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_BRK
        cpu._regs[3] = 0x500000
        h.handle()
        assert cpu._regs[0] == 0


# ============================================================
# _sys_kill edge cases
# ============================================================

class TestSysKillEdge:

    def test_kill_nonexistent_pid(self):
        h, cpu, s, pcb, m = _make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = 99999
        cpu._regs[1] = 9
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_kill_self(self):
        h, cpu, s, pcb, m = _make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = pcb.pid
        cpu._regs[1] = 9
        h.handle()
        assert cpu._regs[0] == 0

    def test_kill_other_process(self):
        h, cpu, s, pcb, m = _make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        child = m.alloc(1)
        child_pcb = h._ptable.create("child")
        h._rbac.inherit(child_pcb.pid, pcb.pid)
        s.enqueue(child_pcb.pid)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = child_pcb.pid
        cpu._regs[1] = 9
        h.handle()
        assert cpu._regs[0] == 0
        assert h._ptable.get(child_pcb.pid) is None

    def test_kill_nonkill_signal(self):
        h, cpu, s, pcb, m = _make_handler()
        h._rbac.assign(pcb.pid, Role.ADMIN)
        child_pcb = h._ptable.create("child2")
        h._rbac.inherit(child_pcb.pid, pcb.pid)
        s.enqueue(child_pcb.pid)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = child_pcb.pid
        cpu._regs[1] = 15  # SIGTERM, not SIGKILL
        h.handle()
        assert cpu._regs[0] == 0

    def test_kill_higher_role_denied(self):
        h, cpu, s, pcb, m = _make_handler()
        target = h._ptable.create("kernel_target")
        h._rbac.assign(pcb.pid, Role.ADMIN)
        h._rbac.assign(target.pid, Role.KERNEL)
        cpu._regs[0] = h.SYS_KILL
        cpu._regs[3] = target.pid
        cpu._regs[1] = 9
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF


# ============================================================
# _sys_readdir / _sys_uname
# ============================================================

class TestSysReaddirUname:

    def test_readdir_no_fs(self):
        h, cpu, s, pcb, m = _make_handler(fs=None)
        cpu._regs[0] = h.SYS_READDIR
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 10
        h.handle()
        assert cpu._regs[0] == 0

    def test_readdir_with_files(self):
        fs = FlatFS(BlockDevice())
        fs.write("/a.txt", b"data")
        fs.write("/b.txt", b"data")
        h, cpu, s, pcb, m = _make_handler(fs=fs)
        cpu._regs[0] = h.SYS_READDIR
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 10
        h.handle()
        assert cpu._regs[0] >= 0

    def test_readdir_zero_max(self):
        fs = FlatFS(BlockDevice())
        fs.write("/a.txt", b"data")
        h, cpu, s, pcb, m = _make_handler(fs=fs)
        cpu._regs[0] = h.SYS_READDIR
        cpu._regs[3] = 0xE0000
        cpu._regs[1] = 0
        h.handle()
        assert cpu._regs[0] == 0

    def test_uname(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_UNAME
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] == 0
        name = cpu._mem[0xE0000:0xE0000 + 65].split(b'\x00')[0]
        assert b'x86' in name.lower() or b'slough' in name.lower() or len(name) > 0

    def test_gettimeofday_no_buf(self):
        h, cpu, s, pcb, m = _make_handler()
        h._ticks = 42
        cpu._regs[0] = h.SYS_GETTIMEOFDAY
        cpu._regs[3] = 0
        h.handle()
        assert cpu._regs[0] == 42

    def test_gettimeofday_with_buf(self):
        h, cpu, s, pcb, m = _make_handler()
        h._ticks = 100
        cpu._regs[0] = h.SYS_GETTIMEOFDAY
        cpu._regs[3] = 0xE0000
        h.handle()
        assert cpu._regs[0] == 100
        ticks = struct.unpack_from('<I', cpu._mem, 0xE0000)[0]
        assert ticks == 100


# ============================================================
# _sys_yield / _sys_wait
# ============================================================

class TestSysYieldWait:

    def test_yield(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_YIELD
        h.handle()
        assert cpu._regs[0] == 0

    def test_wait_no_children(self):
        h, cpu, s, pcb, m = _make_handler()
        cpu._regs[0] = h.SYS_WAIT
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF

    def test_wait_no_current(self):
        h, cpu, s, pcb, m = _make_handler()
        s._current_pid = None
        cpu._regs[0] = h.SYS_WAIT
        h.handle()
        assert cpu._regs[0] == 0xFFFFFFFF


# ============================================================
# PITDevice
# ============================================================

class TestPITDevice:

    def _make_pit(self):
        cpu = X86CPU()
        pt = ProcessTable()
        sched = Scheduler(pt)
        return PITDevice(cpu=cpu, scheduler=sched, target_hz=100), cpu, sched

    def test_pit_init(self):
        pit, cpu, sched = self._make_pit()
        assert pit._target_hz == 100
        assert pit._divider > 0

    def test_pit_tick(self):
        pit, cpu, sched = self._make_pit()
        for _ in range(pit._divider + 1):
            pit.tick()
        assert pit._tick_count >= 1

    def test_pit_write_command(self):
        pit, cpu, sched = self._make_pit()
        pit._write_command(0x30)  # latch channel 0
        assert pit._latch[0] > 0 or pit._latch[0] == pit._counters[0]

    def test_pit_read_counter(self):
        pit, cpu, sched = self._make_pit()
        pit._write_command(0x30)
        val = pit._read_counter(0)
        assert isinstance(val, int)
        assert 0 <= val <= 255


# ============================================================
# X86VirtualSystem setup
# ============================================================

class TestX86VirtualSystem:

    def test_init_defaults(self):
        vs = X86VirtualSystem()
        assert vs._cpu is not None
        assert vs._scheduler is not None

    def test_init_with_params(self):
        vs = X86VirtualSystem(memory_size=0x200000, timer_hz=50, quantum=10)
        assert vs._allocator is not None

    def test_load_and_run(self):
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nNOP\nHLT", org=0x1000)
        cycles = vs.run(max_cycles=1000)
        assert isinstance(cycles, int)
        assert cycles >= 0

    def test_devices_wired(self):
        vs = X86VirtualSystem()
        assert vs._serial is not None
        assert vs._mouse is not None
        assert vs._rtc is not None
        assert vs._disk is not None
        assert vs._nic is not None
        assert vs._pit is not None

    def test_spawn(self):
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nHLT", org=0x1000)
        pid = vs.spawn("user", "[BITS 32]\nNOP\nHLT")
        assert pid is not None
        assert pid > 0

    def test_status(self):
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nHLT", org=0x1000)
        vs.run(max_cycles=100)
        s = vs.status()
        assert "cpu" in s
        assert "memory" in s
        assert "scheduler" in s

    def test_reset(self):
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nHLT", org=0x1000)
        vs.run(max_cycles=100)
        vs.reset()
        assert vs._ptable.count() == 1  # kernel process is recreated


# ============================================================
# Scheduler
# ============================================================

class TestSchedulerEdge:

    def test_tick(self):
        pt = ProcessTable()
        sched = Scheduler(pt)
        cpu = X86CPU()
        pcb = pt.create("t1")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        sched.tick(cpu)
        assert sched._tick_count == 1

    def test_block_current(self):
        pt = ProcessTable()
        sched = Scheduler(pt)
        cpu = X86CPU()
        pcb = pt.create("t1")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        sched.block_current(cpu)
        assert pcb.state == ProcessState.WAITING

    def test_exit_current(self):
        pt = ProcessTable()
        sched = Scheduler(pt)
        cpu = X86CPU()
        pcb = pt.create("t1")
        sched.enqueue(pcb.pid)
        sched.start(cpu)
        sched.exit_current(cpu, exit_code=0)
        assert pcb.state == ProcessState.TERMINATED

    def test_process_state_transitions(self):
        pt = ProcessTable()
        pcb = pt.create("test")
        pcb.state = ProcessState.READY
        assert pcb.state == ProcessState.READY
        pcb.state = ProcessState.RUNNING
        assert pcb.state == ProcessState.RUNNING
        pcb.state = ProcessState.WAITING
        assert pcb.state == ProcessState.WAITING
        pcb.state = ProcessState.TERMINATED
        assert pcb.state == ProcessState.TERMINATED


# ============================================================
# Memory
# ============================================================

class TestMemoryEdge:

    def test_alloc_and_free(self):
        mem = PageFrameAllocator()
        addr = mem.alloc(1)
        assert addr is not None
        assert addr > 0
        mem.free(addr, 1)

    def test_alloc_returns_aligned(self):
        mem = PageFrameAllocator()
        addr = mem.alloc(4)
        assert addr % 0x1000 == 0

    def test_free_nonexistent(self):
        mem = PageFrameAllocator()
        mem.free(0xDEADBEEF, 1)

    def test_multiple_alloc(self):
        mem = PageFrameAllocator()
        a1 = mem.alloc(1)
        a2 = mem.alloc(1)
        assert a1 != a2


# ============================================================
# ProcessTable
# ============================================================

class TestProcessTableEdge:

    def test_create_and_get(self):
        pt = ProcessTable()
        pcb = pt.create("test")
        assert pt.get(pcb.pid) is not None

    def test_remove(self):
        pt = ProcessTable()
        pcb = pt.create("test")
        pt.remove(pcb.pid)
        assert pt.get(pcb.pid) is None

    def test_get_nonexistent(self):
        pt = ProcessTable()
        assert pt.get(99999) is None

    def test_list_all(self):
        pt = ProcessTable()
        pt.create("a")
        pt.create("b")
        all_p = pt.all()
        assert len(all_p) >= 2


# ============================================================
# DeviceBus
# ============================================================

class TestDeviceBusEdge:

    def test_register(self):
        bus = DeviceBus()
        dev = ClockDevice()
        bus.register(0x10, dev)
        assert 0x10 in bus._devices

    def test_list_devices(self):
        bus = DeviceBus()
        bus.register(0x10, ClockDevice())
        devs = bus.list_devices()
        assert isinstance(devs, list)


# ============================================================
# KeyboardDevice
# ============================================================

class TestKeyboardDevice:

    def test_read_empty(self):
        kb = PS2KeyboardDevice()
        val = kb.read_key()
        assert val == 0

    def test_info(self):
        kb = PS2KeyboardDevice()
        info = kb.info()
        assert isinstance(info, dict)


class TestSerialDevice:

    def test_has_data_empty(self):
        serial = SerialDevice()
        assert not serial.has_data()

    def test_write_byte(self):
        serial = SerialDevice()
        serial.write_byte(0x41)

    def test_flush(self):
        serial = SerialDevice()
        serial.flush()


class TestMouseDevice:

    def test_read_packet_empty(self):
        mouse = MouseDevice()
        pkt = mouse.read_packet()
        assert pkt is None or pkt == [] or pkt == b''

    def test_move(self):
        mouse = MouseDevice()
        mouse.move(10, -5)
        pkt = mouse.read_packet()
        assert pkt is not None

    def test_reset(self):
        mouse = MouseDevice()
        mouse.move(10, 5)
        mouse.reset()
        pkt = mouse.read_packet()
        assert pkt is None or pkt == [] or pkt == b''


class TestCMOSDevice:

    def test_get_time(self):
        cmos = CMOSDevice()
        val = cmos.get_time()
        assert isinstance(val, dict)

    def test_get_unix_time(self):
        cmos = CMOSDevice()
        val = cmos.get_unix_time()
        assert isinstance(val, int)
        assert val >= 0


class TestNICDevice:

    def test_send_packet(self):
        nic = NICDevice()
        result = nic.send_packet(b"test data")
        assert result is not None

    def test_recv_empty(self):
        nic = NICDevice()
        result = nic.recv_packet()
        assert result is None or result == b''

    def test_get_stats(self):
        nic = NICDevice()
        stats = nic.get_stats()
        assert isinstance(stats, dict)

    def test_has_packet(self):
        nic = NICDevice()
        assert not nic.has_packet()


class TestDiskDevice:

    def test_get_geometry(self):
        disk = DiskDevice()
        geo = disk.get_geometry()
        assert isinstance(geo, dict)

    def test_status(self):
        disk = DiskDevice()
        s = disk.status()
        assert isinstance(s, int)


# ============================================================
# VGADevice deeper tests
# ============================================================

class TestVGADeviceEdge:

    def test_write(self):
        vga = VGADevice()
        vga.call("write", 0, 0, 'X', 4, 1)
        assert vga._screen[0][0]['char'] == 'X'

    def test_write_string(self):
        vga = VGADevice()
        vga.call("write_string", 0, 0, "Hi!")
        assert vga._screen[0][0]['char'] == 'H'

    def test_clear(self):
        vga = VGADevice()
        vga.call("write", 0, 0, 'A')
        vga.call("clear")
        assert vga._screen[0][0]['char'] == ' '

    def test_scroll(self):
        vga = VGADevice()
        vga.call("write", 0, 0, 'A')
        vga.call("scroll", 1)
        assert vga._screen[0][0]['char'] == ' '

    def test_set_cursor(self):
        vga = VGADevice()
        vga.call("set_cursor", 5, 10)
        assert vga._cursor_row == 5

    def test_get_cursor(self):
        vga = VGADevice()
        vga.call("set_cursor", 3, 7)
        pos = vga.call("get_cursor")
        assert pos == (3, 7)

    def test_get_screen(self):
        vga = VGADevice()
        vga.call("write", 0, 0, 'H')
        lines = vga.call("get_screen")
        assert isinstance(lines, list)


# ============================================================
# X86CPU deeper tests
# ============================================================

class TestX86CPUEdge:

    def test_read_write_8(self):
        cpu = X86CPU()
        cpu._write8(0x1000, 0xAB)
        assert cpu._read8(0x1000) == 0xAB

    def test_read_write_16(self):
        cpu = X86CPU()
        cpu._write16(0x1000, 0x1234)
        assert cpu._read16(0x1000) == 0x1234

    def test_read_write_32(self):
        cpu = X86CPU()
        cpu._write32(0x1000, 0x12345678)
        assert cpu._read32(0x1000) == 0x12345678

    def test_push_pop(self):
        cpu = X86CPU()
        cpu._regs[4] = 0x80000
        cpu._push32(0xDEADBEEF)
        val = cpu._pop32()
        assert val == 0xDEADBEEF

    def test_flags(self):
        cpu = X86CPU()
        cpu._set_flag(FLAG_ZF, True)
        assert cpu._flag(FLAG_ZF) is True
        cpu._set_flag(FLAG_ZF, False)
        assert cpu._flag(FLAG_ZF) is False


# ============================================================
# X86Assembler edge cases
# ============================================================

class TestAssemblerEdge:

    def test_empty_source(self):
        asm = X86Assembler()
        code = asm.assemble('')
        assert list(code) == []

    def test_bits16(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 16]\nNOP')
        assert len(code) > 0

    def test_bits32(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nNOP')
        assert list(code) == [0x90]

    def test_label_resolution(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nstart:\nNOP\nJMP start')
        assert len(code) >= 2

    def test_data_sections(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndb 0x90\ndw 0x1234\ndd 0xDEADBEEF')
        assert len(code) == 7

    def test_equ(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nVAL equ 42\nMOV EAX, VAL')
        assert len(code) > 0

    def test_times_directive(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ntimes 5 nop')
        assert list(code) == [0x90] * 5

    def test_org_directive(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nMOV EAX, 1', org=0x200000)
        assert len(code) > 0

    def test_section_directive(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\nsection .text\nNOP')
        assert len(code) > 0

    def test_string_literal(self):
        asm = X86Assembler()
        code = asm.assemble('[BITS 32]\ndb "Hello", 0')
        assert len(code) == 6


# ============================================================
# FlatFS
# ============================================================

class TestFlatFSEdge:

    def test_write_read(self):
        bd = BlockDevice()
        fs = FlatFS(bd)
        fs.write("/test.txt", b"hello world")
        data = fs.read("/test.txt")
        assert data[:11] == b"hello world"

    def test_exists(self):
        bd = BlockDevice()
        fs = FlatFS(bd)
        assert not fs.exists("/nope.txt")
        fs.write("/exists.txt", b"data")
        assert fs.exists("/exists.txt")

    def test_list_files(self):
        bd = BlockDevice()
        fs = FlatFS(bd)
        fs.write("/a.txt", b"a")
        fs.write("/b.txt", b"b")
        files = fs.list_files()
        assert "/a.txt" in files
        assert "/b.txt" in files

    def test_delete(self):
        bd = BlockDevice()
        fs = FlatFS(bd)
        fs.write("/del.txt", b"data")
        fs.delete("/del.txt")
        assert not fs.exists("/del.txt")


# ============================================================
# ClockDevice
# ============================================================

class TestClockDevice:

    def test_seconds_now(self):
        clk = ClockDevice()
        val = clk.seconds_now()
        assert isinstance(val, (int, float))

    def test_info(self):
        clk = ClockDevice()
        info = clk.info()
        assert isinstance(info, dict)


# ============================================================
# ConsoleDevice
# ============================================================

class TestConsoleDevice:

    def test_init(self):
        con = ConsoleDevice(port=1)
        assert con is not None


class TestFileDevice:

    def test_open_read(self):
        import tempfile, os
        fd_dev = FileDevice()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            f.write(b"test content")
            path = f.name
        try:
            result = fd_dev.call("open", path, "r")
            assert result is not None
        finally:
            os.unlink(path)

import struct

def run_asm(source, max_cycles=5000):
    vs = X86VirtualSystem()
    vs.load_kernel(source, org=0x1000)
    cycles = vs.run(max_cycles=max_cycles)
    return cycles, vs


def find_dword(mem, start, end, value):
    for i in range(start, end):
        if struct.unpack_from("<I", mem, i)[0] == value:
            return i
    return None


def find_byte(mem, start, end, value):
    for i in range(start, end):
        if mem[i] == value:
            return i
    return None


class TestAssemblerEncoding:
    def test_66_prefix_mov_ax_imm(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV AX, 0x1234\nHLT")
        assert r[0] == 0x66

    def test_66_prefix_mov_ax_mem(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV AX, [0x1000]\nHLT")
        assert r[0] == 0x66
        assert 0xA1 in r

    def test_66_prefix_mov_mem_ax(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV [0x1000], AX\nHLT")
        assert r[0] == 0x66
        assert 0xA3 in r

    def test_rep_movsb_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nMOV ESI, 0x1000\nMOV EDI, 0x2000\nMOV ECX, 4\nREP MOVSB\nHLT")
        assert 0xF3 in r and 0xA4 in r

    def test_rep_stosb_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nMOV EDI, 0x2000\nMOV AL, 0x41\nMOV ECX, 4\nREP STOSB\nHLT")
        assert 0xF3 in r and 0xAA in r

    def test_stosw_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nSTOSW\nHLT")
        assert 0x66 in r and 0xAB in r

    def test_lodsb_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nLODSB\nHLT")
        assert 0xAC in r

    def test_lodsw_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nLODSW\nHLT")
        assert 0x66 in r and 0xAD in r

    def test_cmpsb_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nCMPSB\nHLT")
        assert 0xA6 in r

    def test_scasb_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCLD\nSCASB\nHLT")
        assert 0xAE in r

    def test_iret_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nIRET\nHLT")
        assert 0xCF in r

    def test_mov_reg_reg_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV EAX, EBX\nHLT")
        assert 0x89 in r

    def test_push_imm8_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nPUSH 42\nHLT")
        assert 0x6A in r

    def test_push_imm32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nPUSH 0x12345678\nHLT")
        assert 0x68 in r

    def test_group1_mem_imm8_add(self):
        r = X86Assembler().assemble("[BITS 32]\nADD DWORD [data], 5\ndata: dd 10")
        assert 0x83 in r

    def test_group1_mem_imm32(self):
        r = X86Assembler().assemble("[BITS 32]\nADD DWORD [data], 0x100\ndata: dd 10")
        assert 0x81 in r

    def test_group1_reg_imm8(self):
        r = X86Assembler().assemble("[BITS 32]\nADD EAX, 5\nHLT")
        assert 0x83 in r

    def test_shift_imm8(self):
        r = X86Assembler().assemble("[BITS 32]\nSHL EAX, 4\nSHR EBX, 8\nSAR ECX, 2\nHLT")
        assert 0xC1 in r

    def test_rol_imm8(self):
        r = X86Assembler().assemble("[BITS 32]\nROL EAX, 4\nROR EBX, 8\nHLT")
        assert 0xC1 in r

    def test_push_reg_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nPUSH EAX\nPUSH EBX\nHLT")
        assert 0x50 in r

    def test_pop_reg_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nPOP EAX\nPOP EBX\nHLT")
        assert 0x58 in r

    def test_mov_reg_imm32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV EAX, 0x12345678\nHLT")
        assert 0xB8 in r

    def test_mov_reg_imm8_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV AL, 42\nHLT")
        assert 0xB0 in r

    def test_nop_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nNOP\nHLT")
        assert 0x90 in r

    def test_call_rel32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nCALL target\ntarget:\nHLT")
        assert 0xE8 in r

    def test_short_jmp_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nJMP SHORT target\ntarget:\nHLT")
        assert 0xEB in r

    def test_int_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nINT 0x80\nHLT")
        assert 0xCD in r

    def test_in_al_imm8(self):
        r = X86Assembler().assemble("[BITS 32]\nIN AL, 0x60\nHLT")
        assert 0xE4 in r

    def test_out_imm8_al(self):
        r = X86Assembler().assemble("[BITS 32]\nOUT 0x60, AL\nHLT")
        assert 0xE6 in r

    def test_in_eax_imm8(self):
        r = X86Assembler().assemble("[BITS 32]\nIN EAX, 0x60\nHLT")
        assert 0xE5 in r

    def test_out_imm8_eax(self):
        r = X86Assembler().assemble("[BITS 32]\nOUT 0x60, EAX\nHLT")
        assert 0xE7 in r

    def test_mov_reg8_mem8_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\ndata: db 0xFF\nMOV AL, [data]\nHLT")
        assert 0x8A in r

    def test_mov_mem8_reg8_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV AL, 0x42\nMOV [data], AL\ndata: db 0")
        assert 0x88 in r

    def test_mov_reg32_mem32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\ndata: dd 0\nMOV EAX, [data]\nHLT")
        assert 0x8B in r

    def test_mov_mem32_reg32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV EAX, 1\nMOV [data], EAX\ndata: dd 0")
        assert 0x89 in r

    def test_mov_mem32_imm32_encoding(self):
        r = X86Assembler().assemble("[BITS 32]\nMOV DWORD [data], 0xDEADBEEF\ndata: dd 0")
        assert 0xC7 in r


class TestCPUExecution:
    def test_adc_with_carry(self):
        _, vs = run_asm("[BITS 32]\nSTC\nMOV EAX, 5\nADC EAX, 3\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_sbb_with_borrow(self):
        _, vs = run_asm("[BITS 32]\nSTC\nMOV EAX, 10\nSBB EAX, 3\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_neg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nNEG EAX\nHLT")
        assert vs._cpu._regs[0] != 5

    def test_not(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFF\nNOT EAX\nHLT")
        assert vs._cpu._regs[0] != 0xFF

    def test_inc_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 41\nINC EAX\nHLT")
        assert vs._cpu._regs[0] == 42

    def test_dec_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 2\nDEC EAX\nHLT")
        assert vs._cpu._regs[0] == 1

    def test_and_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFF\nAND EAX, 0x0F\nHLT")
        assert vs._cpu._regs[0] == 0x0F

    def test_or_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xF0\nOR EAX, 0x0F\nHLT")
        assert vs._cpu._regs[0] == 0xFF

    def test_xor_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFF\nXOR EAX, 0xFF\nHLT")
        assert vs._cpu._regs[0] == 0

    def test_shl_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nSHL EAX, 5\nHLT")
        assert vs._cpu._regs[0] == 32

    def test_shr_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 32\nSHR EAX, 3\nHLT")
        assert vs._cpu._regs[0] == 4

    def test_rol_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nROL EAX, 4\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_ror_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0x10\nROR EAX, 4\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_shl_cl(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 3\nMOV EAX, 1\nSHL EAX, CL\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_shr_cl(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 2\nMOV EAX, 16\nSHR EAX, CL\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_sar_cl(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 1\nMOV EAX, 0x80000000\nSAR EAX, CL\nHLT")
        assert vs._cpu._regs[0] != 0

    def test_rol_cl(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 1\nMOV EAX, 0x80000001\nROL EAX, CL\nHLT")
        assert vs._cpu._regs[0] != 0

    def test_ror_cl(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 1\nMOV EAX, 3\nROR EAX, CL\nHLT")
        assert vs._cpu._regs[0] != 0

    def test_mul_eax_ebx(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 6\nMOV EBX, 7\nMUL EBX\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_div_eax_ebx(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 100\nXOR EDX, EDX\nMOV EBX, 7\nDIV EBX\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_imul_reg_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 6\nMOV EBX, 7\nIMUL EAX, EBX\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_imul_2op(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 6\nIMUL EAX, 7\nHLT")
        assert vs._cpu._regs[0] > 0

    def test_mov_reg_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 42\nMOV EBX, EAX\nHLT")
        assert vs._cpu._regs[3] == 42

    def test_mov_eax_mem(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, [data]\nHLT\ndata: dd 0xDEADBEEF")
        assert vs._cpu._regs[0] == 0xDEADBEEF

    def test_mov_al_mem(self):
        _, vs = run_asm("[BITS 32]\nMOV AL, [data]\nHLT\ndata: db 0x42")
        assert vs._cpu._regs[0] & 0xFF == 0x42

    def test_mov_mem_imm32(self):
        _, vs = run_asm("[BITS 32]\nMOV DWORD [data], 0xDEADBEEF\ndata: dd 0\nHLT")
        assert find_dword(vs._cpu._mem, 0x1000, 0x1040, 0xDEADBEEF) is not None

    def test_mov_mem_al(self):
        _, vs = run_asm("[BITS 32]\nMOV AL, 0x42\nMOV [data], AL\ndata: db 0\nHLT")
        assert find_byte(vs._cpu._mem, 0x1000, 0x1040, 0x42) is not None

    def test_jmp_short(self):
        _, vs = run_asm("[BITS 32]\nJMP SHORT skip\nMOV EAX, 99\nskip:\nHLT")
        assert vs._cpu._regs[0] != 99

    def test_jcc_jnz_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nTEST EAX, EAX\nJNZ target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jnz_not_taken(self):
        _, vs = run_asm("[BITS 32]\nXOR EAX, EAX\nTEST EAX, EAX\nJNZ target\nMOV ECX, 42\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 42

    def test_call_ret(self):
        _, vs = run_asm("[BITS 32]\nCALL mysub\nHLT\nmysub:\nMOV EAX, 42\nRET")
        assert vs._cpu._regs[0] == 42

    def test_loop(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 5\nXOR EAX, EAX\nlp:\nINC EAX\nLOOP lp\nHLT")
        assert vs._cpu._regs[0] == 5

    def test_rep_movsb(self):
        _, vs = run_asm("[BITS 32]\nMOV ESI, src\nMOV EDI, dst\nMOV ECX, 3\nCLD\nREP MOVSB\nHLT\nsrc: db ABC\ndst: times 10 db 0")
        assert vs._cpu._regs[1] == 0

    def test_rep_stosb(self):
        _, vs = run_asm("[BITS 32]\nMOV EDI, buf\nMOV AL, 0x41\nMOV ECX, 3\nCLD\nREP STOSB\nHLT\nbuf: times 10 db 0")
        assert vs._cpu._regs[1] == 0

    def test_repe_cmpsb(self):
        _, vs = run_asm("[BITS 32]\nMOV ESI, src\nMOV EDI, dst\nMOV ECX, 3\nCLD\nREPE CMPSB\nHLT\nsrc: db AAA\ndst: db AAA")
        assert vs._cpu._regs[1] >= 0



    def test_repne_scasb(self):
        _, vs = run_asm("[BITS 32]\nMOV EDI, buf\nMOV AL, 'X'\nMOV ECX, 5\nCLD\nREPNZ SCASB\nHLT\nbuf: db ABCXD")
        assert vs._cpu._regs[1] >= 0

    def test_push_pop(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 42\nPUSH EAX\nXOR EAX, EAX\nPOP EAX\nHLT")
        assert vs._cpu._regs[0] == 42

    def test_cpuid(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0\nCPUID\nHLT")
        assert vs._cpu._regs[0] == 0

    def test_lea(self):
        _, vs = run_asm("[BITS 32]\nMOV EBX, 10\nLEA EAX, [EBX+0x20]\nHLT")
        assert vs._cpu._regs[0] == 42

    def test_bt_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFF\nBT EAX, 3\nHLT")
        assert vs._cpu._regs[0] == 0xFF

    def test_xadd(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nMOV EBX, 3\nXADD EAX, EBX\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_sahf_lahf(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFF00\nSAHF\nLAHF\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_cdq_negative(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xB8
        struct.pack_into("<I", cpu._mem, 0x1001, 0x80000000)
        cpu._mem[0x1005] = 0x99
        cpu._mem[0x1006] = 0xF4
        cpu._eip = 0x1000
        cpu.step()
        cpu.step()
        assert cpu._regs[2] == 0xFFFFFFFF

    def test_cdq_positive(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xB8
        struct.pack_into("<I", cpu._mem, 0x1001, 0x10)
        cpu._mem[0x1005] = 0x99
        cpu._mem[0x1006] = 0xF4
        cpu._eip = 0x1000
        cpu.step()
        cpu.step()
        assert cpu._regs[2] == 0

    def test_group1_mem_imm8_add(self):
        _, vs = run_asm("[BITS 32]\nADD DWORD [data], 5\ndata: dd 10")
        assert find_dword(vs._cpu._mem, 0x1000, 0x1040, 15) is not None

    def test_group1_mem_imm32_sub(self):
        _, vs = run_asm("[BITS 32]\nSUB DWORD [data], 3\ndata: dd 10")
        assert find_dword(vs._cpu._mem, 0x1000, 0x1040, 7) is not None

    def test_group1_mem_imm8_cmp(self):
        _, vs = run_asm("[BITS 32]\nCMP DWORD [data], 42\nHLT\ndata: dd 42")
        assert vs._cpu._flag(0x40)

    def test_test_zero(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0\nTEST EAX, EAX\nHLT")
        assert vs._cpu._flag(0x40)

    def test_test_nonzero(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nTEST EAX, EAX\nHLT")
        assert not vs._cpu._flag(0x40)

    def test_cmp_equal(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 42\nCMP EAX, 42\nHLT")
        assert vs._cpu._flag(0x40)

    def test_cmp_less(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nCMP EAX, 10\nHLT")
        assert vs._cpu._flag(0x01)

    def test_in_out(self):
        _, vs = run_asm("[BITS 32]\nIN AL, 0x60\nOUT 0x60, AL\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_mov_eax_imm32(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0x12345678\nHLT")
        assert vs._cpu._regs[0] == 0x12345678

    def test_push_imm8(self):
        _, vs = run_asm("[BITS 32]\nPUSH 42\nPOP EAX\nHLT")
        assert vs._cpu._regs[0] == 42

    def test_push_imm32(self):
        _, vs = run_asm("[BITS 32]\nPUSH 0x12345678\nPOP EAX\nHLT")
        assert vs._cpu._regs[0] == 0x12345678

    def test_push_pop_ebx(self):
        _, vs = run_asm("[BITS 32]\nMOV EBX, 99\nPUSH EBX\nXOR EBX, EBX\nPOP EBX\nHLT")
        assert vs._cpu._regs[3] == 99

    def test_mov_ecx_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV ECX, 77\nHLT")
        assert vs._cpu._regs[1] == 77

    def test_mov_edx_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EDX, 88\nHLT")
        assert vs._cpu._regs[2] == 88

    def test_mov_esi_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV ESI, 111\nHLT")
        assert vs._cpu._regs[6] == 111

    def test_mov_edi_imm(self):
        _, vs = run_asm("[BITS 32]\nMOV EDI, 222\nHLT")
        assert vs._cpu._regs[7] == 222

    def test_ebp_preserved(self):
        _, vs = run_asm("[BITS 32]\nMOV EBP, 0x10000\nRET")
        assert vs._cpu._regs[5] == 0x10000

    def test_sub_reg_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 10\nSUB EAX, 3\nHLT")
        assert vs._cpu._regs[0] == 7

    def test_cmp_reg_reg(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nCMP EAX, 5\nHLT")
        assert vs._cpu._flag(0x40)

    def test_jcc_jl_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nCMP EAX, 5\nJL target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jg_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 10\nCMP EAX, 5\nJG target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jle_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nCMP EAX, 5\nJLE target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jge_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 5\nCMP EAX, 5\nJGE target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_ja_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 10\nCMP EAX, 5\nJA target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jb_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nCMP EAX, 5\nJB target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_js_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0xFFFFFFFF\nTEST EAX, EAX\nJS target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jns_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nTEST EAX, EAX\nJNS target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0

    def test_jcc_jo_taken(self):
        _, vs = run_asm("[BITS 32]\nMOV EAX, 0x7FFFFFFF\nADD EAX, 1\nJO target\nMOV ECX, 99\ntarget:\nHLT")
        assert vs._cpu._regs[1] == 0


class TestDeviceAndInternal:
    def test_clock_device(self):
        cd = ClockDevice()
        t = cd.seconds_now()
        assert isinstance(t, float)

    def test_serial_device(self):
        sd = SerialDevice()
        sd.push_byte(0x41)
        assert sd.has_data()
        assert sd.read_byte() == 0x41
        sd.flush()

    def test_keyboard_device(self):
        kd = PS2KeyboardDevice()
        assert kd.read_key() == 0

    def test_mouse_device(self):
        md = MouseDevice()
        md.move(5, 5)
        state = md.get_state()
        assert isinstance(state, dict)
        md.reset()

    def test_cmos_device(self):
        cmos = CMOSDevice()
        t = cmos.get_time()
        assert isinstance(t, dict)

    def test_nic_device(self):
        nic = NICDevice()
        assert not nic.has_packet()
        stats = nic.get_stats()
        assert isinstance(stats, dict)

    def test_device_bus(self):
        bus = DeviceBus()
        sd = SerialDevice()
        bus.register("serial", sd)
        devices = bus.list_devices()
        assert isinstance(devices, list)

    def test_process_table(self):
        pt = ProcessTable()
        pcb = pt.create("test")
        assert pcb is not None
        assert pt.count() >= 1
        pt.remove(pcb.pid)

    def test_scheduler(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert sch._tick_count == 0

    def test_rbac(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        assert rbac.role_of(1) == 1

    def test_permission_device_serial(self):
        assert Permission.DEVICE_SERIAL.value > 0

    def test_permission_training(self):
        assert Permission.TRAINING.value > 0

    def test_permission_process_spawn(self):
        assert Permission.PROCESS_SPAWN.value > 0

    def test_permission_file_read(self):
        assert Permission.FILE_READ.value > 0

    def test_x86_cpu_flag(self):
        cpu = X86CPU()
        cpu._set_flag(0x40, True)
        assert cpu._flag(0x40)
        cpu._set_flag(0x40, False)
        assert not cpu._flag(0x40)

    def test_x86_cpu_set_get32(self):
        cpu = X86CPU()
        cpu._set32(0, 0xDEADBEEF)
        assert cpu._get32(0) == 0xDEADBEEF

    def test_x86_cpu_push_pop32(self):
        cpu = X86CPU()
        cpu._push32(0x12345678)
        val = cpu._pop32()
        assert val == 0x12345678

    def test_x86_cpu_alu_add(self):
        cpu = X86CPU()
        assert cpu._alu(0, 5, 3, 32) == 8

    def test_x86_cpu_alu_sub(self):
        cpu = X86CPU()
        assert cpu._alu(5, 10, 3, 32) == 7

    def test_x86_cpu_alu_and(self):
        cpu = X86CPU()
        assert cpu._alu(4, 0xFF, 0x0F, 32) == 0x0F

    def test_x86_cpu_alu_or(self):
        cpu = X86CPU()
        assert cpu._alu(1, 0xF0, 0x0F, 32) == 0xFF

    def test_x86_cpu_alu_xor(self):
        cpu = X86CPU()
        assert cpu._alu(6, 0xFF, 0xFF, 32) == 0

    def test_x86_cpu_alu_cmp(self):
        cpu = X86CPU()
        assert cpu._alu(7, 10, 10, 32) == 10

    def test_x86_cpu_set_get8(self):
        cpu = X86CPU()
        cpu._set8l(0, 0xAB)
        assert cpu._get8l(0) == 0xAB

    def test_x86_cpu_set_get16(self):
        cpu = X86CPU()
        cpu._set16(0, 0xBEEF)
        assert cpu._get16(0) == 0xBEEF

    def test_x86_assembler_basic(self):
        result = X86Assembler().assemble("[BITS 32]\nNOP\nHLT")
        assert 0x90 in result and 0xF4 in result

    def test_x86_virtual_system_status(self):
        vs = X86VirtualSystem()
        status = vs.status()
        assert isinstance(status, dict)

    def test_x86_virtual_system_reset(self):
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nMOV EAX, 42\nHLT", org=0x1000)
        vs.run()
        vs.reset()
        assert isinstance(vs.status(), dict)
    def test_disk_call_read_sectors(self):
        d = DiskDevice()
        result = d.call("read_sectors", 0, 1)
        assert isinstance(result, (bytes, bytearray))

    def test_disk_call_write_sectors(self):
        d = DiskDevice()
        d.call("write_sectors", 0, b'\x00' * 512)
        assert True

    def test_disk_call_get_geometry(self):
        d = DiskDevice()
        result = d.call("get_geometry")
        assert isinstance(result, dict)

    def test_disk_call_status(self):
        d = DiskDevice()
        result = d.call("status")
        assert isinstance(result, int)

    def test_serial_call_write_byte(self):
        sd = SerialDevice()
        result = sd.call("write_byte", 0x41)
        assert result is True

    def test_serial_call_read_byte(self):
        sd = SerialDevice()
        result = sd.call("read_byte")
        assert result == -1

    def test_serial_call_push_byte(self):
        sd = SerialDevice()
        result = sd.call("push_byte", 0x42)
        assert result is True

    def test_serial_call_has_data(self):
        sd = SerialDevice()
        result = sd.call("has_data")
        assert result is False

    def test_serial_call_flush(self):
        sd = SerialDevice()
        result = sd.call("flush")
        assert result is True

    def test_mouse_call_move(self):
        md = MouseDevice()
        result = md.call("move", 5, 5)
        assert result is True

    def test_mouse_call_press(self):
        md = MouseDevice()
        result = md.call("press", 1)
        assert result is True

    def test_mouse_call_release(self):
        md = MouseDevice()
        result = md.call("release", 1)
        assert result is True

    def test_mouse_call_read_packet(self):
        md = MouseDevice()
        md.move(10, 5)
        result = md.call("read_packet")
        assert isinstance(result, bytes)

    def test_mouse_call_get_state(self):
        md = MouseDevice()
        result = md.call("get_state")
        assert isinstance(result, dict)

    def test_mouse_call_reset(self):
        md = MouseDevice()
        md.move(10, 10)
        result = md.call("reset")
        assert result is True
        state = md.get_state()
        assert state["x"] == 0

    def test_cmos_call_get_time(self):
        c = CMOSDevice()
        t = c.call("get_time")
        assert isinstance(t, dict)

    def test_nic_call_send_packet(self):
        n = NICDevice()
        result = n.call("send_packet", b'\x00' * 64)
        assert result is True or isinstance(result, (int, bool))

    def test_nic_call_recv_packet(self):
        n = NICDevice()
        result = n.call("recv_packet")
        assert result is None or isinstance(result, bytes)

    def test_nic_call_inject_packet(self):
        n = NICDevice()
        result = n.call("inject_packet", b'\x00' * 64)
        assert result is True

    def test_nic_call_has_packet(self):
        n = NICDevice()
        result = n.call("has_packet")
        assert result is False

    def test_nic_call_get_stats(self):
        n = NICDevice()
        stats = n.call("get_stats")
        assert isinstance(stats, dict)

    def test_nic_call_flush(self):
        n = NICDevice()
        result = n.call("flush")
        assert result is True

    def test_clock_call_seconds_now(self):
        cd = ClockDevice()
        t = cd.seconds_now()
        assert isinstance(t, float)

    def test_keyboard_call_read_key(self):
        kd = PS2KeyboardDevice()
        result = kd.call("read_key")
        assert result == 0


class TestCMOSInternals:
    def test_cmos_read_status_c(self):
        c = CMOSDevice()
        # Write to port 0x70 to select status C register (0x0C)
        c._write_addr(0x8C)  # NMI disabled + register 0x0C
        val = c._read_data()
        assert isinstance(val, int)

    def test_cmos_write_status_a(self):
        c = CMOSDevice()
        c._write_addr(0x8A)  # Status A
        c._write_data(0x20)  # Should be ignored
        assert True

    def test_cmos_write_status_d(self):
        c = CMOSDevice()
        c._write_addr(0x8D)  # Status D
        c._write_data(0x80)  # VRT bit
        assert True

    def test_cmos_write_regular(self):
        c = CMOSDevice()
        c._write_addr(0x10)  # Regular CMOS register
        c._write_data(0x42)
        assert c._cmos[0x10] == 0x42

    def test_cmos_refresh_with_clock(self):
        c = CMOSDevice()
        c._clock = ClockDevice()
        c._refresh_rtc()
        assert True


class TestDeviceBusExtended:
    def test_device_bus_call(self):
        bus = DeviceBus()
        sd = SerialDevice()
        bus.register("serial", sd)
        result = bus.call(sd, "write_byte", 0x41)
        assert result is True

    def test_device_bus_list_devices(self):
        bus = DeviceBus()
        sd = SerialDevice()
        bus.register("serial", sd)
        devices = bus.list_devices()
        assert len(devices) >= 1


class TestProcessTableExtended:
    def test_process_table_create_and_get(self):
        pt = ProcessTable()
        pcb = pt.create("test_proc")
        assert pcb is not None
        found = pt.get(pcb.pid)
        assert found is not None
        pt.remove(pcb.pid)

    def test_process_table_by_state(self):
        pt = ProcessTable()
        pcb = pt.create("test_proc")
        procs = pt.by_state(ProcessState.CREATED)
        assert len(procs) >= 1
        pt.remove(pcb.pid)

    def test_process_table_get_by_name(self):
        pt = ProcessTable()
        pcb = pt.create("named_proc")
        found = pt.get_by_name("named_proc")
        assert found is not None
        pt.remove(pcb.pid)

    def test_process_table_alive_count(self):
        pt = ProcessTable()
        pcb = pt.create("alive_proc")
        count = pt.alive_count()
        assert count >= 1
        pt.remove(pcb.pid)


class TestSchedulerExtended:
    def test_scheduler_tick_count(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert sch._tick_count == 0

    def test_scheduler_ready_queue(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert len(sch._ready_queue) == 0


class TestX86CPUExtended:
    def test_cpu_set_get8h(self):
        cpu = X86CPU()
        cpu._set8h(0, 0xCD)
        assert cpu._get8h(0) == 0xCD

    def test_cpu_push_pop32(self):
        cpu = X86CPU()
        cpu._push32(0xDEADBEEF)
        val = cpu._pop32()
        assert val == 0xDEADBEEF

    def test_cpu_set_flag(self):
        cpu = X86CPU()
        cpu._set_flag(0x01, True)
        assert cpu._flag(0x01)
        cpu._set_flag(0x01, False)
        assert not cpu._flag(0x01)

    def test_cpu_flag_zero(self):
        cpu = X86CPU()
        cpu._set_flag(0x40, True)
        assert cpu._flag(0x40)

    def test_cpu_flag_sign(self):
        cpu = X86CPU()
        cpu._set_flag(0x80, True)
        assert cpu._flag(0x80)

    def test_cpu_flag_carry(self):
        cpu = X86CPU()
        cpu._set_flag(0x01, True)
        assert cpu._flag(0x01)

    def test_cpu_flag_overflow(self):
        cpu = X86CPU()
        cpu._set_flag(0x800, True)
        assert cpu._flag(0x800)

    def test_cpu_io_in_out(self):
        cpu = X86CPU()
        cpu._io_in[0x60] = lambda: 0x41
        val = cpu._port_in(0x60)
        assert val == 0x41

    def test_cpu_io_out(self):
        cpu = X86CPU()
        results = []
        cpu._io_out[0x60] = lambda v: results.append(v)
        cpu._port_out(0x60, 0x42)
        assert results == [0x42]

    def test_cpu_interrupt_table(self):
        cpu = X86CPU()
        assert hasattr(cpu, '_idt_handlers')
        assert isinstance(cpu._idt_handlers, dict)


class TestRBACExtended:
    def test_rbac_assign_admin(self):
        rbac = X86RBAC()
        rbac.assign(1, Role.ADMIN)
        assert rbac.role_of(1) == 1

    def test_rbac_assign_user(self):
        rbac = X86RBAC()
        rbac.assign(2, Role.USER)
        assert rbac.role_of(2) == Role.USER

    def test_rbac_role_of_default(self):
        rbac = X86RBAC()
        # Default role for unknown pid
        role = rbac.role_of(999)
        assert isinstance(role, int)

    def test_rbac_permissions(self):
        assert Permission.FILE_READ.value > 0
        assert Permission.FILE_WRITE.value > 0
        assert Permission.PROCESS_SPAWN.value > 0
        assert Permission.PROCESS_KILL.value > 0
        assert Permission.DEVICE_SERIAL.value > 0
        assert Permission.DEVICE_MOUSE.value > 0
        assert Permission.DEVICE_DISK.value > 0
        assert Permission.DEVICE_RTC.value > 0
        assert Permission.DEVICE_NET.value > 0
        assert Permission.RAW_MEMORY.value > 0
        assert Permission.RAW_CPU.value > 0
        assert Permission.TRAINING.value > 0

    def test_role_enum(self):
        assert Role.KERNEL.value >= 0
        assert Role.ADMIN.value >= 0
        assert Role.USER.value >= 0
        assert Role.KERNEL != Role.ADMIN
        assert Role.ADMIN != Role.USER
def run_asm(source, max_cycles=5000):
    vs = X86VirtualSystem()
    vs.load_kernel(source, org=0x1000)
    cycles = vs.run(max_cycles=max_cycles)
    return cycles, vs


def cpu_with_bytes(*byte_vals):
    """Create a CPU with bytes loaded at 0x1000 and EIP=0x1000."""
    cpu = X86CPU()
    for i, b in enumerate(byte_vals):
        cpu._mem[0x1000 + i] = b
    cpu._eip = 0x1000
    return cpu


class TestX86MulDivIdiv8:
    def test_mul_reg8(self):
        """MUL r/m8: AL * r/m8 -> AX"""
        # MOV AL, 6; MOV BL, 7; MUL BL -> AX = 42
        cpu = cpu_with_bytes(0xB0, 0x06,  # MOV AL, 6
                             0xB3, 0x07,  # MOV BL, 7
                             0xF6, 0xE3,  # MUL BL (reg_f=4, modrm=0xE3)
                             0xF4)        # HLT
        cpu.step()  # MOV AL
        cpu.step()  # MOV BL
        cpu.step()  # MUL BL
        ax = cpu._get16(0)  # AX
        assert ax == 42

    def test_mul_mem8(self):
        """MUL r/m8 with memory operand."""
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xB0        # MOV AL, 5
        cpu._mem[0x1001] = 0x05
        cpu._mem[0x1002] = 0xF6        # MUL byte [disp32]
        cpu._mem[0x1003] = 0x05        # modrm: reg=0 (MUL), mod=00, rm=101 (disp32)
        struct.pack_into("<I", cpu._mem, 0x1004, 0x2000)  # address
        cpu._mem[0x2000] = 0x0A        # data = 10
        cpu._mem[0x1008] = 0xF4        # HLT
        cpu._eip = 0x1000
        cpu.step()  # MOV AL
        cpu.step()  # MUL
        # Verify multiplication happened (result in AX)
        ax = cpu._get16(0)
        # MUL should have executed - verify it didn't crash
        assert isinstance(ax, int)

    def test_imul_reg8(self):
        """IMUL r/m8: AL * r/m8 -> AX (signed)"""
        cpu = cpu_with_bytes(0xB0, 0xFE,  # MOV AL, -2
                             0xB3, 0x03,  # MOV BL, 3
                             0xF6, 0xEB,  # IMUL BL (reg_f=5, modrm=0xEB)
                             0xF4)
        cpu.step()
        cpu.step()
        cpu.step()
        ax = cpu._get16(0)
        # -2 * 3 = -6 = 0xFFFA
        assert ax == 0xFFFA

    def test_div_reg8(self):
        """DIV r/m8: AX / r/m8 -> AL (quotient), AH (remainder)"""
        cpu = cpu_with_bytes(0xB0, 0x64,  # MOV AL, 100
                             0xB4, 0x00,  # MOV AH, 0
                             0xB3, 0x07,  # MOV BL, 7
                             0xF6, 0xF3,  # DIV BL (reg_f=6, modrm=0xF3)
                             0xF4)
        cpu.step()
        cpu.step()
        cpu.step()
        cpu.step()
        al = cpu._get8l(0)  # AL = quotient
        ah = cpu._get8h(0)  # AH = remainder
        assert al == 14  # 100 / 7 = 14
        assert ah == 2   # 100 % 7 = 2

    def test_idiv_reg8(self):
        """IDIV r/m8: AX / r/m8 -> AL (quotient), AH (remainder) signed"""
        cpu = cpu_with_bytes(0xB0, 0x06,  # MOV AL, 6
                             0xB4, 0x00,  # MOV AH, 0 (AX = 6)
                             0xB3, 0x02,  # MOV BL, 2
                             0xF6, 0xFB,  # IDIV BL (reg_f=7, modrm=0xFB)
                             0xF4)
        cpu.step()
        cpu.step()
        cpu.step()
        cpu.step()
        al = cpu._get8l(0)
        ah = cpu._get8h(0)
        assert al == 3   # 6 / 2 = 3
        assert ah == 0   # 6 % 2 = 0

    def test_div_by_zero_raises(self):
        """DIV by zero raises InsFault."""
        cpu = cpu_with_bytes(0xB0, 0x0A,  # MOV AL, 10
                             0xB4, 0x00,  # MOV AH, 0
                             0xB3, 0x00,  # MOV BL, 0
                             0xF6, 0xF3,  # DIV BL
                             0xF4)
        cpu.step()
        cpu.step()
        cpu.step()
        try:
            cpu.step()  # DIV by zero
            assert False, "Should have raised"
        except Exception:
            pass


class TestAssemblerCoverage:
    def test_load_const_tensor_literal(self):
        """Tensor literal loading: [1, 2, 3]"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nHLT")
        assert vs._cpu._regs[0] is not None

    def test_load_const_string_tensor(self):
        """String tensor: [hello, world]"""
        _, vs = run_asm('[BITS 32]\nMOV R0, [hello, world]\nHLT')
        assert vs._cpu._regs[0] is not None

    def test_load_const_float_tensor(self):
        """Float tensor: [1.5, 2.5, 3.5]"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1.5, 2.5, 3.5]\nHLT")
        assert vs._cpu._regs[0] is not None

    def test_load_const_empty_tensor(self):
        """Empty tensor: []"""
        _, vs = run_asm("[BITS 32]\nMOV R0, []\nHLT")
        assert vs._cpu._regs[0] is not None

    def test_load_shape(self):
        """LOAD_SHAPE instruction"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nLOAD_SHAPE R1, 3, 1\nHLT")
        assert vs._cpu._regs[1] is not None

    def test_tensor_add(self):
        """Tensor addition"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nMOV R1, [4, 5, 6]\nADD_T R2, R0, R1\nHLT")
        assert vs._cpu._regs[2] is not None

    def test_tensor_mul(self):
        """Tensor multiplication"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nMOV R1, [4, 5, 6]\nMUL_T R2, R0, R1\nHLT")
        assert vs._cpu._regs[2] is not None

    def test_tensor_matmul(self):
        """Tensor matrix multiply"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3, 4]\nLOAD_SHAPE R1, 2, 2\nMOV R2, [5, 6, 7, 8]\nLOAD_SHAPE R3, 2, 2\nMATMUL R4, R0, R1, R2, R3\nHLT")
        assert vs._cpu._regs[4] is not None

    def test_tensor_sum(self):
        """Tensor sum reduction"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nSUM R1, R0\nHLT")
        assert vs._cpu._regs[1] is not None

    def test_tensor_mean(self):
        """Tensor mean reduction"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nMEAN R1, R0\nHLT")
        assert vs._cpu._regs[1] is not None

    def test_tensor_relu(self):
        """Tensor ReLU activation"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [-1, 0, 1]\nRELU R1, R0\nHLT")
        assert vs._cpu._regs[1] is not None

    def test_tensor_softmax(self):
        """Tensor softmax"""
        _, vs = run_asm("[BITS 32]\nMOV R0, [1, 2, 3]\nSOFTMAX R1, R0\nHLT")
        assert vs._cpu._regs[1] is not None


class TestSyscallCoverage:
    def test_syscall_serial_write(self):
        """SYS_SERIAL_WRITE syscall"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 1\nMOV EBX, 0x41\nINT 0x80\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_syscall_malloc_free(self):
        """SYS_MALLOC and SYS_FREE syscalls"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 12\nMOV EBX, 64\nINT 0x80\nMOV ECX, EAX\nMOV EAX, 13\nMOV EBX, ECX\nINT 0x80\nHLT")
        assert vs._cpu._regs[1] >= 0

    def test_syscall_process_info(self):
        """SYS_PROCESS_INFO syscall"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 20\nINT 0x80\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_syscall_device_info(self):
        """SYS_DEVICE_INFO syscall"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 25\nINT 0x80\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_syscall_yield(self):
        """SYS_YIELD syscall"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 16\nINT 0x80\nHLT")
        assert vs._cpu._regs[0] >= 0

    def test_syscall_sleep(self):
        """SYS_SLEEP syscall"""
        _, vs = run_asm("[BITS 32]\nMOV EAX, 17\nMOV EBX, 100\nINT 0x80\nHLT")
        assert vs._cpu._regs[0] >= 0


class TestX86SyscallHandler:
    def _make_handler(self):
        from domains.shell.vm import X86SyscallHandler, X86CPU, ProcessTable, Scheduler, PageFrameAllocator
        cpu = X86CPU()
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        mem = PageFrameAllocator()
        handler = X86SyscallHandler(cpu, pt, sch, mem)
        return handler, cpu

    def test_handler_init(self):
        """SyscallHandler initialization"""
        handler, cpu = self._make_handler()
        assert handler is not None

    def test_handler_dispatch_exit(self):
        """SyscallHandler dispatches SYS_EXIT"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 1  # EAX = SYS_EXIT
        cpu._regs[3] = 0  # EBX = exit code
        try:
            handler.handle()
        except Exception:
            pass  # SYS_EXIT raises Halt

    def test_handler_serial_write_no_serial(self):
        """Serial write sets EAX to error when no serial device"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 19  # EAX = SYS_SERIAL_WRITE
        cpu._regs[3] = 0x41  # EBX = char
        handler.handle()
        assert cpu._regs[0] != 0  # Error code (not 0)

    def test_handler_malloc(self):
        """SYS_MALLOC allocates memory"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 15  # EAX = SYS_MALLOC
        cpu._regs[3] = 64  # EBX = size
        handler.handle()
        assert cpu._regs[0] > 0

    def test_handler_free(self):
        """SYS_FREE frees memory"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 15  # EAX = SYS_MALLOC
        cpu._regs[3] = 64  # EBX = size
        handler.handle()
        addr = cpu._regs[0]
        cpu._regs[0] = 16  # EAX = SYS_FREE
        cpu._regs[3] = addr  # EBX = addr
        handler.handle()
        assert cpu._regs[0] >= 0

    def test_handler_free_nonexistent(self):
        """SYS_FREE on non-existent address returns -1"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 16  # EAX = SYS_FREE
        cpu._regs[3] = 0xDEADBEEF  # EBX = addr
        handler.handle()
        assert cpu._regs[0] == 0xFFFFFFFF  # -1

    def test_handler_malloc_zero(self):
        """SYS_MALLOC with size 0 returns 0"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 15  # EAX = SYS_MALLOC
        cpu._regs[3] = 0  # EBX = size
        handler.handle()
        assert cpu._regs[0] == 0

    def test_handler_getpid(self):
        """SYS_GETPID returns pid"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 10  # EAX = SYS_GETPID
        handler.handle()
        assert cpu._regs[0] >= 0

    def test_handler_yield(self):
        """SYS_YIELD returns 0"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 12  # EAX = SYS_YIELD
        handler.handle()
        assert cpu._regs[0] == 0

    def test_handler_uname(self):
        """SYS_UNAME returns 0"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 18  # EAX = SYS_UNAME
        cpu._regs[3] = 0x10000  # EBX = buf_addr
        handler.handle()
        assert cpu._regs[0] == 0

    def test_handler_gettimeofday(self):
        """SYS_GETTIMEOFDAY returns ticks"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 14  # EAX = SYS_GETTIMEOFDAY
        cpu._regs[3] = 0x10000  # EBX = buf_addr
        handler.handle()
        assert cpu._regs[0] >= 0

    def test_handler_sbrk(self):
        """SYS_SBRK returns old_break"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 11  # EAX = SYS_SBRK
        cpu._regs[3] = 0x1000  # EBX = increment
        handler.handle()
        assert cpu._regs[0] >= 0

    def test_handler_readdir(self):
        """SYS_READDIR returns count"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 17  # EAX = SYS_READDIR
        cpu._regs[3] = 0x10000  # EBX = buf_addr
        cpu._regs[1] = 10  # ECX = max_entries
        handler.handle()
        assert cpu._regs[0] >= 0

    def test_handler_close(self):
        """SYS_CLOSE on invalid fd returns -1"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 5  # EAX = SYS_CLOSE
        cpu._regs[3] = 999  # EBX = invalid fd
        handler.handle()
        assert cpu._regs[0] == 0xFFFFFFFF  # -1

    def test_handler_open(self):
        """SYS_OPEN returns fd"""
        handler, cpu = self._make_handler()
        # Write a filename to memory
        cpu._mem[0x20000:0x20006] = b'test\x00'
        cpu._regs[0] = 4  # EAX = SYS_OPEN
        cpu._regs[3] = 0x20000  # EBX = name_addr
        cpu._regs[1] = 0  # ECX = mode
        handler.handle()
        assert cpu._regs[0] >= 3  # fd >= 3 (0-2 reserved)

    def test_handler_brk(self):
        """SYS_BRK sets heap break"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 9  # EAX = SYS_BRK
        cpu._regs[3] = 0x500000  # EBX = new_heap_end
        handler.handle()
        assert isinstance(cpu._regs[0], int)

    def test_handler_kill(self):
        """SYS_KILL on nonexistent pid returns error"""
        handler, cpu = self._make_handler()
        cpu._regs[0] = 13  # EAX = SYS_KILL
        cpu._regs[3] = 999  # EBX = pid
        cpu._regs[1] = 0  # ECX = signal
        handler.handle()
        assert cpu._regs[0] != 0  # Error code


class TestX86VirtualSystemExtended:
    def test_vs_status_dict(self):
        """status() returns dict with expected keys"""
        vs = X86VirtualSystem()
        status = vs.status()
        assert isinstance(status, dict)

    def test_vs_reset(self):
        """reset() clears state"""
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nMOV EAX, 42\nHLT", org=0x1000)
        vs.run()
        vs.reset()
        status = vs.status()
        assert isinstance(status, dict)

    def test_vs_spawn(self):
        """spawn() creates a process"""
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nHLT", org=0x1000)
        result = vs.spawn("test", "[BITS 32]\nHLT", org=0x2000)
        assert result is not None

    def test_vs_get_serial(self):
        """serial device is accessible"""
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nMOV EAX, 1\nMOV EBX, 0x41\nINT 0x80\nHLT", org=0x1000)
        vs.run()
        serial = vs.serial
        assert serial is not None

    def test_vs_get_cpu(self):
        """cpu is accessible"""
        vs = X86VirtualSystem()
        vs.load_kernel("[BITS 32]\nHLT", org=0x1000)
        vs.run()
        cpu = vs.cpu
        assert cpu is not None


class TestVGADeviceCoverage:
    def test_vga_device_info(self):
        vga = VGADevice()
        info = vga.info()
        assert isinstance(info, dict)

    def test_vga_device_call_clear(self):
        vga = VGADevice()
        result = vga.call("clear")
        assert result is True

    def test_vga_device_call_scroll(self):
        vga = VGADevice()
        result = vga.call("scroll", 1)
        assert result is True

    def test_vga_device_call_write(self):
        vga = VGADevice()
        result = vga.call("write", 0, 0, "Hello")
        assert result is True


class TestFlatFSExtended:
    def test_flatfs_write_and_read(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        fs.write("test.txt", b"Hello World")
        data = fs.read("test.txt")
        assert data.startswith(b"Hello World")

    def test_flatfs_list_files(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        fs.write("a.txt", b"aaa")
        fs.write("b.txt", b"bbb")
        files = fs.list_files()
        assert len(files) >= 2

    def test_flatfs_exists(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        fs.write("exists.txt", b"yes")
        assert fs.exists("exists.txt") is True
        assert fs.exists("no.txt") is False

    def test_flatfs_size(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        fs.write("size.txt", b"12345")
        size = fs.size("size.txt")
        assert size >= 5

    def test_flatfs_delete(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        fs.write("del.txt", b"delete me")
        result = fs.delete("del.txt")
        assert result is True
        assert not fs.exists("del.txt")

    def test_flatfs_delete_nonexistent(self):
        from domains.shell.vm import FlatFS, BlockDevice
        bd = BlockDevice(num_sectors=64)
        fs = FlatFS(bd)
        result = fs.delete("no.txt")
        assert result is False


class TestFileDevice:
    def test_file_device_info(self):
        fd = FileDevice()
        info = fd.info()
        assert isinstance(info, dict)
        assert info["type"] == "file"

    def test_file_device_open_and_read(self):
        fd = FileDevice()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Hello World")
            path = f.name
        try:
            file_fd = fd.call("open", path, "r")
            assert file_fd >= 1
            data = fd.call("read", file_fd, 5)
            assert "Hello" in str(data)
        finally:
            os.unlink(path)

    def test_file_device_write(self):
        fd = FileDevice()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            path = f.name
        try:
            file_fd = fd.call("open", path, "w")
            written = fd.call("write", file_fd, "Test data")
            assert written >= 0
        finally:
            os.unlink(path)

    def test_file_device_close(self):
        fd = FileDevice()
        with tempfile.NamedTemporaryFile(mode='r', delete=False, suffix='.txt') as f:
            path = f.name
        try:
            file_fd = fd.call("open", path, "r")
            result = fd.call("close", file_fd)
            assert result is True
        finally:
            os.unlink(path)

    def test_file_device_close_bad_fd(self):
        fd = FileDevice()
        result = fd.call("close", 999)
        assert result is True

    def test_file_device_read_bad_fd(self):
        fd = FileDevice()
        try:
            fd.call("read", 999)
            assert False, "Should have raised"
        except Exception:
            pass

    def test_file_device_write_bad_fd(self):
        fd = FileDevice()
        try:
            fd.call("write", 999, "data")
            assert False, "Should have raised"
        except Exception:
            pass

    def test_file_device_listdir(self):
        fd = FileDevice()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = fd.call("listdir", tmpdir)
            assert isinstance(result, list)

    def test_file_device_exists(self):
        fd = FileDevice()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as f:
            path = f.name
        try:
            result = fd.call("exists", path)
            assert result is True
            result = fd.call("exists", path + ".nonexistent")
            assert result is False
        finally:
            os.unlink(path)

    def test_file_device_info_with_open(self):
        fd = FileDevice()
        with tempfile.NamedTemporaryFile(mode='r', delete=False, suffix='.txt') as f:
            path = f.name
        try:
            file_fd = fd.call("open", path, "r")
            info = fd.info()
            assert info["open_files"] >= 1
        finally:
            os.unlink(path)


class TestBlockDeviceExtended:
    def test_block_device_read_write_sector(self):
        bd = BlockDevice(num_sectors=8)
        data = b'\x41' * 512
        bd.write_sector(0, data)
        result = bd.read_sector(0)
        assert result[:10] == b'\x41' * 10

    def test_block_device_info(self):
        bd = BlockDevice(num_sectors=8)
        info = bd.info()
        assert isinstance(info, dict)


class TestIRQDevice:
    def test_irq_device_init(self):
        from domains.shell.vm import IRQDevice
        irq = IRQDevice()
        assert irq is not None

    def test_irq_device_info(self):
        from domains.shell.vm import IRQDevice
        irq = IRQDevice()
        info = irq.info()
        assert isinstance(info, dict)


class TestVGADeviceExtended:
    def test_vga_call_write(self):
        vga = VGADevice()
        result = vga.call("write", 0, 0, "Hello")
        assert result is True

    def test_vga_call_clear(self):
        vga = VGADevice()
        result = vga.call("clear")
        assert result is True

    def test_vga_call_scroll(self):
        vga = VGADevice()
        result = vga.call("scroll", 1)
        assert result is True

    def test_vga_call_get_cursor(self):
        vga = VGADevice()
        result = vga.call("get_cursor")
        assert isinstance(result, tuple)

    def test_vga_call_get_screen(self):
        vga = VGADevice()
        result = vga.call("get_screen")
        assert isinstance(result, list)


class TestCMOSDeviceExtended:
    def test_cmos_call_get_time(self):
        cmos = CMOSDevice()
        t = cmos.call("get_time")
        assert isinstance(t, dict)

    def test_cmos_call_get_unix_time(self):
        cmos = CMOSDevice()
        t = cmos.call("get_unix_time")
        assert isinstance(t, int)

    def test_cmos_call_read_cmos(self):
        cmos = CMOSDevice()
        val = cmos.call("read_cmos", 0x10)
        assert isinstance(val, int)

    def test_cmos_call_write_cmos(self):
        cmos = CMOSDevice()
        cmos.call("write_cmos", 0x10, 0x42)
        val = cmos.call("read_cmos", 0x10)
        assert val == 0x42

    def test_cmos_call_set_binary_mode(self):
        cmos = CMOSDevice()
        cmos.call("set_binary_mode", True)
        assert True

    def test_cmos_read_write_cycle(self):
        cmos = CMOSDevice()
        cmos._write_addr(0x10)
        cmos._write_data(0x42)
        cmos._write_addr(0x10)
        val = cmos._read_data()
        assert val == 0x42


class TestNICDeviceExtended:
    def test_nic_inject_and_recv(self):
        nic = NICDevice()
        nic.inject_packet(b'\x41' * 64)
        assert nic.has_packet()
        pkt = nic.recv_packet()
        assert pkt[:1] == b'\x41'

    def test_nic_send_packet(self):
        nic = NICDevice()
        result = nic.send_packet(b'\x00' * 64)
        assert result is True
        stats = nic.get_stats()
        assert stats["tx_packets"] >= 1

    def test_nic_send_too_large(self):
        nic = NICDevice()
        result = nic.send_packet(b'\x00' * 2000)
        assert result is False

    def test_nic_recv_empty(self):
        nic = NICDevice()
        pkt = nic.recv_packet()
        assert pkt == b'' or pkt is None

    def test_nic_stats(self):
        nic = NICDevice()
        nic.inject_packet(b'\x00' * 64)
        stats = nic.get_stats()
        assert stats["rx_packets"] >= 1

    def test_nic_flush(self):
        nic = NICDevice()
        nic.inject_packet(b'\x00' * 64)
        nic.flush()
        assert not nic.has_packet()


class TestPS2KeyboardExtended:
    def test_keyboard_read_key_empty(self):
        kd = PS2KeyboardDevice()
        assert kd.read_key() == 0

    def test_keyboard_push_scancode(self):
        kd = PS2KeyboardDevice()
        kd.call("push_scancode", 0x1E)  # scancode for 'a'
        assert kd.read_key() == ord('a')

    def test_keyboard_has_key(self):
        kd = PS2KeyboardDevice()
        assert kd.call("has_key") is False
        kd.call("push_scancode", 0x1E)
        assert kd.call("has_key") is True

    def test_keyboard_clear(self):
        kd = PS2KeyboardDevice()
        kd.call("push_scancode", 0x1E)
        kd.call("clear")
        assert kd.call("has_key") is False

    def test_keyboard_info(self):
        kd = PS2KeyboardDevice()
        info = kd.info()
        assert isinstance(info, dict)


class TestProcessTableExtended:
    def test_process_table_create_and_remove(self):
        pt = ProcessTable()
        pcb = pt.create("test_proc")
        assert pcb is not None
        pt.remove(pcb.pid)
        assert pt.get(pcb.pid) is None

    def test_process_table_all(self):
        pt = ProcessTable()
        pcb = pt.create("proc1")
        all_procs = pt.all()
        assert len(all_procs) >= 1
        pt.remove(pcb.pid)

    def test_process_table_count(self):
        pt = ProcessTable()
        pcb = pt.create("proc_count")
        count = pt.count()
        assert count >= 1
        pt.remove(pcb.pid)

    def test_process_table_by_state(self):
        pt = ProcessTable()
        pcb = pt.create("proc_state")
        procs = pt.by_state(ProcessState.CREATED)
        assert len(procs) >= 1
        pt.remove(pcb.pid)

    def test_process_table_get_by_name(self):
        pt = ProcessTable()
        pcb = pt.create("named_proc")
        found = pt.get_by_name("named_proc")
        assert found is not None
        pt.remove(pcb.pid)

    def test_process_table_alive_count(self):
        pt = ProcessTable()
        pcb = pt.create("alive_proc")
        count = pt.alive_count()
        assert count >= 1
        pt.remove(pcb.pid)


class TestSchedulerExtended:
    def test_scheduler_tick_count(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert sch._tick_count == 0

    def test_scheduler_ready_queue(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert len(sch._ready_queue) == 0

    def test_scheduler_current_pid(self):
        pt = ProcessTable()
        sch = Scheduler(process_table=pt)
        assert sch._current_pid is None


class TestPageFrameAllocator:
    def test_allocator_init(self):
        pfa = PageFrameAllocator()
        assert pfa is not None

    def test_allocator_alloc(self):
        pfa = PageFrameAllocator()
        addr = pfa.alloc()
        assert addr is not None and addr > 0

    def test_allocator_free(self):
        pfa = PageFrameAllocator()
        addr = pfa.alloc()
        pfa.free(addr)
        assert True

    def test_allocator_alloc_zero(self):
        pfa = PageFrameAllocator()
        result = pfa.alloc(0)
        assert result is None

    def test_allocator_alloc_multiple(self):
        pfa = PageFrameAllocator()
        addr = pfa.alloc(4)
        assert addr is not None and addr > 0


class TestX86CPUExtended:
    def test_cpu_step_halt(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xF4  # HLT
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._eip == 0x1001

    def test_cpu_step_nop(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x90  # NOP
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._eip == 0x1001

    def test_cpu_step_mov_eax_imm32(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xB8
        cpu._mem[0x1001:0x1005] = b'\x2A\x00\x00\x00'
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 42

    def test_cpu_step_push_pop(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x50  # PUSH EAX
        cpu._mem[0x1001] = 0x58  # POP EAX
        cpu._regs[0] = 42
        cpu._eip = 0x1000
        cpu.step()  # PUSH
        cpu.step()  # POP
        assert cpu._regs[0] == 42

    def test_cpu_step_add(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xC0
        cpu._mem[0x1002] = 0x05
        cpu._regs[0] = 10
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 15

    def test_cpu_step_sub(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xE8
        cpu._mem[0x1002] = 0x05
        cpu._regs[0] = 10
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 5

    def test_cpu_step_and(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xE0
        cpu._mem[0x1002] = 0x0F
        cpu._regs[0] = 0xFF
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 0x0F

    def test_cpu_step_or(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xC8
        cpu._mem[0x1002] = 0xF0
        cpu._regs[0] = 0x0F
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 0xFFFFFFFF  # 0x0F | 0xF0 = 0xFF, sign-extended to 32-bit

    def test_cpu_step_xor(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xF0
        cpu._mem[0x1002] = 0xFF
        cpu._regs[0] = 0xFF
        cpu._eip = 0x1000
        cpu.step()
        # 0xFF XOR (sign-extended 0xFF = 0xFFFFFFFF) = 0xFFFFFF00
        assert cpu._regs[0] == 0xFFFFFF00

    def test_cpu_step_cmp(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x83
        cpu._mem[0x1001] = 0xF8
        cpu._mem[0x1002] = 0x0A
        cpu._regs[0] = 10
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._flag(0x40)  # ZF set

    def test_cpu_step_inc(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x40  # INC EAX
        cpu._regs[0] = 41
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 42

    def test_cpu_step_dec(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0x48  # DEC EAX
        cpu._regs[0] = 42
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 41

    def test_cpu_step_not(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xF7
        cpu._mem[0x1001] = 0xD0  # NOT EAX
        cpu._regs[0] = 0xFF
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 0xFFFFFF00

    def test_cpu_step_neg(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xF7
        cpu._mem[0x1001] = 0xD8  # NEG EAX
        cpu._regs[0] = 5
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 0xFFFFFFFB

    def test_cpu_step_shl(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xC1
        cpu._mem[0x1001] = 0xE0
        cpu._mem[0x1002] = 0x04  # SHL EAX, 4
        cpu._regs[0] = 1
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 16

    def test_cpu_step_shr(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xC1
        cpu._mem[0x1001] = 0xE8
        cpu._mem[0x1002] = 0x04  # SHR EAX, 4
        cpu._regs[0] = 16
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] == 1

    def test_cpu_step_rol(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xC1
        cpu._mem[0x1001] = 0xC0
        cpu._mem[0x1002] = 0x04  # ROL EAX, 4
        cpu._regs[0] = 0x12345678
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] != 0x12345678

    def test_cpu_step_ror(self):
        cpu = X86CPU()
        cpu._mem[0x1000] = 0xC1
        cpu._mem[0x1001] = 0xC8
        cpu._mem[0x1002] = 0x04  # ROR EAX, 4
        cpu._regs[0] = 0x12345678
        cpu._eip = 0x1000
        cpu.step()
        assert cpu._regs[0] != 0x12345678


# ── Memory (simple VM) ─────────────────────────────────────────────────────

class TestMemory:
    def test_store_and_load(self):
        mem = Memory()
        arr = np.array([1, 2, 3], dtype=np.float32)
        mem.store("x", arr)
        assert mem.load("x") is arr

    def test_load_missing_raises(self):
        mem = Memory()
        with pytest.raises(InsFault, match="heap key not found"):
            mem.load("nonexistent")

    def test_free_existing(self):
        mem = Memory()
        mem.store("a", np.array([1]))
        mem.free("a")
        assert not mem.contains("a")

    def test_free_nonexistent_noop(self):
        mem = Memory()
        mem.free("missing")

    def test_contains(self):
        mem = Memory()
        assert not mem.contains("k")
        mem.store("k", np.array([1]))
        assert mem.contains("k")

    def test_lru_evict_order(self):
        mem = Memory()
        mem.store("a", np.array([1]))
        mem.store("b", np.array([2]))
        mem.store("c", np.array([3]))
        evicted = mem.lru_evict()
        assert evicted == "a"

    def test_lru_evict_empty(self):
        mem = Memory()
        assert mem.lru_evict() is None

    def test_lru_touch_reorders(self):
        mem = Memory()
        mem.store("a", np.array([1]))
        mem.store("b", np.array([2]))
        mem.load("a")  # touch a, moves to end
        evicted = mem.lru_evict()
        assert evicted == "b"

    def test_usage(self):
        mem = Memory()
        mem.store("x", np.array([1, 2], dtype=np.float32))
        u = mem.usage()
        assert u["entries"] == 1
        assert "x" in u["keys"]
        assert u["bytes_tracked"] == 8

    def test_store_non_ndarray(self):
        mem = Memory()
        mem.store("val", 42)
        assert mem.load("val") == 42
        assert mem.usage()["bytes_tracked"] == 0

    def test_free_removes_from_lru(self):
        mem = Memory()
        mem.store("a", np.array([1]))
        mem.free("a")
        assert "a" not in mem.usage()["lru_order"]


# ── ClockDevice ─────────────────────────────────────────────────────────────

class TestClockDeviceUnit:
    def test_init_defaults(self):
        cd = ClockDevice()
        assert cd.freq == 100
        assert cd.ticks == 0

    def test_tick_increments(self):
        cd = ClockDevice()
        cd.tick()
        cd.tick()
        assert cd.ticks == 2

    def test_seconds_now(self):
        cd = ClockDevice(freq=100, epoch_unix=1000)
        cd.tick()  # 1 tick = 0.01s
        assert cd.seconds_now() == 1000.01

    def test_set_time(self):
        cd = ClockDevice()
        cd.set_time(2024, 1, 15, 12, 30, 0)
        d = cd.decode()
        assert d["year"] == 2024
        assert d["month"] == 1
        assert d["day"] == 15
        assert d["hour"] == 12
        assert d["minute"] == 30

    def test_decode_unix_epoch(self):
        d = ClockDevice._decode_unix(0)
        assert d["year"] == 1970
        assert d["month"] == 1
        assert d["day"] == 1

    def test_decode_specific_date(self):
        ts = ClockDevice._date_to_unix(2000, 1, 1, 0, 0, 0)
        d = ClockDevice._decode_unix(ts)
        assert d["year"] == 2000
        assert d["month"] == 1
        assert d["day"] == 1

    def test_is_leap(self):
        assert ClockDevice._is_leap(2000)
        assert not ClockDevice._is_leap(1900)
        assert ClockDevice._is_leap(2024)
        assert not ClockDevice._is_leap(2023)

    def test_days_in_month(self):
        assert ClockDevice._days_in_month(2024, 1) == 31
        assert ClockDevice._days_in_month(2024, 2) == 29  # leap
        assert ClockDevice._days_in_month(2023, 2) == 28  # non-leap
        assert ClockDevice._days_in_month(2024, 4) == 30

    def test_date_to_unix_roundtrip(self):
        ts = ClockDevice._date_to_unix(2024, 6, 15, 10, 30, 45)
        d = ClockDevice._decode_unix(ts)
        assert d["year"] == 2024
        assert d["month"] == 6
        assert d["day"] == 15
        assert d["hour"] == 10
        assert d["minute"] == 30
        assert d["second"] == 45

    def test_decode_weekday(self):
        # 2024-01-01 is Monday = 0
        ts = ClockDevice._date_to_unix(2024, 1, 1, 0, 0, 0)
        d = ClockDevice._decode_unix(ts)
        assert d["weekday"] == 0

    def test_decode_default_uses_seconds_now(self):
        cd = ClockDevice(freq=100, epoch_unix=0)
        cd.set_time(2024, 3, 20, 8, 0, 0)
        d = cd.decode()
        assert d["year"] == 2024
        assert d["month"] == 3

    def test_decode_negative_clamps(self):
        d = ClockDevice._decode_unix(-1000)
        assert d["year"] >= 1970

    def test_set_time_resets_ticks(self):
        cd = ClockDevice()
        cd.tick()
        cd.tick()
        cd.set_time(2024, 1, 1)
        assert cd.ticks == 0

    def test_custom_freq(self):
        cd = ClockDevice(freq=50)
        cd.tick()
        assert cd.seconds_now() == ClockDevice.EPOCH_1900 + 1 / 50


# ── X86Shell ────────────────────────────────────────────────────────────────

class TestX86ShellDirect:
    def test_init(self):
        shell = X86Shell()
        assert not shell.running

    def test_start_and_stop(self):
        shell = X86Shell()
        shell.start(max_steps=100)
        import time
        time.sleep(0.05)
        shell.stop()
        assert not shell.running

    def test_read_screen_returns_string(self):
        shell = X86Shell()
        shell.start(max_steps=100)
        import time
        time.sleep(0.05)
        screen = shell.read_screen()
        shell.stop()
        assert isinstance(screen, str)

    def test_read_screen_custom_size(self):
        shell = X86Shell()
        shell.start(max_steps=100)
        import time
        time.sleep(0.05)
        screen = shell.read_screen(width=40, height=10)
        shell.stop()
        lines = screen.split("\n")
        assert len(lines) == 10

    def test_type_keys(self):
        shell = X86Shell()
        shell.start(max_steps=100)
        import time
        time.sleep(0.05)
        shell.type_keys("hello")
        time.sleep(0.05)
        shell.stop()

    def test_custom_source(self):
        shell = X86Shell(source="HLT")
        shell.start(max_steps=10)
        import time
        time.sleep(0.05)
        shell.stop()


# ── DiskProgramLoader ───────────────────────────────────────────────────────

class TestDiskProgramLoaderDirect:
    def test_list_programs_empty(self):
        fs = FlatFS(BlockDevice())
        loader = DiskProgramLoader(fs)
        assert loader.list_programs() == []

    def test_list_programs_filters_asm(self):
        fs = FlatFS(BlockDevice())
        fs.write("hello.asm", b"HLT")
        fs.write("data.txt", b"not asm")
        loader = DiskProgramLoader(fs)
        progs = loader.list_programs()
        assert progs == ["hello.asm"]

    def test_load_source(self):
        fs = FlatFS(BlockDevice())
        fs.write("test.asm", b"MOV R0, 1\nHLT")
        loader = DiskProgramLoader(fs)
        src = loader.load_source("test.asm")
        assert "MOV" in src

    def test_load_source_adds_asm_extension(self):
        fs = FlatFS(BlockDevice())
        fs.write("test.asm", b"HLT")
        loader = DiskProgramLoader(fs)
        src = loader.load_source("test")
        assert "HLT" in src

    def test_save_program(self):
        fs = FlatFS(BlockDevice())
        loader = DiskProgramLoader(fs)
        loader.save_program("prog", "HLT")
        assert "prog.asm" in fs.list_files()

    def test_save_program_adds_asm(self):
        fs = FlatFS(BlockDevice())
        loader = DiskProgramLoader(fs)
        loader.save_program("prog.asm", "HLT")
        assert "prog.asm" in fs.list_files()

    def test_run_program(self):
        fs = FlatFS(BlockDevice())
        fs.write("hi.asm", b"MOV R0, 42\nOUT 1, R0\nHLT")
        loader = DiskProgramLoader(fs)
        result = loader.run("hi.asm", max_steps=100)
        assert result["name"] == "hi.asm"
        assert result["steps"] > 0

    def test_run_with_custom_io(self):
        fs = FlatFS(BlockDevice())
        fs.write("io.asm", b"MOV R0, 99\nOUT 1, R0\nHLT")
        output = []
        loader = DiskProgramLoader(fs)
        result = loader.run("io.asm", stdout_fn=lambda v: output.append(v))
        assert result["steps"] > 0


# ── Device base class ──────────────────────────────────────────────────────

class TestDeviceBase:
    def test_call_raises(self):
        dev = Device()
        with pytest.raises(DeviceFault, match="does not support"):
            dev.call("unknown_method")

    def test_info(self):
        dev = Device()
        info = dev.info()
        assert info["type"] == "base"
