"""
Tests for Shell Virtual Machine — CPU, assembler, syscall dispatch, sandbox.
"""

import pytest
from domains.shell.vm import (
    ProgramLoader, VirtualCPU, VMRunner, VMFault, Halt, MemFault, InsFault,
    HELLO_ASM, NUM_REGS, MEM_SIZE, STACK_BASE, F_ZERO, F_NEG,
)


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
