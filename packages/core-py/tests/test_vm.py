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
            HLT
        """)
        # JMP target → target resolves to index 2
        assert len(insts) == 3
        assert insts[0].opcode == "JMP"
        assert insts[0].operands == [2]
        assert insts[2].opcode == "HLT"

    def test_forward_label(self):
        loader = ProgramLoader()
        insts = loader.load("""
        start:
            JZ end
            NOP
        end:
            HLT
        """)
        assert insts[0].opcode == "JZ"
        assert insts[0].operands == [2]

    def test_data_section_labels(self):
        loader = ProgramLoader()
        insts = loader.load("""
        .data
            msg: db 72, 73, 0
        .text
        start:
            MOV R0, msg
            HLT
        """)
        assert len(insts) == 2
        assert insts[0].operands[1] == 0  # msg address = 0
        assert loader.data_segment == [72, 73, 0]

    def test_data_str_directive(self):
        loader = ProgramLoader()
        loader.load("""
        .data
            msg: str "AB"
        .text
        start:
            HLT
        """)
        assert loader.data_segment == [65, 66, 0]  # 'A'=65, 'B'=66, null

    def test_parse_bytes_quoted(self):
        from domains.shell.vm import parse_bytes
        result = parse_bytes('"abc", 10, 0')
        assert result == [97, 98, 99, 10, 0]

    def test_parse_bytes_hex(self):
        from domains.shell.vm import parse_bytes
        result = parse_bytes('0x41, 0x42')
        assert result == [65, 66]

    def test_hello_asm_loads(self):
        loader = ProgramLoader()
        insts = loader.load(HELLO_ASM)
        assert len(insts) >= 3
        assert insts[0].opcode == "MOV"


# ── VirtualCPU ─────────────────────────────────────────────────────────────


class TestVirtualCPU:
    def test_create_cpu(self):
        cpu = VirtualCPU()
        assert len(cpu.regs) == NUM_REGS
        assert cpu.sp == STACK_BASE
        assert cpu.pc == 0

    def test_load_program_sets_pc(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("start: MOV R0, 0\nHLT")
        cpu.load_program(insts, labels={"start": 0})
        assert cpu.pc == 0

    def test_execute_mov_immediate(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R5, 42\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[5] == 42

    def test_execute_mov_register(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 10\nMOV R1, R0\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[1] == 10

    def test_execute_add(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 3\nADD R0, R0, 4\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 7

    def test_execute_sub(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 10\nSUB R0, R0, 3\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 7

    def test_execute_mul(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 6\nMUL R0, R0, 7\nHLT")
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
            HLT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42  # skipped MOV R0, 99

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
            HLT
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
            HLT
        skip:
            MOV R0, 42
            HLT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 42

    def test_push_pop(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            MOV R0, 42
            PUSH R0
            MOV R0, 0
            POP R1
            HLT
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu.regs[0] == 0   # was overwritten
        assert cpu.regs[1] == 42  # popped from stack

    def test_call_ret(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("""
            CALL fn
            HLT
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
        # Infinite loop
        insts = loader.load("""
        loop:
            JMP loop
        """)
        cpu.load_program(insts)
        cpu.run()
        assert cpu._step_count == 10

    def test_sandbox_syscall_limit(self):
        cpu = VirtualCPU()
        cpu._max_syscalls = 3
        loader = ProgramLoader()
        # SYSCALL 2 (write char) so it doesn't exit
        insts = loader.load("""
        loop:
            MOV R0, 2
            SYSCALL
            JMP loop
        """)
        cpu.load_program(insts, labels={"loop": 0})
        cpu.run()
        assert cpu._syscall_count == 3

    def test_alu_operations(self):
        for op, a, b, expected in [("AND", 0xFF, 0x0F, 0x0F),
                                    ("OR", 0xF0, 0x0F, 0xFF),
                                    ("XOR", 0xFF, 0x0F, 0xF0),
                                    ("SHL", 1, 3, 8),
                                    ("SHR", 8, 3, 1)]:
            cpu = VirtualCPU()
            loader = ProgramLoader()
            code = f"MOV R0, {a}\n{op} R0, R0, {b}\nHLT"
            insts = loader.load(code)
            cpu.load_program(insts)
            cpu.run()
            assert cpu.regs[0] == expected, f"{op}: got {cpu.regs[0]} expected {expected}"

    def test_flags_set_on_cmp(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 5\nCMP R0, 5\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.flags & F_ZERO

    def test_flags_set_on_sub(self):
        cpu = VirtualCPU()
        loader = ProgramLoader()
        insts = loader.load("MOV R0, 3\nSUB R0, R0, 5\nHLT")
        cpu.load_program(insts)
        cpu.run()
        assert cpu.flags & F_NEG

    def test_conditional_jumps(self):
        for jmp, a, b, should_jump in [
            ("JL", 3, 5, True),
            ("JL", 5, 3, False),
            ("JLE", 5, 5, True),
            ("JG", 7, 3, True),
            ("JG", 3, 3, False),
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
            HLT
        jump:
            MOV R0, 1
            HLT
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
        output = runner.assemble_and_run("MOV R0, 42\nHLT")
        assert runner.cpu.regs[0] == 42

    def test_register_to_register(self):
        runner = VMRunner()
        output = runner.assemble_and_run("MOV R0, 10\nMOV R1, R0\nHLT")
        assert runner.cpu.regs[1] == 10

    def test_infinite_loop_terminates(self):
        runner = VMRunner()
        output = runner.assemble_and_run("loop: JMP loop")
        assert runner.cpu._step_count > 0

    def test_memory_store_load(self):
        runner = VMRunner()
        output = runner.assemble_and_run("""
            MOV R0, 42
            STORE 100, R0
            MOV R1, 0
            LOAD R1, 100
            HLT
        """)
        assert runner.cpu.regs[1] == 42

    def test_disassemble(self):
        runner = VMRunner()
        listing = runner.disassemble(HELLO_ASM)
        assert any("MOV" in line for line in listing)
        assert any("SYSCALL" in line for line in listing)

    def test_self_test(self):
        from domains.shell.vm import self_test
        results = self_test()
        assert len(results) >= 3

    def test_cpu_get_state(self):
        runner = VMRunner()
        runner.assemble_and_run("MOV R0, 42\nHLT")
        state = runner.cpu.get_state()
        assert state["regs"][0] == 42
        assert not state["running"]
