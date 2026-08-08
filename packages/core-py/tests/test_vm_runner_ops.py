"""
Tests for the VMRunner high-level bytecode op handlers in domains/shell/vm.py.

Covers the integer ALU, I/O, tensor ALU, comparison, control-flow, data
movement, and device-bus ops by executing real assembled programs through
the VMRunner public entry point, plus direct handler/helper calls where a
runner path is impractical (unreachable via valid assembly).
"""

import pytest
import numpy as np

import domains.shell.vm as vm
from domains.shell.vm import (
    VMRunner,
    CPU,
    DeviceBus,
    ConsoleDevice,
    ClockDevice,
    InsFault,
    DeviceFault,
    MAX_CALL_DEPTH,
)


def _run(program, devices=None):
    """Assemble + run a program and return its CPU."""
    runner = VMRunner(devices=devices)
    runner.assemble_and_run(program)
    return runner.cpu


# ══════════════════════════════════════════════════════════════════════════════
# Integer ALU
# ══════════════════════════════════════════════════════════════════════════════

def test_integer_alu_ops():
    cpu = _run("""
        LOAD_CONST R1, 10
        LOAD_CONST R2, 3
        ISUB R3, R1, R2
        IMUL R4, R1, R2
        IDIV R5, R1, R2
        IDIV R6, R1, R0
        INEG R7, R1
        INC R1
        DEC R1
        ICMP R1, R2
        HALT
    """)
    assert cpu.regs[3] == 7
    assert cpu.regs[4] == 30
    assert cpu.regs[5] == 3
    assert cpu.regs[6] == 0
    assert cpu.regs[7] == -10
    assert cpu.regs[1] == 10
    assert cpu._cmp_flag == 1


# ══════════════════════════════════════════════════════════════════════════════
# I/O ops (IN / OUT)
# ══════════════════════════════════════════════════════════════════════════════

def _console_bus(stdin=None, stdout=None):
    bus = DeviceBus()
    bus.register_console(stdin_fn=stdin or (lambda: ""), stdout_fn=stdout or (lambda v: None))
    return bus


def test_in_reads_int():
    cpu = _run("IN R0, 0\nHALT", devices=_console_bus(stdin=lambda: "42"))
    assert cpu.regs[0] == 42


def test_in_reads_float():
    cpu = _run("IN R0, 0\nHALT", devices=_console_bus(stdin=lambda: "3.5"))
    assert cpu.regs[0] == 3.5


def test_in_reads_non_numeric():
    cpu = _run("IN R0, 0\nHALT", devices=_console_bus(stdin=lambda: "abc"))
    assert cpu.regs[0] == "abc"


def test_in_unknown_port_returns_zero():
    cpu = _run("IN R0, 7\nHALT", devices=_console_bus(stdin=lambda: "42"))
    assert cpu.regs[0] == 0


def test_in_device_without_read_uses_info_status():
    bus = DeviceBus()
    bus.register("5", ClockDevice())
    cpu = _run("IN R0, 5\nHALT", devices=bus)
    assert cpu.regs[0] == 0


def test_in_device_read_exception_returns_zero():
    def boom():
        raise RuntimeError("no data")
    cpu = _run("IN R0, 0\nHALT", devices=_console_bus(stdin=boom))
    assert cpu.regs[0] == 0


def test_out_writes_and_swallows_exception():
    seen = []
    cpu = _run("OUT 1, 42\nHALT", devices=_console_bus(stdout=seen.append))
    assert seen == [42]

    def boom(v):
        raise RuntimeError("write fail")
    cpu = _run("OUT 1, 7\nHALT", devices=_console_bus(stdout=boom))
    assert cpu.regs is not None


# ══════════════════════════════════════════════════════════════════════════════
# Tensor ALU
# ══════════════════════════════════════════════════════════════════════════════

def test_tensor_arith_add_sub_mul():
    cpu = _run("""
        LOAD_CONST R1, [1.0, 2.0, 3.0]
        LOAD_CONST R2, [4.0, 5.0, 6.0]
        ADD R3, R1, R2
        SUB R4, R2, R1
        MUL R5, R1, R2
        HALT
    """)
    np.testing.assert_allclose(cpu.regs[3], [5.0, 7.0, 9.0])
    np.testing.assert_allclose(cpu.regs[4], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(cpu.regs[5], [4.0, 10.0, 18.0])


def test_scalar_arith_and_neg_abs():
    cpu = _run("""
        ADD R0, 5, 3
        SUB R1, 5, 3
        MUL R2, 5, 3
        NEG R3, 5
        ABS R4, -5
        HALT
    """)
    assert cpu.regs[0] == 8
    assert cpu.regs[1] == 2
    assert cpu.regs[2] == 15
    assert cpu.regs[3] == -5
    assert cpu.regs[4] == 5


def test_div_normal_and_zero_guard():
    cpu = _run("""
        DIV R0, 6.0, 2.0
        DIV R1, 1.0, 0.0
        DIV R2, 0.0, 0.0
        HALT
    """)
    assert float(cpu.regs[0]) == 3.0
    assert float(cpu.regs[1]) == 0.0
    assert float(cpu.regs[2]) == 0.0


def test_matmul_all_ndims():
    cpu = _run("""
        LOAD_CONST R1, [[1.0, 2.0], [3.0, 4.0]]
        LOAD_CONST R2, [[5.0, 6.0], [7.0, 8.0]]
        MATMUL R3, R1, R2
        MATMUL R4, 2.0, 4.0
        LOAD_CONST R5, [1.0, 2.0, 3.0]
        LOAD_CONST R6, [4.0, 5.0, 6.0]
        MATMUL R7, R5, R6
        HALT
    """)
    np.testing.assert_allclose(cpu.regs[3], [[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_allclose(cpu.regs[4], [[8.0]])
    np.testing.assert_allclose(cpu.regs[7], [[32.0]])


def test_transpose_and_reductions():
    cpu = _run("""
        LOAD_CONST R1, [[1.0, 2.0], [3.0, 4.0]]
        LOAD_CONST R2, [1.0, 2.0, 3.0]
        TRANSPOSE R3, R1
        DOT R4, R2, R2
        NORM R5, R2
        SUM R6, R2
        MEAN R7, R2
        MAX R8, R2
        ARGMAX R9, R2
        HALT
    """)
    np.testing.assert_allclose(cpu.regs[3], [[1.0, 3.0], [2.0, 4.0]])
    assert float(cpu.regs[4]) == 14.0
    assert abs(float(cpu.regs[5]) - np.sqrt(14.0)) < 1e-9
    assert float(cpu.regs[6]) == 6.0
    assert float(cpu.regs[7]) == 2.0
    assert float(cpu.regs[8]) == 3.0
    assert int(cpu.regs[9]) == 2


def test_reshape_shape_size():
    cpu = _run("""
        LOAD_CONST R1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        RESHAPE R2, R1, 2, 3
        RESHAPE R3, R1, R0, 3
        SHAPE R4, R2
        SIZE R5, R2
        HALT
    """)
    assert cpu.regs[2].shape == (2, 3)
    assert cpu.regs[3].shape == (2, 3)
    assert cpu.regs[4] == [2, 3]
    assert cpu.regs[5] == 6


def test_activation_ops():
    cpu = _run("""
        LOAD_CONST R1, [-1.0, 0.0, 2.0]
        RELU R2, R1
        GELU R3, R1
        SIGMOID R4, R1
        TANH R5, R1
        SOFTMAX R6, R1
        LAYERNORM R7, R1
        RMSNORM R8, R1
        HALT
    """)
    np.testing.assert_allclose(cpu.regs[2], [0.0, 0.0, 2.0])
    assert cpu.regs[3].shape == (3,)
    assert cpu.regs[4].shape == (3,)
    assert cpu.regs[5].shape == (3,)
    assert abs(float(np.sum(cpu.regs[6])) - 1.0) < 1e-9
    assert cpu.regs[7].shape == (3,)
    assert cpu.regs[8].shape == (3,)


def test_softmax_values():
    cpu = _run("LOAD_CONST R1, [1.0, 2.0, 3.0]\nSOFTMAX R2, R1\nHALT")
    np.testing.assert_allclose(
        cpu.regs[2], [0.09003057, 0.24472847, 0.66524096], rtol=1e-5
    )


def test_random_ops_shapes_and_ranges():
    cpu = _run("RANDN R0, 2, 3\nRANDUNIF R1, 2, 3, 0.0, 1.0\nHALT")
    assert cpu.regs[0].shape == (2, 3)
    assert cpu.regs[1].shape == (2, 3)
    assert np.all(cpu.regs[1] >= 0.0)
    assert np.all(cpu.regs[1] < 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison / TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_cmp_scalar():
    cpu = _run("CMP 5, 3\nHALT")
    assert cpu._cmp_flag == 1
    cpu = _run("CMP 3, 5\nHALT")
    assert cpu._cmp_flag == -1
    cpu = _run("CMP 5, 5\nHALT")
    assert cpu._cmp_flag == 0


def test_cmp_ndarray():
    cpu = _run("LOAD_CONST R1, [1.0, 2.0]\nLOAD_CONST R2, [3.0, 4.0]\nCMP R1, R2\nHALT")
    assert cpu._cmp_flag == -1
    cpu = _run("LOAD_CONST R1, [3.0, 4.0]\nLOAD_CONST R2, [1.0, 2.0]\nCMP R1, R2\nHALT")
    assert cpu._cmp_flag == 1
    cpu = _run("LOAD_CONST R1, [3.0, 4.0]\nLOAD_CONST R2, [3.0, 4.0]\nCMP R1, R2\nHALT")
    assert cpu._cmp_flag == 0
    cpu = _run("LOAD_CONST R1, [1.0, 4.0]\nLOAD_CONST R2, [2.0, 3.0]\nCMP R1, R2\nHALT")
    assert cpu._cmp_flag == 0


def test_cmp_non_numeric():
    cpu = _run('LOAD_CONST R1, "abc"\nLOAD_CONST R2, "abd"\nCMP R1, R2\nHALT')
    assert cpu._cmp_flag == -1


def test_cmp_updates_flags_sub():
    runner = VMRunner()
    cpu = CPU(devices=runner._devices)
    cpu._update_flags_sub = lambda a, b, diff, bits: None
    cpu.load_program(runner._assembler.assemble("CMP 10, 4\nHALT"))
    cpu.run()
    assert cpu._cmp_flag == 1


def test_test_op():
    cpu = _run("TEST R1\nHALT")
    assert cpu._cmp_flag == 0
    cpu = _run("LOAD_CONST R1, 5\nTEST R1\nHALT")
    assert cpu._cmp_flag == 1
    cpu = _run("LOAD_CONST R1, [0.0, 0.0]\nTEST R1\nHALT")
    assert cpu._cmp_flag == 0
    cpu = _run("LOAD_CONST R1, [1.0, 0.0]\nTEST R1\nHALT")
    assert cpu._cmp_flag == 1


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: _truthy / _parse_tensor
# ══════════════════════════════════════════════════════════════════════════════

def test_truthy_helper():
    cpu = CPU()
    assert cpu._truthy(True) is True
    assert cpu._truthy(False) is False
    assert cpu._truthy(1) is True
    assert cpu._truthy(0.0) is False
    assert cpu._truthy(np.array([0, 0])) is False
    assert cpu._truthy(np.array([0, 1])) is True
    assert cpu._truthy("x") is True
    assert cpu._truthy("") is False
    assert cpu._truthy(None) is False
    assert cpu._truthy([]) is False


def test_parse_tensor_helper():
    cpu = CPU()
    arr = np.array([1.0, 2.0])
    assert cpu._parse_tensor(arr) is arr
    np.testing.assert_allclose(cpu._parse_tensor([1, 2]), [1.0, 2.0])
    assert float(cpu._parse_tensor(3)) == 3.0
    assert float(cpu._parse_tensor(2.5)) == 2.5
    with pytest.raises(InsFault):
        cpu._parse_tensor({"x": 1})


# ══════════════════════════════════════════════════════════════════════════════
# Control flow
# ══════════════════════════════════════════════════════════════════════════════

def test_jmp_and_resolve_label_int():
    cpu = _run("""
        JMP 3
        LOAD_CONST R1, 99
        HALT
        LOAD_CONST R2, 42
        HALT
    """)
    assert cpu.regs[2] == 42
    assert cpu.regs[1] == 0


def test_resolve_label_direct():
    cpu = CPU()
    assert vm._resolve_label(cpu, 5) == 5
    assert vm._resolve_label(cpu, "7") == 7
    with pytest.raises(InsFault):
        vm._resolve_label(cpu, "foo")


def test_call_ret_normal():
    cpu = _run("""
        CALL 3
        LOAD_CONST R1, 99
        HALT
        LOAD_CONST R2, 42
        RET
    """)
    assert cpu.regs[1] == 99
    assert cpu.regs[2] == 42


def test_call_stack_overflow():
    runner = VMRunner()
    runner.assemble_and_run("CALL 0\nHALT")
    runner.cpu._call_stack = [0] * MAX_CALL_DEPTH
    runner.cpu.load_program(runner._assembler.assemble("CALL 1\nHALT"))
    runner.cpu.run()
    assert not runner.cpu._running
    assert "[VM] call stack overflow" in runner.cpu._output


def test_ret_empty_stack():
    with pytest.raises(InsFault):
        _run("RET\nHALT")


# ══════════════════════════════════════════════════════════════════════════════
# Data movement
# ══════════════════════════════════════════════════════════════════════════════

def test_load_const_tensor_json():
    cpu = _run("LOAD_CONST R0, [1, 2, 3]\nHALT")
    np.testing.assert_allclose(cpu.regs[0], [1.0, 2.0, 3.0])


def test_load_const_tensor_fallback():
    cpu = _run("LOAD_CONST R0, [01, 2.5]\nHALT")
    np.testing.assert_allclose(cpu.regs[0], [1.0, 2.5])


def test_load_const_empty_tensor():
    cpu = _run("LOAD_CONST R0, [\u00a0]\nHALT")
    assert cpu.regs[0].shape == (0,)


def test_load_const_scalar():
    cpu = _run("LOAD_CONST R0, 42\nLOAD_CONST R1, 3.5\nHALT")
    assert cpu.regs[0] == 42
    assert cpu.regs[1] == 3.5


def test_load_const_fallback_non_numeric_crash():
    cpu = CPU()
    with pytest.raises(ValueError):
        vm._op_load_const(cpu, ["R0", "[foo]"])


def test_data_movement_ops():
    cpu = _run("""
        LOAD_SHAPE R0, 2, 3
        LOAD_SHAPE R5, R4, R4
        LOAD_CONST R1, [1.0, 2.0]
        STORE R1, "vec"
        LOAD R2, "vec"
        MOV R3, R2
        FREE "vec"
        PRINT R1
        PRINT R4
        HALT
    """)
    assert cpu.regs[0].shape == (2, 3)
    assert cpu.regs[5].shape == (1, 1)
    np.testing.assert_allclose(cpu.regs[2], [1.0, 2.0])
    np.testing.assert_allclose(cpu.regs[3], [1.0, 2.0])
    assert "vec" not in cpu._memory._heap
    assert len(cpu._output) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Device bus ops
# ══════════════════════════════════════════════════════════════════════════════

def test_dev_ops():
    bus = DeviceBus()
    bus.register("console", ConsoleDevice(0, stdin_fn=lambda: "7", stdout_fn=lambda v: None))
    cpu = _run("""
        DEV_OPEN R0, "console"
        DEV_CALL R1, R0, "read"
        DEV_CALL R2, R0, "write", 99
        DEV_INFO R3, R0
        DEV_CLOSE R0
        HALT
    """, devices=bus)
    assert cpu.regs[0] == "console"
    assert cpu.regs[1] == "7"
    assert cpu.regs[2] is None
    assert cpu.regs[3]["type"] == "console"


def test_dev_open_unknown_device():
    with pytest.raises(DeviceFault):
        _run('DEV_OPEN R0, "nope"\nHALT')
