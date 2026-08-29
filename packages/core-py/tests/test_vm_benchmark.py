"""CPU benchmarks for the sloughGPT tensor VM and x86 CPU."""

import time
import numpy as np
from domains.shell.vm import CPU, X86CPU, X86Assembler, Assembler


def bench(label, fn, iterations=1000):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    elapsed_ns = time.perf_counter_ns() - start
    per_op = elapsed_ns / iterations
    ops_sec = 1e9 / per_op if per_op > 0 else float('inf')
    print(f"  {label}: {per_op:,.0f} ns/op  ({ops_sec:,.0f} ops/sec)")
    return per_op


def bench_cpu(name, source, iterations=100, max_steps=100_000):
    asm = Assembler()
    instructions = asm.assemble(source)
    times = []
    for _ in range(iterations):
        cpu = CPU()
        cpu.load_program(instructions)
        start = time.perf_counter_ns()
        cpu.run(max_steps=max_steps)
        times.append(time.perf_counter_ns() - start)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:,.0f} ns/run  ({1e9/avg:,.0f} runs/sec)  [{len(instructions)} insns]")
    return avg


def bench_x86(name, source, iterations=100, max_steps=100_000):
    asm = X86Assembler()
    binary = asm.assemble(source)
    times = []
    for _ in range(iterations):
        cpu = X86CPU()
        cpu.load(binary)
        start = time.perf_counter_ns()
        cpu.run(max_steps=max_steps)
        times.append(time.perf_counter_ns() - start)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:,.0f} ns/run  ({1e9/avg:,.0f} runs/sec)  [{len(binary)} bytes]")
    return avg


class TestTensorIntALU:
    def test_iadd(self):
        bench_cpu("IADD", "LOAD_CONST R0, 1\nLOAD_CONST R1, 2\nIADD R2, R0, R1", iterations=500)
    def test_isub(self):
        bench_cpu("ISUB", "LOAD_CONST R0, 10\nLOAD_CONST R1, 3\nISUB R2, R0, R1", iterations=500)
    def test_imul(self):
        bench_cpu("IMUL", "LOAD_CONST R0, 7\nLOAD_CONST R1, 6\nIMUL R2, R0, R1", iterations=500)
    def test_idiv(self):
        bench_cpu("IDIV", "LOAD_CONST R0, 100\nLOAD_CONST R1, 7\nIDIV R2, R0, R1", iterations=500)
    def test_iand(self):
        bench_cpu("IAND", "LOAD_CONST R0, 255\nLOAD_CONST R1, 15\nIAND R2, R0, R1", iterations=500)
    def test_ior(self):
        bench_cpu("IOR", "LOAD_CONST R0, 240\nLOAD_CONST R1, 15\nIOR R2, R0, R1", iterations=500)
    def test_ixor(self):
        bench_cpu("IXOR", "LOAD_CONST R0, 255\nLOAD_CONST R1, 170\nIXOR R2, R0, R1", iterations=500)
    def test_ishl(self):
        bench_cpu("ISHL", "LOAD_CONST R0, 1\nLOAD_CONST R1, 8\nISHL R2, R0, R1", iterations=500)
    def test_ishr(self):
        bench_cpu("ISHR", "LOAD_CONST R0, 256\nLOAD_CONST R1, 4\nISHR R2, R0, R1", iterations=500)
    def test_icmp(self):
        bench_cpu("ICMP", "LOAD_CONST R0, 5\nLOAD_CONST R1, 5\nICMP R2, R0, R1", iterations=500)
    def test_inc(self):
        bench_cpu("INC", "LOAD_CONST R0, 0\nINC R0", iterations=500)
    def test_dec(self):
        bench_cpu("DEC", "LOAD_CONST R0, 100\nDEC R0", iterations=500)


class TestTensorFloatALU:
    def test_fadd(self):
        bench_cpu("FADD", "LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.71\nFADD R2, R0, R1", iterations=500)
    def test_fsub(self):
        bench_cpu("FSUB", "LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.71\nFSUB R2, R0, R1", iterations=500)
    def test_fmul(self):
        bench_cpu("FMUL", "LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.71\nFMUL R2, R0, R1", iterations=500)
    def test_fdiv(self):
        bench_cpu("FDIV", "LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.71\nFDIV R2, R0, R1", iterations=500)
    def test_fcmp(self):
        bench_cpu("FCMP", "LOAD_CONST R0, 3.14\nLOAD_CONST R1, 2.71\nFCMP R2, R0, R1", iterations=500)


class TestTensorOps:
    def test_add_1d(self):
        bench_cpu("ADD 1D [100]", "RANDN R0, 100, 1\nRANDN R1, 100, 1\nADD R2, R0, R1", iterations=200)
    def test_add_2d(self):
        bench_cpu("ADD 2D [32x32]", "RANDN R0, 32, 32\nRANDN R1, 32, 32\nADD R2, R0, R1", iterations=200)
    def test_mul_1d(self):
        bench_cpu("MUL 1D [100]", "RANDN R0, 100, 1\nRANDN R1, 100, 1\nMUL R2, R0, R1", iterations=200)
    def test_matmul_32(self):
        bench_cpu("MATMUL 32x32", "RANDN R0, 32, 32\nRANDN R1, 32, 32\nMATMUL R2, R0, R1", iterations=100)
    def test_matmul_64(self):
        bench_cpu("MATMUL 64x64", "RANDN R0, 64, 64\nRANDN R1, 64, 64\nMATMUL R2, R0, R1", iterations=50)
    def test_matmul_128(self):
        bench_cpu("MATMUL 128x128", "RANDN R0, 128, 128\nRANDN R1, 128, 128\nMATMUL R2, R0, R1", iterations=20)
    def test_matmul_256(self):
        bench_cpu("MATMUL 256x256", "RANDN R0, 256, 256\nRANDN R1, 256, 256\nMATMUL R2, R0, R1", iterations=5)
    def test_transpose(self):
        bench_cpu("TRANSPOSE 32x32", "RANDN R0, 32, 32\nTRANSPOSE R1, R0", iterations=200)
    def test_dot(self):
        bench_cpu("DOT 1000", "RANDN R0, 1000, 1\nRANDN R1, 1000, 1\nDOT R2, R0, R1", iterations=200)
    def test_norm(self):
        bench_cpu("NORM 1000", "RANDN R0, 1000, 1\nNORM R1, R0", iterations=200)
    def test_sum(self):
        bench_cpu("SUM 10000", "RANDN R0, 10000, 1\nSUM R1, R0", iterations=200)
    def test_mean(self):
        bench_cpu("MEAN 10000", "RANDN R0, 10000, 1\nMEAN R1, R0", iterations=200)
    def test_max(self):
        bench_cpu("MAX 10000", "RANDN R0, 10000, 1\nMAX R1, R0", iterations=200)
    def test_argmax(self):
        bench_cpu("ARGMAX 10000", "RANDN R0, 10000, 1\nARGMAX R1, R0", iterations=200)
    def test_reshape(self):
        bench_cpu("RESHAPE 4x4->16", "RANDN R0, 4, 4\nRESHAPE R1, R0, 16", iterations=500)


class TestTensorActivations:
    def test_relu(self):
        bench_cpu("RELU 1000", "RANDN R0, 1000, 1\nRELU R1, R0", iterations=500)
    def test_gelu(self):
        bench_cpu("GELU 1000", "RANDN R0, 1000, 1\nGELU R1, R0", iterations=500)
    def test_sigmoid(self):
        bench_cpu("SIGMOID 1000", "RANDN R0, 1000, 1\nSIGMOID R1, R0", iterations=500)
    def test_tanh(self):
        bench_cpu("TANH 1000", "RANDN R0, 1000, 1\nTANH R1, R0", iterations=500)
    def test_softmax(self):
        bench_cpu("SOFTMAX 1000", "RANDN R0, 1000, 1\nSOFTMAX R1, R0", iterations=200)
    def test_layernorm(self):
        bench_cpu("LAYERNORM 1000", "RANDN R0, 1000, 1\nLAYERNORM R1, R0", iterations=200)
    def test_rmsnorm(self):
        bench_cpu("RMSNORM 1000", "RANDN R0, 1000, 1\nRMSNORM R1, R0", iterations=200)
    def test_relu_2d(self):
        bench_cpu("RELU 32x32", "RANDN R0, 32, 32\nRELU R1, R0", iterations=500)
    def test_softmax_2d(self):
        bench_cpu("SOFTMAX 32x32", "RANDN R0, 32, 32\nSOFTMAX R1, R0", iterations=200)
    def test_layernorm_2d(self):
        bench_cpu("LAYERNORM 32x32", "RANDN R0, 32, 32\nLAYERNORM R1, R0", iterations=200)


class TestTensorControlFlow:
    def test_loop_100(self):
        bench_cpu("LOOP 100", "LOAD_CONST R0, 0\nLOAD_CONST R1, 100\nloop:\nINC R0\nICMP R2, R0, R1\nJLT loop\nHALT", iterations=500)
    def test_loop_1000(self):
        bench_cpu("LOOP 1000", "LOAD_CONST R0, 0\nLOAD_CONST R1, 1000\nloop:\nINC R0\nICMP R2, R0, R1\nJLT loop\nHALT", iterations=200)
    def test_call_return(self):
        bench_cpu("CALL/RET 100", "LOAD_CONST R0, 0\nLOAD_CONST R1, 100\nloop:\nCALL noop\nINC R0\nICMP R2, R0, R1\nJLT loop\nHALT\nnoop:\nRET", iterations=200)
    def test_dense_branches(self):
        bench_cpu("dense branches 50", "LOAD_CONST R0, 0\nLOAD_CONST R1, 50\nloop:\nINC R0\nICMP R2, R0, R1\nJGE done\nJMP loop\ndone:\nHALT", iterations=500)


class TestTensorMemory:
    def test_store_load(self):
        bench_cpu("STORE+LOAD", 'LOAD_CONST R0, 42\nSTORE R0, "key"\nLOAD R0, "key"', iterations=500)
    def test_push_pop(self):
        bench_cpu("PUSH+POP", "LOAD_CONST R0, 42\nPUSH R0\nPOP R0", iterations=500)
    def test_alloc_free(self):
        bench_cpu("ALLOC+FREE", "ALLOC R0, 1024", iterations=500)


class TestTensorComposite:
    def test_matmul_relu(self):
        bench_cpu("MATMUL->RELU 32x32", "RANDN R0, 32, 32\nRANDN R1, 32, 32\nMATMUL R2, R0, R1\nRELU R3, R2", iterations=100)
    def test_attention(self):
        bench_cpu("attention 32x32", "RANDN R0, 32, 32\nRANDN R1, 32, 32\nMATMUL R2, R0, R1\nLOAD_CONST R3, 5.656\nFDIV R4, R2, R3\nSOFTMAX R5, R4", iterations=50)
    def test_layernorm_matmul(self):
        bench_cpu("LAYERNORM->MATMUL 32x32", "RANDN R0, 32, 32\nRANDN R1, 32, 32\nLAYERNORM R2, R0\nMATMUL R3, R2, R1", iterations=50)
    def test_transformer_block(self):
        bench_cpu("transformer block 16x16", "RANDN R0, 16, 16\nRANDN R1, 16, 16\nRANDN R2, 16, 16\nMATMUL R3, R0, R1\nSOFTMAX R4, R3\nMATMUL R5, R4, R2\nADD R6, R5, R2\nLAYERNORM R7, R6\nMATMUL R8, R7, R1\nRELU R9, R8\nMATMUL R10, R9, R2\nADD R11, R10, R2\nLAYERNORM R12, R11", iterations=20)


class TestX86Basic:
    def test_nop(self):
        bench_x86("NOP x1000", "[BITS 32]\n[ORG 0x100000]\ntimes 1000 nop\nhlt", iterations=100)
    def test_mov(self):
        bench_x86("MOV x1000", "[BITS 32]\n[ORG 0x100000]\ntimes 500 mov eax, 42\ntimes 500 mov ebx, 99\nhlt", iterations=100)
    def test_add_sub(self):
        bench_x86("ADD/SUB x1000", "[BITS 32]\n[ORG 0x100000]\nmov eax, 0\ntimes 1000 add eax, 1\ntimes 1000 sub eax, 1\nhlt", iterations=50)
    def test_cmp_jnz(self):
        bench_x86("CMP+JNZ 1000", "[BITS 32]\n[ORG 0x100000]\nmov ecx, 1000\n.loop:\ndec ecx\njnz .loop\nhlt", iterations=100)
    def test_push_pop(self):
        bench_x86("PUSH/POP x1000", "[BITS 32]\n[ORG 0x100000]\ntimes 500 push eax\ntimes 500 pop ebx\nhlt", iterations=100)
    def test_mul(self):
        bench_x86("MUL x100", "[BITS 32]\n[ORG 0x100000]\nmov eax, 7\nmov ebx, 6\ntimes 100 mul ebx\nhlt", iterations=50)
    def test_div(self):
        bench_x86("DIV x100", "[BITS 32]\n[ORG 0x100000]\nmov eax, 1000\nmov ebx, 7\ntimes 100 div ebx\nhlt", iterations=50)
    def test_shift(self):
        bench_x86("SHL/SHR x1000", "[BITS 32]\n[ORG 0x100000]\nmov eax, 1\ntimes 500 shl eax, 1\ntimes 500 shr eax, 1\nhlt", iterations=50)
    def test_call_ret(self):
        bench_x86("CALL/RET x100", "[BITS 32]\n[ORG 0x100000]\nmov ecx, 100\n.loop:\ncall .noop\ndec ecx\njnz .loop\nhlt\n.noop:\nret", iterations=100)
    def test_lea(self):
        bench_x86("LEA x1000", "[BITS 32]\n[ORG 0x100000]\nmov ebx, 8\ntimes 1000 lea eax, [ebx+ecx*4+16]\nhlt", iterations=50)


class TestX86Complex:
    def test_fibonacci(self):
        bench_x86("fibonacci(20)", "[BITS 32]\n[ORG 0x100000]\nmov eax, 0\nmov ebx, 1\nmov ecx, 20\n.loop:\nmov edx, eax\nadd edx, ebx\nmov eax, ebx\nmov ebx, edx\ndec ecx\njnz .loop\nhlt", iterations=200)
    def test_collatz(self):
        bench_x86("collatz(27)", "[BITS 32]\n[ORG 0x100000]\nmov eax, 27\n.loop:\ncmp eax, 1\nje .done\ntest eax, 1\njnz .odd\nshr eax, 1\njmp .loop\n.odd:\nmov ebx, eax\nshl eax, 1\nadd eax, ebx\ninc eax\njmp .loop\n.done:\nhlt", iterations=200)
    def test_rep_movsd(self):
        bench_x86("rep movsd 64", "[BITS 32]\n[ORG 0x100000]\nlea esi, [0x10000]\nlea edi, [0x20000]\nmov ecx, 64\nrep movsd\nhlt", iterations=200)
    def test_sum_array(self):
        bench_x86("sum array[64]", "[BITS 32]\n[ORG 0x100000]\nlea esi, [0x10000]\nmov ecx, 64\nxor eax, eax\n.loop:\nadd eax, [esi]\nadd esi, 4\ndec ecx\njnz .loop\nhlt", iterations=200)


class TestOverhead:
    def test_tensor_assembler(self):
        src = "LOAD_CONST R0, 1\nLOAD_CONST R1, 2\nIADD R2, R0, R1\nHALT"
        asm = Assembler()
        bench("tensor assembler", lambda: asm.assemble(src), iterations=5000)
    def test_tensor_cpu_load(self):
        asm = Assembler()
        insns = asm.assemble("LOAD_CONST R0, 1\nHALT")
        cpu = CPU()
        bench("tensor cpu.load", lambda: cpu.load_program(insns), iterations=5000)
    def test_x86_assembler(self):
        src = "[BITS 32]\n[ORG 0x100000]\nmov eax, 1\nhlt"
        asm = X86Assembler()
        bench("x86 assembler", lambda: asm.assemble(src), iterations=2000)
    def test_x86_cpu_load(self):
        asm = X86Assembler()
        binary = asm.assemble("[BITS 32]\n[ORG 0x100000]\nmov eax, 1\nhlt")
        cpu = X86CPU()
        bench("x86 cpu.load", lambda: cpu.load(binary), iterations=2000)
