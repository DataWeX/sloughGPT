"""
Simulation harness — boot kernel, open NPU, load model, run inference/training.

Tests the full stack: Kernel → NPU device → SloNet provider → metrics.
"""

import time
import numpy as np
import pytest
from domains.shell.kernel import Kernel


# ---------------------------------------------------------------------------
# Mock model for testing (no real weights needed)
# ---------------------------------------------------------------------------

class MockTransformer:
    """Minimal transformer-like model for simulation testing."""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, num_layers: int = 2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self._call_count = 0
        self._total_tokens = 0

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        self._call_count += 1
        self._total_tokens += input_ids.size
        # Simple embedding lookup + linear projection
        logits = np.random.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size).astype(np.float32)
        return logits

    def generate_numpy(self, prompt: str, max_tokens: int = 10, temperature: float = 1.0, **kwargs) -> list[int]:
        self._call_count += 1
        tokens = list(range(10, 10 + max_tokens))
        self._total_tokens += max_tokens
        return tokens

    def forward(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._call_count += 1
        input_ids = inputs.get("input_ids", np.zeros((1, 10), dtype=np.int64))
        logits = np.random.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size).astype(np.float32)
        self._total_tokens += input_ids.size
        return {"logits": logits}


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestKernelBoot:
    """Test kernel lifecycle."""

    def test_boot_and_shutdown(self):
        k = Kernel()
        msg = k.boot()
        assert "Kernel booted" in msg
        assert k.running
        assert k.uptime >= 0
        msg = k.shutdown()
        assert "shut down" in msg
        assert not k.running

    def test_double_boot_is_idempotent(self):
        k = Kernel()
        k.boot()
        msg = k.boot()
        assert "Already booted" in msg
        k.shutdown()

    def test_tick_advances(self):
        k = Kernel()
        k.boot()
        before = k.tick_count
        result = k.tick()
        assert result["tick_count"] == before + 1
        k.shutdown()

    def test_run_executes_ticks(self):
        k = Kernel()
        k.boot()
        results = k.run(max_ticks=10)
        assert len(results) > 0
        assert k.tick_count > 0
        k.shutdown()


class TestProcessManagement:
    """Test process spawning and lifecycle."""

    def test_spawn_process(self):
        from domains.shell.kernel_process import Priority
        k = Kernel()
        k.boot()
        proc = k.spawn_process("test-job", Priority.NORMAL)
        assert proc.pid > 0
        assert proc.name == "test-job"
        assert proc in k.list_processes()
        k.shutdown()

    def test_kill_process(self):
        from domains.shell.kernel_process import Priority, ProcessState
        k = Kernel()
        k.boot()
        proc = k.spawn_process("doomed", Priority.LOW)
        pid = proc.pid
        assert k.kill_process(pid)
        assert k.get_process(pid).state == ProcessState.STOPPED
        k.shutdown()

    def test_process_count_in_stats(self):
        from domains.shell.kernel_process import Priority
        k = Kernel()
        k.boot()
        k.spawn_process("a", Priority.NORMAL)
        k.spawn_process("b", Priority.HIGH)
        stats = k.stats()
        assert stats["process_count"] == 3  # init + a + b
        k.shutdown()


class TestNeuralCapabilities:
    """Test neural-specific kernel features."""

    def test_tokenize(self):
        k = Kernel()
        k.boot()
        tokens = k.tokenize("hello world")
        assert tokens == list(b"hello world")
        k.shutdown()

    def test_detokenize(self):
        k = Kernel()
        k.boot()
        text = k.detokenize([104, 101, 108, 108, 111])
        assert text == "hello"
        k.shutdown()

    def test_create_neural_process(self):
        from domains.shell.kernel_neural import NeuralProcessType
        k = Kernel()
        k.boot()
        proc = k.create_neural_process("infer", NeuralProcessType.INFERENCE, model_name="mock")
        assert proc.neural_type == NeuralProcessType.INFERENCE
        assert proc.model_name == "mock"
        k.shutdown()

    def test_create_embedding_store(self):
        k = Kernel()
        k.boot()
        store = k.create_embedding_store("test", 1000, 64)
        assert store.vocab_size == 1000
        vecs = k.embed(np.array([1, 2, 3]), "test")
        assert vecs is not None
        assert vecs.shape == (3, 64)
        k.shutdown()

    def test_create_kv_cache(self):
        k = Kernel()
        k.boot()
        cache = k.create_kv_cache("turn1", num_layers=4, head_dim=32)
        cache.initialize(num_heads=8)
        k0 = np.random.randn(8, 32)
        v0 = np.random.randn(8, 32)
        cache.update(0, k0, v0)
        cache.advance(1)
        kr, vr = cache.get(0, 0, 1)
        np.testing.assert_array_almost_equal(kr[:, 0, :], k0)
        k.shutdown()


class TestNPUDevice:
    """Test NPU device through kernel's engine."""

    def test_load_and_generate(self):
        k = Kernel()
        k.boot()
        model = MockTransformer()
        k.engine.load_model("mock", model)
        result = k.generate("mock", "test", max_tokens=5)
        assert result is not None
        assert result["token_count"] == 5
        k.shutdown()

    def test_load_and_forward(self):
        from domains.shell.kernel_neural import NeuralProcessType
        k = Kernel()
        k.boot()
        model = MockTransformer()
        k.engine.load_model("mock", model)
        proc = k.create_neural_process("fwd", NeuralProcessType.INFERENCE, model_name="mock")
        proc.model_ref = model
        inputs = {"input_ids": np.array([[1, 2, 3, 4, 5]])}
        outputs = k.forward(proc, inputs)
        assert "logits" in outputs
        assert outputs["logits"].shape == (1, 5, 256)
        k.shutdown()

    def test_unload_model(self):
        k = Kernel()
        k.boot()
        k.engine.load_model("tmp", MockTransformer())
        assert "tmp" in k.engine.info()["model_names"]
        k.engine.unload_model("tmp")
        assert "tmp" not in k.engine.info()["model_names"]
        k.shutdown()


class TestSyscalls:
    """Test kernel syscall dispatch."""

    def test_tensor_alloc_syscall(self):
        from domains.shell.kernel_syscall import SyscallNumber
        k = Kernel()
        k.boot()
        result = k.syscall(SyscallNumber.TENSOR_ALLOC, (32, 64), "float32")
        assert result.success
        assert result.value["shape"] == (32, 64)
        k.shutdown()

    def test_neural_tokenize_syscall(self):
        from domains.shell.kernel_neural import NeuralSyscall
        k = Kernel()
        k.boot()
        result = k.syscall(NeuralSyscall.TOKENIZE, "hello")
        assert result.success
        assert result.value["tokens"] == list(b"hello")
        k.shutdown()


class TestKernelStats:
    """Test kernel info and stats."""

    def test_info_snapshot(self):
        k = Kernel()
        k.boot()
        info = k.info()
        assert "uptime_s" in info
        assert "process_count" in info
        assert "memory" in info
        assert "devices" in info
        k.shutdown()

    def test_neural_stats(self):
        k = Kernel()
        k.boot()
        k.create_kv_cache("c1", 4, 32)
        k.create_embedding_store("e1", 500, 128)
        ns = k.neural_stats()
        assert ns["kv_caches"] == 1
        assert ns["embedding_stores"] == 1
        assert ns["engine"]["models_loaded"] == 0
        k.shutdown()


class TestDeviceRegistration:
    """Test that neural devices are registered with the kernel."""

    def test_register_devices(self):
        k = Kernel()
        k.boot()
        k.register_devices()
        stats = k.devices.stats()
        assert stats["total_devices"] >= 4  # null + engine + tokenizer + embedding + mha
        k.shutdown()


class TestEndToEndSimulation:
    """Full simulation: boot → load → infer → metrics → shutdown."""

    def test_full_simulation(self):
        from domains.shell.kernel_neural import NeuralProcessType
        k = Kernel()
        k.boot()
        k.register_devices()

        # Load model
        model = MockTransformer(vocab_size=128, d_model=32, num_layers=2)
        k.engine.load_model("sim-model", model)

        # Create inference process
        proc = k.create_neural_process("sim-infer", NeuralProcessType.INFERENCE, model_name="sim-model")
        proc.model_ref = model

        # Tokenize
        tokens = k.tokenize("the quick brown fox")
        assert len(tokens) > 0

        # Forward pass
        input_ids = np.array([tokens])
        outputs = k.forward(proc, {"input_ids": input_ids})
        assert "logits" in outputs

        # Generate
        gen_result = k.generate("sim-model", "once upon a time", max_tokens=20)
        assert gen_result["token_count"] == 20

        # KV cache
        cache = k.create_kv_cache("sim-cache", num_layers=2, head_dim=32)
        cache.initialize(num_heads=4)
        k0 = np.random.randn(4, 32)
        v0 = np.random.randn(4, 32)
        cache.update(0, k0, v0)
        cache.advance(1)

        # Stats
        info = k.info()
        ns = k.neural_stats()
        assert info["process_count"] >= 2
        assert ns["engine"]["models_loaded"] == 1

        k.shutdown()


class TestBootAndShell:
    def test_boot_program(self):
        from domains.shell.vm import VirtualSystem
        from domains.shell.vm_programs import BOOT_ASM

        out = []
        vs = VirtualSystem(enable_block=True, stdout_fn=lambda v: out.append(str(v)))
        vs.load_program(BOOT_ASM)
        vs.run()
        assert any("AI Compteur" in line for line in out)
        assert any("Ready" in line for line in out)

    def test_shell_echoes_command(self):
        from domains.shell.vm import VirtualSystem
        from domains.shell.vm_programs import SHELL_ASM

        out = []
        inputs = ['test-cmd']
        input_iter = iter(inputs)

        vs = VirtualSystem(
            stdin_fn=lambda: next(input_iter, 'exit'),
            stdout_fn=lambda v: out.append(str(v)),
        )
        vs.load_program(SHELL_ASM)
        vs.run(max_steps=30)
        assert any("test-cmd" in line for line in out)
        assert any("ai-compteur>" in line for line in out)

    def test_kernel_boots_and_spawns_shell(self):
        import time as _time
        k = Kernel()
        k.boot()
        k.register_devices()
        proc = k.spawn_shell()
        assert proc.name == "shell"
        k.tick()
        _time.sleep(0.05)
        assert proc.state.name in ("RUNNING", "READY")
        k.shutdown()
