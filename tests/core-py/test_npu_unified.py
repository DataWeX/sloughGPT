"""Unified NPU — one device, both seams (card 20261104_090736).

The duplication this pins away: two NPUDevice classes with drifted ioctl
vocabularies (PIPE vs PIPELINE), reversed __call__ argument orders, and a
boot path that registered TWO distinct NPU instances under the name "npu"
so a model loaded from one seam was invisible from the other.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from domain.shell._internal import kernel_npu, npu_device
from domain.shell._internal.device_system import get_device_system, reset_device_system
from domain.shell._internal.ioctl import IoctlCommand
from domain.shell._internal.kernel_npu import NPUDevice
from domain.shell._internal.kernel_syscall import SyscallResult
from domain.shell._internal.vm_devices import NPUVMDevice


class _StubProvider:
    """Zero-cost provider that records the budget it was called with."""

    def __init__(self):
        self.last_max_tokens = None

    def tokenize(self, text: str):
        return [1, 2, 3]

    def detokenize(self, ids):
        return "stub-output"

    def generate_numpy(self, ids, max_new_tokens=100, **kw):
        self.last_max_tokens = max_new_tokens
        return np.array([[7, 8]])

    def forward_numpy(self, ids):
        return np.zeros((1, 2, 4))

    def embed(self, text, layer=-1):
        return np.zeros(8)


def _npu_with_stub() -> tuple[NPUDevice, _StubProvider]:
    npu = NPUDevice(name="npu")
    stub = _StubProvider()
    npu._models["stub"] = stub
    npu._default_model = "stub"
    return npu, stub


# ── One class, both import paths ──────────────────────────────────────────


class TestOneClassBothImports:
    def test_npu_device_import_is_the_unified_class(self):
        """The standalone path must not grow a second implementation again."""
        assert npu_device.NPUDevice is kernel_npu.NPUDevice

    def test_npu_model_reexported(self):
        assert npu_device.NPUModel is kernel_npu.NPUModel

    def test_public_shell_export(self):
        """`from domain.shell import NPUDevice` was the documented import and
        cli.py used it — it was never exported, so the CLI crashed on line 1.
        """
        from domain.shell import NPUDevice as PublicNPU

        assert PublicNPU is kernel_npu.NPUDevice


# ── Merged ioctl vocabulary ───────────────────────────────────────────────


class TestVocabularyMerge:
    @pytest.mark.parametrize("cmd", ["PIPE", "PIPELINE"])
    def test_both_chain_names_run_the_same_operation(self, cmd):
        npu, _ = _npu_with_stub()
        npu._models["stub2"] = _StubProvider()
        result = npu.ioctl(cmd, ["stub", "stub2"], "hi")
        assert result.success is True
        assert len(result.value["trace"]) == 2

    def test_unknown_command_speaks_both_dialects(self):
        """kernel face pins 'unknown ioctl', standalone face pins 'unknown command'."""
        npu, _ = _npu_with_stub()
        result = npu.ioctl("NONEXISTENT")
        assert result.success is False
        assert "unknown command" in result.error
        assert "unknown ioctl" in result.error

    def test_enum_command_accepted(self):
        npu, _ = _npu_with_stub()
        result = npu.ioctl(IoctlCommand.CALL, "stub", "hi")
        assert result.success is True

    def test_list_commands_is_the_merged_surface(self):
        npu, _ = _npu_with_stub()
        cmds = npu.list_commands()
        for expected in ("PIPE", "PIPELINE", "COMPUTE", "LIST_COMMANDS", "LOAD", "MEMORY"):
            assert expected in cmds

    def test_info_carries_both_key_vocabs(self):
        npu, _ = _npu_with_stub()
        info = npu.info()
        assert info["device"] == "npu"  # kernel/DeviceBus face
        assert info["name"] == "npu"  # standalone/dev face
        assert info["type"] == "npu"
        assert isinstance(info["compute_ops"], list)
        assert info["models"] == 1


# ── Both __call__ shapes ──────────────────────────────────────────────────


class TestBothCallShapes:
    def test_canonical_positional_shape(self):
        """Kernel face: npu(model_name, input)."""
        npu, _ = _npu_with_stub()
        assert "text" in npu("stub", "hi")

    def test_standalone_keyword_shape(self):
        """Old standalone face: npu(input, model_name=...)."""
        npu, _ = _npu_with_stub()
        assert "text" in npu("hi", model_name="stub")

    def test_single_positional_uses_default_model(self):
        npu, _ = _npu_with_stub()
        assert "text" in npu("hi")

    def test_missing_input_raises(self):
        npu, _ = _npu_with_stub()
        with pytest.raises(TypeError):
            npu()


# ── Cross-seam model registry ─────────────────────────────────────────────


class TestCrossSeamRegistry:
    def test_vm_face_and_raw_face_share_one_registry(self):
        """A model loaded through either face is visible from the other."""
        npu, _ = _npu_with_stub()
        vm_face = NPUVMDevice(npu)

        # raw/kernel face loads → VM face sees it
        info = vm_face.call("info")
        assert "stub" in info["names"]

        # VM face loads → raw face sees it (same object, not a copy)
        with patch.object(npu, "_load_numpy", return_value=_StubProvider()):
            vm_face.call("load_model", "second", "/tmp/model.npy")
        assert "second" in npu._models
        assert npu.ioctl("INFO").value["models"] == 2

    def test_cli_hf_id_load_fails_cleanly_into_fallback(self):
        """cli.py's non-file model ids must land in its mock fallback:
        a failed SyscallResult, not an exception."""
        npu, _ = _npu_with_stub()
        result = npu.ioctl("LOAD", "huggingface:some/model", "some/model")
        assert isinstance(result, SyscallResult)
        assert result.success is False
        assert "unsupported" in result.error


# ── max_tokens reaches the model through both faces ───────────────────────


class TestGenerateBudget:
    def test_ioctl_call_third_arg_sets_budget(self):
        npu, stub = _npu_with_stub()
        result = npu.ioctl("CALL", "stub", "hi", 5)
        assert result.success is True
        assert stub.last_max_tokens == 5

    def test_vm_generate_passes_budget_through(self):
        """The NPUVMDevice docstring promises ``generate, qwen, Hello, 50`` —
        the third argument used to be silently dropped."""
        npu, stub = _npu_with_stub()
        vm_face = NPUVMDevice(npu)
        text = vm_face.call("generate", "stub", "hi", 7)
        assert text == "stub-output"
        assert stub.last_max_tokens == 7

    def test_vm_generate_default_budget_when_omitted(self):
        npu, stub = _npu_with_stub()
        vm_face = NPUVMDevice(npu)
        vm_face.call("generate", "stub", "hi")
        assert stub.last_max_tokens == 100


# ── Provider-level ops: the four the VM docstring advertises ───────────────


# (VM op, ioctl command, args after the model name, expected value key)
_PROVIDER_OPS = [
    ("tokenize", "TOKENIZE", ("hi",), "token_ids"),
    ("detokenize", "DETOKENIZE", ([1, 2, 3],), "text"),
    ("embed", "EMBED", ("hi",), "embedding"),
    ("forward", "FORWARD", ([1, 2, 3],), "logits"),
]


class TestProviderLevelOps:
    """tokenize / detokenize / embed / forward on the unified device.

    The NPUVMDevice docstring advertises all four as guest-callable
    (``DEV_CALL R2, R0, tokenize, qwen, Hello world``), but neither
    historical NPU class ever implemented them — verified against the real
    device with the real model, all four raised ``DeviceFault: op not
    supported``. They are reachable from every seam now.
    """

    @pytest.mark.parametrize("vm_op,cmd,tail,key", _PROVIDER_OPS)
    def test_kernel_face_reaches_the_same_op(self, vm_op, cmd, tail, key):
        """One contract: ioctl is the same operation as the VM's DEV_CALL."""
        npu, _ = _npu_with_stub()
        result = npu.ioctl(cmd, "stub", *tail)
        assert result.success is True, result.error
        assert key in result.value

    def test_vm_face_unwraps_to_the_guest_shapes(self):
        """DEV_CALL hands the guest the bare value, not the SyscallResult
        box — that unwrap is what the VM wrappers are for, and it is the
        shape the assembly-facing tests already pin."""
        npu, _ = _npu_with_stub()
        vm = NPUVMDevice(npu)

        ids = vm.call("tokenize", "stub", "hi")
        assert isinstance(ids, list) and ids == [1, 2, 3]

        assert vm.call("detokenize", "stub", ids) == "stub-output"

        emb = vm.call("embed", "stub", "hi")
        assert isinstance(emb, np.ndarray) and emb.size > 0

        logits = vm.call("forward", "stub", [1, 2, 3])
        assert isinstance(logits, np.ndarray) and logits.size > 0

    @pytest.mark.parametrize("vm_op,cmd,tail,key", _PROVIDER_OPS)
    def test_guest_call_face_translates_lower_to_upper(self, vm_op, cmd, tail, key):
        """DeviceDriver.call is the DeviceBus face — same op, lowercase."""
        npu, _ = _npu_with_stub()
        assert key in npu.call(vm_op, "stub", *tail)

    @pytest.mark.parametrize("vm_op,cmd,tail,key", _PROVIDER_OPS)
    def test_missing_args_fail_instead_of_raising(self, vm_op, cmd, tail, key):
        npu, _ = _npu_with_stub()
        result = npu.ioctl(cmd)
        assert result.success is False
        assert "requires" in result.error

    @pytest.mark.parametrize("vm_op,cmd,tail,key", _PROVIDER_OPS)
    def test_unloaded_model_is_a_clean_failure(self, vm_op, cmd, tail, key):
        npu, _ = _npu_with_stub()
        result = npu.ioctl(cmd, "ghost", *tail)
        assert result.success is False
        assert "not loaded" in result.error

    def test_vm_face_and_kernel_face_share_the_registry(self):
        """Load through the VM face, serve through the kernel face."""
        npu, _ = _npu_with_stub()
        vm_face = NPUVMDevice(npu)
        with patch.object(npu, "_load_numpy", return_value=_StubProvider()):
            vm_face.call("load_model", "second", "/tmp/model.npy")

        ids = vm_face.call("tokenize", "second", "hi")
        assert isinstance(ids, list) and ids  # VM face unwraps for the guest
        assert "token_ids" in npu.ioctl("TOKENIZE", "second", "hi").value

    def test_new_commands_are_in_the_declared_surface(self):
        npu, _ = _npu_with_stub()
        cmds = npu.list_commands()
        for expected in ("TOKENIZE", "DETOKENIZE", "EMBED", "FORWARD"):
            assert expected in cmds

    def test_embed_layer_is_forwarded_only_when_asked(self):
        class LayerSpy(_StubProvider):
            def __init__(self):
                super().__init__()
                self.seen = None

            def embed(self, text, layer=-1):
                self.seen = layer
                return np.zeros(4)

        npu = NPUDevice(name="npu")
        spy = LayerSpy()
        npu._models["stub"] = spy
        npu._default_model = "stub"

        npu.embed("stub", "hi")
        assert spy.seen == -1  # default: plain call, no kwarg forced through
        npu.embed("stub", "hi", layer=3)
        assert spy.seen == 3


# ── Boot wiring: one instance, both seams ─────────────────────────────────


class TestBootSharedInstance:
    @pytest.fixture(autouse=True)
    def clean_device_system(self):
        reset_device_system()
        yield
        reset_device_system()

    def test_boot_registers_one_npu_behind_its_vm_face(self):
        """The old boot registered a second, distinct NPUDevice and silently
        overwrote the shell one — two registries under one name. The VM face
        must wrap the very instance _devices serves."""
        from domain.shell._internal.runtime import DaitRuntime

        rt = DaitRuntime()
        try:
            rt.boot()
            shell_npu = rt.devices.get("npu")
            vm_npu = get_device_system().get("npu")
            assert shell_npu is not None
            assert isinstance(vm_npu, NPUVMDevice)
            assert vm_npu._npu is shell_npu  # one shared instance, not a copy
        finally:
            rt.shutdown()
