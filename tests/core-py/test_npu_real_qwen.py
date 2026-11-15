"""Slow gate: REAL inference through the NPU device — VM face and raw face.

Card 20261104_090736: the VM's NPU device is a scaled-down real machine, so
it is validated with actual inference on real-world prompts against the
cached Qwen2.5-0.5B .slnc — not a mock that agrees with itself. Mocked tests
elsewhere prove dispatch branching; this proves the device does the work.

Ground rules (user-approved plan):
- offline: the cached model file only, never the network;
- property assertions (non-empty output, budget honored, stops), never
  golden text — exact strings on real model output are flakiness;
- slow-marked: excluded from the default suite (pytest.ini -m "not slow"),
  run with:  ./scripts/python -m pytest tests/core-py/test_npu_real_qwen.py -o addopts=''
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    # pytest.ini pins timeout=120 globally so hangs fail fast; loading a real
    # 0.5B .slnc (weight plan + tied-weight assign) legitimately exceeds that,
    # so this module opts into a longer budget rather than the suite slowing
    # down for everyone.
    pytest.mark.timeout(600),
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QWEN = (
    _REPO_ROOT
    / "models"
    / "hf-cache"
    / "hub"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "model.slnc"
)
_QWEN = Path(os.environ.get("SLO_TEST_SLNC") or _DEFAULT_QWEN)

if not _QWEN.exists():
    pytest.skip(
        f"real Qwen .slnc not found at {_QWEN} (set SLO_TEST_SLNC)", allow_module_level=True
    )

from domain.shell._internal.kernel_npu import NPUDevice  # noqa: E402
from domain.shell._internal.vm_devices import NPUVMDevice  # noqa: E402

# Real-world prompts: a fact question, a generation task, a completion, and a
# long-context turn — the classes of workload the device will actually carry.
FACT_PROMPT = "What is the capital of France?"
GEN_PROMPT = "Write one short sentence about the ocean."
COMPLETION_PROMPT = "The quick brown fox"
LONG_PROMPT = (
    "Context: The Dait kernel exposes devices through a bus. Programs open a "
    "device, call its operations, and close it. The NPU loads a model file "
    "that already contains the graph, executes prompts, and reports memory. "
    "Question: how does a program reach the NPU? Answer briefly. "
) * 4  # ~600 chars — a realistic context, not a toy one

BUDGET = 16  # tokens per generation: real inference, bounded runtime


@pytest.fixture(scope="module")
def npu():
    dev = NPUDevice(name="qwen_npu")
    result = dev.ioctl("LOAD", str(_QWEN), "qwen")
    assert result.success, f"real model failed to load: {result.error}"
    yield dev
    dev.unload("qwen")


def test_vm_face_generates_real_text(npu):
    """The DEV_CALL-shaped face: load was through ioctl, generate through the
    guest op vocabulary — the exact path an assembly program takes."""
    vm_face = NPUVMDevice(npu)

    info = vm_face.call("info")
    assert "qwen" in info["names"], "VM face cannot see the loaded model"

    text = vm_face.call("generate", "qwen", FACT_PROMPT, BUDGET)
    assert isinstance(text, str)
    assert text.strip(), "real model produced empty output"


def test_raw_face_real_generation_within_budget(npu):
    out = npu.execute("qwen", GEN_PROMPT, max_tokens=BUDGET)
    assert isinstance(out["text"], str)
    assert out["text"].strip(), "real model produced empty output"
    assert 0 < out["tokens"] <= BUDGET, "generation exceeded the requested budget"
    assert out["ms"] > 0


def test_long_context_real_generation(npu):
    out = npu.execute("qwen", LONG_PROMPT, max_tokens=8)
    assert out["text"].strip()
    assert 0 < out["tokens"] <= 8


def test_cross_seam_visibility_with_a_real_model(npu):
    """One registry: the model loaded behind the kernel face is visible to
    the VM face and to the raw info face without a second load."""
    vm_face = NPUVMDevice(npu)
    assert "qwen" in npu.info()["names"]
    assert "qwen" in vm_face.call("info")["names"]

    # a second face-level call runs on the SAME resident weights
    text = vm_face.call("generate", "qwen", COMPLETION_PROMPT, 8)
    assert text.strip()


def test_real_model_unload_is_clean(npu):
    """A real 0.5B provider must leave the registry through the VM face."""
    vm_face = NPUVMDevice(npu)
    result = npu.ioctl("LOAD", str(_QWEN), "qwen_tmp")
    assert result.success, result.error
    assert "qwen_tmp" in npu.info()["names"]
    assert "qwen_tmp" in vm_face.call("info")["names"]

    unloaded = npu.ioctl("UNLOAD", "qwen_tmp")
    assert unloaded.success is True
    assert unloaded.value == {"unloaded": True}
    assert "qwen_tmp" not in npu.info()["names"]
    # the VM face is the same registry, not a snapshot
    assert "qwen_tmp" not in vm_face.call("info")["names"]
    # and the surviving model still generates after a neighbour unload
    assert vm_face.call("generate", "qwen", COMPLETION_PROMPT, 8).strip()


# ---------------------------------------------------------------------------
# The four provider-level ops the VM docstring advertises — real model,
# both seams. These raised "DeviceFault: op not supported" before this card.
# ---------------------------------------------------------------------------


def test_real_tokenize_and_detokenize_roundtrip(npu):
    vm_face = NPUVMDevice(npu)

    tok = vm_face.call("tokenize", "qwen", FACT_PROMPT)
    assert isinstance(tok, list) and tok, "real tokenizer returned nothing"
    assert all(isinstance(t, int) for t in tok)

    text = vm_face.call("detokenize", "qwen", tok)
    assert isinstance(text, str) and text.strip()

    # kernel face serves the same op from the same registry
    same = npu.ioctl("TOKENIZE", "qwen", FACT_PROMPT).value
    assert same["token_ids"] == tok
    assert same["count"] == len(tok)


def test_real_embed_is_a_finite_vector(npu):
    vm_face = NPUVMDevice(npu)

    vec = vm_face.call("embed", "qwen", FACT_PROMPT)
    assert isinstance(vec, np.ndarray)
    assert vec.size > 0
    assert np.isfinite(vec).all(), "real embedding contains NaN/inf"

    # an explicit layer must survive the trip (the VM passes 3 args)
    layered = npu.ioctl("EMBED", "qwen", FACT_PROMPT, -1).value
    assert layered["shape"] == list(vec.shape)


def test_real_forward_returns_well_formed_logits(npu):
    vm_face = NPUVMDevice(npu)
    ids = vm_face.call("tokenize", "qwen", COMPLETION_PROMPT)
    assert ids

    logits = vm_face.call("forward", "qwen", ids)
    assert isinstance(logits, np.ndarray)
    assert logits.ndim == 3, "logits must be (batch, seq, vocab)"
    assert logits.shape[0] == 1
    assert logits.shape[1] == len(ids), "logits must cover every input token"
    assert logits.shape[2] > 1
    assert np.isfinite(logits).all(), "real logits contain NaN/inf"

    # same op through the kernel face
    via_ioctl = npu.ioctl("FORWARD", "qwen", ids).value
    assert via_ioctl["shape"] == list(logits.shape)
