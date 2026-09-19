"""RSS budget guards for model inference (sub-1GB program).

Loads Qwen2.5-0.5B exactly the way the server does (int8 + freed fp32
originals) and asserts:
  - load footprint stays bounded (no duplicate weight copies),
  - repeated generations don't accumulate state (retention guard),
  - long-prefill temporaries (logits) are freed after the call.

Skips when the SLNC weights aren't cached locally or the platform has
no /proc (non-Linux).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "packages" / "core-py")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

QWEN_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SLNC_NAME = "models--Qwen--Qwen2.5-0.5B-Instruct"


def _slnc_path() -> Path | None:
    """Resolve the cached .slnc file (flat project layout first)."""
    candidates = [
        _REPO_ROOT / "models" / "hf-cache" / "hub" / SLNC_NAME / "model.slnc",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _rss_gb() -> float:
    """Current process RSS in GiB (Linux only)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    raise AssertionError("unreachable")


needs_proc = pytest.mark.skipif(
    not Path("/proc/self/status").exists(), reason="requires Linux /proc"
)


@pytest.fixture(scope="module")
def provider():
    """Load the model once per module, mirroring server flags."""
    slnc = _slnc_path()
    if slnc is None:
        pytest.skip(f"{QWEN_ID} .slnc not cached locally")
    from domain.inference._internal.slonet_provider import SloNetChatProvider

    prov = SloNetChatProvider.from_slnc(
        str(slnc),
        model_id=QWEN_ID,
        quantize=True,
        quant_bits=8,
        quant_mode="symmetric",
        free_quantized_originals=True,
    )
    gc.collect()
    return prov


@needs_proc
def test_load_stays_within_budget(provider):
    rss = _rss_gb()
    assert rss < 2.0, f"model load RSS {rss:.2f}GB exceeds 2.0GB budget"


@needs_proc
def test_no_growth_across_generations(provider):
    base = _rss_gb()
    for _ in range(3):
        provider.generate("The capital of France is", max_tokens=8, temperature=0.0)
    gc.collect()
    growth = _rss_gb() - base
    assert growth < 0.5, f"3 generations grew RSS by {growth:.2f}GB (retention?)"


@needs_proc
def test_long_prefill_temporaries_freed(provider):
    base = _rss_gb()
    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    assert len(provider._tokenizer.encode(prompt)) > 800
    provider.generate(prompt, max_tokens=4, temperature=0.0)
    gc.collect()
    growth = _rss_gb() - base
    assert growth < 0.5, f"900-token prefill retained {growth:.2f}GB (logits leak?)"
