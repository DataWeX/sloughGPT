"""Tests for scripts/benchmark_v86_factor.py (card 20261103_121721, C1).

Card A measured decode on bare metal (368 tok/s @141K params) and set the
verdict against a 3.0 tok/s v86 floor — but never measured the target
machine. C1 closes that hole by *calibrating the emulation factor*: the
same static x86 workload runs native on the host and inside the v86
guest (driven over serial), host wall-clock times both, and the ratio
scales Card A's bare-metal number into an in-guest estimate.

The node driver (scripts/v86_factor_guest.mjs) is a dumb serial pipe —
all logic lives here so it can be tested without booting a VM.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_v86_factor as vf  # noqa: E402

# ── factor arithmetic ────────────────────────────────────────────────────────


def test_compute_factor_simple_ratio():
    assert vf.compute_factor(native_s=1.0, guest_s=25.0) == 25.0
    assert vf.compute_factor(native_s=2.5, guest_s=10.0) == 4.0


def test_compute_factor_rejects_nonpositive():
    """A zero or negative time is a broken measurement, not a fast machine."""
    with pytest.raises(ValueError):
        vf.compute_factor(native_s=0.0, guest_s=10.0)
    with pytest.raises(ValueError):
        vf.compute_factor(native_s=1.0, guest_s=-1.0)


# ── baseline application ─────────────────────────────────────────────────────


def test_apply_factor_scales_rate():
    # 10x slower machine -> 1/10 the tokens per second
    assert vf.apply_factor(baseline_tps=368.0, factor=10.0) == pytest.approx(36.8)
    # factor 1 (native == guest) -> unchanged
    assert vf.apply_factor(baseline_tps=368.0, factor=1.0) == 368.0


def test_apply_factor_rejects_bad_factor():
    with pytest.raises(ValueError):
        vf.apply_factor(baseline_tps=368.0, factor=0.5)
    with pytest.raises(ValueError):
        vf.apply_factor(baseline_tps=368.0, factor=0.0)


# ── verdict ──────────────────────────────────────────────────────────────────


def test_verdict_meets_and_misses():
    assert vf.verdict(estimate_tps=36.8, slo_tps=3.0)["meets_slo"] is True
    assert vf.verdict(estimate_tps=1.2, slo_tps=3.0)["meets_slo"] is False


def test_verdict_carries_margin_and_numbers():
    v = vf.verdict(estimate_tps=6.0, slo_tps=3.0)
    assert v["estimate_tps"] == 6.0
    assert v["slo_tps"] == 3.0
    assert v["margin_x"] == pytest.approx(2.0)


def test_verdict_rejects_bad_estimate():
    with pytest.raises(ValueError):
        vf.verdict(estimate_tps=0.0, slo_tps=3.0)


# ── guest runner contract ────────────────────────────────────────────────────


def test_parse_guest_output_valid():
    out = vf.parse_guest_output(
        json.dumps({"guest_s": 12.5, "method": "base64", "marker": "V86FACTOR_OK 42"})
    )
    assert out["guest_s"] == 12.5
    assert out["method"] == "base64"


def test_parse_guest_output_rejects_bad_json():
    with pytest.raises(ValueError):
        vf.parse_guest_output("not json at all")
    with pytest.raises(ValueError):
        vf.parse_guest_output(json.dumps({"guest_s": 12.5}))  # missing method


def test_parse_guest_output_rejects_nonpositive_time():
    with pytest.raises(ValueError):
        vf.parse_guest_output(json.dumps({"guest_s": 0, "method": "base64"}))


# ── report ───────────────────────────────────────────────────────────────────


def test_build_report_full_shape():
    rep = vf.build_report(
        baseline_decode_tps=368.0,
        baseline_source="tests/core-py recorded (Card A, own-docs corpus)",
        native_s=[1.0, 1.1, 1.05],
        guest_s=27.5,
        method="base64",
        slo_tps=3.0,
    )
    assert rep["factor"] == pytest.approx(27.5 / 1.05)  # native = median
    assert rep["native_s_median"] == pytest.approx(1.05)
    assert rep["native_repeats"] == 3
    assert rep["guest_s"] == 27.5
    assert rep["method"] == "base64"
    # estimate scales the bare-metal number down by the factor
    assert rep["v86_decode_tps"] == pytest.approx(368.0 / (27.5 / 1.05))
    assert rep["verdict"]["meets_slo"] in (True, False)
    assert rep["verdict"]["slo_tps"] == 3.0
    assert rep["baseline_decode_tps"] == 368.0


def test_build_report_records_assumption():
    """The report must state what the number IS — a calibrated estimate.

    The guest cannot run numpy, so the factor comes from a same-binary
    C float workload; applying it to decode is an assumption the report
    carries explicitly, never silently.
    """
    rep = vf.build_report(
        baseline_decode_tps=368.0,
        baseline_source="x",
        native_s=[1.0],
        guest_s=10.0,
        method="base64",
        slo_tps=3.0,
    )
    assert "calibrated" in rep["assumption"].lower()
    assert rep["workload"]  # the C workload is named


def test_build_report_rejects_empty_native():
    with pytest.raises(ValueError):
        vf.build_report(
            baseline_decode_tps=368.0,
            baseline_source="x",
            native_s=[],
            guest_s=10.0,
            method="base64",
            slo_tps=3.0,
        )


# ── workload compile ─────────────────────────────────────────────────────────


def test_compile_workload_produces_static_binary(tmp_path):
    """The guest has no glibc — the workload must be a static binary, and
    the same binary runs both sides so the factor is apples-to-apples."""
    binary = tmp_path / "bm"
    out = vf.compile_workload(binary, cc="gcc")
    assert Path(out).exists() and Path(out).stat().st_size > 0
    assert not vf._is_dynamically_linked(out)  # static, guest-runnable


def test_compile_workload_missing_cc(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        vf.compile_workload(tmp_path / "bm", cc="definitely-not-a-compiler")


# ── guest driver invocation (contract, not a boot) ───────────────────────────


def test_guest_driver_args_contract():
    """The python side builds the node driver's argv from explicit inputs —
    asserted so a flag rename on either side fails loudly here."""
    argv = vf.guest_driver_argv(
        node="node",
        driver=str(Path("scripts") / "v86_factor_guest.mjs"),
        wasm="/w/v86.wasm",
        kernel="/k/bzimage68.bin",
        bios="/b/seabios.bin",
        vga_bios="/b/vgabios.bin",
        binary="/tmp/bm",
        timeout_s=300,
    )
    assert argv[0] == "node"
    assert "--wasm" in argv and "/w/v86.wasm" in argv
    assert "--kernel" in argv and "/k/bzimage68.bin" in argv
    assert "--bios" in argv
    assert "--vga-bios" in argv
    assert "--binary" in argv and "/tmp/bm" in argv
    assert "--timeout" in argv and "300" in argv


def test_compile_workload_sse2_variant(tmp_path):
    """The guest has SSE2 (probe: cpuflags sse2) — a -msse2 build must
    also compile static. This is the instruction set an in-guest decode
    would realistically be compiled for; x87 is the conservative
    baseline. The verdict must not hinge on an unmeasured choice."""
    binary = tmp_path / "bm_sse2"
    out = vf.compile_workload(binary, cc="gcc", sse2=True)
    assert out.exists() and out.stat().st_size > 0
    assert not vf._is_dynamically_linked(out)
