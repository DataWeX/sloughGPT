"""Tests for scripts/compat_track.py — the VM compat-matrix harness.

The harness is the measurement instrument for emulator-readiness: these
tests pin its verdict taxonomy, its asm-gap line attribution (how gaps get
named for kanban cards), and a real VM run of the hello control.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import compat_track as ct  # noqa: E402

# ── manifest ────────────────────────────────────────────────────────────────


def test_manifest_syscall_surface_is_complete():
    manifest = ct.load_manifest()
    names = ct.syscall_names(manifest)
    assert len(names) == 31
    assert names[1] == "EXIT" and names[3] == "WRITE" and names[31] == "UNLINK"


def test_manifest_covers_standard_contracts_and_planned_tiers():
    manifest = ct.load_manifest()
    contracts = [d["contract"] for d in manifest["devices"]]
    for standard in (
        "8250 UART serial",
        "8254 PIT system timer",
        "VGA text buffer @0xB8000",
        "PS/2 keyboard",
        "ATA-like block storage",
    ):
        assert standard in contracts
    planned = [d for d in manifest["devices"] if d.get("status") == "planned"]
    assert any("BIOS video" in d["contract"] for d in planned)
    assert any(form["status"] == "planned" for form in manifest["intake"])


# ── assembler probe + gap attribution ───────────────────────────────────────


def test_probe_assemble_accepts_known_good_source():
    ok, ms, size, err, line, text = ct.probe_assemble("[BITS 32]\nstart:\n  nop\n  jmp start\n")
    assert ok and size > 0 and err == "" and line == 0


def test_probe_16bit_forward_branch_now_assembles():
    # Card A regression: pass-1 placeholder rel16 converges (was OverflowError
    # at org 0x100000 — see matrix.json / kanban gap cards for the history).
    src = "[BITS 16]\nstart:\n  call far_fn\nfar_fn:\n  nop\n"
    ok, _, size, err, line, _ = ct.probe_assemble(src)
    assert ok and size > 0 and err == "" and line == 0


def test_probe_attributes_named_gate_to_its_line():
    # Line attribution still works against an open intake gate (bits 64).
    src = "[BITS 16]\nnop\n  bits 64\n  nop\n"
    ok, _, _, err, line, text = ct.probe_assemble(src)
    assert not ok
    assert "BITS 64" in err
    assert line == 3 and "bits" in text


def test_probe_equ_hex_expression_now_assembles():
    # Card C regression: hex-leading equ RHS evaluates (was ValueError).
    ok, _, size, err, line, _ = ct.probe_assemble("A equ 0x8000 + 4\n  nop\n")
    assert ok and size > 0 and err == "" and line == 0


# ── a real run ──────────────────────────────────────────────────────────────


def test_hello_control_runs_and_reports_syscalls():
    from domain.shell._internal.repl import ShellREPL

    manifest = ct.load_manifest()
    row = ct.run_program(ShellREPL.HELLO_X86, "hello_x86", manifest, 50_000)
    assert row.verdict == "ok"
    assert row.exit_code == 0
    assert row.syscalls.get("WRITE") == 1
    assert row.syscalls.get("EXIT") == 1
    assert "Hello from x86 VM!" in row.stdout_head


def test_interactive_program_hits_step_budget():
    from domain.shell._internal.vm_programs import PROGRAMS

    manifest = ct.load_manifest()
    row = ct.run_program(PROGRAMS["test_syscalls"], "test_syscalls", manifest, 5_000)
    assert row.verdict == "budget"
    assert row.steps >= 5_000
    assert row.syscalls, "budget runs must still report observed coverage"


def test_vga_memory_diff_and_port_io_are_observed():
    manifest = ct.load_manifest()
    src = (
        "[BITS 32]\n"
        "mov edi, 0xB8000\n"
        "mov dword [edi], 0x03414141\n"
        "mov dx, 0x3F8\n"
        "mov al, 'H'\n"
        "out dx, al\n"
        "jmp $\n"
    )
    row = ct.run_program(src, "vga_port_probe", manifest, 5_000)
    assert row.vga_bytes_changed > 0, "VGA text-buffer writes must be diffed"
    assert "0x3F8" in row.port_io, "direct serial out must be counted"
    assert row.verdict == "budget"


# ── coverage + rendering ────────────────────────────────────────────────────


def _row(**kw):
    base = dict(
        name="x",
        tier="tier2",
        source="x.asm",
        verdict="asm_gap",
        reason="ValueError: nope @ line 7",
    )
    base.update(kw)
    return ct.Row(**base)


def test_coverage_summary_counts_gaps_and_unobserved_syscalls():
    manifest = ct.load_manifest()
    rows = [
        ct.Row(
            name="ctl",
            tier="control",
            source="ctl",
            verdict="ok",
            syscalls={"EXIT": 1, "WRITE": 2, "SERIAL_WRITE": 1},
        ),
        _row(),
        _row(name="b", verdict="fault", reason="MemFault: bad"),
    ]
    cov = ct.coverage_summary(rows, manifest)
    assert "EXIT" in cov["syscalls_observed"]
    assert "NET_RECV" in cov["syscalls_unobserved"]
    assert len(cov["gaps"]) == 2  # asm_gap + fault; ok controls are not gaps
    assert any(d["contract"] == "8250 UART serial" and d["observed"] for d in cov["devices"])


def test_render_md_emits_matrix_and_gap_sections():
    manifest = ct.load_manifest()
    rows = [ct.Row(name="ctl", tier="control", source="ctl", verdict="ok", steps=8, wall_ms=1.0)]
    cov = ct.coverage_summary(rows, manifest)
    md = ct.render_md(manifest, rows, cov, 300_000)
    assert "| entry | tier | verdict |" in md
    assert "## Named gaps" in md
    assert "## Coverage summary" in md


def test_probe_bweeper_runs_to_budget():
    """Card d475eae9: bweeper must leave the fault verdict.

    It assembles (562 B) and its org is honored, but pre-unification BITS 16
    decode faulted at step 4 (`b8 03 00 cd 10` read as mov eax, imm32).
    After VM-targeted 16-bit emission the interactive main loop should run
    to the step budget instead: verdict `budget` (or `ok` if it halts).
    """
    src = (ct.CORPUS_DIR / "bweeper.asm").read_text()
    row = ct.run_program(src, "bweeper", ct.load_manifest(), budget=20000)
    assert row.verdict in ("budget", "ok"), f"{row.verdict}: {row.reason}"
    assert row.steps > 100
