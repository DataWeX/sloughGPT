#!/usr/bin/env python3
"""Track VM/device capability coverage by running programs against the VM.

The compat matrix answers one question per run: *which standard contracts can
third-party code actually drive today, and what blocks the rest?*

Tier ladder (per matrix row):
  control - in-house x86 programs (ShellREPL.HELLO_X86, test_syscalls)
            proving harness health
  tier1   - third-party source assembled + ran on our VM (ok/budget verdict)
  tier2   - third-party source blocked at intake or runtime (named-gap row)

Verdicts:
  ok        - assembled, ran, exited (exit_code = EAX at HLT)
  budget    - assembled, ran, stopped at the step budget (loop/interactive)
  asm_gap   - X86Assembler rejected the source (reason = first error)
  fault     - VM fault ended the run (reason = fault class or captured log)
  error     - harness/host problem (spawn failure, unexpected exception)

Coverage signals collected per run (see scripts/compat/manifest.json):
  syscalls        int 0x80 dispatch counts (RBAC denials counted separately)
  unhandled_irqs  interrupts invoked with NO registered handler
                  (this is how BIOS INT 10h/16h/13h gaps show up: the CPU
                  treats missing vectors as silent no-ops)
  vga             byte diff of the 0xB8000 text buffer
  stdout          SYS_WRITE fd1/2 capture (the `vmrun` pattern)

Usage:
  python scripts/compat_track.py                          # full run, writes matrix
  python scripts/compat_track.py --only hello             # single entry filter
  python scripts/compat_track.py --steps 300000           # per-entry step budget
  python scripts/compat_track.py --json X.json --md X.md  # redirect artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "apps" / "api" / "server"))

CORPUS_DIR = REPO / "scripts" / "compat" / "corpus"
MANIFEST_PATH = REPO / "scripts" / "compat" / "manifest.json"
DEFAULT_JSON = REPO / "scripts" / "compat" / "matrix.json"
DEFAULT_MD = REPO / "docs" / "VM_COMPAT.md"

VERDICTS = ("ok", "budget", "asm_gap", "fault", "error")


@dataclass
class Row:
    name: str
    tier: str  # control | tier1 | tier2
    source: str  # corpus file or registry key
    verdict: str = ""
    reason: str = ""
    asm_ms: float = 0.0
    asm_bytes: int = 0
    wall_ms: float = 0.0
    steps: int = 0
    exit_code: int | None = None
    syscalls: dict = field(default_factory=dict)  # syscall name -> count
    denials: int = 0
    port_io: dict = field(default_factory=dict)  # "0x3F8" -> in/out count
    unhandled_irqs: dict = field(default_factory=dict)  # "INT 0x10" -> count
    vga_bytes_changed: int = 0
    stdout_lines: int = 0
    stdout_head: str = ""
    asm_error_line: int | None = None  # 1-based source line of an asm_gap
    asm_error_src: str = ""  # that line's text (trimmed)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def syscall_names(manifest: dict) -> dict[int, str]:
    # EXIT=1 .. UNLINK=31 (array index + 1 == syscall number)
    return {i + 1: name for i, name in enumerate(manifest["abi"]["syscalls"])}


def _asm_error(src: str, org: int) -> tuple[str, int, str] | None:
    """Try to assemble; on failure return (message, line_no, line_text).

    X86Assembler raises without line context, so we bisect on line prefixes:
    the first error dominates, so "prefix fails with the identical error" is
    monotone and binary search lands on the offending line.
    """
    from domain.shell._internal.vm import X86Assembler

    try:
        X86Assembler().assemble(src, org=org)
        return None
    except TypeError:
        try:
            X86Assembler().assemble(src)
            return None
        except Exception as exc:
            full = (type(exc).__name__, str(exc))
    except Exception as exc:
        full = (type(exc).__name__, str(exc))

    def err_at(k: int) -> tuple[str, int, str] | None:
        prefix = "\n".join(src.splitlines()[:k])
        try:
            asm = X86Assembler()
            try:
                asm.assemble(prefix, org=org)
            except TypeError:
                asm.assemble(prefix)
            return None
        except Exception as exc:
            return (type(exc).__name__, str(exc), prefix)

    # Any error persists as lines are added (errors are line-local), so
    # binary search on "prefix fails" finds the FIRST failing line.
    lines = src.splitlines()
    lo, hi = 1, len(lines)  # invariant: err_at(hi) is not None
    while lo < hi:
        mid = (lo + hi) // 2
        if err_at(mid) is not None:
            hi = mid
        else:
            lo = mid + 1
    got = err_at(lo)
    if got is not None:
        return (f"{got[0]}: {got[1]}", lo, lines[lo - 1].strip())
    return (f"{full[0]}: {full[1]}", 0, "")


def probe_assemble(src: str, org: int = 0x100000) -> tuple[bool, float, int, str, int, str]:
    """Assemble-only probe so asm gaps are attributed before any run starts.

    Returns (ok, asm_ms, size, error, error_line, error_line_text).
    """
    from domain.shell._internal.vm import X86Assembler

    asm = X86Assembler()
    t0 = time.perf_counter()
    try:
        res = asm.assemble(src, org=org)
    except TypeError:
        res = asm.assemble(src)  # older signature without org
    except Exception:
        ms = (time.perf_counter() - t0) * 1000.0
        found = _asm_error(src, org)
        if found:
            msg, ln, text = found
            return False, ms, 0, f"{msg} @ line {ln}" if ln else msg, ln, text
        return False, ms, 0, "assemble failed (unattributed)", 0, ""
    code = res[0] if isinstance(res, tuple) else res
    return True, (time.perf_counter() - t0) * 1000.0, len(code), "", 0, ""


def _capture_stdout(vs) -> tuple[list[str], object]:
    """Capture SYS_WRITE to fd 1/2 (the canonical `vmrun` pattern)."""
    chunks: list[str] = []
    orig_write = vs._syscall._sys_write

    def cap_write(fd, buf_addr, count):
        if fd in (1, 2):
            data = bytes(vs.cpu._read8(buf_addr + i) for i in range(count))
            chunks.append(data.decode("ascii", errors="replace"))
            return count
        return orig_write(fd, buf_addr, count)

    vs._syscall._sys_write = cap_write
    return chunks, orig_write


class _ErrCap(logging.Handler):
    """Capture CPU-fault log records (the CPU swallows non-InsFault errors)."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record) -> None:
        self.messages.append(record.getMessage())


def run_program(src: str, name: str, manifest: dict, budget: int) -> Row:
    """Spawn + run one program on X86VirtualSystem with full instrumentation."""
    from domain.shell._internal.vm import VMFault, X86VirtualSystem
    from domain.shell._internal.vm_permissions import Role

    names = syscall_names(manifest)
    vs = X86VirtualSystem()

    pid = vs.spawn(name, src)
    if pid is None:
        return Row(name=name, tier="", source="", verdict="error", reason="spawn returned None")
    vs._syscall._rbac.assign(pid, Role.USER)

    syscall_counts: Counter = Counter()
    unhandled: Counter = Counter()
    denials = 0

    # -- instrumentation (all restored in `finally`) ------------------------
    orig_handle = vs._syscall.handle

    def counting_handle():
        syscall_counts[vs.cpu._regs[0] & 0xFFFFFFFF] += 1
        return orig_handle()

    vs._syscall.handle = counting_handle

    orig_perm = vs._syscall._check_perm

    def counting_perm(scn):
        nonlocal denials
        ok = orig_perm(scn)
        if not ok:
            denials += 1
        return ok

    vs._syscall._check_perm = counting_perm

    # Interrupt spy: unregistered vectors are silent no-ops in the CPU, so
    # counting them is the only way to evidence missing BIOS/IRQ contracts.
    orig_raise = vs.cpu._raise_interrupt

    def spy_raise(int_num):
        if int_num not in vs.cpu._idt_handlers:
            unhandled[int_num] += 1
        return orig_raise(int_num)

    vs.cpu._raise_interrupt = spy_raise

    # Port-I/O spy: direct `out dx` traffic never touches a syscall, so
    # UART/PIT-class contracts are observed by wrapping the CPU's port
    # methods (catches every port, handled or not).
    port_counts: Counter = Counter()
    orig_port_out = vs.cpu._port_out
    orig_port_in = vs.cpu._port_in

    def spy_port_out(port, val):
        port_counts[port] += 1
        return orig_port_out(port, val)

    def spy_port_in(port):
        port_counts[port] += 1
        return orig_port_in(port)

    vs.cpu._port_out = spy_port_out
    vs.cpu._port_in = spy_port_in

    stdout_chunks, orig_write = _capture_stdout(vs)
    err_cap = _ErrCap()
    root = logging.getLogger()
    root.addHandler(err_cap)

    mem = vs.cpu._mem
    vga_before = bytes(mem[0xB8000 : 0xB8000 + 4000])

    row = Row(name=name, tier="", source="")
    steps_done = 0
    fatal: str | None = None
    try:
        vs.scheduler.start(vs.cpu)
        current = vs.scheduler.current
        if current is None:
            fatal = "scheduler has no current process"
        else:
            current.restore_to_cpu(vs.cpu)
            t0 = time.perf_counter()
            try:
                steps_done = vs.cpu.run(max_steps=budget)
            except BaseException as exc:  # VMFault subclasses propagate
                fatal = (
                    f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, VMFault)
                    else f"harness: {type(exc).__name__}: {exc}"
                )
            finally:
                row.wall_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        root.removeHandler(err_cap)
        vs._syscall._sys_write = orig_write
        vs._syscall.handle = orig_handle
        vs._syscall._check_perm = orig_perm
        vs.cpu._raise_interrupt = orig_raise
        vs.cpu._port_out = orig_port_out
        vs.cpu._port_in = orig_port_in

    row.steps = steps_done
    vga_after = bytes(mem[0xB8000 : 0xB8000 + 4000])
    row.vga_bytes_changed = sum(1 for a, b in zip(vga_before, vga_after) if a != b)

    row.syscalls = {names.get(n, f"SYS_{n}"): c for n, c in sorted(syscall_counts.items())}
    row.denials = denials
    row.port_io = {f"0x{p:03X}": c for p, c in sorted(port_counts.items())}
    row.unhandled_irqs = {f"INT 0x{n:02X}": c for n, c in sorted(unhandled.items())}
    out = "".join(stdout_chunks)
    row.stdout_lines = out.count("\n") + (1 if out and not out.endswith("\n") else 0)
    row.stdout_head = out[:200].replace("\n", "\\n")

    if fatal is not None:
        row.verdict = "fault" if not fatal.startswith("harness:") else "error"
        row.reason = fatal[:300]
    elif err_cap.messages:
        row.verdict = "fault"
        row.reason = err_cap.messages[0][:300]
    elif steps_done >= budget:
        row.verdict = "budget"
        row.reason = f"step budget {budget} reached (interactive/loop program)"
    else:
        row.verdict = "ok"
        row.exit_code = vs.cpu._regs[0] & 0xFFFFFFFF
    return row


def tier_for(row: Row, is_control: bool) -> str:
    if is_control:
        return "control"
    return "tier1" if row.verdict in ("ok", "budget") else "tier2"


def corpus_entries(only: str | None) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    if CORPUS_DIR.is_dir():
        for path in sorted(CORPUS_DIR.glob("*.asm")):
            entries.append((path.stem, path))
    if only:
        entries = [e for e in entries if only in e[0]]
    return entries


def control_entries(only: str | None) -> list[tuple[str, str]]:
    """Harness-health controls: x86 sources that must assemble and run.

    `hello_x86` is ShellREPL.HELLO_X86 (SYS_WRITE -> SYS_EXIT, used by
    `vmrun hello`); `test_syscalls` exercises the INT 0x80 surface.
    Legacy inner-ISA programs (LOAD_CONST/PRINT) are deliberately NOT
    controls — they are a different machine than X86VirtualSystem.
    """
    from domain.shell._internal.repl import ShellREPL
    from domain.shell._internal.vm_programs import PROGRAMS

    controls: list[tuple[str, str]] = []
    if getattr(ShellREPL, "HELLO_X86", None):
        controls.append(("hello_x86", ShellREPL.HELLO_X86))
    if "test_syscalls" in PROGRAMS:
        controls.append(("test_syscalls", PROGRAMS["test_syscalls"]))
    if "cave_game" in PROGRAMS:
        # renders to the VGA text buffer and writes the serial port directly,
        # proving the memory-diff and port-I/O observation paths
        controls.append(("cave_game", PROGRAMS["cave_game"]))
    if only:
        controls = [c for c in controls if only in c[0]]
    return controls


def collect(manifest: dict, budget: int, only: str | None) -> list[Row]:
    rows: list[Row] = []

    for key, src in control_entries(only):
        rows.append(_entry(key, src, "control", manifest, budget))

    for stem, path in corpus_entries(only):
        rows.append(_entry(stem, path.read_text(), None, manifest, budget, source=path.name))

    return rows


def _entry(
    name: str, src: str, forced_tier: str | None, manifest: dict, budget: int, source: str = ""
) -> Row:
    ok, asm_ms, asm_bytes, err, err_line, err_src = probe_assemble(src)
    if not ok:
        row = Row(
            name=name,
            tier="",
            source=source or name,
            verdict="asm_gap",
            reason=err,
            asm_ms=round(asm_ms, 3),
            asm_error_line=err_line or None,
            asm_error_src=err_src[:120],
        )
        row.tier = tier_for(row, forced_tier == "control")
        return row

    row = run_program(src, name, manifest, budget)
    row.source = source or name
    row.asm_ms = round(asm_ms, 3)
    row.asm_bytes = asm_bytes
    row.tier = tier_for(row, forced_tier == "control")
    if forced_tier == "control" and row.verdict not in ("ok", "budget"):
        # a broken control is a harness-health signal, not a tier fact
        row.reason = f"[control-health] {row.reason}"
    return row


def coverage_summary(rows: list[Row], manifest: dict) -> dict:
    names = syscall_names(manifest)
    seen: set[str] = set()
    for r in rows:
        seen.update(r.syscalls)
    syscalls_obs = sorted(seen)
    syscalls_unobs = sorted(n for n in names.values() if n not in seen)

    unhandled: Counter = Counter()
    for r in rows:
        for k, v in r.unhandled_irqs.items():
            unhandled[k] += v

    devices = []
    ports_seen: set = set()
    for r in rows:
        ports_seen.update(k.upper() for k in r.port_io)
    for dev in manifest["devices"]:
        obs = dev.get("observe") or []
        hit = False
        how = []
        for o in obs:
            if o.startswith("syscall:"):
                sn = o.split(":", 1)[1]
                if sn in seen:
                    hit = True
                    how.append(f"syscall:{sn}")
            elif o.startswith("port:"):
                port = o.split(":", 1)[1].upper()
                if port in ports_seen:
                    hit = True
                    how.append(o)
            elif o == "memory-diff" and any(r.vga_bytes_changed for r in rows):
                hit = True
                how.append("memory-diff")
            elif o == "interrupt-spy" and unhandled:
                hit = True
                how.append("interrupt-spy")
            elif o == "always-on":
                how.append("always-on")
            elif o == "engine-tier":
                how.append("engine-tier")
        devices.append(
            {
                "contract": dev["contract"],
                "status": dev.get("status") or ("wired" if dev.get("wired") else "standalone"),
                "observed": hit,
                "via": how,
            }
        )

    gaps = []
    for r in rows:
        if r.verdict in ("asm_gap", "fault", "error"):
            gaps.append({"entry": r.name, "verdict": r.verdict, "reason": r.reason})
        elif r.verdict == "budget" and r.tier != "control":
            gaps.append(
                {
                    "entry": r.name,
                    "verdict": "budget",
                    "reason": "no input injection; program loops until the step budget",
                }
            )

    return {
        "syscalls_observed": syscalls_obs,
        "syscalls_unobserved": syscalls_unobs,
        "unhandled_interrupts": dict(unhandled),
        "devices": devices,
        "gaps": gaps,
    }


def render_md(manifest: dict, rows: list[Row], cov: dict, budget: int) -> str:
    lines = [
        "# VM compat matrix — third-party programs on the x86 VM",
        "",
        "*Generated by `scripts/compat_track.py` — do not hand-edit the table.*",
        "",
        "The VM tracks its emulator-readiness by running outside programs against",
        "**standard device contracts** (see `scripts/compat/manifest.json`). Each",
        "row is one program: in-house controls prove the harness, third-party",
        "sources probe the intake path. A failing row is a *finding*, not a test",
        "failure — its reason becomes a kanban gap card.",
        "",
        f"Run: `python scripts/compat_track.py` (per-entry step budget {budget:,})",
        "",
        "## Tier ladder",
        "",
        "| tier | meaning |",
        "|------|---------|",
        "| control | in-house program; harness health |",
        "| tier1 | third-party source assembled **and ran** on our VM |",
        "| tier2 | third-party source blocked — reason below is the named gap |",
        "",
        "## Matrix",
        "",
        "| entry | tier | verdict | steps | wall ms | syscalls (hits) | unhandled ints | VGA bytes | reason / evidence |",
        "|-------|------|---------|------:|--------:|-----------------|----------------|----------:|-------------------|",
    ]
    for r in rows:
        sc = ", ".join(f"{k}:{v}" for k, v in list(r.syscalls.items())[:6]) or "-"
        if len(r.syscalls) > 6:
            sc += f" (+{len(r.syscalls) - 6})"
        if r.port_io:
            sc += "; " + ", ".join(f"out {k}:{v}" for k, v in list(r.port_io.items())[:4])
        unh = ", ".join(f"{k}:{v}" for k, v in r.unhandled_irqs.items()) or "-"
        reason = (r.reason or r.stdout_head)[:110].replace("|", "/")
        lines.append(
            f"| {r.name} | {r.tier} | {r.verdict} | {r.steps:,} | {r.wall_ms:,.0f} "
            f"| {sc} | {unh} | {r.vga_bytes_changed:,} | {reason} |"
        )

    sys_total = len(cov["syscalls_observed"]) + len(cov["syscalls_unobserved"])
    unobs = (
        f" — never observed: {', '.join(cov['syscalls_unobserved'])}"
        if cov["syscalls_unobserved"]
        else ""
    )

    lines += [
        "",
        "## Coverage summary",
        "",
        f"- **syscalls observed**: {len(cov['syscalls_observed'])}/{sys_total}{unobs}",
        "- **unhandled interrupts invoked**: "
        + (", ".join(f"{k} (x{v})" for k, v in cov["unhandled_interrupts"].items()) or "none"),
        "",
        "| device contract | status | observed | via |",
        "|------------------|--------|----------|-----|",
    ]
    for d in cov["devices"]:
        lines.append(
            f"| {d['contract']} | {d['status']} | "
            f"{'yes' if d['observed'] else 'no'} | {', '.join(d['via']) or '-'} |"
        )

    lines += ["", "## Named gaps (→ kanban cards)", ""]
    if cov["gaps"]:
        for g in cov["gaps"]:
            lines.append(f"- **{g['entry']}** [{g['verdict']}]: {g['reason']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--steps", type=int, default=300_000, help="per-entry step budget (default 300000)"
    )
    ap.add_argument("--only", default=None, help="substring filter on entry names")
    ap.add_argument("--json", dest="json_path", default=str(DEFAULT_JSON))
    ap.add_argument("--md", dest="md_path", default=str(DEFAULT_MD))
    ap.add_argument("--no-write", action="store_true", help="print only, write nothing")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    rows = collect(manifest, args.steps, args.only)
    cov = coverage_summary(rows, manifest)

    payload = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest_version": manifest["version"],
        "steps_budget": args.steps,
        "rows": [asdict(r) for r in rows],
        "coverage": cov,
    }

    if not args.no_write:
        Path(args.json_path).write_text(json.dumps(payload, indent=2) + "\n")
        Path(args.md_path).write_text(render_md(manifest, rows, cov, args.steps))

    for r in rows:
        print(
            f"{r.tier:8s} {r.name:24s} {r.verdict:8s} "
            f"steps={r.steps:<9,d} {r.wall_ms:8.0f}ms  {r.reason[:70]}"
        )
    tail = f"gaps: {len(cov['gaps'])} | syscalls observed: {len(cov['syscalls_observed'])}/31"
    if not args.no_write:
        tail += f" | written: {args.json_path}, {args.md_path}"
    print(f"\n{tail}")

    if not rows:
        print("no entries matched", file=sys.stderr)
        return 1
    # Exit code reflects HARNESS health (controls), not third-party gaps:
    # findings are the product of this tool, not failures.
    controls = [r for r in rows if r.tier == "control"]
    if args.only is None and any(r.verdict not in ("ok", "budget") for r in controls):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
