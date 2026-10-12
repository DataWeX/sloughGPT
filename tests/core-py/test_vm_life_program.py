"""life — Conway's Game of Life builtin for the x86 VM.

Runner contract (same as cave_game): [BITS 32], no ORG directive, zero
syscalls — raw VGA text-buffer pokes at 0xB8000 plus the BIOS keyboard byte
at [0x400], hlt on exit.  The same source therefore runs on the bare
VMEngine (CLI ``vm run life`` / shell ``vmrun``) and on X86VirtualSystem
(API POST /vm/run, console session).

The startup prompt asks for the initial live-cell count (max 1200 = the
whole 60x20 board) and VALIDATES it: anything outside 1..1200, any
non-digit, or an empty entry prints an error and re-prompts.  That is the
defect the user's screenshot shows in the third-party MikeOS life.bin:
type 1212 at its "Enter number of lifes(max=1200): " prompt and the
screen dies.

Batch mode (no key ever typed) auto-starts with DEFAULT_LIVES cells and
halts after BATCH_GENS generations, so headless runs terminate.
"""

from __future__ import annotations

import pytest

from domain.shell._internal.vm import X86Assembler, X86VirtualSystem
from domain.shell._internal.vm_permissions import Role
from domain.shell._internal.vm_programs import LIFE_ASM, PROGRAMS

GRID_W, GRID_H = 60, 20
GRID_CELLS = GRID_W * GRID_H  # 1200 — also the maximum live count
PROMPT = "Enter number of lifes(max=1200): "
DEFAULT_LIVES = 240
BATCH_GENS = 4

ROW2 = 0xB8000 + 2 * 160  # prompt + digit echo
ROW3 = 0xB8000 + 3 * 160  # error line
ROW21 = 0xB8000 + 21 * 160  # HUD
ECHO = ROW2 + len(PROMPT) * 2

_LABELS_CACHE: dict[str, int] | None = None


def _labels() -> dict[str, int]:
    """Label table for the canonical spawn org (0x100000) — cached (assembly is slow)."""
    global _LABELS_CACHE
    if _LABELS_CACHE is None:
        asm = X86Assembler()
        asm.assemble(LIFE_ASM, org=0x100000)
        _LABELS_CACHE = dict(asm._labels)
    return _LABELS_CACHE


def _text(mem, addr: int, cells: int) -> str:
    raw = bytes(mem[addr : addr + cells * 2])
    return "".join(chr(raw[i * 2]) if 32 <= raw[i * 2] < 127 else " " for i in range(cells))


def _dword(vs: X86VirtualSystem, addr: int) -> int:
    m = vs.cpu._mem
    return m[addr] | (m[addr + 1] << 8) | (m[addr + 2] << 16) | (m[addr + 3] << 24)


def _spawn() -> X86VirtualSystem:
    vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
    pid = vs.spawn("life", LIFE_ASM)
    assert pid is not None, "spawn failed"
    vs._syscall._rbac.assign(pid, Role.USER)
    vs.scheduler.start(vs.cpu)
    vs.scheduler.current.restore_to_cpu(vs.cpu)
    return vs


def _drain(vs: X86VirtualSystem, budget: int = 400_000) -> None:
    """Step until the keyboard byte at [0x400] is consumed (or budget spent).

    Then settle a little: the byte clears inside ``poll_key``, before the
    consuming program finishes acting on it (echoing the digit, updating
    state), so post-consume steps must run before callers inspect memory.
    """
    limit = vs.cpu._step_count + budget
    while vs.cpu._mem[0x400] != 0 and vs.cpu._step_count < limit:
        vs.cpu.step()
    settle = vs.cpu._step_count + 2_000
    while vs.cpu._step_count < settle and vs.cpu.step():
        pass


def _feed(vs: X86VirtualSystem, keys: str, budget: int = 400_000) -> None:
    """Type `keys`, waiting for each byte to be consumed before the next."""
    for ch in keys:
        vs.cpu.push_key(ch)
        vs.cpu.transfer_key()
        _drain(vs, budget)


def _backspace(vs: X86VirtualSystem, budget: int = 400_000) -> None:
    """Backspace has no char mapping — push its raw make-scancode (0x0E -> chr(8))."""
    vs.cpu.push_scancode(0x0E)
    vs.cpu.transfer_key()
    _drain(vs, budget)


def _run_until(vs: X86VirtualSystem, predicate, budget: int = 3_000_000) -> bool:
    """Step until predicate() holds or the budget is spent. True = predicate held."""
    limit = vs.cpu._step_count + budget
    while vs.cpu._step_count < limit:
        if predicate():
            return True
        if not vs.cpu.step():  # halted
            return predicate()
    return predicate()


def _halted(vs: X86VirtualSystem, budget: int = 3_000_000) -> bool:
    """Run to halt; True when the CPU stopped before exhausting the budget."""
    limit = vs.cpu._step_count + budget
    return vs.cpu.run(max_steps=limit) < limit


def _grid(vs: X86VirtualSystem) -> set[tuple[int, int]]:
    cur = _labels()["cur"]
    mem = vs.cpu._mem
    return {(i % GRID_W, i // GRID_W) for i in range(GRID_CELLS) if mem[cur + i]}


def _poke(vs: X86VirtualSystem, cells: list[tuple[int, int]]) -> None:
    """Overwrite the live grid with an exact pattern (rule tests)."""
    cur = _labels()["cur"]
    mem = vs.cpu._mem
    for i in range(GRID_CELLS):
        mem[cur + i] = 0
    for x, y in cells:
        mem[cur + y * GRID_W + x] = 1


def _xorshift(s: int) -> int:
    """Mirror of the program's xorshift32 (shl13, shr17, shl5)."""
    s &= 0xFFFFFFFF
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= s >> 17
    s ^= (s << 5) & 0xFFFFFFFF
    return s & 0xFFFFFFFF


def _reference(n: int, gens: int, seed: int) -> set[tuple[int, int]]:
    """Independent Python model of seeding + toroidal B3/S23."""
    cur = [0] * GRID_CELLS
    s = seed
    placed = 0
    while placed < n:
        s = _xorshift(s)
        idx = s % GRID_CELLS
        if cur[idx] == 0:
            cur[idx] = 1
            placed += 1
    for _ in range(gens):
        nxt = [0] * GRID_CELLS
        for y in range(GRID_H):
            for x in range(GRID_W):
                count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx or dy:
                            count += cur[((y + dy) % GRID_H) * GRID_W + ((x + dx) % GRID_W)]
                alive = cur[y * GRID_W + x]
                nxt[y * GRID_W + x] = int(count == 3 or (alive and count == 2))
        cur = nxt
    return {(i % GRID_W, i // GRID_W) for i, v in enumerate(cur) if v}


class TestLifeAssembly:
    def test_assembles_at_both_loader_orgs(self):
        code_1k = X86Assembler().assemble(LIFE_ASM, org=0x1000)
        code_100k = X86Assembler().assemble(LIFE_ASM, org=0x100000)
        assert len(code_1k) > 2000
        assert len(code_100k) == len(code_1k)

    def test_no_org_directive_in_source(self):
        # loaders must own the org so label addresses match the load address
        # (comment lines are stripped — prose may legitimately mention it)
        code_lines = [
            line for line in LIFE_ASM.splitlines() if not line.lstrip().startswith(";")
        ]
        assert "[ORG" not in "\n".join(code_lines)

    def test_no_syscalls(self):
        # the bare VMEngine has no INT 0x80 handler — the game must not need one
        assert "int 0x80" not in LIFE_ASM.lower()

    def test_registered_in_programs(self):
        assert PROGRAMS.get("life") is LIFE_ASM

    def test_registered_as_api_builtin(self):
        builtins = pytest.importorskip("vm_builtins")  # needs apps/api/server on path
        assert builtins.get_builtin("life") == LIFE_ASM
        assert "life" in builtins.BUILTIN_PROGRAMS

    def test_no_silently_dropped_instructions(self):
        # The assembler currently emits a bare NOP for mnemonic lines it does
        # not recognise (movzx did exactly this — every neighbor load was a
        # no-op).  Every code line of the program must therefore produce real
        # bytes; label/jump/directive lines are checked as a group.
        jumpy = ("jmp", "je ", "jne", "jl ", "jg ", "jge", "jle", "ja ", "jb ",
                 "call", "loop")
        for raw in LIFE_ASM.splitlines():
            line = raw.strip()
            if not line or line.startswith(";") or line.startswith("["):
                continue
            if line.endswith(":"):
                continue
            if any(line.startswith(m) for m in jumpy):
                continue  # forward label refs don't resolve standalone
            if line.split()[0] in ("db", "dd", "times"):
                continue
            code = X86Assembler().assemble(f"[BITS 32]\nstart:\n{line}\n", org=0x100000)
            assert code not in (b"", b"\x90"), f"instruction silently dropped: {line!r}"


class TestPromptPhase:
    def test_banner_and_prompt_drawn(self):
        vs = _spawn()
        vs.cpu.run(max_steps=150_000)
        assert "Conway" in _text(vs.cpu._mem, 0xB8000, 80)
        assert PROMPT in _text(vs.cpu._mem, ROW2, 80)

    def test_digits_echo_at_the_prompt(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "24")
        assert _text(vs.cpu._mem, ECHO, 4) == "24  "


class TestInputValidation:
    """The contract the third-party MikeOS binary lacks: bad input re-prompts."""

    def test_over_max_reprompts_then_accepts_valid(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "1212\r")  # the screenshot's input: over max=1200
        vs.cpu.run(max_steps=60_000)
        assert "Out of range" in _text(vs.cpu._mem, ROW3, 60)
        assert _text(vs.cpu._mem, ECHO, 8) == " " * 8  # entry cleared
        assert "LIVE" not in _text(vs.cpu._mem, ROW21, 80)  # still at the prompt
        _feed(vs, "300\r")
        _run_until(vs, lambda: "LIVE" in _text(vs.cpu._mem, ROW21, 80))
        assert _dword(vs, _labels()["lives"]) == 300

    def test_zero_reprompts(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "0\r")
        vs.cpu.run(max_steps=60_000)
        assert "Out of range" in _text(vs.cpu._mem, ROW3, 60)
        assert "LIVE" not in _text(vs.cpu._mem, ROW21, 80)

    def test_empty_entry_reprompts(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "\r")
        vs.cpu.run(max_steps=60_000)
        assert "Nothing typed" in _text(vs.cpu._mem, ROW3, 60)

    def test_letters_rejected_then_valid_accepted(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "abc")
        vs.cpu.run(max_steps=60_000)
        assert "Digits only" in _text(vs.cpu._mem, ROW3, 60)
        assert _text(vs.cpu._mem, ECHO, 4) == "    "  # letters never echo
        _feed(vs, "42\r")
        _run_until(vs, lambda: _dword(vs, _labels()["lives"]) == 42)

    def test_backspace_edits_the_prefix_exactly(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "12")
        _backspace(vs)  # "12" -> "1"
        _feed(vs, "0\r")  # -> 10, not 120 or 1
        _run_until(vs, lambda: _dword(vs, _labels()["lives"]) == 10)
        assert _dword(vs, _labels()["lives"]) == 10

    def test_max_accepted_fills_the_whole_board(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "1200\r")
        _feed(vs, " ")  # pause before the first generation
        assert _dword(vs, _labels()["lives"]) == 1200
        assert _dword(vs, _labels()["gen"]) == 0
        assert _dword(vs, _labels()["livecount"]) == 1200
        assert len(_grid(vs)) == GRID_CELLS

    def test_typed_count_seeds_exactly_n_cells(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "240\r")
        _feed(vs, " ")  # pause before the first generation
        assert _dword(vs, _labels()["lives"]) == 240
        assert _dword(vs, _labels()["gen"]) == 0
        assert _dword(vs, _labels()["livecount"]) == 240
        assert len(_grid(vs)) == 240

    def test_no_dword_wraparound_on_long_input(self):
        # 12 digits must not overflow a dword into a value that passes the check
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        _feed(vs, "123456789012\r")
        vs.cpu.run(max_steps=60_000)
        assert "Out of range" in _text(vs.cpu._mem, ROW3, 60)
        assert "LIVE" not in _text(vs.cpu._mem, ROW21, 80)


class TestConwayRules:
    """Known patterns, poked straight into the grid, must follow B3/S23."""

    def _live_and_step(self, vs: X86VirtualSystem, cells: list[tuple[int, int]]) -> set:
        """Seed+start, poke `cells`, run exactly one generation, return the grid."""
        _feed(vs, "1\r")
        assert _run_until(vs, lambda: _dword(vs, _labels()["gen"]) >= 1)
        _poke(vs, cells)
        labels = _labels()
        gen0 = _dword(vs, labels["gen"])
        assert _run_until(
            vs, lambda: _dword(vs, labels["gen"]) == gen0 + 1, budget=1_000_000
        )
        return _grid(vs)

    def test_block_is_a_still_life(self):
        vs = _spawn()
        after = self._live_and_step(vs, [(10, 10), (11, 10), (10, 11), (11, 11)])
        assert {(10, 10), (11, 10), (10, 11), (11, 11)} <= after

    def test_blinker_oscillates(self):
        vs = _spawn()
        after = self._live_and_step(vs, [(19, 11), (20, 11), (21, 11)])
        assert {(20, 10), (20, 11), (20, 12)} <= after
        assert (19, 11) not in after and (21, 11) not in after

    def test_lone_cell_dies(self):
        vs = _spawn()
        after = self._live_and_step(vs, [(30, 10)])
        assert (30, 10) not in after

    def test_birth_with_exactly_three_neighbors(self):
        vs = _spawn()
        # L-tromino: the fourth corner of the 2x2 box is born
        after = self._live_and_step(vs, [(5, 5), (6, 5), (5, 6)])
        assert {(5, 5), (6, 5), (5, 6), (6, 6)} <= after


class TestControls:
    def test_pause_step_reseed_quit(self):
        vs = _spawn()
        _feed(vs, "3\r")
        _feed(vs, " ")  # pause before the first generation
        labels = _labels()
        assert _dword(vs, labels["gen"]) == 0
        _feed(vs, " ")  # resume
        assert _run_until(vs, lambda: _dword(vs, labels["gen"]) >= 1)
        _feed(vs, " ")  # pause again
        frozen = _dword(vs, labels["gen"])
        vs.cpu.run(max_steps=vs.cpu._step_count + 200_000)
        assert _dword(vs, labels["gen"]) == frozen  # paused: no evolution
        _feed(vs, "s")  # single step while paused
        assert _run_until(
            vs, lambda: _dword(vs, labels["gen"]) == frozen + 1, budget=600_000
        )
        _feed(vs, "r")  # reseed: generation resets, live count returns to N
        assert _run_until(vs, lambda: _dword(vs, labels["gen"]) == 0, budget=600_000)
        assert _run_until(vs, lambda: _dword(vs, labels["livecount"]) == 3, budget=600_000)
        _feed(vs, "q")
        assert _halted(vs, 1_000_000)


class TestBatchMode:
    def test_no_input_runs_and_halts(self):
        vs = _spawn()
        seed0 = _dword(vs, _labels()["seed"])
        assert seed0 != 0  # xorshift32 must never start at zero
        assert _halted(vs, 4_000_000)
        labels = _labels()
        assert _dword(vs, labels["gen"]) == BATCH_GENS
        assert _dword(vs, labels["lives"]) == DEFAULT_LIVES
        # seeding + rules end-to-end: the on-screen grid equals the model
        assert _grid(vs) == _reference(DEFAULT_LIVES, BATCH_GENS, seed0)

    def test_keyboard_input_only_batch_still_halts(self):
        # CLI/API journey: keyboard_input="30\r" then nothing -> must halt
        vs = _spawn()
        _feed(vs, "30\r")
        assert _halted(vs, 4_000_000)
        assert _dword(vs, _labels()["gen"]) == BATCH_GENS
        assert _dword(vs, _labels()["lives"]) == 30
