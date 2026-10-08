"""cave_game — Minecraft iteration-1 voxel runner for the x86 VM.

The program renders a first-person heightmap raycaster into the 80x25 VGA
text buffer. Runner contract: zero syscalls (raw VGA pokes + keyboard buffer
poll at 0x400 + HLT), no [ORG] directive — so the same source runs on the
bare VMEngine (CLI `vm run cave_game`) and on X86VirtualSystem (shell vmrun,
API /vm/run, web panel).

Batch mode: with no input for 40 frames the game halts on its own, so a
headless run terminates. Any keypress switches to interactive mode where
only 'q' quits.
"""

from __future__ import annotations

import pytest

from domain.shell._internal.vm import X86Assembler, X86VirtualSystem
from domain.shell._internal.vm_permissions import Role
from domain.shell._internal.vm_programs import CAVE_GAME_ASM, PROGRAMS

HUD_ADDR = 0xB8F00  # row 24 = 0xB8000 + 24*160
VIEW_BYTES = 80 * 24 * 2


def _labels() -> dict[str, int]:
    """Label table for the canonical spawn org (0x100000)."""
    asm = X86Assembler()
    asm.assemble(CAVE_GAME_ASM, org=0x100000)
    return asm._labels


def _text(mem, addr: int, cells: int) -> str:
    raw = bytes(mem[addr : addr + cells * 2])
    return "".join(chr(raw[i * 2]) if 32 <= raw[i * 2] < 127 else " " for i in range(cells))


def _spawn() -> X86VirtualSystem:
    vs = X86VirtualSystem(memory_size=4 * 1024 * 1024)
    pid = vs.spawn("cave_game", CAVE_GAME_ASM)
    assert pid is not None, "spawn failed"
    vs._syscall._rbac.assign(pid, Role.USER)
    vs.scheduler.start(vs.cpu)
    vs.scheduler.current.restore_to_cpu(vs.cpu)
    return vs


def _hud(vs: X86VirtualSystem) -> str:
    return _text(vs.cpu._mem, HUD_ADDR, 80)


def _view(vs: X86VirtualSystem) -> str:
    return _text(vs.cpu._mem, 0xB8000, 80 * 24)


def _press(vs: X86VirtualSystem, key: str, budget: int = 200_000) -> bool:
    """Feed one key and run; returns True if the game halted early."""
    vs.cpu.push_key(key)
    vs.cpu.transfer_key()
    limit = vs.cpu._step_count + budget
    return vs.cpu.run(max_steps=limit) < limit


class TestCaveGameAssembly:
    def test_assembles_at_both_loader_orgs(self):
        code_1k = X86Assembler().assemble(CAVE_GAME_ASM, org=0x1000)
        code_100k = X86Assembler().assemble(CAVE_GAME_ASM, org=0x100000)
        assert len(code_1k) > 1000
        assert len(code_100k) == len(code_1k)

    def test_no_org_directive_in_source(self):
        # loaders must own the org so label addresses match the load address
        assert "[ORG" not in CAVE_GAME_ASM

    def test_no_syscalls(self):
        # the bare VMEngine has no INT 0x80 handler — the game must not need one
        assert "int 0x80" not in CAVE_GAME_ASM.lower()

    def test_registered_in_programs(self):
        assert PROGRAMS.get("cave_game") is CAVE_GAME_ASM

    def test_registered_as_api_builtin(self):
        builtins = pytest.importorskip("vm_builtins")  # needs apps/api/server on path
        assert builtins.get_builtin("cave_game") == CAVE_GAME_ASM
        assert "cave_game" in builtins.BUILTIN_PROGRAMS


class TestWorldGen:
    def test_borders_walls_and_spawn_clearing(self):
        vs = _spawn()
        vs.cpu.run(max_steps=60_000)
        hm = _labels()["heightmap"]
        mem = vs.cpu._mem
        assert mem[hm] == 12  # top-left corner wall
        assert mem[hm + 31] == 12  # top-right
        assert mem[hm + 992] == 12  # bottom-left
        assert mem[hm + 1023] == 12  # bottom-right
        assert mem[hm + 10 * 32 + 10] == 2  # spawn clearing
        for i in range(1, 31):  # inner rows stay walkable (0..5)
            for j in range(1, 31):
                assert mem[hm + i * 32 + j] <= 5


class TestCaveGameRuntime:
    def test_renders_first_frame(self):
        vs = _spawn()
        vs.cpu.run(max_steps=120_000)
        hud = _hud(vs)
        assert "CRAFT x=10 y=10 a=00 m=00 p=00" in hud
        assert "w/s mv a/d turn m/p dig q quit" in hud
        view = _view(vs)
        # terrain columns ('0'..'5') and sky (blank top rows) both present
        assert any(ch in "012345" for ch in view)
        assert view[:80].strip() == ""  # top row is sky

    def test_turn_key_changes_heading(self):
        vs = _spawn()
        vs.cpu.run(max_steps=120_000)
        assert "a=00" in _hud(vs)
        _press(vs, "d")  # turn must NOT halt the game
        assert " a=01 " in _hud(vs)

    def test_forward_key_steps_half_cell(self):
        vs = _spawn()
        vs.cpu.run(max_steps=120_000)
        _press(vs, "w")
        hud = _hud(vs)
        assert "x=11" in hud and "y=10" in hud

    def test_place_then_mine_updates_counters(self):
        vs = _spawn()
        vs.cpu.run(max_steps=120_000)
        _press(vs, "p")
        assert " p=01 " in _hud(vs)
        _press(vs, "m")
        hud = _hud(vs)
        assert " m=01 " in hud and " p=01 " in hud

    def test_quit_key_halts(self):
        vs = _spawn()
        vs.cpu.run(max_steps=120_000)
        assert _press(vs, "q") is True

    def test_batch_idle_exit(self):
        """No input ever -> HLT after 40 frames (headless CLI termination)."""
        vs = _spawn()
        idle_addr = _labels()["idle"]
        vs.cpu._mem[idle_addr : idle_addr + 4] = (39).to_bytes(4, "little")
        limit = 150_000  # one frame (~40k steps) then the idle check trips
        assert vs.cpu.run(max_steps=limit) < limit

    def test_moves_blocked_by_border_wall(self):
        vs = _spawn()
        px_addr = _labels()["px"]
        # stand at x=30.75 facing +x (ang 0); border wall (height 12) at x=31
        vs.cpu._mem[px_addr : px_addr + 4] = (0x1EC000).to_bytes(4, "little")
        vs.cpu.run(max_steps=120_000)
        before = bytes(vs.cpu._mem[px_addr : px_addr + 4])
        _press(vs, "w")
        assert bytes(vs.cpu._mem[px_addr : px_addr + 4]) == before  # wall blocked
        _press(vs, "a")
        assert " a=63 " in _hud(vs)  # left turn: 0 -> 63, input loop was live


class TestBareVMEnginePath:
    """The CLI (`vm run cave_game`) runs on VMEngine — no syscall handler."""

    def test_renders_on_vmengine(self):
        from domain.shell._internal.vm_engine import VMEngine

        eng = VMEngine(memory_size=0x400000)
        eng.load_source(CAVE_GAME_ASM, org=0x1000)
        eng.run(max_steps=120_000)
        hud = _text(eng._cpu._mem, HUD_ADDR, 80)
        assert "CRAFT x=10" in hud
        assert any(ch in "012345#" for ch in _text(eng._cpu._mem, 0xB8000, 80 * 24))
