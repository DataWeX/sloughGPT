"""BIOS service tier (kanban card 508245c0) — INT 10h/16h/1Ah minimal stubs.

Corpus-evidence scoped: every service here has a call site in
scripts/compat/corpus/*.asm (bweeper, nasm_tetris, bootmine). Handlers
register in X86VirtualSystem.__init__ next to INT 0x80 so the compat
harness sees them with no harness change. Video char writes land in the
flat VGA text buffer at 0xB8000 (the harness's vga_bytes_changed window).

Flag returns (CF/ZF) patch the stacked EFLAGS at [esp+8] because the
simulated IRET in _raise_interrupt restores EFLAGS from the stack.
"""

from domain.shell._internal.vm import FLAG_CF, FLAG_ZF, X86Assembler, X86VirtualSystem

_KEY_PROGRAM = "mov ah, 0\nint 0x16\nhlt"
_PEEK_PROGRAM = "mov ah, 0x01\nint 0x16\nhlt"
_CLOCK_PROGRAM = "mov ah, 0\nint 0x1A\nhlt"


def _run(source: str, max_steps: int = 2000) -> X86VirtualSystem:
    """Assemble + execute a tiny program on a fresh virtual system."""
    vs = X86VirtualSystem()
    vs.cpu.load(X86Assembler().assemble(source), org=0)
    vs.cpu.run(max_steps=max_steps)
    return vs


def _run_program_on(vs: X86VirtualSystem, source: str, max_steps: int = 200) -> None:
    vs.cpu.load(X86Assembler().assemble(source), org=0)
    vs.cpu.run(max_steps=max_steps)


class TestBiosRegistration:
    def test_bios_vectors_registered_at_boot(self):
        vs = X86VirtualSystem()
        for vec in (0x10, 0x16, 0x1A):
            assert vec in vs.cpu._idt_handlers, f"INT {vec:#x} not registered"

    def test_dos_vector_stays_unregistered(self):
        # INT 21h is the DOS tier — deliberately silent-unregistered.
        vs = X86VirtualSystem()
        assert 0x21 not in vs.cpu._idt_handlers


class TestInt10Video:
    def test_set_mode_then_get_mode(self):
        vs = _run("mov ax, 0x0003\nint 0x10\nmov ah, 0x0F\nint 0x10\nhlt")
        assert vs.cpu._regs[0] & 0xFF == 3  # AL = current mode
        assert (vs.cpu._regs[0] >> 8) & 0xFF == 80  # AH = screen columns

    def test_mode_set_clears_text_buffer(self):
        vs = X86VirtualSystem()
        vs.cpu._write8(0xB8000, ord("X"))
        # 66-prefixed mov ax,3 (VM 16-bit convention) — the bare
        # `b8 03 00 cd 10` would decode as 32-bit mov eax, imm32 and
        # swallow the int.
        vs.cpu.load(b"\x66\xb8\x03\x00\xcd\x10\xf4", org=0)
        vs.cpu.run(max_steps=100)
        assert vs.cpu._read8(0xB8000) == ord(" ")

    def test_set_and_get_cursor_position(self):
        vs = _run("mov ah, 0x02\nxor bx, bx\nmov dx, 0x050A\nint 0x10\nmov ah, 0x03\nint 0x10\nhlt")
        assert (vs.cpu._regs[2] >> 8) & 0xFF == 5  # DH = row
        assert vs.cpu._regs[2] & 0xFF == 10  # DL = col

    def test_set_and_get_cursor_type(self):
        vs = _run("mov ah, 0x01\nmov cx, 0x0607\nint 0x10\nmov ah, 0x03\nint 0x10\nhlt")
        assert (vs.cpu._regs[1] >> 8) & 0xFF == 6  # CH = start line
        assert vs.cpu._regs[1] & 0xFF == 7  # CL = end line

    def test_telemype_writes_vga_and_advances(self):
        vs = _run("mov ah, 0x0E\nmov al, 'A'\nint 0x10\nmov al, 'B'\nint 0x10\nhlt")
        assert vs.cpu._read8(0xB8000) == ord("A")
        assert vs.cpu._read8(0xB8002) == ord("B")
        assert vs.cpu._read8(0xB8001) == 0x07  # default attribute

    def test_telemype_wraps_at_column_80(self):
        src = [
            "mov ah, 0x02",
            "xor bx, bx",
            "mov dx, 0x004F",
            "int 0x10",
            "mov ah, 0x0E",
            "mov al, 'Z'",
            "int 0x10",
            "mov al, 'Y'",
            "int 0x10",
            "hlt",
        ]
        vs = _run("\n".join(src))
        assert vs.cpu._read8(0xB8000 + 2 * 79) == ord("Z")
        assert vs.cpu._read8(0xB8000 + 2 * 80) == ord("Y")  # wrapped to row 1

    def test_write_char_with_attr_does_not_advance(self):
        # AH=09: AL=char, BL=attr, CX=count — cursor stays put.
        vs = _run("mov ah, 0x09\nmov al, '*'\nmov bl, 0x1F\nmov cx, 3\nint 0x10\nhlt")
        for i in range(3):
            assert vs.cpu._read8(0xB8000 + 2 * i) == ord("*")
            assert vs.cpu._read8(0xB8000 + 2 * i + 1) == 0x1F
        # cursor untouched: a later teletype overwrites cell 0
        vs2 = _run(
            "mov ah, 0x09\nmov al, '*'\nmov bl, 0x1F\nmov cx, 1\nint 0x10\n"
            "mov ah, 0x0E\nmov al, '#'\nint 0x10\nhlt"
        )
        assert vs2.cpu._read8(0xB8000) == ord("#")

    def test_write_char_only_ah0a_keeps_attr(self):
        vs = _run(
            # mode set first: it fills the buffer with attr 0x07, so the
            # "untouched" assertion has a defined baseline
            "mov ax, 0x0003\nint 0x10\nmov ah, 0x0A\nmov al, 'q'\nmov cx, 1\nint 0x10\nhlt"
        )
        assert vs.cpu._read8(0xB8000) == ord("q")
        assert vs.cpu._read8(0xB8001) == 0x07  # attr untouched


class TestInt16Keyboard:
    def test_read_key_pops_queue(self):
        vs = X86VirtualSystem()
        vs._bios.feed_key(ord("w"), 0x11)
        _run_program_on(vs, _KEY_PROGRAM)
        assert vs.cpu._regs[0] & 0xFF == ord("w")  # AL = ASCII
        assert (vs.cpu._regs[0] >> 8) & 0xFF == 0x11  # AH = scan code

    def test_empty_queue_returns_zero(self):
        vs = _run("mov ax, 0xFFFF\nmov ah, 0\nint 0x16\nhlt")
        assert vs.cpu._regs[0] & 0xFFFF == 0

    def test_peek_sets_zf_when_empty(self):
        vs = _run(_PEEK_PROGRAM)
        assert vs.cpu._flag(FLAG_ZF) is True

    def test_peek_clears_zf_and_reports_next_key(self):
        vs = X86VirtualSystem()
        vs._bios.feed_key(ord("q"), 0x10)
        _run_program_on(vs, _PEEK_PROGRAM)
        assert vs.cpu._flag(FLAG_ZF) is False
        assert vs.cpu._regs[0] & 0xFF == ord("q")
        # peek is non-destructive: the key is still in the queue
        vs.cpu._eip = 0
        _run_program_on(vs, _KEY_PROGRAM)
        assert vs.cpu._regs[0] & 0xFF == ord("q")


class TestInt1aClock:
    def test_read_ticks_returns_pit_count(self):
        vs = X86VirtualSystem()
        vs._pit._tick_count = 0x12345678
        _run_program_on(vs, _CLOCK_PROGRAM)
        assert vs.cpu._regs[1] & 0xFFFF == 0x1234  # CX = high word
        assert vs.cpu._regs[2] & 0xFFFF == 0x5678  # DX = low word

    def test_set_clock_is_accepted(self):
        vs = _run("mov ah, 0x01\nmov cx, 1\nmov dx, 2\nint 0x1A\nhlt")
        assert vs.cpu._flag(FLAG_CF) is False


class TestBiosUnknownServices:
    def test_unknown_ah_is_counted_not_faulted(self):
        vs = _run("mov ah, 0x7F\nint 0x10\nhlt")
        assert vs._bios.unknown[0x10] == 1
        assert (vs.cpu._regs[0] >> 8) & 0xFF == 0x7F  # program continued
