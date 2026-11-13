"""
Terminal session protocol — one owner for stdin, one decoder for key sequences.

Two problems motivated this module, and both were caused by the same mistake:
asking the terminal what it is instead of being told.

`InteractivePrompt` decided whether it could do raw input by probing
`os.isatty(sys.stdin.fileno())`. A curses session *is* a tty, so in TUI mode the
prompt layer concluded it owned stdin, put the terminal into raw mode behind
curses' back, and called `os.read` on the same descriptor the event loop was
already polling. Two readers, one descriptor, keystrokes destroyed. Probing
cannot distinguish "curses owns this descriptor" from "nobody owns it" — there
is no probe that answers that. So ownership is declared here instead: a session
either owns stdin or it does not, and `InteractiveBase` asks the session rather
than the terminal.

The second problem is that the byte-level decoder lived in two places. The line
mode had `_read_raw_key`, a 125-line CSI parser over `os.read`. The TUI had
`_read_escape_remainder` plus a chain of `ch in (curses.KEY_UP, ...)` tests,
returning action strings like ``alt:f`` and ``seq:ctrl-left``. Both decoded the
same wire format into different vocabularies, so `prompt_select` (curses) and
`_select_raw` (line mode) implemented the same widget against incompatible key
types and could never be swapped. The invariant fixed here is that both backends
emit the *same* `Key` type. The parser itself stays backend-specific — curses
does that work in C and hands over `KEY_UP` already decoded — but the
normalisation is shared, and `from_curses` is the table that does it.

The protocol, in full:

    owns_stdin          declaration, not a probe
    read_key(timeout)   blocking (or bounded) read, returns Key or None on timeout
    raw()               context manager for raw-mode entry/exit
    size()              (rows, cols)
    write(text)         direct output

`TtySession` is the line-mode implementation and is the only thing in the tree
that calls `os.read` on a terminal. `CursesSession` lives in
`tui_session_curses.py` so that this module stays free of curses, which keeps it
importable — and testable — on a machine with no terminal at all.

Blocking is deliberate. Turning the prompts into non-blocking state machines
would buy nothing here: the TUI already runs each command on its own thread and
already blocks it on an Event, exactly as `TuiRepl.prompt_select` does. That
shape is kept, so the eventual move to a different TUI library is a change of
renderer rather than a change of control flow.

Usage:
    from domain.shell._internal.tui_session import TtySession
    session = TtySession()
    with session.raw():
        key = session.read_key(timeout=1.0)
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Callable
from typing import Protocol

# ── Key vocabulary ───────────────────────────────────────────────────────────
# These values are the contract between a backend and the prompt layer. They are
# moved here, not renamed: interactive_term re-exports them, so no prompt method
# changes a single comparison.

KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENTER = "enter"
KEY_ESC = "esc"
KEY_BACKSPACE = "backspace"
KEY_DELETE = "delete"
KEY_HOME = "home"
KEY_END = "end"
KEY_CTRL_C = "ctrl_c"
KEY_CTRL_A = "ctrl_a"
KEY_CTRL_E = "ctrl_e"
KEY_PAGE_UP = "page_up"
KEY_PAGE_DOWN = "page_down"
KEY_SPACE = "space"
KEY_TAB = "tab"
KEY_CHAR = "char"
KEY_RESIZE = "resize"
"""
The one key the byte decoder has no counterpart for, and it is not symmetric.

A terminal reports a window resize as a pseudo-key, and the curses backend
receives it as `KEY_RESIZE`. It must not be folded into the default branch,
because the default is KEY_ESC and every prompt treats Esc as *cancel* — a
resize during a select would dismiss it, which is the bug this constant exists
to prevent.

The byte decoder has nothing to mirror here because the two backends get resize
differently: the TUI polls `os.get_terminal_size()` and `stdscr.getmaxyx()` on
its idle tick, because in practice ncurses rarely delivers KEY_RESIZE at all.
So this is a backend-specific event surfaced through the shared vocabulary
rather than a disagreement between two decoders, and the mirror check in
.splitwork/probe_session.py exempts it for that reason.
"""

KEY_CTRL = "ctrl"
"""A control chord not otherwise named, with the letter it carries in `char`.

Byte 16 is Ctrl+P, so `Key(KEY_CTRL, "p")`. One kind plus a letter rather than
twenty `KEY_CTRL_P` constants: the vocabulary would double for no gain, and the
dispatch reads better as `kind == KEY_CTRL and char == "p"` than as a name the
reader has to recall.

Produced for every byte in 1..31 that has no specific meaning above -- which is
the change that unblocks the TUI port. Those bytes used to fall through to the
final `return Key(KEY_ESC)`, so Ctrl+P, Ctrl+N, Ctrl+R, Ctrl+U, Ctrl+K and the
rest all reported *Esc*. Two consequences, neither designed:

* in `tui_repl` they were unreachable by kind, so the dispatch read raw integer
  codes and the byte decoder could never serve it;
* in a prompt, Esc is **cancel** -- so Ctrl+U dismissed a select menu, and in
  search mode Ctrl+R cancelled the search it was supposed to reverse.

Prompt cancel sets are spelled `(_KEY_CTRL_C, _KEY_ESC)` in all eight
`interactive_*` modules, so the designed contract was always "Ctrl+C or Esc".
Making the other control chords their own kind restores that contract instead of
routing them through a fallback that happened to mean cancel. Chords the prompts
do not handle now fall through to "ignored", which is what an unhandled key
should do.
"""

KEY_ALT = "alt"
"""An Alt/Meta chord, with the character it carries in `char`.

`ESC` followed by a printable byte: Alt+f is `Key(KEY_ALT, "f")`. `tui_repl`
binds Alt+f/b/d to word-forward, word-back and delete-word, and those three
could not be reached either -- the decoder reported Esc for all of them.
"""

# Arrow keys with Ctrl held were named here until this change: KEY_CTRL_LEFT,
# _RIGHT, _UP and _DOWN. They are `Key(KEY_LEFT, ctrl=True)` and friends now,
# for two reasons that are really one. A modifier inside the constant name
# forces a constant per (base x modifier) pair -- four of them for one axis,
# while `KEY_CTRL` covered twenty-six letters with a single constant -- and it
# makes the *encoding* decide the answer: `ESC [ 5 D` reached a branch that
# `ESC [ 1 ; 5 D` did not, so one chord decoded two ways. The modifier is read
# in exactly one place (`_modifiers`) and carried on `Key`, so no pair of
# encodings can disagree.
#
# The defect those constants were named around is recorded where it was fixed,
# in `_decode_csi`: the old parse returned Esc for `ESC [ 1 ; 5 D` *while
# leaving `;5D` in the stream*, so Ctrl+Left both cancelled the menu and typed
# "5D" into whatever read next.


KEY_UNKNOWN = "unknown"
"""A sequence was read successfully but names no key this vocabulary has.

The distinction is the whole point of the constant. `KEY_ESC` means *the user
pressed Esc* — a key every prompt binds as cancel. `KEY_UNKNOWN` means *I read
a key I do not model*: F1, Shift+Left, Shift+Tab, `ESC [ 99 ~`. Collapsing the
second into the first made an unrecognised keystroke into a cancel, which for
a confirm dialog is not "nothing happened" but "answered the opposite of its
default". It was latent while `tui_repl` cancelled on the raw code 27, because
the code for F1 is 265 and no branch matched it; moving overlay cancel onto
`key.kind` is what turned the collapse into a live bug.

A `Key` of this kind must be inert: nothing binds it, so an unmodelled press
does nothing anywhere. That is the behaviour the decoder's old fallback was
mistakenly believed to have.
"""


class Key:
    """A decoded key press, normalised across every terminal backend.

    `kind` names the base key, and the three modifier dimensions ride beside it
    rather than inside it. That split is the whole design: a modifier folded
    into the name makes the vocabulary grow as base x modifiers, which is how
    `ctrl_left`, `ctrl_right`, `ctrl_up` and `ctrl_down` came to exist as four
    constants for one axis while `KEY_CTRL` covered twenty-six letters with one.
    With the dimensions separate, Ctrl+Left is `KEY_LEFT` with `ctrl` set and
    Shift+Left is `KEY_LEFT` with `shift` set -- representable without naming
    either, so no modified press has to degrade to KEY_UNKNOWN for want of a
    constant.

    `matches()` is the only place a modifier is ever compared. Consumers ask for
    the modifiers they mean, which is what keeps a caret-moved-by-one branch
    from swallowing a word-motion chord now that both arrive as `kind="left"`.
    """

    __slots__ = ("alt", "char", "ctrl", "kind", "shift")

    def __init__(
        self,
        kind: str,
        char: str = "",
        *,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
    ) -> None:
        self.kind = kind
        self.char = char
        self.ctrl = ctrl
        self.alt = alt
        self.shift = shift

    def matches(
        self,
        *kinds: str,
        char: str | None = None,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
    ) -> bool:
        """Is this press exactly one of *kinds*, with no modifier but those named?

        Modifiers are opt-in, and that is the point. A caller that does not ask
        for `ctrl` does not get a Ctrl chord: `matches(KEY_LEFT)` is False for
        Ctrl+Left, so the plain-arrow branch cannot fire on a word-motion chord
        regardless of where it sits in the chain, and `matches(KEY_CHAR)` is
        False for Ctrl+P, so a prompt never types the letter a chord carries.
        Kinds are positional, mirroring the `key.kind in (...)` this replaces.
        """
        if self.kind not in kinds:
            return False
        if char is not None and self.char != char:
            return False
        return self.ctrl == ctrl and self.alt == alt and self.shift == shift

    def __repr__(self) -> str:
        mods = "".join(
            flag for flag, on in (("c", self.ctrl), ("a", self.alt), ("s", self.shift)) if on
        )
        if mods and self.char:
            return f"Key({self.kind!r}, {self.char!r}, {mods!r})"
        if self.char:
            return f"Key({self.kind!r}, {self.char!r})"
        if mods:
            return f"Key({self.kind!r}, {mods!r})"
        return f"Key({self.kind!r})"


# ── Byte-level decoder (line mode) ───────────────────────────────────────────


def decode_key(read_byte: Callable[[], bytes]) -> Key:
    """Read one key press, decoding escape sequences into named keys.

    Args:
        read_byte: returns the next byte from the terminal, or b"" at the
            end of an escape sequence. A source whose *stream* has ended
            raises EOFError — see `terminal_bytes` — and that propagates
            here rather than being decoded as a key. Taking a byte source
            rather than a descriptor is what makes the decoder testable
            without a pty — a test passes a scripted stream.

    Returns:
        Key with kind and optional char. Blocks until a key arrives.
    """
    ch = read_byte()
    if not ch:
        return Key(KEY_ESC)
    byte = ch[0]
    # Ctrl+C
    if byte == 3:
        return Key(KEY_CTRL_C)
    # Ctrl+A — Home
    if byte == 1:
        return Key(KEY_HOME)
    # Ctrl+E — End
    if byte == 5:
        return Key(KEY_END)
    # Enter
    if byte in (10, 13):
        return Key(KEY_ENTER)
    # Escape — everything after it belongs to the same key press.
    if byte == 27:
        return _decode_escape(read_byte)
    # Backspace / Delete. Before the chord branch below: byte 8 is Ctrl+H and
    # would otherwise be reported as a chord rather than the deletion it is.
    if byte in (127, 8):
        return Key(KEY_BACKSPACE)
    # Ctrl+G — Esc, deliberately. Not because the byte is Esc but because both
    # consumers ask for a cancel chord by name and neither can see the raw byte
    # after this port: `tui_repl` cancels overlays on `(KEY_ESC, KEY_CTRL_C)`
    # (replacing `_CANCEL_CHORDS = (27, 7, 3)`, which covered Esc, Ctrl+G and
    # Ctrl+C), and every `interactive_*` prompt cancels on `(_KEY_CTRL_C,
    # _KEY_ESC)`. Reporting `ctrl`/`g` would drop Ctrl+G — readline's `abort`
    # binding — from all eight prompt modules at once, and no test covers it.
    # Named here rather than silently folded in, like Tab and Ctrl+A below.
    if byte == 7:
        return Key(KEY_ESC)
    # Tab. Byte 9 is Ctrl+I and a terminal cannot tell them apart, so this is
    # also the chord mapping's one deliberate casualty — and Tab now has a
    # producer at last. `tui_repl` needs it for completion and can no longer see
    # the raw byte, so leaving it as Esc would have made Esc trigger completion.
    if byte == 9:
        return Key(KEY_TAB)
    # The remaining control bytes are chords. 1..26 are letters — Ctrl+P is
    # byte 16, so ctrl/"p". The rest have no letter: 0 is Ctrl+@ and 28..31 are
    # Ctrl+\ Ctrl+] Ctrl+^ Ctrl+_, and chr(byte + 64) is precisely that symbol.
    # Reaching them is not just symmetry. Byte 0 and 28..31 used to fall
    # through to Esc, so once the overlay handlers cancelled on `key.kind`,
    # Ctrl+\ would dismiss a dialog that only Esc, Ctrl+G and Ctrl+C were
    # ever meant to dismiss — and Ctrl+\ is a perfectly likely keystroke.
    if 0 <= byte <= 31:
        return Key(KEY_CTRL, chr(byte + 96) if 1 <= byte <= 26 else chr(byte + 64))
    # Printable ASCII
    if 32 <= byte < 127:
        return Key(KEY_CHAR, chr(byte))
    # Non-ASCII (utf-8 multi-byte) — read remaining bytes
    if byte >= 128:
        remaining = []
        if byte < 224:
            count = 1
        elif byte < 240:
            count = 2
        elif byte < 248:
            count = 3
        else:
            count = 4
        for _ in range(count):
            more = read_byte()
            if more:
                remaining.append(more[0])
        try:
            char = bytes([byte] + remaining).decode("utf-8", errors="replace")
        except Exception:
            char = ""
        return Key(KEY_CHAR, char)
    # Unreachable for a byte in 0..255 — every one is classified above. Kept as
    # KEY_UNKNOWN so the rule has no second spelling if a byte is ever added.
    return Key(KEY_UNKNOWN)


def _decode_escape(read_byte: Callable[[], bytes]) -> Key:
    """Resolve what follows ESC: a CSI sequence, an SS3 sequence, or Alt.

    Called only from `decode_key` for byte 27, so the "is there a next byte"
    decision stays in one place. Three shapes, distinguished by their introducer:

    * `ESC [` — a CSI sequence. `ESC` alone with nothing following is a lone
      Esc and never reaches here.
    * `ESC O` — SS3: application-cursor arrows, `kent`, and rxvt's Ctrl+arrows.
    * `ESC <printable>` — Alt/Meta, which is just the byte with a bit stuck on
      front of it. This is the only case where the *character* follows, so it
      is the one branch that returns a chord with a `char`.
    """
    seq = read_byte()
    if not seq:
        # A lone Esc. Only reachable if the caller's byte source does not block;
        # TtySession bounds continuation reads for exactly this reason.
        return Key(KEY_ESC)
    c = seq[0]
    if c == ord("["):
        return _decode_csi(read_byte)
    if c == ord("O"):
        return _decode_ss3(read_byte)
    if 32 <= c < 127:
        return Key(KEY_ALT, chr(c))
    # ESC followed by a control byte — Alt+Ctrl+X on some terminals, or two
    # unrelated bytes that arrived together. Not Esc, and not a key anyone
    # binds, so it reads as unknown rather than stealing Esc's meaning.
    return Key(KEY_UNKNOWN)


def _modifiers(field: str) -> tuple[bool, bool, bool] | None:
    """Decode xterm's `1 + bitmask` modifier field as `(ctrl, alt, shift)`.

    Returns None when the field is not a number or names a modifier this
    vocabulary has no flag for. The mask runs shift=1, alt=2, ctrl=4, meta=8
    above that, so anything outside the low three bits has no home here and the
    caller reports KEY_UNKNOWN rather than guessing -- a chord the system
    cannot name must stay inert, never become a cancel.

    An empty field is no modifier at all, which is what makes a bare
    `ESC [ D` and a `ESC [ 1 ; 1 D` land on the same plain arrow.
    """
    if not field:
        return False, False, False
    if not field.isdigit():
        return None
    bits = int(field) - 1
    if bits < 0 or bits & ~0b111:
        return None
    return bool(bits & 4), bool(bits & 2), bool(bits & 1)


def _decode_csi(read_byte: Callable[[], bytes]) -> Key:
    """Decode `ESC [ ... final`, consuming every byte through the final one.

    This replaces a chain of "read one byte and see if it happens to be what I
    expect" branches, and the difference is not cosmetic. The old parse read the
    `1` of `ESC [ 1 ; 5 D`, failed to match it against its short list, and
    returned Esc **leaving `;5D` in the stream**. The next read then handed the
    prompt a literal `5`, then `D`, then whatever followed — so Ctrl+Left both
    cancelled the menu and typed text into the field underneath it.

    A CSI sequence is defined as parameters, then a final byte in `@`..`~`, so
    that is exactly what this reads: loop until a final byte shows up, then
    interpret. `ESC [ 99 ~` now consumes the `99` and the `~` instead of leaving
    `9~` behind; `ESC [ 1 ; 2 D` likewise.

    Modifiers follow xterm: the field is `1 + bitmask`, so 5 is Ctrl (4 | 1) and
    2 is Shift. On an arrow the field stands alone as well as in a list -- see
    the branch below for why -- while on `~`, `H` and `F` a lone field is a key
    number and only a second field carries the modifier. A modifier with no
    name in the vocabulary reads as KEY_UNKNOWN, same as any unmodelled
    sequence, rather than inventing a chord the rest of the system does not
    know how to receive, and *not* as Esc, which would turn Shift+Left into a
    cancel.
    """
    params = ""
    while True:
        b = read_byte()
        if not b:
            # Truncated: the final byte never arrived. A half-read sequence is
            # not an Esc press either, so it does not claim to be one.
            return Key(KEY_UNKNOWN)
        c = b[0]
        if 0x40 <= c <= 0x7E:  # final byte: @A..Z[\]^_`a..z{|}~
            final = chr(c)
            break
        params += chr(c)

    parts = params.split(";")
    # A modifier written as a second field -- the form `H`, `F` and the `~`
    # family use, where a lone field is a key *number* (`ESC [ 5 ~` is PageUp)
    # and the last field of a longer list is the modifier.
    mod = None
    if len(parts) > 1 and parts[-1].isdigit():
        mod = int(parts[-1])

    if final in "ABCD":
        # On an arrow a lone field is the modifier too, so `ESC [ 5 D` and
        # `ESC [ 1 ; 5 D` are one chord in two encodings. Reading the first as
        # CUB -- "repeat 5" -- is what let them disagree: no installed terminfo
        # binds `\E[5D`, five bind `\E[1;5D`, and a cursor-motion CSI is an
        # *output* sequence an app sends to a terminal, never something a
        # terminal sends back as a keystroke. One arriving as input can only be
        # a key encoding, so this is the reading that matches the wire.
        mods = _modifiers(parts[-1])
        if mods is None:
            # Meta/Super, a non-numeric field, or a negative mask: unmodelled,
            # so inert -- not Esc, which would turn Shift+Left into a cancel.
            return Key(KEY_UNKNOWN)
        return Key(
            {"A": KEY_UP, "B": KEY_DOWN, "C": KEY_RIGHT, "D": KEY_LEFT}[final],
            ctrl=mods[0],
            alt=mods[1],
            shift=mods[2],
        )
    if final in "HF" and mod is None:
        return Key(KEY_HOME if final == "H" else KEY_END)
    if final == "~":
        if mod is not None:
            # Shift+F1 and friends: a `~` sequence with a modifier has no name
            # in the vocabulary, and inventing one would be worse than ignoring.
            return Key(KEY_UNKNOWN)
        num = parts[0] if parts[0].isdigit() else ""
        kind = {
            "1": KEY_HOME,
            "7": KEY_HOME,
            "3": KEY_DELETE,
            "4": KEY_END,
            "8": KEY_END,
            "5": KEY_PAGE_UP,
            "6": KEY_PAGE_DOWN,
        }.get(num)
        # Includes F1-F4's `ESC [ 1 1 ~`-style numbers, which have no kind.
        return Key(kind) if kind else Key(KEY_UNKNOWN)
    # `ESC [ Z` (shift-Tab) and any final byte with no case above.
    return Key(KEY_UNKNOWN)


def _decode_ss3(read_byte: Callable[[], bytes]) -> Key:
    r"""Decode `ESC O <final>`: application-cursor keys, Enter, rxvt Ctrl+arrows.

    `xterm-256color` names its arrows `kcuu1=\EOA` and its Enter `kent=\EOM`, so
    these arrive whenever the terminal is in application mode — which is not an
    exotic state, it is what `smkx` turns on. The old parse handled only `OH`
    and `OF`; a plain `ESC O A` fell through to Esc, and `ESC O M` was read as an
    X10 mouse report, which then **blocked reading two bytes that were never
    coming** before reporting Esc.

    Mouse reporting is never enabled, so `ESC O M` cannot be a mouse report
    here. Treating it as Enter is both correct and the choice that cannot hang:
    reading on for three more bytes waits for input that will not arrive.
    """
    b = read_byte()
    if not b:
        # Truncated SS3 — no final byte, so nothing was actually pressed that
        # this parser can name.
        return Key(KEY_UNKNOWN)
    c = chr(b[0])
    if c in "ABCD":
        return Key({"A": KEY_UP, "B": KEY_DOWN, "C": KEY_RIGHT, "D": KEY_LEFT}[c])
    if c == "H":
        return Key(KEY_HOME)
    if c == "F":
        return Key(KEY_END)
    if c == "M":
        return Key(KEY_ENTER)
    # rxvt: kLFT5=\EOd, kRIT5=\EOc (terminfo modifier suffix, 5 = Ctrl).
    if c == "d":
        return Key(KEY_LEFT, ctrl=True)
    if c == "c":
        return Key(KEY_RIGHT, ctrl=True)
    return Key(KEY_UNKNOWN)


# ── Normalisation from curses ────────────────────────────────────────────────

# Every entry here has a counterpart in `decode_key`, and the pairing is the
# whole point of this module: a prompt must behave the same whether it is fed
# bytes or handed `getch()` output. So this table mirrors the byte decoder's
# actual decisions, including the ones that look wrong.
#
# Three of them do look wrong, and are preserved on purpose:
#
#   Ctrl+A (1)     -> KEY_HOME  a long-standing shortcut, not a ctrl key
#   Ctrl+E (5)     -> KEY_END   likewise
#   Ctrl+G (7)     -> KEY_ESC   so it remains a cancel chord in every consumer
#
# KEY_CTRL_A, KEY_CTRL_E and KEY_SPACE are therefore still vocabulary with no
# producer, and say so rather than quietly being given a meaning. The one that
# used to be in that list, KEY_TAB, now has one: byte 9 reports Tab, because
# `tui_repl` binds Tab to completion and could no longer reach the raw byte once
# the port moved it onto the shared vocabulary. Leaving it as Esc would have
# made Esc trigger completion.
#
# The chord kinds below are keyed on the byte, not on a curses name: ncurses
# folds Ctrl+P into a raw 16 before `getch()` ever returns, so byte 16 is the
# only representation either backend sees.
#
# Curses constants are read off `keypad` by name rather than hardcoded, so the
# table stays right on any ncurses build and a test can pass a stub.
# (ctrl, alt, shift) triples for the `mods` column below. Every plain key takes
# the first; only ncurses' Ctrl+arrow codes take the second. Named rather than
# written inline so the table reads as data and the bit order is stated once.
_MODS_NONE = (False, False, False)
_MODS_CTRL = (True, False, False)

# `name, kind, char, mods`, where `mods` is `(ctrl, alt, shift)`. The column is
# empty for every plain key and set only for the two Ctrl+arrow codes ncurses
# exposes; keeping it in the same tuple rather than a second table is what lets
# the byte cases and the named cases stay one table with one precedence rule.
_CURSES_KEYMAP = (
    ("KEY_UP", KEY_UP, "", _MODS_NONE),
    ("KEY_DOWN", KEY_DOWN, "", _MODS_NONE),
    ("KEY_LEFT", KEY_LEFT, "", _MODS_NONE),
    ("KEY_RIGHT", KEY_RIGHT, "", _MODS_NONE),
    ("KEY_HOME", KEY_HOME, "", _MODS_NONE),
    ("KEY_END", KEY_END, "", _MODS_NONE),
    ("KEY_PPAGE", KEY_PAGE_UP, "", _MODS_NONE),
    ("KEY_NPAGE", KEY_PAGE_DOWN, "", _MODS_NONE),
    ("KEY_BACKSPACE", KEY_BACKSPACE, "", _MODS_NONE),
    ("KEY_DC", KEY_DELETE, "", _MODS_NONE),
    ("KEY_RESIZE", KEY_RESIZE, "", _MODS_NONE),
    # Absent from older ncurses builds — `getattr` skips them when missing, and
    # xterm delivers these chords as CSI anyway, so the byte decoder is the
    # path that matters. Reported as a modifier on the base arrow rather than
    # as a kind of its own, so both backends answer `Key(KEY_LEFT, ctrl=True)`
    # for one physical keystroke.
    ("KEY_CTRL_LEFT", KEY_LEFT, "", _MODS_CTRL),
    ("KEY_CTRL_RIGHT", KEY_RIGHT, "", _MODS_CTRL),
    # Claimed so it mirrors the byte decoder rather than falling into Esc. The
    # two disagreed the moment `decode_key` learned that `ESC O M` is terminfo's
    # `kent` and returned Enter for it: ncurses folds that same sequence into
    # 343, so an unclaimed KEY_ENTER made `from_curses` say Esc while
    # `decode_key` said Enter for one physical keystroke. Enter cancelling a
    # dialog was the visible half of that.
    ("KEY_ENTER", KEY_ENTER, "", _MODS_NONE),
)

# Single bytes, in the same order and with the same outcomes as decode_key.
# Triples rather than pairs because a chord carries a letter: Ctrl+P is byte 16
# and reports kind "ctrl" with char "p", and the char has to survive the
# translation or the two backends stop agreeing.
_CURSES_BYTES = (
    (3, KEY_CTRL_C, ""),
    (1, KEY_HOME, ""),
    (5, KEY_END, ""),
    (10, KEY_ENTER, ""),
    (13, KEY_ENTER, ""),
    (27, KEY_ESC, ""),
    (127, KEY_BACKSPACE, ""),
    (8, KEY_BACKSPACE, ""),
    (9, KEY_TAB, ""),
    (7, KEY_ESC, ""),
)

# Byte -> (KEY_CTRL, letter), for every control byte that is not one of the
# specific cases above. The letters come from chr(byte + 96) for 1..26; 0 and
# 28..31 have no letter, so they use chr(byte + 64) — @ \ ] ^ _ — exactly as
# the byte decoder does. Excluded here rather than filtered: 1, 3, 5, 7, 8,
# 9, 10, 13, and 27 — Esc is excluded too, or it would be silently rebuilt as
# ctrl/"[" by the later entry, because the dict build takes the last word on
# a duplicate byte.
_CURSES_EXCLUDED_BYTES = frozenset((1, 3, 5, 7, 8, 9, 10, 13, 27))
_CURSES_CTRL_BYTES = tuple(
    (byte, KEY_CTRL, chr(byte + 96) if 1 <= byte <= 26 else chr(byte + 64))
    for byte in range(32)
    if byte not in _CURSES_EXCLUDED_BYTES
)
_CURSES_BYTES = _CURSES_BYTES + _CURSES_CTRL_BYTES


# A resolved table row: base kind, the letter a chord carries, then the three
# modifier flags in (ctrl, alt, shift) order. Five fields rather than a nested
# tuple so `from_curses` can build the Key with one explicit unpack.
_CursesEntry = tuple[str, str, bool, bool, bool]

_CURSES_TABLE_CACHE: dict[int, tuple[object, dict[int, _CursesEntry]]] = {}
_CURSES_TABLE_NEXT = [0]


def _curses_table(keypad: object) -> dict[int, _CursesEntry]:
    """`code -> (kind, char, ctrl, alt, shift)` for one keypad, resolved once.

    This runs on every keypress, and deriving it per call meant eleven `getattr`
    lookups against the curses module plus a second linear scan for the byte
    cases — measured at ~1 µs, which is forty times the cost of the dispatch it
    feeds. The constants are fixed for a given keypad, so the table is built
    once and reused; this is the same discipline as the interpreter's symbol
    table being interned rather than re-read on every LOAD_NAME.

    Values are `(kind, char, ctrl, alt, shift)` rows because the chord kinds
    carry a letter and the arrows carry a modifier: byte 16 is
    `("ctrl", "p")` and ncurses' Ctrl+Left is `("left", "", True, False,
    False)`. Dropping either field would make `from_curses` disagree with
    `decode_key` on the very inputs that motivate the whole mirror.

    Cached by `id`, and the key object is held alongside so its id cannot be
    recycled by a later, unrelated object while the entry is still live.

    Raises:
        AssertionError: if a byte case collides with a named key. The two
            tables were separate scans before, so a collision would silently
            change which kind wins; merging them into one dict would make that
            silent rather than accidental, so it is checked here.
    """
    hit = _CURSES_TABLE_CACHE.get(id(keypad))
    if hit is not None and hit[0] is keypad:
        return hit[1]

    table: dict[int, _CursesEntry] = {}
    for byte, kind, char in _CURSES_BYTES:
        # Byte chords never carry a modifier flag: the byte decoder reports
        # Ctrl+P as `Key(KEY_CTRL, "p")`, a chord kind plus a letter, so the
        # three flags are False here by construction rather than by omission.
        table[byte] = (kind, char, False, False, False)
    for name, kind, char, mods in _CURSES_KEYMAP:
        value = getattr(keypad, name, None)
        if value is None:
            continue
        assert value not in table, (
            f"{name}={value} collides with a byte case in _CURSES_BYTES; "
            "the two tables must stay disjoint so precedence is unambiguous"
        )
        table[value] = (kind, char, *mods)

    _CURSES_TABLE_CACHE[id(keypad)] = (keypad, table)
    return table


def from_curses(code: int | str, keypad: object) -> Key:
    """Normalise one value from `window.getch()` into a `Key`.

    Args:
        code: an int as returned by `getch()`, or a one-character str when the
            window is in `get_wch` mode.
        keypad: an object exposing curses' `KEY_*` constants — the curses module
            itself, or a stand-in in tests.

    Returns:
        Key. Anything unrecognised becomes KEY_UNKNOWN, exactly as the byte
        decoder reports a sequence it cannot parse — so a prompt behaves
        identically across backends for input neither decoder knows, and a key
        nobody binds stays inert instead of reading as a cancel.
    """
    if isinstance(code, str):
        if not code:
            # No input at all, mirroring `decode_key(b"")`. Not a key press.
            return Key(KEY_ESC)
        code = ord(code[0])

    entry = _curses_table(keypad).get(code)
    if entry is not None:
        kind, char, ctrl, alt, shift = entry
        return Key(kind, char, ctrl=ctrl, alt=alt, shift=shift)

    # ncurses reserves everything from KEY_MIN up for named keys, and this build
    # fills the band from 257 to 511 without a gap — 156 of them, including
    # KEY_RESIZE (410) and 63 function keys. A code landing here that the table
    # did not claim is therefore never a character, whatever it looks like
    # numerically: chr(343) is a Cyrillic letter and chr(410) is 'Ċ', and
    # letting either through turns a window resize into a letter in the
    # prompt's input buffer. It is an unhandled special key, and an unhandled
    # special key is what KEY_UNKNOWN is for.
    if code >= 256:
        return Key(KEY_UNKNOWN)

    if 32 <= code < 127 or 128 <= code < 256:
        return Key(KEY_CHAR, chr(code))
    # Unreachable: every byte in 0..31 is claimed by _CURSES_BYTES. Kept so a
    # future gap reads as unknown rather than as a cancel.
    return Key(KEY_UNKNOWN)


# ── The protocol ─────────────────────────────────────────────────────────────


class TerminalSession(Protocol):
    """A terminal that exactly one component reads from."""

    owns_stdin: bool
    """Whether this session is the sole reader of the input descriptor.

    Declared, never probed. `InteractiveBase` consults this instead of calling
    `os.isatty`, because a tty check cannot tell "curses owns it" from "free".
    """

    def read_key(self, timeout: float | None = None) -> Key | None:
        """Block for one key, or return None if *timeout* seconds elapse first."""

    def raw(self) -> object:
        """Context manager entering and leaving raw mode."""

    def size(self) -> tuple[int, int]:
        """Terminal size as (rows, columns), falling back to (24, 80)."""

    def write(self, text: str) -> None:
        """Write straight to the terminal, bypassing the output pane."""


class _RawMode:
    """Raw-mode scope for a descriptor, restoring whatever was there before."""

    __slots__ = ("_fd", "_old")

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._old: list | None = None

    def __enter__(self) -> _RawMode:
        try:
            self._old = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
        except termios.error:
            # Not a terminal, or the mode is already raw. Reading still works;
            # only the echo suppression is unavailable, so carry on.
            self._old = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except termios.error:
                pass


ESC_DELAY = 0.025
"""How long to wait for the rest of an escape sequence, in seconds.

Same figure as the `curses.set_escdelay(25)` the TUI used while curses owned
the screen, so removing curses does not change when a lone Esc resolves.

A terminal writes a whole escape sequence in one go, so this is not a guess
about interleaving — if the rest has not arrived within a frame or two, there is
no rest. What it buys is the ability to report Esc *at all*: `decode_key`
returns Esc only once it knows nothing follows, and the only way to know that is
to stop waiting.
"""


def terminal_bytes(fd: int) -> Callable[[], bytes]:
    """Build the byte source `decode_key` reads from, with a bounded tail.

    The first byte waits indefinitely — that is "wait for a key press". Every
    byte after it waits at most `ESC_DELAY`, because by then a sequence is
    known to be in progress and its remainder either arrived with it or is not
    coming.

    Unbounded continuation reads are how a decoder hangs. The two shapes that
    did it: a lone Esc, where the prompt could not return until the *next*
    keypress arrived to prove nothing followed; and `ESC O M`, which read on
    for two mouse bytes that no enabled mouse would ever send. Both were
    waiting for input that could not arrive, which is the one failure mode a
    deadline fixes and a retry does not.

    Args:
        fd: a terminal descriptor, already in raw mode.

    Returns:
        A callable for `decode_key`. Returns b"" at the deadline or on a
        descriptor that has stopped accepting reads mid-sequence — nothing
        more of *this keypress* is coming. A true EOF on the *first* byte
        raises `EOFError` instead: the input stream itself is finished.
    """
    first = True

    def read_byte() -> bytes:
        nonlocal first
        if first:
            first = False
            try:
                b = os.read(fd, 1)
            except OSError:
                # EAGAIN on a non-blocking descriptor, EINTR from a signal:
                # "no byte for you" — the decoder turns b"" into Esc rather
                # than raising into a prompt.
                return b""
            if b:
                return b
            # True EOF: the unbounded first read yielded nothing, so the far
            # end is gone (terminal hangup, drained pipe). Returning b"" here
            # would decode as a phantom Esc on every poll tick — forever, at
            # full CPU. EOFError matches input()'s contract and lets the
            # session end.
            raise EOFError
        ready, _, _ = select.select([fd], [], [], ESC_DELAY)
        if not ready:
            return b""
        try:
            return os.read(fd, 1) or b""
        except OSError:
            # EINTR, EAGAIN on a continuation byte: the sequence stalled, and
            # the decoder treats that as its end.
            return b""

    return read_byte


class TtySession:
    """Line mode: owns the descriptor, enters raw mode, decodes bytes itself.

    This is the only implementation that touches `os.read` on a terminal. The
    curses backend cannot, because it does not own the descriptor.
    """

    def __init__(self, io: object) -> None:
        self._io = io
        self._raw: _RawMode | None = None

    @property
    def owns_stdin(self) -> bool:
        # ConsoleIO decides this per instance, not as a class constant: a piped
        # stdin is not owned, and a /dev/tty or a real stdin is. A constant here
        # would be a probe wearing a disguise.
        return bool(getattr(self._io, "owns_stdin", False))

    @property
    def fd(self) -> int:
        """The descriptor this session owns.

        Mirrors what `InteractivePrompt._get_fd` did: a dedicated /dev/tty when
        one was opened, otherwise stdin. It is the only place in the tree that
        asks the operating system for a terminal descriptor, which is the point —
        everyone else asks the session.
        """
        handle = getattr(self._io, "_tty", None)
        if handle is not None:
            return handle.fileno()
        try:
            return sys.stdin.fileno()
        except (AttributeError, ValueError, OSError):
            # `sys.stdin` is a Python object, and it may be a stand-in that has
            # no descriptor of its own: pytest's capture layer swaps in a
            # pseudofile whose `fileno()` raises, an embedder may have replaced
            # it, and a closed stream raises ValueError. The process still has
            # descriptor 0, and that -- not the object -- is what "stdin" means
            # to raw mode, select and os.read. Same guard
            # `InteractiveBase._detect_tty` already wraps around this exact
            # call, so the two agree instead of one crashing where the other
            # degrades. `UnsupportedOperation` is both an OSError and a
            # ValueError, so the tuple above covers it.
            return 0

    def read_key(self, timeout: float | None = None) -> Key | None:
        """Read one key, honouring a deadline.

        The deadline is here rather than at the call sites because seven prompt
        loops already pass one to a decoder that had no such parameter, so every
        one of those countdown paths died on a TypeError before it could render.

        It governs the *first* byte only — "is a key arriving?". Bytes after it
        belong to a sequence already in progress and are bounded by
        `terminal_bytes`, not by this deadline.
        """
        if timeout is not None:
            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                return None
        return decode_key(terminal_bytes(self.fd))

    def raw(self) -> _RawMode:
        return _RawMode(self.fd)

    def size(self) -> tuple[int, int]:
        try:
            s = os.get_terminal_size()
            return s.lines, s.columns
        except (AttributeError, ValueError, OSError):
            return 24, 80

    def write(self, text: str) -> None:
        self._io.write(text, "")
