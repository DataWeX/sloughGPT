"""The two rules the retired tests guarded, re-aimed at `decode_key`.

`TestReadEscapeRemainder` (in `test_tui_repl.py`) and `test_esc_finals_is_frozen`
(in `test_tui_repl_utils.py`) both asserted on helpers the curses port removed.
What they protected still exists — only the shape changed — so these pin the
behaviour rather than the representation.

Two of the retired cases were the sole coverage of the *bare* modifier form:

    ESC [ 1 ; 5 D   the xterm two-part form, named by terminfo
    ESC [ 5 D       the bare form, which no installed terminfo binds

They are not the same statement, and treating them as one is exactly the bug the
fold had to decide: `_decode_csi` used to read a modifier only when there was
more than one parameter, so the two-part form reached Ctrl+Left while the bare
form fell through to plain Left. A cursor-motion CSI is an *output* sequence — an
application sends `ESC [ 5 D` to a terminal, a terminal never sends one back as
a keystroke — so one arriving as input can only be a key encoding, and its lone
parameter can only be a modifier.

The second half of this module is the reason that fold could be written at all:
modifiers are flags on `Key`, not extra kinds, so the two encodings land on one
value and `matches()` can tell a plain arrow from a chorded one without either
needing a name of its own.
"""

from domain.shell._internal.tui_session import (
    KEY_ALT,
    KEY_CHAR,
    KEY_CTRL,
    KEY_ESC,
    KEY_HOME,
    KEY_LEFT,
    KEY_PAGE_UP,
    KEY_RIGHT,
    KEY_UNKNOWN,
    Key,
    decode_key,
)


def _decode(stream: str) -> Key:
    """Decode one scripted press, one byte per read, as a terminal delivers it."""
    data = stream.encode("latin-1")
    pos = 0

    def read_byte() -> bytes:
        nonlocal pos
        if pos >= len(data):
            return b""
        byte = data[pos]
        pos += 1
        return bytes([byte])

    return decode_key(read_byte)


# ── The two rules the retired tests guarded ─────────────────────────────────


def test_bare_csi_ctrl_right_folds_to_word_motion():
    # Was `result == "seq:ctrl-right"` from `_read_escape_remainder`.
    key = _decode("\x1b[5C")
    assert (key.kind, key.char, key.ctrl) == (KEY_RIGHT, "", True)


def test_bare_csi_ctrl_left_folds_to_word_motion():
    # Was `result == "seq:ctrl-left"`. Without the fold this reads as plain
    # Left, which the live suite surfaces as a caret one character short.
    key = _decode("\x1b[5D")
    assert (key.kind, key.char, key.ctrl) == (KEY_LEFT, "", True)


def test_bare_csi_without_modifier_stays_plain():
    # The bare form must not fold unconditionally: parameter 1 is "no modifier".
    assert _decode("\x1b[D").matches(KEY_LEFT)
    assert _decode("\x1b[1D").matches(KEY_LEFT)


def test_csi_final_outside_old_set_is_unknown_not_esc():
    """`q` is not in `ABCDEFHZ~`, but it is a CSI final byte (`@`..`~`).

    The old frozenset would keep reading past it and time out to `None`, which
    the caller read as Esc — so an unmodelled sequence cancelled. The range rule
    terminates on it and reports unknown: the bytes are consumed (none leak into
    the input field) and nothing is claimed the decoder cannot name.
    """
    assert _decode("\x1b[5q").kind == KEY_UNKNOWN


def test_truncated_csi_is_unknown_not_esc():
    # A half-read sequence is not an Esc press either.
    assert _decode("\x1b[5").kind == KEY_UNKNOWN
    assert _decode("\x1b").kind == KEY_ESC


# ── One chord, two encodings ────────────────────────────────────────────────


def test_the_two_csi_encodings_of_ctrl_arrow_agree():
    bare, named = _decode("\x1b[5D"), _decode("\x1b[1;5D")
    assert (bare.kind, bare.ctrl) == (named.kind, named.ctrl) == (KEY_LEFT, True)
    assert bare.matches(KEY_LEFT, ctrl=True) and named.matches(KEY_LEFT, ctrl=True)


def test_lone_parameter_is_a_key_number_on_the_tilde_family():
    # The fold is scoped to arrow finals: `ESC [ 5 ~` is PageUp, and reading its
    # 5 as a modifier would turn PageUp into an unnameable chord.
    assert _decode("\x1b[5~").kind == KEY_PAGE_UP


def test_a_lone_field_on_home_or_end_is_not_a_modifier():
    # H and F deliberately do not take the arrow treatment: a lone field is a
    # key number there, not a modifier, so `ESC [ 5 H` is still Home. A real
    # Ctrl+Home has no binding in this vocabulary, so it stays unmodelled
    # rather than folding into a chord nothing expects.
    assert _decode("\x1b[5H").matches(KEY_HOME)
    assert _decode("\x1b[1;5H").kind == KEY_UNKNOWN


def test_rxvt_ctrl_arrow_carries_the_flag_not_a_kind():
    # The SS3 path with terminfo's modifier suffix (5 = Ctrl).
    assert _decode("\x1bOd").matches(KEY_LEFT, ctrl=True)
    assert _decode("\x1bOc").matches(KEY_RIGHT, ctrl=True)


# ── Modifiers are flags, not kinds ──────────────────────────────────────────


def test_shift_and_alt_arrows_decode_instead_of_being_dropped():
    # Both were KEY_UNKNOWN before flags existed: the vocabulary had no name
    # for them, so a keystroke arrived as "unmodelled" and was discarded.
    assert _decode("\x1b[2D").matches(KEY_LEFT, shift=True)
    assert _decode("\x1b[1;2D").matches(KEY_LEFT, shift=True)
    assert _decode("\x1b[3D").matches(KEY_LEFT, alt=True)


def test_a_modifier_above_shift_is_unknown_never_esc():
    # meta/super sit above shift in the same bitmask. Nothing here names them,
    # so they stay inert — Esc would have turned them into a cancel.
    assert _decode("\x1b[9D").kind == KEY_UNKNOWN
    assert _decode("\x1b[99D").kind == KEY_UNKNOWN
    assert _decode("\x1b[0D").kind == KEY_UNKNOWN


def test_chord_kinds_keep_their_letter_and_stay_off_the_char_kind():
    # Ctrl+P is KEY_CTRL with a letter, not KEY_CHAR, so a prompt never types
    # the letter a chord carries.
    ctrl_p = Key(KEY_CTRL, "p")
    assert ctrl_p.matches(KEY_CTRL, char="p")
    assert not ctrl_p.matches(KEY_CHAR, char="p")
    assert not Key(KEY_CHAR, "p").matches(KEY_CTRL, char="p")


def test_alt_letter_is_distinct_from_the_plain_letter():
    assert _decode("\x1bd").matches(KEY_ALT, char="d")
    assert not _decode("d").matches(KEY_ALT, char="d")
    assert _decode("d").matches(KEY_CHAR, char="d")


class TestMatches:
    """`matches()` is the only place a modifier is ever compared."""

    def test_a_plain_call_refuses_every_modifier(self):
        # This is the invariant the 156 converted sites rely on: naming no
        # modifier means no modifier, so branch order cannot make a word-motion
        # chord move the caret by one.
        for mod in (dict(ctrl=True), dict(alt=True), dict(shift=True)):
            assert not Key(KEY_LEFT, **mod).matches(KEY_LEFT)
        assert Key(KEY_LEFT).matches(KEY_LEFT)

    def test_named_modifiers_must_all_hold(self):
        assert Key(KEY_LEFT, ctrl=True).matches(KEY_LEFT, ctrl=True)
        assert not Key(KEY_LEFT, ctrl=True, alt=True).matches(KEY_LEFT, ctrl=True)
        assert not Key(KEY_LEFT, ctrl=True).matches(KEY_LEFT, ctrl=True, shift=True)

    def test_kinds_are_positional_mirroring_the_membership_it_replaced(self):
        assert Key(KEY_RIGHT).matches(KEY_LEFT, KEY_RIGHT)
        assert Key(KEY_LEFT).matches(KEY_LEFT, KEY_RIGHT)
        assert not Key(KEY_ESC).matches(KEY_LEFT, KEY_RIGHT)

    def test_no_kinds_at_all_matches_nothing(self):
        assert not Key(KEY_LEFT).matches()

    def test_char_filter_combines_with_modifiers(self):
        assert Key(KEY_CTRL, "w").matches(KEY_CTRL, char="w")
        assert not Key(KEY_CTRL, "w").matches(KEY_CTRL, char="d")
        # An arrow carries no letter, so a char filter cannot match one even
        # when the modifier the caller asked for does hold.
        assert not Key(KEY_LEFT, ctrl=True).matches(KEY_LEFT, char="w")
        assert Key(KEY_LEFT, ctrl=True).matches(KEY_LEFT, ctrl=True)

    def test_repr_names_the_modifiers_it_carries(self):
        assert "c" in repr(Key(KEY_LEFT, ctrl=True))
        assert "s" in repr(Key(KEY_LEFT, shift=True))
        assert repr(Key(KEY_LEFT)) == "Key('left')"
        assert repr(Key(KEY_CHAR, "a")) == "Key('char', 'a')"
