"""Word-wise caret motion and word deletion over a list-backed edit buffer.

Every text field in the shell edits the same shape of thing: a `list[str]`
holding one character per element, plus an integer caret index into it. That
shape is why this module exists. It used to exist only inside `tui_repl`,
as four private methods that scanned a buffer for word boundaries, and the
prompts had none at all — so Ctrl+Left was a word jump in the shell's own
prompt and a dead key in every other input field. One scan, one definition of
where a word starts, means the caret lands in the same place in all of them.

The four functions are the readline commands under their plain names:

    word_forward     forward-word     Alt+F, Ctrl+Right
    word_backward    backward-word    Alt+B, Ctrl+Left
    delete_back      unix-word-rubout Ctrl+W
    delete_forward   kill-word        Alt+D

They are pure with one deliberate exception: the two deletes mutate the buffer
they are handed, because copying and reassigning a buffer at every keystroke
would make every caller's reference to it a bug. The deletes report what they
removed so the caller can decide what it is worth — the shell pushes it onto
its kill ring, the prompts have no kill ring and drop it.

A "word" here is a maximal run of non-space characters, which is readline's
definition of a word for ``forward-word`` and ``backward-word``. Only the
plain space counts as a separator: these are single-line ASCII fields, and
`interactive_input` accepts nothing below 32, so a tab or newline can reach a
buffer only through a `default` string that already contained one.
"""

from __future__ import annotations

__all__ = [
    "delete_back",
    "delete_forward",
    "word_backward",
    "word_forward",
]


def word_forward(buf: list[str], i: int) -> int:
    """Caret position after moving right past the end of the next word.

    From `i`, skip spaces, then skip non-spaces; the result is one past the
    last character of that word. From mid-word this lands at the end of the
    current word rather than the start of the next, which is readline's
    ``forward-word``. Returns `len(buf)` at the end of the line.
    """
    n = len(buf)
    while i < n and buf[i] == " ":
        i += 1
    while i < n and buf[i] != " ":
        i += 1
    return i


def word_backward(buf: list[str], i: int) -> int:
    """Caret position after moving left to the start of the current or previous word.

    Mirrors readline's ``backward-word``: from `i`, step back over any spaces,
    then back over the word itself, landing on its first character. Returns 0
    at the start of the line.
    """
    while i > 0 and buf[i - 1] == " ":
        i -= 1
    while i > 0 and buf[i - 1] != " ":
        i -= 1
    return i


def delete_back(buf: list[str], i: int) -> tuple[int, str]:
    """Remove the word before the caret, returning `(new_caret, removed_text)`.

    Scans exactly as `word_backward` does, so Ctrl+W erases precisely the run
    Ctrl+Left would have jumped over — the two commands agreeing on where a
    word starts is the whole point of keeping one scan. Matches readline's
    ``unix-word-rubout``, including the trailing spaces it takes with the word.

    Mutates `buf` in place.
    """
    end = i
    while i > 0 and buf[i - 1] == " ":
        i -= 1
    while i > 0 and buf[i - 1] != " ":
        i -= 1
    removed = "".join(buf[i:end])
    del buf[i:end]
    return i, removed


def delete_forward(buf: list[str], i: int) -> tuple[int, str]:
    """Remove the word after the caret, returning `(caret, removed_text)`.

    Scans exactly as `word_forward` does, so Alt+D erases precisely the run
    Alt+F would have jumped over. Matches readline's ``kill-word``: the caret
    stays where it was, which is why the returned caret is `i` unchanged — the
    pair of deletes share one shape so callers need no special case.

    Mutates `buf` in place.
    """
    n = len(buf)
    end = i
    while end < n and buf[end] == " ":
        end += 1
    while end < n and buf[end] != " ":
        end += 1
    removed = "".join(buf[i:end])
    del buf[i:end]
    return i, removed
