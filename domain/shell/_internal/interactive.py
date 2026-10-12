"""InteractivePrompt — assembled from one mixin per feature family.

The public surface is unchanged: still constructed as `InteractivePrompt(io)`, still carrying all 126 public prompts and their raw paths, and still the name every caller imports.

The MRO below *is* the layering. `_InputPrompts` is the substrate every other family composes; the rest are ordered so a family only ever inherits from families declared before it.

The module-level names below are re-exported verbatim from `interactive_term`, which holds them unchanged. There is deliberately no `__all__` here either, matching the original: `import *` must keep yielding the same surface.
"""

from __future__ import annotations

from domain.shell._internal.interactive_confirm import _ConfirmPrompts
from domain.shell._internal.interactive_input import _InputPrompts
from domain.shell._internal.interactive_picker import _PickerPrompts
from domain.shell._internal.interactive_progress import _ProgressPrompts
from domain.shell._internal.interactive_prompt import _PromptPrompts
from domain.shell._internal.interactive_screen import _ScreenPrompts
from domain.shell._internal.interactive_select import _SelectPrompts
from domain.shell._internal.interactive_spinner import (
    _InteractiveSpinner,  # noqa: F401
    _SpinnerPrompts,
)
from domain.shell._internal.interactive_table import _TablePrompts
from domain.shell._internal.interactive_term import (  # noqa: F401
    _BOLD,
    _CURSOR_UP,
    _CYAN,
    _DIM,
    _ERASE_LINE,
    _GREEN,
    _HIDE_CURSOR,
    _KEY_BACKSPACE,
    _KEY_CHAR,
    _KEY_CTRL_A,
    _KEY_CTRL_C,
    _KEY_CTRL_E,
    _KEY_DELETE,
    _KEY_DOWN,
    _KEY_END,
    _KEY_ENTER,
    _KEY_ESC,
    _KEY_HOME,
    _KEY_LEFT,
    _KEY_PAGE_DOWN,
    _KEY_PAGE_UP,
    _KEY_RIGHT,
    _KEY_SPACE,
    _KEY_TAB,
    _KEY_UP,
    _RED,
    _RESET,
    _RESET_REVERSE,
    _REVERSE,
    _SHOW_CURSOR,
    _YELLOW,
    _dash_row,
    _RawKey,
    _RawTerminal,
    _read_raw_key,
    _terminal_height,
    _terminal_width,
    _truncate,
)
from domain.shell._internal.interactive_widget import _WidgetPrompts


class InteractivePrompt(
    _ScreenPrompts,
    _SpinnerPrompts,
    _ProgressPrompts,
    _TablePrompts,
    _PickerPrompts,
    _SelectPrompts,
    _ConfirmPrompts,
    _WidgetPrompts,
    _PromptPrompts,
    _InputPrompts,
):
    """Every interactive prompt, one mixin per feature family.

    Construct exactly as before: `InteractivePrompt(io)`.
    """
