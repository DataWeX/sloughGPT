"""
ANSI Markdown renderer for terminal output.

Renders basic Markdown to ANSI-colored terminal text:
- **bold** → bold
- *italic* → dim
- `code` → colored
- ```code blocks``` → colored block
- # Headers → bold + color
- - lists → bullet
- > blockquotes → dim
"""

from __future__ import annotations

import os
import re

# ANSI codes
_COLOR_ENABLED = not os.environ.get("NO_COLOR")
if _COLOR_ENABLED:
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _ITALIC = "\033[3m"
    _UNDERLINE = "\033[4m"
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _BLUE = "\033[34m"
    _MAGENTA = "\033[35m"
    _CYAN = "\033[36m"
    _WHITE = "\033[37m"
    _RESET = "\033[0m"
    _BG_DIM = "\033[2;22m"
else:
    _BOLD = _DIM = _ITALIC = _UNDERLINE = ""
    _RED = _GREEN = _YELLOW = _BLUE = _MAGENTA = _CYAN = _WHITE = _RESET = _BG_DIM = ""


def render_markdown(text: str, width: int = 0) -> str:
    """Render markdown text to ANSI terminal output.

    Args:
        text: Markdown text to render.
        width: Max line width (0 = no wrapping).

    Returns:
        ANSI-colored string ready for terminal output.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result = []
    in_code_block = False
    code_lang = ""
    code_lines = []

    for line in lines:
        # Fenced code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block
                result.append(_render_code_block(code_lines, code_lang))
                code_lines = []
                code_lang = ""
                in_code_block = False
            else:
                # Start code block
                in_code_block = True
                code_lang = line.strip()[3:].strip()
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Empty line
        if not line.strip():
            result.append("")
            continue

        # Headers
        if line.startswith("# "):
            result.append(f"{_BOLD}{_CYAN}{line[2:]}{_RESET}")
            continue
        if line.startswith("## "):
            result.append(f"{_BOLD}{_GREEN}{line[3:]}{_RESET}")
            continue
        if line.startswith("### "):
            result.append(f"{_BOLD}{_YELLOW}{line[4:]}{_RESET}")
            continue

        # Blockquote
        if line.startswith("> "):
            content = line[2:]
            result.append(f"  {_DIM}{_ITALIC}{_render_inline(content)}{_RESET}")
            continue

        # Unordered list
        if re.match(r"^[-*]\s", line):
            content = line[2:]
            result.append(f"  {_CYAN}\u2022{_RESET} {_render_inline(content)}")
            continue

        # Ordered list
        m = re.match(r"^(\d+)\.\s", line)
        if m:
            num = m.group(1)
            content = line[len(num) + 2 :]
            result.append(f"  {_YELLOW}{num}.{_RESET} {_render_inline(content)}")
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", line.strip()):
            result.append(f"  {_DIM}{'─' * 40}{_RESET}")
            continue

        # Regular line
        result.append(f"  {_render_inline(line)}")

    # If we ended inside a code block
    if in_code_block and code_lines:
        result.append(_render_code_block(code_lines, code_lang))

    return "\n".join(result)


def _render_inline(text: str) -> str:
    """Render inline markdown (bold, italic, code, links)."""
    # Bold: **text** or __text__
    text = re.sub(
        r"\*\*(.+?)\*\*",
        f"{_BOLD}\\1{_RESET}",
        text,
    )
    text = re.sub(
        r"__(.+?)__",
        f"{_BOLD}\\1{_RESET}",
        text,
    )

    # Italic: *text* or _text_
    text = re.sub(
        r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)",
        f"{_ITALIC}\\1{_RESET}",
        text,
    )
    text = re.sub(
        r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",
        f"{_ITALIC}\\1{_RESET}",
        text,
    )

    # Inline code: `text`
    text = re.sub(
        r"`([^`]+)`",
        f"{_YELLOW}\\1{_RESET}",
        text,
    )

    # Links: [text](url) → text (url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        f"{_CYAN}\\1{_RESET} {_DIM}(\\2){_RESET}",
        text,
    )

    return text


def _render_code_block(lines: list[str], lang: str = "") -> str:
    """Render a fenced code block."""
    if not lines:
        return ""

    # Language tag
    header = ""
    if lang:
        header = f"  {_DIM}{lang}{_RESET}\n"

    # Indented code block with dim color
    rendered = []
    for line in lines:
        rendered.append(f"  {_DIM}{line}{_RESET}")

    return header + "\n".join(rendered)
