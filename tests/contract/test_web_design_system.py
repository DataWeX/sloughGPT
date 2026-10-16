"""Design-system conformance ratchet for apps/web (card 289a8e6c).

The audit of 2026-10-16 found the app carrying a large ad-hoc layer BESIDE
@sloughgpt/strui: raw <button> sites, hardcoded hex/rgb literals, and
div/span onClick "fake buttons". Much of it is legitimate (chrome-free
toggle links, canvas/chart renderers where CSS tokens do not reach a 2d
context, the magazine design showcase), so a blanket ban would be wrong.
This gate freezes the layer: per-file baselines may SHRINK, never grow.
New UI is built from the strui library elements; conversions lower the
baseline (this file then fails "baseline not inflated" until it is
regenerated lower - that failure is the ratchet working).

Regenerate the baseline after a conversion sweep:
    python tests/contract/test_web_design_system.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "apps" / "web"
BASELINE_PATH = Path(__file__).resolve().parent / "web_design_system_baseline.json"

# Test files assert against mocks; .gen.* is emitted by scripts; node_modules
# and dist are not source.
SKIP_SUBSTRINGS = ("node_modules", "/dist/", ".test.", ".gen.", "__snapshots__")

# Longest-first hex alternation with a lookahead so '#ffffff' is one hit and
# never a partial '#fff' hit; rgb()/rgba() calls cover JS-composed colors.
COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])|rgba?\(")

# A div/span with an onClick is an interactive element wearing presentational
# clothing: no button semantics, no keyboard, no focus. Two in-tree patterns
# are NOT fake buttons and are skipped by shape, not by allowlist:
#   - modal backdrops carrying aria-hidden (click-outside-to-close)
#   - content wrappers whose handler is a bare stopPropagation inside a row
FAKE_BUTTON_RE = re.compile(r"<(?:div|span)\b[^>]*onClick\s*=", re.S)


def _source_files():
    for path in sorted(WEB.rglob("*.tsx")):
        text = str(path)
        if any(s in text for s in SKIP_SUBSTRINGS):
            continue
        yield path


def scan() -> dict[str, dict[str, int]]:
    raw_button_files: dict[str, int] = {}
    color_literal_files: dict[str, int] = {}
    fake_button_files: dict[str, int] = {}
    for path in _source_files():
        text = path.read_text(errors="ignore")
        rel = str(path.relative_to(REPO))

        buttons = text.count("<button")
        if buttons:
            raw_button_files[rel] = buttons

        colors = len(COLOR_RE.findall(text))
        if colors:
            color_literal_files[rel] = colors

        fake = 0
        for match in FAKE_BUTTON_RE.finditer(text):
            tag = match.group(0)
            if "aria-hidden" in tag:
                continue
            if "stopPropagation" in tag:
                continue
            fake += 1
        if fake:
            fake_button_files[rel] = fake

    return {
        "raw_button_files": raw_button_files,
        "color_literal_files": color_literal_files,
        "fake_button_files": fake_button_files,
    }


def _load_baseline() -> dict[str, dict[str, int]]:
    if not BASELINE_PATH.exists():
        return {"raw_button_files": {}, "color_literal_files": {}, "fake_button_files": {}}
    return json.loads(BASELINE_PATH.read_text())


def _assert_frozen(rule: str, section: str) -> None:
    baseline = _load_baseline().get(section, {})
    current = scan()[section]
    grown = {f: (baseline.get(f, 0), n) for f, n in current.items() if n > baseline.get(f, 0)}
    assert not grown, (
        f"design-system ratchet ({rule}): ad-hoc layer GREW in {sorted(grown)} "
        f"(was, now)={grown}. Build new UI from @sloughgpt/strui elements "
        f"(Button, Input, ...) instead of raw markup - or, if this is a "
        f"conversion, regenerate the baseline with:\n"
        f"    python tests/contract/test_web_design_system.py"
    )


def _assert_not_inflated(section: str) -> None:
    baseline = _load_baseline().get(section, {})
    current = scan()[section]
    shrunk = {f: (n, current.get(f, 0)) for f, n in baseline.items() if current.get(f, 0) < n}
    assert not shrunk, (
        f"design-system ratchet: baseline is INFLATED for {sorted(shrunk)} "
        f"(baseline, now)={shrunk} - a conversion happened without lowering the "
        f"bar. Regenerate with:\n    python tests/contract/test_web_design_system.py"
    )


def test_raw_button_counts_frozen() -> None:
    """No file may gain raw <button> markup; new files must start at zero."""
    _assert_frozen("raw buttons", "raw_button_files")


def test_hardcoded_color_literals_frozen() -> None:
    """No file may gain hex/rgb literals; new pages use Noir Violet tokens."""
    _assert_frozen("hardcoded colors", "color_literal_files")


def test_no_new_fake_interactive_elements() -> None:
    """div/span onClick (non-backdrop, non-stopPropagation) may not grow."""
    _assert_frozen("fake buttons", "fake_button_files")


def test_baseline_not_inflated_raw_buttons() -> None:
    _assert_not_inflated("raw_button_files")


def test_baseline_not_inflated_colors() -> None:
    _assert_not_inflated("color_literal_files")


def test_baseline_not_inflated_fake_buttons() -> None:
    _assert_not_inflated("fake_button_files")


if __name__ == "__main__":
    BASELINE_PATH.write_text(json.dumps(scan(), indent=2, sort_keys=True) + "\n")
    print(f"baseline written: {BASELINE_PATH.relative_to(REPO)}")
