"""Journey suites must skip cleanly when playwright is absent (card 6e826226).

The canonical test env mandates PYTHONNOUSERSITE=1 (a stray user-site
``tests`` package shadows the repo tests/). When playwright only exists in
user-site, every browser journey errored at fixture setup — 175 ERRORs of
pure capability noise. Two invariants keep that from recurring:

1. every browser journey wires the capability guard BEFORE any driver import,
2. a poisoned playwright (import raises) turns the suite into a clean SKIP
   with a capability reason, never an error.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTEST_INI = ROOT / "pytest.ini"

JOURNEY_FILES = [
    "packages/core-py/tests/test_user_journeys.py",
    "packages/core-py/tests/test_comprehensive_training_journeys.py",
    "packages/core-py/tests/test_computer_use_training_integration.py",
    "packages/core-py/tests/test_e2e_training_trigger.py",
]

# Imports that can transitively pull playwright into fixture setup.
DRIVER_NEEDLES = ("from avion import", "ComputerUseAgent", "sync_playwright")

GUARD_RE = re.compile(r'importorskip\(\s*"playwright\.sync_api"')


def test_guard_wired_before_driver_imports():
    for rel in JOURNEY_FILES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        match = GUARD_RE.search(src)
        assert match, f"{rel} lacks the playwright capability guard (card 6e826226)"
        guard_pos = match.start()
        for needle in DRIVER_NEEDLES:
            pos = src.find(needle)
            if pos != -1:
                assert guard_pos < pos, (
                    f"{rel}: capability guard must precede `{needle}` "
                    "so a missing playwright skips instead of erroring"
                )


def test_poisoned_playwright_skips_with_capability_reason(tmp_path):
    # Model TRUE ABSENCE: a meta_path blocker makes `import playwright`
    # raise ModuleNotFoundError the way a missing package would.
    poison = tmp_path / "poison"
    poison.mkdir(parents=True)
    (poison / "sitecustomize.py").write_text(
        "import sys\n"
        "class _BlockPlaywright:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'playwright' or name.startswith('playwright.'):\n"
        "            raise ModuleNotFoundError(\n"
        "                f'No module named {name!r}', name=name\n"
        "            )\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockPlaywright())\n",
        encoding="utf-8",
    )
    target = ROOT / JOURNEY_FILES[3]  # e2e trigger: 10 tests, no live data deps

    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(poison), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYTEST_INI),
            str(target),
            "-x",
            "-q",
            "-rs",
            "--timeout=60",
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = proc.stdout + proc.stderr
    summary = next(
        (ln for ln in reversed(output.splitlines()) if ln.strip()), ""
    )
    assert "playwright not installed" in output, (
        "poisoned playwright must produce a capability SKIP with reason, "
        f"got exit={proc.returncode}:\n{output[-2500:]}"
    )
    # 0 = clean run, 5 = NO_TESTS_COLLECTED (module skipped at collection —
    # expected when the guarded file is the only target of the invocation).
    assert proc.returncode in (0, 5) and "error" not in summary.lower(), (
        f"capability absence must never surface as an error: {summary!r}\n"
        f"{output[-2500:]}"
    )
