#!/usr/bin/env python3
"""test-doctor — Fast test failure diagnosis for sloughGPT.

Expo-style: one line per result, expand only on failure.

Usage:
    python scripts/test-doctor.py                              # health scan
    python scripts/test-doctor.py <file>                      # diagnose file
    python scripts/test-doctor.py <file>::TestClass::test_fn  # single test
    python scripts/test-doctor.py --recent                    # recent failures
    python scripts/test-doctor.py --flake                     # flaky detector
    python scripts/test-doctor.py --fix-hint <test>           # suggest fix
    python scripts/test-doctor.py --mock-drift                # dead @patch targets
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = REPO_ROOT / ".test-history.jsonl"

# ── ANSI ───────────────────────────────────────────────────────────────

_NC = os.environ.get("NO_COLOR") or not sys.stdout.isatty()
_b = "" if _NC else "\033[1m"
_d = "" if _NC else "\033[2m"
_r = "" if _NC else "\033[0m"
_rd = "" if _NC else "\033[31m"
_gn = "" if _NC else "\033[32m"
_y = "" if _NC else "\033[33m"
_c = "" if _NC else "\033[36m"
_gr = "" if _NC else "\033[90m"

_ok = f"{_gn}✓{_r}"
_no = f"{_rd}✖{_r}"
_co = f"{_gr}›{_r}"


class Spinner:
    """No-op spinner — just runs the task."""

    def __init__(self, msg: str):
        self.msg = msg

    def start(self) -> Spinner:
        return self

    def stop(self) -> None:
        pass


def _p(t: str = "", end: str = "\n") -> None:
    sys.stdout.write(t + end)
    sys.stdout.flush()


def _line(icon: str, msg: str) -> None:
    """One line: icon + message."""
    _p(f"  {icon} {msg}")


def _dim(msg: str) -> str:
    return f"{_d}{msg}{_r}"


def _red(msg: str) -> str:
    return f"{_rd}{msg}{_r}"


def _green(msg: str) -> str:
    return f"{_gn}{msg}{_r}"


# ── Pytest ─────────────────────────────────────────────────────────────


def _run(args: list[str], timeout: int = 120, tb: str = "short") -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "pytest"] + args + [f"--tb={tb}", "-q", "--no-header"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT))
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"TIMEOUT after {timeout}s"


def _counts(stdout: str, stderr: str) -> dict[str, int]:
    c = {"p": 0, "f": 0, "e": 0, "s": 0}
    for line in stdout.splitlines() + stderr.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m:
            c["p"] = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m:
            c["f"] = int(m.group(1))
        m = re.search(r"(\d+) error", line)
        if m:
            c["e"] = int(m.group(1))
        m = re.search(r"(\d+) skipped", line)
        if m:
            c["s"] = int(m.group(1))
    return c


def _results(stdout: str) -> list[tuple[str, str]]:
    out = []
    for line in stdout.splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line.strip())
        if m:
            out.append((m.group(2), m.group(1).lower()))
    return out


# ── Diagnosis ──────────────────────────────────────────────────────────


def _source(nodeid: str, n: int = 6) -> str:
    parts = nodeid.split("::")
    if len(parts) < 2:
        return ""
    fp = REPO_ROOT / parts[0]
    if not fp.exists():
        return ""
    try:
        lines = fp.read_text().splitlines()
    except Exception:
        return ""
    name = parts[-1]
    for i, line in enumerate(lines):
        if f"def {name}(" in line or f"def {name}:" in line:
            s, e = max(0, i - n), min(len(lines), i + n + 1)
            return "\n".join(
                f"  {'→' if j == i else ' '} {j + 1:4d} │ {lines[j]}" for j in range(s, e)
            )
    return ""


def _git_log(filepath: str, n: int = 3) -> list[str]:
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", f"-{n}", "--", filepath],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(REPO_ROOT),
        )
        return [l for l in r.stdout.strip().splitlines() if l]
    except Exception:
        return []


def _related(nodeid: str, lim: int = 5) -> list[str]:
    parts = nodeid.split("::")
    if len(parts) < 2:
        return []
    fp = REPO_ROOT / parts[0]
    if not fp.exists():
        return []
    try:
        text = fp.read_text()
    except Exception:
        return []
    results = [
        f"{parts[0]}::{m.group(1)}"
        for m in re.finditer(r"def (test_\w+)\(", text)
        if f"{parts[0]}::{m.group(1)}" != nodeid
    ]
    return results[:lim]


# ── Failure patterns ───────────────────────────────────────────────────

_PAT: list[tuple[str, str, str]] = [
    (
        r"ImportError: cannot import name '(\w+)' from '(\w+)'",
        "import",
        "'{0}' not in '{1}' — check __init__.py exports",
    ),
    (
        r"ModuleNotFoundError: No module named '(\S+)'",
        "module",
        "'{0}' not installed — pip install {0}",
    ),
    (r"AssertionError: (.+)", "assert", "{0}"),
    (r"TypeError: .+takes (\d+) positional.*?but (\d+)", "arity", "expects {0} args, got {1}"),
    (r"KeyError: (.+)", "key", "{0} not found — check spelling"),
    (r"AttributeError: (.+)", "attr", "{0}"),
    (r"FileNotFoundError:.*'(.+?)'", "file", "{0} not found"),
    (r"(asyncio\.)?TimeoutError", "timeout", "timed out — check for deadlock or increase timeout"),
    (r"DID NOT RAISE", "no-raise", "expected exception but none was raised"),
    (r"SystemExit\((\d+)\)", "exit", "sys.exit({0}) called — use exception instead"),
]


def _diagnose(output: str) -> tuple[str, str] | None:
    for pat, label, msg in _PAT:
        m = re.search(pat, output)
        if m:
            try:
                return label, msg.format(*m.groups())
            except (IndexError, KeyError):
                return label, msg
    return None


# ── History ────────────────────────────────────────────────────────────


def _append(results: list[tuple[str, str]], dur: float) -> None:
    entry = {"ts": time.time(), "dur": round(dur, 2), "r": [{"n": n, "o": o} for n, o in results]}
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def _history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    out = []
    with open(HISTORY_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return out


# ═══════════════════════════════════════════════════════════════════════
# Commands — one line per result
# ═══════════════════════════════════════════════════════════════════════


def cmd_health(quiet: bool = False, summary: bool = False, verbose: bool = False) -> int:
    areas = [
        ("CLI framework", "tests/cli/test_slo_cli.py"),
        ("CLI commands", "tests/cli/"),
        ("Core Python", "tests/core-py/"),
        ("Root tests", "tests/"),
    ]

    if not quiet and not summary:
        _p(f"\n  {_b}{_c}Test Health{_r}")

    all_ok = True
    total_p = total_f = total_e = 0

    for name, path in areas:
        full = REPO_ROOT / path
        if not full.exists():
            if not quiet and not summary:
                _line(_co, f"{name:20s} {_dim(path + ' not found')}")
            continue

        sp = Spinner(f"running {name}...").start()
        t0 = time.time()
        code, out, err = _run([path], timeout=120)
        dur = time.time() - t0
        c = _counts(out, err)
        total_p += c["p"]
        total_f += c["f"]
        total_e += c["e"]
        sp.stop()

        if code == 0:
            msg = f"{name:20s} {c['p']} passed  {_dim(f'{dur:.1f}s')}"
            if not quiet and not summary:
                _line(_ok, msg)
        else:
            all_ok = False
            parts = [f"{c['p']} passed"]
            if c["f"]:
                parts.append(_red(f"{c['f']} failed"))
            if c["e"]:
                parts.append(_red(f"{c['e']} error"))
            msg = f"{name:20s} {', '.join(parts)}  {_dim(f'{dur:.1f}s')}"
            if not quiet and not summary:
                _line(_no, msg)
                if verbose:
                    for line in err.splitlines()[-5:]:
                        _p(f"           {_dim(line.strip())}")

    _p()
    if all_ok:
        _line(_ok, f"{_green('All healthy')}  {_dim(f'{total_p} passed')}")
    else:
        _line(_no, f"{_red(f'{total_f + total_e} failures')}")
        if not quiet and not summary:
            _p(f"           {_dim('run: test-doctor <file> to diagnose')}")
    _p()
    return 0 if all_ok else 1


def cmd_diagnose(target: str) -> int:
    fp = REPO_ROOT / target.split("::")[0]
    if not fp.exists():
        _p(f"\n  {_no} {_red(target + ' not found')}\n")
        return 1

    _p(f"\n  {_b}{_c}Test Doctor{_r}  {target}")

    sp = Spinner("running...").start()
    t0 = time.time()
    code, out, err = _run([target, "--tb=short", "-v"], timeout=60)
    dur = time.time() - t0
    c = _counts(out, err)
    res = _results(out)
    sp.stop()

    parts = []
    if c["p"]:
        parts.append(_green(f"{c['p']}P"))
    if c["f"]:
        parts.append(_red(f"{c['f']}F"))
    if c["e"]:
        parts.append(_red(f"{c['e']}E"))
    if c["s"]:
        parts.append(f"{c['s']}S")
    parts.append(_dim(f"{dur:.1f}s"))
    _line(" ", " ".join(parts))

    # Failures — one line each, then expand first
    if code != 0:
        fails = [(n, o) for n, o in res if o in ("failed", "error")]
        for n, _ in fails:
            _line(_no, n)

        # Expand first failure
        if fails:
            first = fails[0][0]
            combined = out + "\n" + err

            # Pattern diagnosis
            diag = _diagnose(combined)
            if diag:
                label, detail = diag
                _p(f"           {_c}{label}{_r}: {detail}")

            # Source
            src = _source(first)
            if src:
                _p()
                for line in src.splitlines():
                    _p(f"  {line}")

            # Git log — one line
            glog = _git_log(first.split("::")[0])
            if glog:
                _p(f"           {_dim(glog[0])}")

            # Related — one line
            rel = _related(first)
            if rel:
                _p(f"           {_dim('also: ' + ', '.join(t.split('::')[-1] for t in rel))}")

    _p(f"\n  {_b}re-run:{_r} python3 -m pytest {target} -v --tb=short\n")

    if res:
        _append(res, dur)

    return code


def cmd_fix_hint(target: str) -> int:
    _p(f"\n  {_b}{_c}Fix Hint{_r}  {target}")

    sp = Spinner("running with full traceback...").start()
    t0 = time.time()
    code, out, err = _run([target, "--tb=long", "-v"], timeout=60)
    dur = time.time() - t0
    sp.stop()

    if code == 0:
        _line(_ok, f"{_green('passing')}  {_dim(f'{dur:.1f}s')}")
        _p()
        return 0

    combined = out + "\n" + err

    # Compact traceback — only the interesting lines
    _p(f"  {_dim('traceback:')}")
    capture = False
    for line in combined.splitlines():
        if "FAILED" in line or "Traceback" in line:
            capture = True
        if capture:
            if line.strip().startswith("===") or line.strip().startswith("---"):
                capture = False
                continue
            # Skip pytest internal frames
            if "site-packages" in line or "_pytest" in line:
                continue
            _p(f"    {line}")

    # Diagnosis
    diag = _diagnose(combined)
    if diag:
        label, detail = diag
        _p(f"\n  {_y}{label}{_r}: {detail}")

    # Source — wider context
    src = _source(target, n=10)
    if src:
        _p()
        for line in src.splitlines():
            _p(f"  {line}")

    # Related
    rel = _related(target)
    if rel:
        _p(f"\n  {_dim('also: ' + ', '.join(t.split('::')[-1] for t in rel))}")

    _p(f"\n  {_b}re-run:{_r} python3 -m pytest {target} -v --tb=long\n")
    return 1


def cmd_recent(n: int = 20) -> int:
    history = _history()
    if not history:
        _p(f"\n  {_dim('no history yet')}\n")
        return 0

    _p(f"\n  {_b}{_c}Recent{_r}  last {n} runs\n")

    recent = history[-n:]
    by_test: dict[str, list[str]] = defaultdict(list)
    for entry in recent:
        ts = time.strftime("%H:%M", time.localtime(entry.get("ts", 0)))
        for r in entry.get("r", []):
            if r.get("o") in ("failed", "error"):
                by_test[r["n"]].append(ts)

    if not by_test:
        _line(_ok, _green("no failures"))
    else:
        for nid, times in sorted(by_test.items(), key=lambda x: -len(x[1])):
            _line(_no, f"{nid}  {_dim(f'{len(times)}x, last {times[-1]}')}")

    _p()
    return 0


def cmd_flake() -> int:
    history = _history()
    _p(f"\n  {_b}{_c}Flaky{_r}\n")

    if len(history) < 2:
        _p(f"  {_dim('need 2+ runs to detect')}\n")
        return 0

    outcomes: dict[str, list[bool]] = defaultdict(list)
    for entry in history:
        seen = set()
        for r in entry.get("r", []):
            nid = r["n"]
            if nid not in seen:
                outcomes[nid].append(r["o"] == "passed")
                seen.add(nid)

    flaky = [
        (nid, sum(r) / len(r), len(r))
        for nid, r in outcomes.items()
        if len(r) >= 2 and any(r) and not all(r)
    ]

    if not flaky:
        _line(_ok, _green("no flakes"))
    else:
        flaky.sort(key=lambda x: x[1])
        for nid, rate, _runs in flaky:
            bar_len = 16
            filled = int(rate * bar_len)
            clr = _gn if rate > 0.8 else _y if rate > 0.5 else _rd
            bar = f"{clr}{'█' * filled}{_d}{'░' * (bar_len - filled)}{_r}"
            _p(f"  {_no} {bar} {rate:.0%}  {nid}")

    _p()
    return 0


# ── Mock drift ─────────────────────────────────────────────────────────
#
# The bug class: a test patches the *definition* site
#
#     @patch("domain.feedback._internal.per_user_lora.get_per_user_lora")
#
# while production reads the *re-export* at package level
#
#     from domain.feedback import get_per_user_lora     # routers/user_adapters.py:54
#
# If `domain/feedback/__init__.py` binds that name EAGERLY (a plain
# `from ... import ...` with no `__getattr__`), the package attribute is frozen
# at import time and the patch never reaches the reader — the test silently runs
# against live state. Lazy re-export (`__getattr__` that re-imports per access)
# does follow the patch.
#
# So a target is only worth reporting when BOTH hold:
#   (1) EMPIRICAL — patching `_internal.X.attr` does not move `reader.attr`, and
#   (2) RELEVANT  — some test that patches the target imports the module doing
#                   the non-patched read.
# (1) alone over-reports: if the code under test imports `_internal` directly
# (function-level import), the patch lands and the package binding is irrelevant.

# One test root since 2026-10-08 (was: packages/core-py/tests, tests,
# apps/cli/tests, apps/api/server/tests — four roots, 86 colliding basenames).
# `tests` rglobs the three subdirs, so a single entry is the non-duplicative
# form; listing them separately would rescan the same files.
_MOCK_DRIFT_TEST_DIRS = ("tests",)
_MOCK_DRIFT_PROD_ROOTS = ("domain", "apps/api/server", "apps/cli/src", "packages")
# Longest-prefix-wins roots, mirroring pytest.ini's `pythonpath`.
# NOTE: `domain` is deliberately NOT a root — it is a package directory under
# the repo root (pytest.ini puts `.` on the path), so `domain/feedback/` maps
# to `domain.feedback`. Treating it as a root dropped the prefix and broke
# relative-import resolution across all of `domain/`.
_MOCK_DRIFT_PATH_ROOTS = (
    ("apps", "api", "server"),
    ("apps", "cli", "src"),
    ("packages", "core-py"),
    ("packages", "mogdb", "src"),
    ("packages", "chargectl", "src"),
    ("packages", "downcraft"),
    ("packages", "infra-lib"),
    (),
)
_MD_PATCH_RE = re.compile(r'patch\(\s*["\']([A-Za-z0-9_.]+)["\']')
_MD_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$")
_MD_ANY_IMPORT_RE = re.compile(
    r"^[ \t]*(?:from[ \t]+([A-Za-z0-9_.]+)[\t ]+import|import[ \t]+([A-Za-z0-9_.]+))",
    re.MULTILINE,
)
_MD_MISSING = object()


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _md_setup_path() -> None:
    """Put pytest.ini's shared pythonpath on sys.path — one copy, one config."""
    ini = REPO_ROOT / "pytest.ini"
    if not ini.exists():
        return
    for line in ini.read_text().splitlines():
        if line.strip().startswith("pythonpath"):
            _, _, val = line.partition("=")
            for part in val.split():
                s = str((REPO_ROOT / part).resolve())
                if s not in sys.path:
                    sys.path.insert(0, s)
            return


def _md_path_module(p: Path) -> str | None:
    """Map a repo file to its dotted module path."""
    s = str(p.relative_to(REPO_ROOT))
    if s.endswith("/__init__.py"):
        s = s[: -len("/__init__.py")]
    elif s.endswith(".py"):
        s = s[: -len(".py")]
    else:
        return None
    parts = s.split("/")
    for root in _MOCK_DRIFT_PATH_ROOTS:
        if tuple(parts[: len(root)]) == root and len(parts) > len(root):
            return ".".join(parts[len(root) :])
    return None


def _md_resolve(module: str, importer: Path) -> str:
    """Resolve a (possibly relative) import to an absolute dotted path."""
    if not module.startswith("."):
        return module
    base = _md_path_module(importer.with_name("__init__.py")) or str(
        importer.parent.relative_to(REPO_ROOT)
    ).replace("/", ".")
    dots = len(module) - len(module.lstrip("."))
    rest = module.lstrip(".")
    parts = base.split(".")
    if dots - 1 > 0:
        parts = parts[: len(parts) - (dots - 1)]
    if rest:
        parts += rest.split(".")
    return ".".join(x for x in parts if x)


def _md_targets() -> dict[str, set[Path]]:
    """patch target -> the test files that patch it (definition-site only)."""
    out: dict[str, set[Path]] = {}
    for d in _MOCK_DRIFT_TEST_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            for m in _MD_PATCH_RE.finditer(txt):
                t = m.group(1)
                if "._internal." in t:
                    out.setdefault(t, set()).add(f)
    return out


def _md_readers() -> dict[str, dict[str, set[str]]]:
    """attr -> reader module -> the file-modules doing `from reader import attr`."""
    out: dict[str, dict[str, set[str]]] = {}
    for root_s in _MOCK_DRIFT_PROD_ROOTS:
        root = REPO_ROOT / root_s
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            if "/tests/" in str(f):
                continue
            try:
                lines = f.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            me = _md_path_module(f)
            for line in lines:
                if line.lstrip().startswith("#"):
                    continue
                m = _MD_FROM_IMPORT_RE.match(line)
                if not m:
                    continue
                mod_raw, names = m.group(1), m.group(2)
                if "#" in names:
                    names = names.split("#", 1)[0]
                mod = _md_resolve(mod_raw, f)
                for part in re.split(r"[(),]", names):
                    name = part.strip().split(" as ")[0].strip()
                    if name and name.isidentifier() and name != "*":
                        out.setdefault(name, {}).setdefault(mod, set()).add(me or str(f))
    return out


def _md_test_imports(test: Path, module: str) -> bool:
    """Does the test import `module` (or a package containing it)?"""
    try:
        txt = test.read_text(errors="ignore")
    except OSError:
        return False
    for m in _MD_ANY_IMPORT_RE.finditer(txt):
        imported = m.group(1) or m.group(2) or ""
        if not imported:
            continue
        if (
            module == imported
            or module.startswith(imported + ".")
            or imported.startswith(module + ".")
        ):
            return True
    return False


def _md_reaches(target: str, reader_mod: str, attr: str) -> bool | None:
    """Does patching `target` move `reader_mod.attr`?

    True  = patch lands (reader is lazy / reads the definition site directly)
    False = patch is dead for that reader (eager re-export frozen at import)
    None  = could not evaluate (module missing / attr absent / patch failed)
    """
    tgt_mod = target.rsplit(".", 1)[0]
    try:
        importlib.import_module(tgt_mod)
        reader = importlib.import_module(reader_mod)
        before = getattr(reader, attr)
    except Exception:
        return None
    try:
        with mock.patch(target):
            after = getattr(reader, attr, _MD_MISSING)
    except Exception:
        return None
    return after is not before


def _md_create_true(txt: str, target: str) -> bool:
    """True if the patch() call naming `target` passes create=True.

    `create=True` means "add the attribute if absent" — an intentional mock of
    a not-yet-existing name (e.g. domain.shell._internal.repl.TuiRepl). Those
    must not be reported as broken.
    """
    i = -1
    for q in ('"', "'"):
        i = txt.find(f"{q}{target}{q}")
        if i != -1:
            break
    if i == -1:
        return False
    j = txt.rfind("patch(", 0, i)
    if j == -1:
        return False
    depth = 0
    for k in range(j + len("patch"), len(txt)):
        if txt[k] == "(":
            depth += 1
        elif txt[k] == ")":
            depth -= 1
            if depth == 0:
                return bool(re.search(r"create\s*=\s*True", txt[j : k + 1]))
    return False


def _md_unresolvable(
    targets: dict[str, set[Path]],
) -> list[tuple[str, str, list[str]]]:
    """Targets where `patch()` itself raises — the test errors at setup.

    These are loud, not silent, but they can be invisible for just that reason:
    a `slow`-marked file is deselected, so its ERRORs never reach a normal run
    (that is exactly how test_chat_loop_e2e hid 6 of them).
    """
    out: list[tuple[str, str, list[str]]] = []
    for target, tfiles in sorted(targets.items()):
        texts = []
        for f in tfiles:
            try:
                texts.append(f.read_text(errors="ignore"))
            except OSError:
                continue
        if any(_md_create_true(t, target) for t in texts):
            continue
        try:
            with mock.patch(target):
                pass
        except Exception as exc:
            out.append(
                (
                    target,
                    f"{type(exc).__name__}: {exc}",
                    sorted(_rel(f) for f in tfiles),
                )
            )
    return out


def _md_import_graph() -> dict[str, set[str]]:
    """module -> repo modules it imports (static scan of production files).

    Used to follow a test's imports past the first hop: a test may import
    router A, which imports router B, and B is the module that reads the
    un-patched path. Direct-import matching alone cannot see that.
    """
    graph: dict[str, set[str]] = {}
    for root_s in _MOCK_DRIFT_PROD_ROOTS:
        root = REPO_ROOT / root_s
        if not root.exists():
            continue
        for f in root.rglob("*.py"):
            if "/tests/" in str(f):
                continue
            mod = _md_path_module(f)
            if not mod:
                continue
            try:
                txt = f.read_text(errors="ignore")
            except OSError:
                continue
            deps: set[str] = set()
            for m in _MD_ANY_IMPORT_RE.finditer(txt):
                imp = m.group(1) or m.group(2) or ""
                if imp:
                    deps.add(imp)
            graph[mod] = deps
    return graph


def _md_direct_imports(path: Path) -> set[str]:
    try:
        txt = path.read_text(errors="ignore")
    except OSError:
        return set()
    out: set[str] = set()
    for m in _MD_ANY_IMPORT_RE.finditer(txt):
        imp = m.group(1) or m.group(2) or ""
        if imp:
            out.add(imp)
    return out


def _md_reachable(start: set[str], graph: dict[str, set[str]], depth: int) -> set[str]:
    seen = set(start)
    frontier = set(start)
    for _ in range(depth):
        nxt = set()
        for mod in frontier:
            for dep in graph.get(mod, ()):
                if dep not in seen:
                    seen.add(dep)
                    nxt.add(dep)
        if not nxt:
            break
        frontier = nxt
    return seen


def cmd_mock_drift(verbose: bool = False) -> int:
    _p(f"\n  {_b}{_c}Mock drift{_r}\n")
    _p(f"  {_dim('patch() targets that production no longer reads')}\n")
    _md_setup_path()

    targets = _md_targets()
    readers = _md_readers()

    confirmed: list[tuple[str, str, list[str]]] = []
    # (target, reader_mod, qualifying reader file-modules, patching tests)
    latent: list[tuple[str, str, set[str], set[Path]]] = []
    ok = 0

    for target, tfiles in sorted(targets.items()):
        mod, attr = target.rsplit(".", 1)
        by_mod = readers.get(attr, {})
        others = {r: fs for r, fs in by_mod.items() if r != mod and "._internal." not in r}
        if not others:
            continue  # nothing in production reads this any other way
        for reader_mod, reader_files in others.items():
            # Files that read via reader_mod. If the ONLY such file is the
            # module the patch lands in, the patch already covers it: it
            # replaces the module-global those reads use. The empirical fact
            # "patching doesn't move reader_mod.attr" is then irrelevant --
            # nothing in the code under test reads reader_mod.attr directly.
            # (e.g. patching <pkg>._internal.training.MogDB while training.py
            # itself does `from mogdb import MogDB`: the local rebinding IS
            # the mock.) Without this filter that case reports as confirmed.
            qualifying = {rf for rf in reader_files if rf != mod}
            if not qualifying:
                continue
            reaches = _md_reaches(target, reader_mod, attr)
            if reaches is None:
                continue
            if reaches:
                ok += 1
                continue
            hits = sorted(
                {
                    f"{rf}  <-  {_rel(tf)}"
                    for rf in qualifying
                    for tf in tfiles
                    if _md_test_imports(tf, rf)
                }
            )
            if hits:
                confirmed.append((target, reader_mod, hits))
            else:
                latent.append((target, reader_mod, qualifying, set(tfiles)))

    # Third class — ADVISORY. The reader is not imported directly by the test,
    # but is reachable within two imports (test -> router A -> router B, where
    # B reads the un-patched path). Import reach is weaker evidence than
    # runtime use, so these are leads, not verdicts: they never set the exit
    # code. Without this hop, the latent count reads as "all clear" when a
    # second router is quietly using live state.
    suspects: list[tuple[str, str, list[str], list[str]]] = []
    if latent:
        graph = _md_import_graph()
        for target, reader_mod, qualifying, tfiles in latent:
            reach: set[str] = set()
            for tf in tfiles:
                reach |= _md_reachable(_md_direct_imports(tf), graph, 2)
            hit = qualifying & reach
            if hit:
                suspects.append((target, reader_mod, sorted(hit), sorted(_rel(t) for t in tfiles)))

    for target, reader_mod, hits in confirmed:
        _line(_no, f"{_red(target)}")
        _p(f"      read via {_b}{reader_mod}{_r}  (patch never reaches it)")
        for h in hits[:3]:
            _p(f"      {_dim(h)}")
        _p(f'      {_gn}fix{_r}: patch({_c}"{reader_mod}.{target.rsplit(".", 1)[1]}"{_r})')

    if confirmed:
        _p()
        _line(
            _no,
            _red(
                f"{len(confirmed)} confirmed  ({ok} targets follow the patch, "
                f"{len(latent)} latent, {len(suspects)} suspect)"
            ),
        )
    else:
        _line(
            _ok,
            _green(f"no drift  ({ok} targets follow the patch, {len(suspects)} suspect)"),
        )

    # Second class: the target itself cannot be resolved, so patch() raises
    # and the test errors at setup. Loud — unless the file is deselected.
    unresolvable = _md_unresolvable(targets)
    if unresolvable:
        _p()
        _p(f"  {_b}{_rd}Unresolvable{_r}  {_dim('patch() raises at setup — the test errors')}")
        for target, err, files in unresolvable:
            _line(_no, f"{_red(target)}")
            _p(f"      {_dim(err)}")
            for f in files[:2]:
                _p(f"      {_dim(f)}")

    if suspects:
        _p()
        _p(
            f"  {_b}{_y}Suspects{_r}  "
            f"{_dim('reader reachable within 2 imports — verify, not verdicts')}"
        )
        for target, reader_mod, hits, files in suspects:
            _line(_no, f"{_y}{target}{_r}")
            _p(f"      read via {_b}{reader_mod}{_r}")
            for h in hits[:2]:
                _p(f"      {_dim(h)}")
            for f in files[:1]:
                _p(f"      {_dim(f)}")

    if latent:
        _p(
            f"  {_dim(f'{len(latent)} latent — reader exists but no test imports it')}"
            + (f"{_dim(f', {len(suspects)} of them reachable within 2 hops')}" if suspects else "")
        )
        if verbose:
            for target, reader_mod, _q, _t in latent:
                _p(f"    {_dim(f'{target}  (via {reader_mod})')}")

    _p()
    return 1 if (confirmed or unresolvable) else 0


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="test-doctor",
        description="Fast test failure diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              test-doctor                              health scan
              test-doctor tests/test_foo.py            diagnose file
              test-doctor --fix-hint <test>::test_fn   suggest fix
              test-doctor --mock-drift                 dead @patch targets
        """),
    )
    ap.add_argument("target", nargs="?")
    ap.add_argument("--recent", "-r", action="store_true")
    ap.add_argument("--flake", "-f", action="store_true")
    ap.add_argument("--fix-hint", metavar="TEST")
    ap.add_argument("--mock-drift", action="store_true", help="dead @patch targets")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--quiet", "-q", action="store_true", help="minimal output")
    ap.add_argument("--summary", "-s", action="store_true", help="summary only")
    ap.add_argument("--verbose", "-v", action="store_true", help="detailed output")
    args = ap.parse_args()

    if args.no_color:
        global _b, _d, _r, _rd, _gn, _y, _c, _gr, _NC
        _NC = True
        _b = _d = _r = _rd = _gn = _y = _c = _gr = ""

    if args.mock_drift:
        return cmd_mock_drift(verbose=args.verbose)
    if args.fix_hint:
        return cmd_fix_hint(args.fix_hint)
    if args.flake:
        return cmd_flake()
    if args.recent:
        return cmd_recent()
    if args.target:
        return cmd_diagnose(args.target)
    return cmd_health(quiet=args.quiet, summary=args.summary)


if __name__ == "__main__":
    sys.exit(main())
