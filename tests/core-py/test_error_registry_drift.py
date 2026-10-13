"""ErrorCode is the contract for every E_ code the system can put on the wire.

``domain/infrastructure/_internal/errors.py`` documents ``ErrorCode`` as
"Every E_ code used in the system lives here", but nothing enforced it. A scan
found 36 codes in active use that were absent, and ``raise_error()`` catches
``(ValueError, KeyError)`` and silently downgrades an unknown code to a generic
``AppError`` with status 400 — losing the class, the ``recoverable`` flag and
the real status. A 503 readiness code and a 500 upstream failure both arrived
at the client as a 400 with no log line anywhere.

These tests close that hole by scanning every path that can emit a code:

1. ``raise_error(msg, "E_X", ...)``   — the resolver that consults the registry
2. ``error_response(msg, "E_X", ...)`` — the hand-built envelope constructor
3. class-level ``code = "E_X"`` on AppError subclasses — the typed layer
4. ``{"code": "E_X"}`` dict literals — readiness/status payloads built by hand

Out of scope on purpose: ``domain/logging`` keeps its own log-event codespace
(``E_MODEL_LOAD``, ``E_AUTH_EXPIRED`` …) which merely shares the ``E_``
prefix, and tests/ may use invented codes to exercise fallback behaviour.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "apps" / "api" / "server"

_CODE_RE = re.compile(r'["\'](E_[A-Z_]{2,})["\']')

# Scanners must find a plausible amount of input. A scan that matches nothing
# passes vacuously — this happened during development when an exclusion pattern
# (``/.wt-``) matched the worktree's own name and zeroed the file list.
_FLOORS = {
    "raise_error": 40,
    "error_response": 5,
    "class_level": 20,
    "dict_envelope": 5,
}


def _registered_codes() -> set[str]:
    """Codes the registry accepts, by value *and* by member name.

    Name matters: ``ErrorCode`` carries legacy aliases declared as
    ``E_CREATE_FAILED = "E_BAD_REQUEST"``, and callers write the member name.
    """
    from domain.infrastructure._internal.errors import ErrorCode

    names = set(ErrorCode.__members__)
    values = {member.value for member in ErrorCode}
    return names | values


def _python_sources() -> list[Path]:
    """Repo-owned Python sources, excluding anything we do not own.

    Two exclusions matter, both learned the hard way:

    * no ``/.wt-`` pattern — the scan is rooted at REPO_ROOT so it cannot reach
      sibling worktrees anyway, and matching that pattern silently filtered out
      *this* worktree (``.wt-guard``) and returned zero files, which the floors
      in ``_FLOORS`` then caught;
    * never *descend* into a virtualenv, vendored tree or cache (``.venv``,
      ``.git``, ``.pytest-libs``, ``site-packages``, ``node_modules``). Filtering
      the results afterwards is not enough: ``rglob`` still walks every directory
      inside ``.venv`` (10 s per run), and a third-party ``code = "E_…"`` would
      fail a gate over code this project does not control.
    """
    skip = {"__pycache__", "node_modules", "site-packages"}
    sources: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # In-place prune: os.walk honours the mutation and skips the subtree.
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip]
        for name in filenames:
            if name.endswith(".py"):
                sources.append(Path(dirpath) / name)
    return sources


def _raise_error_codes() -> dict[str, list[str]]:
    """Every code passed to raise_error(), keyed by code, valued by file."""
    found: dict[str, list[str]] = {}
    for path in sorted(API_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        for match in re.finditer(r"raise_error\(", text):
            window = text[match.end() : match.end() + 400]
            code = _CODE_RE.search(window)
            if code:
                found.setdefault(code.group(1), []).append(str(path.relative_to(REPO_ROOT)))
    return found


def _error_response_codes() -> dict[str, list[str]]:
    """Every code passed to error_response(), keyed by code, valued by file."""
    found: dict[str, list[str]] = {}
    for path in sorted(API_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        for match in re.finditer(r"error_response\(", text):
            window = text[match.end() : match.end() + 400]
            code = _CODE_RE.search(window)
            if code:
                found.setdefault(code.group(1), []).append(str(path.relative_to(REPO_ROOT)))
    return found


def _class_level_codes() -> dict[str, list[str]]:
    """Class-level ``code = "E_X"`` attributes declared outside logging."""
    found: dict[str, list[str]] = {}
    for path in _python_sources():
        rel = str(path.relative_to(REPO_ROOT))
        if any(part in f"/{rel}/" for part in ("/tests/", "/domain/logging/")):
            continue
        text = path.read_text(errors="ignore")
        for match in re.finditer(r'^\s+code(?::\s*str)?\s*=\s*(["\']E_[A-Z_]+["\'])', text, re.M):
            code = _CODE_RE.search(match.group(1))
            if code:
                found.setdefault(code.group(1), []).append(rel)
    return found


def _dict_envelope_codes() -> dict[str, list[str]]:
    """Hand-built ``{"code": "E_X"}`` payloads in the API layer."""
    found: dict[str, list[str]] = {}
    for path in sorted(API_ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        for match in re.finditer(r'["\']code["\']\s*:\s*(["\']E_[A-Z_]+["\'])', text):
            code = _CODE_RE.search(match.group(1))
            if code:
                found.setdefault(code.group(1), []).append(str(path.relative_to(REPO_ROOT)))
    return found


def _unregistered(
    codes: dict[str, list[str]], registered: set[str], *, path: str, effect: str
) -> str:
    offenders = {
        code: sorted(set(files)) for code, files in codes.items() if code not in registered
    }
    if not offenders:
        return ""
    lines = [
        f"  {code}  ({len(files)} site(s): {', '.join(files)})"
        for code, files in sorted(offenders.items())
    ]
    return (
        f"{len(offenders)} code(s) emitted through {path} but absent from ErrorCode — "
        f"{effect}:\n" + "\n".join(lines)
    )


# Each rule names the path the code travels through and the concrete harm of
# leaving it unregistered, so a red run says where to look and why it matters.
_RULES = {
    "raise_error": (
        _raise_error_codes,
        "raise_error()",
        "each was silently downgraded to a generic AppError with status 400, "
        "losing its class, recoverable flag and real status",
    ),
    "error_response": (
        _error_response_codes,
        "error_response()",
        "the envelope advertises a code that no client or monitor can resolve",
    ),
    "class_level": (
        _class_level_codes,
        "an AppError subclass's code attribute",
        "the typed layer raises a code the contract does not define",
    ),
    "dict_envelope": (
        _dict_envelope_codes,
        'a hand-built {"code": ...} payload',
        "the client receives an undefined code outside the structured envelope",
    ),
}


def test_raise_error_codes_are_registered():
    scan, path, effect = _RULES["raise_error"]
    codes = scan()
    assert len(codes) >= _FLOORS["raise_error"], f"scan found only {len(codes)} codes"
    assert not (msg := _unregistered(codes, _registered_codes(), path=path, effect=effect)), msg


def test_error_response_codes_are_registered():
    scan, path, effect = _RULES["error_response"]
    codes = scan()
    assert len(codes) >= _FLOORS["error_response"], f"scan found only {len(codes)} codes"
    assert not (msg := _unregistered(codes, _registered_codes(), path=path, effect=effect)), msg


def test_class_level_codes_are_registered():
    scan, path, effect = _RULES["class_level"]
    codes = scan()
    assert len(codes) >= _FLOORS["class_level"], f"scan found only {len(codes)} codes"
    assert not (msg := _unregistered(codes, _registered_codes(), path=path, effect=effect)), msg


def test_hand_built_envelope_codes_are_registered():
    scan, path, effect = _RULES["dict_envelope"]
    codes = scan()
    assert len(codes) >= _FLOORS["dict_envelope"], f"scan found only {len(codes)} codes"
    assert not (msg := _unregistered(codes, _registered_codes(), path=path, effect=effect)), msg


def test_registry_covers_every_registered_code():
    """The enum and the registry must not drift apart from each other either."""
    from domain.infrastructure._internal.errors import ERROR_REGISTRY, ErrorCode

    missing = [code.name for code in ErrorCode if code not in ERROR_REGISTRY]
    assert not missing, f"ErrorCode members with no ERROR_REGISTRY entry: {missing}"


def test_every_registered_code_resolves_by_name():
    """``get_error_info`` is the lookup every non-router consumer uses (CLI,
    health/event surfaces, monitors).

    It must accept a code written as a member *name*: legacy aliases such as
    ``E_CREATE_FAILED = "E_BAD_REQUEST"`` are only ever written that way at
    call sites, and a value-only lookup (``ErrorCode(code)``) misses them —
    which is how an alias silently degrades to a generic AppError.
    """
    from domain.infrastructure._internal.errors import ErrorCode, get_error_info

    unresolved = [name for name in ErrorCode.__members__ if get_error_info(name) is None]
    assert not unresolved, f"get_error_info() returned None for: {unresolved}"


def test_legacy_alias_classifies_like_the_canonical_code():
    """The alias must resolve to the same class and status as its target."""
    from domain.infrastructure._internal.errors import get_error_info

    alias = get_error_info("E_CREATE_FAILED")
    canonical = get_error_info("E_BAD_REQUEST")
    assert alias is not None, "alias E_CREATE_FAILED does not resolve"
    assert canonical is not None, "E_BAD_REQUEST does not resolve"
    assert (alias[0], alias[1]) == (canonical[0], canonical[1]), (alias, canonical)
