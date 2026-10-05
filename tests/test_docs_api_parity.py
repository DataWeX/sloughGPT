"""The `parity-claims` architecture contract is an executable gate.

Two behaviours are worth pinning: a synced block passes, and a drifted block
alarms. The second is the whole point — this exact gate sat red long enough
for its claims to drift roughly 3x without anyone noticing, so a test that
only checks the happy path would be testing decoration.

The drift test monkeypatches ``CLAIMS_MD`` at a temp file rather than editing
``docs/DOC_VS_CODE_GAP_AUDIT.md``, so a failed assertion cannot leave the
tracked doc modified.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_docs_api_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("docs_api_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the script defines dataclasses, and dataclasses
    # resolves annotations via sys.modules[cls.__module__], which is None if
    # the module was never inserted.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parity():
    """Import the script as a module (it is __main__-guarded)."""
    return _load()


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=180,
    )


def _measure(parity):
    f = parity.Findings()
    parity.collect_architecture(f)
    return f


def test_claims_block_exists_and_parses(parity):
    f = _measure(parity)
    assert f.arch_claims, "no parity-claims block parsed from DOC_VS_CODE_GAP_AUDIT.md"


def test_every_measured_key_is_claimed(parity):
    """The block must cover the whole measured surface, not a subset.

    A partial block would let half the numbers drift silently — the exact
    failure the unclaimed half caused the first time around.
    """
    f = _measure(parity)
    assert set(f.arch_claims) == set(f.arch_measured), (
        f"claims={sorted(f.arch_claims)} measured={sorted(f.arch_measured)}"
    )


def test_synced_block_passes(parity):
    f = _measure(parity)
    assert not f.arch_mismatch, f.arch_mismatch
    assert not f.arch_missing, f.arch_missing


def test_claim_raised_by_one_is_an_alarm(parity, tmp_path, monkeypatch):
    """The ratchet must actually bite: raise a claim by one and it alarms.

    This guards the gate's reason for existing — once the numbers are in sync
    they may only go DOWN, and any climb has to be reported rather than
    re-synced away.
    """
    f = _measure(parity)
    key = "controllers_internal_stmts"
    assert key in f.arch_claims

    bumped = dict(f.arch_claims)
    bumped[key] = f.arch_claims[key] + 1
    body = "\n".join(f"{k}={v}" for k, v in bumped.items())
    claims = tmp_path / "audit.md"
    claims.write_text(f"<!-- parity-claims:v1\n{body}\n-->\n", encoding="utf-8")

    monkeypatch.setattr(parity, "CLAIMS_MD", claims)
    g = _measure(parity)
    expected = f"{key}: doc={bumped[key]} code={f.arch_measured[key]}"
    assert g.arch_mismatch == [expected]


def test_missing_block_is_unresolved_not_silent(parity, tmp_path, monkeypatch):
    """A doc without the block must fail closed, not pass vacuously."""
    absent = tmp_path / "no-claims.md"
    absent.write_text("# no contract here\n", encoding="utf-8")
    monkeypatch.setattr(parity, "CLAIMS_MD", absent)
    g = _measure(parity)
    assert g.arch_claims == {}
    assert g.arch_missing
    assert any("no parity-claims block" in row for row in g.arch_missing)


def test_architecture_only_is_green():
    """The gate must be runnable independently of the route-doc truth-up.

    Route docs are a separate, larger workstream; without this flag the
    architecture contract could never be enforced until they landed, which is
    how it stayed unenforced the first time.
    """
    proc = _run("--architecture-only")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "0 mismatched" in proc.stdout
    assert "0 unresolved" in proc.stdout


def test_architecture_only_json_reports_claims_equal_measured():
    proc = _run("--architecture-only", "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["mismatch"] == []
    assert data["missing"] == []
    assert data["claims"] == data["measured"]
    assert data["claims"]
