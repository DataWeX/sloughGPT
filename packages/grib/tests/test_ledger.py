"""Tests for the ledger subcommand (ledger.py) — docs/AGENT_SYNC.md stays the
single hand-written source; grib parses it into an index and gates its shape.

Entry = a line starting with a bold ISO date (``**YYYY-MM-DD — …``). The 23
mid-entry emphasis lines (``**Gates:**`` etc.) are not entries. The prepend
invariant — dates strictly non-increasing top-to-bottom — is the check gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from grib.cli import main
from grib.ledger import check, fingerprint, parse, query

# Two entries in prepend order (newest first), one mid-entry emphasis line
# that must NOT parse as an entry, one continuation line per entry.
_CLEAN = """\
**2026-11-21 — OPENCODE (kv-cache lane): card `20261121_102152_kv-cache-nativepy-pointer-arithmetic-breaks-on-linux-so-13-t` — pushed `4bac44e02..0c1222d72` → HEAD:main, PR #372.**

What landed: `_layer_ptr` integer-address arithmetic.
**Gates:** 111 passed / 0 failed.

**2026-11-19 — OPENCODE (route-parity lane): card `20261119_105501` — one router assembler — landed `94941b8e1` via PR #358.**

Second entry body with sha `d89357fad` and short card `20261119_132133`.
"""


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "AGENT_SYNC.md"
    p.write_text(_CLEAN, encoding="utf-8")
    return p


# --- parse -----------------------------------------------------------------


def test_parse_entries_in_order_with_metadata(doc):
    entries = parse(doc)
    assert len(entries) == 2  # **Gates:** line is NOT an entry
    assert entries[0]["date"] == "2026-11-21"
    assert entries[1]["date"] == "2026-11-19"
    assert entries[0]["line"] == 1
    assert entries[1]["line"] == 6


def test_parse_extracts_cards_shas_prs_and_lane(doc):
    entries = parse(doc)
    first = entries[0]
    assert (
        "20261121_102152_kv-cache-nativepy-pointer-arithmetic-breaks-on-linux-so-13-t"
        in first["cards"]
    )
    assert first["shas"] == ["4bac44e02", "0c1222d72"]  # range endpoints, in order
    assert first["prs"] == [372]
    assert "kv-cache lane" in first["lane"]
    second = entries[1]
    assert "20261119_105501" in second["cards"]  # short 8_6 form
    assert "20261119_132133" in second["cards"]  # short form in body
    assert "d89357fad" in second["shas"]
    assert second["prs"] == [358]


def test_parse_header_is_the_only_entry_signal(doc):
    # The **Gates:** line sits inside entry 1 and must not split it.
    entries = parse(doc)
    assert entries[0]["line"] == 1
    assert entries[1]["line"] == 6  # 4 (Gates) + blank + header = 6


# --- check -----------------------------------------------------------------


def test_check_clean_doc_passes(doc):
    ok, violations = check(doc)
    assert ok is True
    assert violations == []


def test_check_flags_malformed_bold_date_header(tmp_path):
    p = tmp_path / "bad.md"
    p.write_text(
        "**2026-11-21 — fine entry**\n\nbody\n\n**2026-13-45 — month 13 day 45**\n",
        encoding="utf-8",
    )
    ok, violations = check(p)
    assert ok is False
    assert any(v["line"] == 5 for v in violations)
    assert any("malformed" in v["kind"] for v in violations)


def test_check_flags_bottom_append_breaking_prepended_order(tmp_path):
    p = tmp_path / "append.md"
    p.write_text(
        "**2026-11-19 — older entry**\n\nbody\n\n**2026-11-21 — appended at the bottom**\n",
        encoding="utf-8",
    )
    ok, violations = check(p)
    assert ok is False
    assert any(v["kind"] == "order" and v["line"] == 5 for v in violations)


def test_check_flags_missing_closing_bold(tmp_path):
    p = tmp_path / "unclosed.md"
    p.write_text("**2026-11-21 — never closed\n", encoding="utf-8")
    ok, violations = check(p)
    assert ok is False
    assert violations[0]["line"] == 1


# --- query -----------------------------------------------------------------


def test_query_by_card_matches_long_and_short_forms(doc):
    assert len(query(doc, card="20261121_102152")) == 1
    assert len(query(doc, card="20261119_132133")) == 1  # short form in body
    assert query(doc, card="20261119_999999") == []


def test_query_by_sha_and_lane_and_grep(doc):
    assert len(query(doc, sha="0c1222d72")) == 1
    assert len(query(doc, sha="d89357fad")) == 1
    assert len(query(doc, lane="kv-cache")) == 1
    assert len(query(doc, grep="router assembler")) == 1
    assert query(doc, grep="nonexistent phrase") == []


def test_query_combines_filters_with_and(doc):
    # sha from entry 1 + lane from entry 2 -> no match
    assert query(doc, sha="0c1222d72", lane="route-parity") == []


# --- CLI -------------------------------------------------------------------


def test_cli_parse_outputs_json_array(doc, capsys):
    rc = main(["ledger", "parse", "--doc", str(doc)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 2
    assert out[0]["date"] == "2026-11-21"


def test_cli_check_exit_codes(doc, tmp_path, capsys):
    assert main(["ledger", "check", "--doc", str(doc)]) == 0
    bad = tmp_path / "bad.md"
    bad.write_text("**2026-11-19 — old**\n\n**2026-11-21 — bottom append**\n", encoding="utf-8")
    assert main(["ledger", "check", "--doc", str(bad)]) == 1
    assert "order" in capsys.readouterr().out


def test_cli_discovers_doc_by_walking_up_from_cwd(tmp_path, monkeypatch, capsys):
    (tmp_path / "docs").mkdir()
    target = tmp_path / "docs" / "AGENT_SYNC.md"
    target.write_text(_CLEAN, encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    rc = main(["ledger", "check"])
    assert rc == 0
    capsys.readouterr()  # flush check's ok-line so only parse's JSON remains
    rc = main(["ledger", "parse"])
    assert rc == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_cli_errors_cleanly_when_no_doc_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = main(["ledger", "check"])
    assert rc == 2
    assert "AGENT_SYNC.md" in capsys.readouterr().out


# --- baseline (the CI gate: known findings, shrink-only) -------------------

_ORDER_DOC = """\
**2026-11-19 — OPENCODE (older lane): card `20261119_000000` — older entry.**

body

**2026-11-21 — OPENCODE (newer lane): card `20261121_000000` — newer below.**

body
"""


def test_order_violation_carries_stable_identity_fields(tmp_path):
    p = tmp_path / "o.md"
    p.write_text(_ORDER_DOC, encoding="utf-8")
    ok, violations = check(p)
    assert ok is False
    v = violations[0]
    assert v["kind"] == "order"
    assert v["line"] == 5
    assert v["date"] == "2026-11-21"  # the NEWER entry's date (below)
    assert "newer lane" in v["head"]


def test_fingerprint_survives_a_prepend(tmp_path):
    """Lines shift on every prepend — a baseline keyed on line numbers would
    churn daily. The fingerprint must not."""
    p = tmp_path / "o.md"
    p.write_text(_ORDER_DOC, encoding="utf-8")
    _, v1 = check(p)
    fp1 = fingerprint(v1[0])
    p.write_text(
        "**2026-11-21 — OPENCODE (top lane): card `20261121_111111` — fresh top.**\n\n"
        + p.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _, v2 = check(p)
    assert v2[0]["line"] == 7  # shifted by the prepend…
    assert fingerprint(v2[0]) == fp1  # …identity unchanged


def test_cli_check_baseline_accepts_known_findings(tmp_path, capsys):
    p = tmp_path / "o.md"
    p.write_text(_ORDER_DOC, encoding="utf-8")
    _, violations = check(p)
    bp = tmp_path / "baseline.json"
    bp.write_text(json.dumps({"violations": [fingerprint(violations[0])]}), encoding="utf-8")
    rc = main(["ledger", "check", "--doc", str(p), "--baseline", str(bp)])
    assert rc == 0
    assert "baseline aligned" in capsys.readouterr().out


def test_cli_check_baseline_rejects_a_new_violation(tmp_path, capsys):
    p = tmp_path / "o.md"
    p.write_text(_ORDER_DOC, encoding="utf-8")
    bp = tmp_path / "baseline.json"
    bp.write_text(json.dumps({"violations": []}), encoding="utf-8")
    rc = main(["ledger", "check", "--doc", str(p), "--baseline", str(bp)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "NEW" in out
    assert "new" in out  # the counts line


def test_cli_check_baseline_rejects_a_stale_entry(tmp_path, capsys):
    # Doc is clean; the baseline still pins a finding that no longer occurs ->
    # strict shrink: fixing is not enough, the line must be deleted.
    p = tmp_path / "clean.md"
    p.write_text("**2026-11-21 — one fine entry.**\n", encoding="utf-8")
    bp = tmp_path / "baseline.json"
    bp.write_text(
        json.dumps({"violations": ["order|2026-11-19|OPENCODE (older lane): gone"]}),
        encoding="utf-8",
    )
    rc = main(["ledger", "check", "--doc", str(p), "--baseline", str(bp)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "delete" in out


# --- seal: the ledger's hash trail (entries seal the entry below them) ------

_SEAL = "⟨seal:"


def _sealed(tmp_path, date_s, text, name="s.md"):
    """Build a doc whose top entry was written by the helper (or raise in
    test bodies that the helper is missing — that IS the red)."""
    p = tmp_path / name
    p.write_text("", encoding="utf-8")  # explicit --doc requires an existing file
    rc = main(["ledger", "entry", "--doc", str(p), "--date", date_s, "--text", text])
    assert rc == 0, "entry helper must exist and succeed"
    return p


def test_entry_helper_prepends_a_sealed_entry_at_line_one(tmp_path):
    p = tmp_path / "s.md"
    p.write_text("**2026-11-10 — OPENCODE (old lane): first entry.**\n", encoding="utf-8")
    rc = main(
        [
            "ledger",
            "entry",
            "--doc",
            str(p),
            "--date",
            "2026-11-24",
            "--text",
            "OPENCODE (seal lane): second entry.",
        ]
    )
    assert rc == 0
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("**2026-11-24 — OPENCODE (seal lane)")
    assert _SEAL in lines[0]
    assert lines[1] == ""  # blank separator, prepended not appended
    assert lines[2].startswith("**2026-11-10")  # history untouched below
    ok, violations = check(p)
    assert ok is True, violations


def test_entry_helper_seals_from_genesis_on_an_empty_doc(tmp_path):
    p = tmp_path / "e.md"
    p.write_text("", encoding="utf-8")
    assert main(["ledger", "entry", "--doc", str(p), "--date", "2026-11-24", "--text", "x."]) == 0
    ok, violations = check(p)
    assert ok is True, violations


def test_post_cut_entry_without_a_seal_is_a_violation(tmp_path):
    p = tmp_path / "n.md"
    p.write_text("**2026-11-24 — OPENCODE (bare lane): no seal here.**\n", encoding="utf-8")
    ok, violations = check(p)
    assert ok is False
    v = violations[0]
    assert v["kind"] == "seal"
    assert v["date"] == "2026-11-24"  # date-anchored identity, never line-anchored
    assert "seal" in v["msg"]
    assert "entry" in v["msg"]  # actionable: points at the helper


def test_grandfathered_entries_need_no_seal(tmp_path):
    p = tmp_path / "o.md"
    p.write_text(
        "**2026-11-22 — OPENCODE (also-old lane): still grandfathered.**\n\n"
        "**2026-11-10 — OPENCODE (old lane): predates the seal cut.**\n",
        encoding="utf-8",
    )
    ok, violations = check(p)
    assert ok is True, violations


def test_tampered_sealed_entry_is_flagged(tmp_path):
    p = _sealed(tmp_path, "2026-11-24", "OPENCODE (edit lane): steady.")
    p.write_text(
        p.read_text(encoding="utf-8").replace("steady.", "steady EDITED."),
        encoding="utf-8",
    )
    ok, violations = check(p)
    assert ok is False
    assert any(v["kind"] == "seal" and v["date"] == "2026-11-24" for v in violations)


def test_tampered_grandfathered_content_below_breaks_the_sealed_entry_above(tmp_path):
    # The tripwire reaches past the grandfather line: a sealed entry hashes
    # the content below it, so editing an UNSEALED entry is caught by the
    # first sealed entry above.
    p = tmp_path / "u.md"
    p.write_text("**2026-11-10 — OPENCODE (old lane): soft target.**\n", encoding="utf-8")
    assert (
        main(
            [
                "ledger",
                "entry",
                "--doc",
                str(p),
                "--date",
                "2026-11-24",
                "--text",
                "OPENCODE (seal lane): watcher.",
            ]
        )
        == 0
    )
    p.write_text(
        p.read_text(encoding="utf-8").replace("soft target.", "soft target TAMPERED."),
        encoding="utf-8",
    )
    ok, violations = check(p)
    assert ok is False
    assert any(v["kind"] == "seal" and v["date"] == "2026-11-24" for v in violations)


def test_reordered_entries_break_their_seals(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("", encoding="utf-8")
    assert (
        main(["ledger", "entry", "--doc", str(p), "--date", "2026-11-23", "--text", "older."]) == 0
    )
    assert (
        main(["ledger", "entry", "--doc", str(p), "--date", "2026-11-24", "--text", "newer."]) == 0
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    # swap the two entries physically (each still holds its original seal):
    # [newer, "", older] -> [older, "", newer]
    top, sep, below, *rest = lines
    assert sep == ""
    p.write_text("\n".join([below, "", top, *rest]) + "\n", encoding="utf-8")
    ok, violations = check(p)
    assert any(v["kind"] == "seal" for v in violations), violations


def test_cli_check_exits_1_on_a_seal_violation(tmp_path, capsys):
    p = tmp_path / "c.md"
    p.write_text("**2026-11-24 — OPENCODE (bare lane): no seal.**\n", encoding="utf-8")
    rc = main(["ledger", "check", "--doc", str(p)])
    assert rc == 1
    assert "seal" in capsys.readouterr().out


def test_parse_and_query_do_not_leak_the_seal_token(tmp_path):
    p = _sealed(tmp_path, "2026-11-24", "OPENCODE (idx lane): short entry.")
    entries = parse(p)
    assert len(entries) == 1
    assert _SEAL not in entries[0]["head"]
    assert _SEAL not in json.dumps(entries)
    hits = query(p, grep="short entry")
    assert len(hits) == 1
    assert _SEAL not in hits[0]["head"]


# --- the ledger must never be reformatted ----------------------------------


def test_repo_ledger_is_excluded_from_prettier():
    """Seals hash each entry's exact bytes, so a markdown formatter corrupts them.

    2026-11-27: lint-staged's ``*.{json,md}`` -> ``prettier --write`` escaped
    underscores (``_x`` -> ``\\_x``) and rewrote emphasis (``**b**`` ->
    ``_*b*_``) across 73 lines of the repo ledger, invalidating 11 seals.
    docs/AGENT_SYNC.md must stay listed in .prettierignore.
    """
    repo = Path(__file__).resolve().parents[3]
    ignore = (repo / ".prettierignore").read_text(encoding="utf-8")
    patterns = {
        line.strip()
        for line in ignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "docs/AGENT_SYNC.md" in patterns, (
        "docs/AGENT_SYNC.md must be in .prettierignore — prettier rewrites "
        "sealed entry bytes and breaks grib ledger seals"
    )
