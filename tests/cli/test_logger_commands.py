"""Tests for the logger group — live error catch bundles."""

import sys
from pathlib import Path

# Add CLI src to path (mirrors other CLI tests)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "cli" / "src"))

from groups.logger import (
    AUTOFIX_FILE,
    UI_FILE,
    append_entry,
    build_entry,
    bundle_path,
    classify,
    extract_file,
    fingerprint,
    is_frontend,
    load_entries,
)


def test_filenames_match_error_fixer_contract():
    assert AUTOFIX_FILE == ".opencode-autofix-log.json"
    assert UI_FILE == ".opencode-ui-error-log.json"
    assert bundle_path("autofix", Path("/tmp")).name == AUTOFIX_FILE
    assert bundle_path("ui", Path("/tmp")).name == UI_FILE


def test_fingerprint_prefers_backend_id():
    assert fingerprint({"fingerprint": "abc123", "message": "x"}) == "abc123"


def test_fingerprint_stable_without_backend_id():
    a = fingerprint({"message": "boom 42"})
    b = fingerprint({"message": "boom 42"})
    assert a == b
    assert fingerprint({"message": "different"}) != a


def test_is_frontend_sources():
    assert is_frontend({"source": "window.onerror", "message": "x"})
    assert is_frontend({"source": "unhandledrejection", "message": "x"})
    assert is_frontend({"source": "web", "url": "http://x/app", "message": "x"})
    assert is_frontend({"source": "web", "message": "x", "stack": "at render (app.tsx:10:2)"})
    assert not is_frontend({"source": "web", "message": "plain backend-ish"})
    assert not is_frontend({"message": "no source at all"})


def test_classify_playbook_categories():
    assert classify({"message": "error TS2322: bad", "stack": ""}) == "typescript"
    assert (
        classify({"message": "crash", "stack": "Traceback (most recent call last):\n boom"})
        == "python-error"
    )
    assert classify({"message": "ECONNREFUSED 127.0.0.1", "stack": ""}) == "network"
    assert classify({"message": "Script error.", "stack": "hydration mismatch"}) == "hydration"
    assert classify({"source": "window.onerror", "message": "boom", "stack": ""}) == "frontend"
    assert classify({"message": "something odd", "stack": ""}) == "unknown"


def test_extract_file_prefers_url_line():
    assert extract_file({"url": "http://x/app", "line": 7}) == "http://x/app:7"


def test_extract_file_python_frame():
    rec = {"stack": 'Traceback:\n  File "/a/b.py", line 12, in f\nValueError'}
    assert extract_file(rec) == "/a/b.py:12"


def test_extract_file_js_frame():
    rec = {"stack": "Error\n    at render (/app/page.tsx:10:2)"}
    assert extract_file(rec) == "/app/page.tsx:10"


def test_extract_file_none():
    assert extract_file({"message": "no location"}) is None


def test_build_entry_schema():
    entry = build_entry(
        {
            "fingerprint": "fp1",
            "message": "boom",
            "source": "web",
            "url": "http://x",
            "line": 3,
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
    )
    assert entry["id"] == "fp1"
    assert entry["resolved"] is False
    assert entry["resolvedAt"] is None
    assert "boom" in entry["snippet"]
    assert "http://x:3" in entry["snippet"]
    assert entry["file"] == "http://x:3"


def test_append_dedups_open_entries(tmp_path):
    path = tmp_path / "bundle.json"
    entry = build_entry({"message": "boom"})
    assert append_entry(path, entry) is True
    assert append_entry(path, dict(entry)) is False
    assert len(load_entries(path)) == 1


def test_append_reopens_resolved(tmp_path):
    path = tmp_path / "bundle.json"
    entry = build_entry({"message": "boom"})
    append_entry(path, entry)
    entries = load_entries(path)
    entries[0]["resolved"] = True
    entries[0]["resolvedAt"] = "2026-01-02T00:00:00+00:00"
    path.write_text(__import__("json").dumps(entries))
    assert append_entry(path, build_entry({"message": "boom"})) is True
    assert len(load_entries(path)) == 2


def test_append_caps_length(tmp_path):
    import string

    path = tmp_path / "bundle.json"
    # Letter pairs — digit-only suffixes would normalize to one fingerprint
    names = [a + b for a in string.ascii_lowercase[:15] for b in string.ascii_lowercase[:15]]
    assert len(names) >= 205
    for name in names[:205]:
        append_entry(path, build_entry({"message": f"err {name}"}), cap=200)
    assert len(load_entries(path)) == 200


def test_load_tolerates_missing_and_corrupt(tmp_path):
    assert load_entries(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json{{{")
    assert load_entries(bad) == []
