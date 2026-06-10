"""Tests for bawl.cli — tests entry() with various argv patterns."""

import json
import sys
import tempfile
from pathlib import Path

from bawl.cli import entry
from bawl.fetch import _hits


def _run(*args: str, stdin: str = "") -> tuple[int, str, str]:
    """Run entry() with given argv. Returns (exit_code, stdout, stderr)."""
    old_argv = sys.argv
    old_out = sys.stdout
    old_err = sys.stderr
    old_in = sys.stdin
    try:
        sys.argv = ["bawl"] + list(args)
        out = tempfile.TemporaryFile("w+")
        err = tempfile.TemporaryFile("w+")
        sys.stdout = out
        sys.stderr = err
        sys.stdin = tempfile.TemporaryFile("r+")
        sys.stdin.write(stdin)
        sys.stdin.seek(0)
        code = entry()
        out.seek(0)
        err.seek(0)
        return code, out.read(), err.read()
    finally:
        sys.argv = old_argv
        sys.stdout = old_out
        sys.stderr = old_err
        sys.stdin = old_in


def test_help():
    code, out, err = _run("--help")
    assert code == 0
    assert "bawl" in out


def test_help_short():
    code, out, err = _run("-h")
    assert code == 0


def test_help_no_args():
    code, out, err = _run()
    assert code == 0


def test_shorthand_url():
    code, out, err = _run("https://example.com")
    assert code == 0
    assert out.strip()
    data = json.loads(out)
    assert data["title"] == "Example Domain"


def test_page_json():
    code, out, err = _run("page", "https://example.com")
    assert code == 0
    data = json.loads(out)
    assert data["title"] == "Example Domain"


def test_page_to_file():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    code, out, err = _run("page", "https://example.com", "-o", str(tmp))
    assert code == 0
    assert tmp.stat().st_size > 0
    data = json.loads(tmp.read_text())
    assert data["title"] == "Example Domain"
    tmp.unlink()


def test_page_text_format():
    code, out, err = _run("page", "https://example.com", "-f", "text")
    assert code == 0
    data = json.loads(out)
    assert "text" in data
    assert "title" in data


def test_crawl_depth_0():
    code, out, err = _run("crawl", "https://example.com", "--depth", "0")
    assert code == 0
    data = json.loads(out)
    assert data["url"] == "https://example.com"


def test_crawl_max_pages():
    code, out, err = _run("crawl", "https://example.com", "--max", "1")
    assert code == 0
    data = json.loads(out)
    assert data["url"] == "https://example.com"


def test_crawl_to_file():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    code, out, err = _run("crawl", "https://example.com", "--depth", "0", "-o", str(tmp))
    assert code == 0
    assert tmp.stat().st_size > 0
    data = json.loads(tmp.read_text())
    assert data["title"] == "Example Domain"
    tmp.unlink()


def test_cat_empty_pipe():
    code, out, err = _run("cat")
    assert code == 0
    assert out == ""


def test_bad_cmd():
    code, out, err = _run("nope")
    assert code == 1
    assert "unknown command" in err


def test_version():
    code, out, err = _run("--version")
    assert code == 0
    assert "bawl" in out


def test_version_short():
    code, out, err = _run("-V")
    assert code == 0
    assert "bawl" in out


def test_completion_bash():
    code, out, err = _run("completion", "bash")
    assert code == 0
    assert "complete -F" in out


def test_completion_zsh():
    code, out, err = _run("completion", "zsh")
    assert code == 0
    assert "compdef" in out


def test_completion_bad_shell():
    code, out, err = _run("completion", "fish")
    assert code == 1


def test_page_json_array_format():
    """bawl page -f json should output a JSON array, not JSONL."""
    code, out, err = _run("page", "https://example.com", "-f", "json")
    assert code == 0
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "Example Domain"


def test_crawl_from_file():
    """bawl crawl @urls.txt with a URL file."""
    tmp = Path(tempfile.mktemp(suffix=".txt"))
    tmp.write_text("https://example.com\n")
    code, out, err = _run("crawl", f"@{str(tmp)}", "--depth", "0")
    assert code == 0, f"stderr: {err}"
    data = json.loads(out)
    assert data["url"] == "https://example.com"
    tmp.unlink()


def test_bad_url():
    code, out, err = _run("page", "https://nosuchdomain99999.xyz")
    assert code == 1


def test_rate_flag():
    code, out, err = _run("--rate", "1.0", "https://example.com")
    assert code == 0
    assert json.loads(out)["title"] == "Example Domain"


def test_timeout_flag():
    code, out, err = _run("--timeout", "30", "page", "https://example.com")
    assert code == 0
    assert json.loads(out)["title"] == "Example Domain"
