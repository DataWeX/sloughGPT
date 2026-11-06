"""Raw-thread gate — the stage-2 lint hint of the execution-consolidation spec.

`docs/plans/2026-09-20-execution-consolidation.md` (stage 2) prescribes a
lint hint against raw ``threading.Thread(target=`` in ``apps/`` + ``domain/``
with an allowlist file; stage 3 then migrated the candidates and ruled the
rest grandfathered. This gate makes the rule mechanical so the migration
ratchets instead of regressing (page-baseline / command-parity precedent:

- detection is AST-based: docstrings *mentioning* ``threading.Thread(...)``
  are not call sites, and ``from threading import Thread`` / ``_threading``
  aliases are as much a raw site as ``threading.Thread(``;
- allowlist entries are ``path::qualname`` with a declared ``count`` —
  a new site in an allowlisted function changes the count (red), and a
  migrated site lowers it (red until the entry is shrunk), so the
  grandfathered set can only move by explicit, reviewed edit;
- entries carry ``reason`` + ``kind`` (``grandfathered`` / ``deferred`` /
  ``followup``) — kind ``followup`` marks stage-3 migration candidates the
  audit deliberately parked (greppable backlog).

Baseline: audit 2026-11-04 + 2026-11-05 allowlist pass (54 sites).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_raw_threads.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_raw_threads", _SCRIPT)
    assert spec is not None and spec.loader is not None, (
        f"checker missing at {_SCRIPT} — stage-2 lint hint not built"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


class TestDetection:
    def test_checker_module_loads(self):
        module = _load_checker()
        assert hasattr(module, "scan"), "checker must expose scan()"

    def test_flags_unlisted_raw_thread(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/svc.py": (
                    "import threading\n"
                    "def boot():\n"
                    "    threading.Thread(target=run, daemon=True).start()\n"
                ),
            },
        )
        report = checker.scan(tree, allowlist={})
        assert len(report["violations"]) == 1
        v = report["violations"][0]
        assert v["path"] == "apps/x/svc.py"
        assert v["qualname"] == "boot"
        assert v["declared"] is None

    def test_flags_bare_and_aliased_thread(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/a.py": (
                    "from threading import Thread\n"
                    "def go():\n"
                    "    Thread(target=f).start()\n"
                ),
                "domain/y/b.py": (
                    "import threading as th\n"
                    "def go():\n"
                    "    th.Thread(target=f).start()\n"
                ),
            },
        )
        report = checker.scan(tree, allowlist={})
        paths = {v["path"] for v in report["violations"]}
        assert paths == {"apps/x/a.py", "domain/y/b.py"}

    def test_ignores_docstring_mentions(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "domain/y/doc.py": (
                    '"""Notes about ``threading.Thread(...)`` call sites."""\n'
                    "def ok():\n"
                    "    return 1\n"
                ),
            },
        )
        report = checker.scan(tree, allowlist={})
        assert report["violations"] == []

    def test_ignores_test_files(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/tests/test_thing.py": (
                    "import threading\n"
                    "def test_x():\n"
                    "    threading.Thread(target=f).start()\n"
                ),
                "apps/x/conftest.py": (
                    "import threading\n"
                    "threading.Thread(target=f).start()\n"
                ),
            },
        )
        report = checker.scan(tree, allowlist={})
        assert report["violations"] == []


class TestAllowlistRatchet:
    def test_allowlisted_site_passes(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/svc.py": (
                    "import threading\n"
                    "def boot():\n"
                    "    threading.Thread(target=run).start()\n"
                ),
            },
        )
        allowlist = {
            "apps/x/svc.py::boot": {
                "reason": "pump bound to live handle",
                "kind": "grandfathered",
                "count": 1,
            }
        }
        report = checker.scan(tree, allowlist=allowlist)
        assert report["violations"] == []

    def test_new_site_in_allowlisted_func_changes_count(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/svc.py": (
                    "import threading\n"
                    "def boot():\n"
                    "    threading.Thread(target=run).start()\n"
                    "    threading.Thread(target=run2).start()\n"
                ),
            },
        )
        allowlist = {
            "apps/x/svc.py::boot": {
                "reason": "pump",
                "kind": "grandfathered",
                "count": 1,
            }
        }
        report = checker.scan(tree, allowlist=allowlist)
        assert len(report["violations"]) == 1
        assert report["violations"][0]["declared"] == 1
        assert report["violations"][0]["actual"] == 2

    def test_migrated_site_must_shrink_entry(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/svc.py": (
                    "import threading\n"
                    "def boot():\n"
                    "    pass\n"
                ),
            },
        )
        allowlist = {
            "apps/x/svc.py::boot": {
                "reason": "pump",
                "kind": "grandfathered",
                "count": 1,
            }
        }
        report = checker.scan(tree, allowlist=allowlist)
        assert len(report["violations"]) == 1
        assert report["violations"][0]["actual"] == 0

    def test_stale_entry_is_reported(self, checker, tmp_path):
        tree = _write_tree(tmp_path, {"apps/x/svc.py": "def ok():\n    return 1\n"})
        allowlist = {
            "apps/x/gone.py::boot": {"reason": "old", "kind": "grandfathered", "count": 1}
        }
        report = checker.scan(tree, allowlist=allowlist)
        assert any(v["stale"] for v in report["violations"])

    def test_entry_without_reason_or_kind_rejected(self, checker, tmp_path):
        tree = _write_tree(
            tmp_path,
            {
                "apps/x/svc.py": (
                    "import threading\n"
                    "def boot():\n"
                    "    threading.Thread(target=run).start()\n"
                ),
            },
        )
        allowlist = {"apps/x/svc.py::boot": {"count": 1}}
        report = checker.scan(tree, allowlist=allowlist)
        assert any(v["schema"] for v in report["violations"])


class TestRealTree:
    def test_repo_tree_is_clean(self, checker):
        """Integration: the committed allowlist covers every real site."""
        report = checker.scan(_REPO_ROOT, allowlist=checker.load_allowlist(_REPO_ROOT))
        assert report["violations"] == [], (
            "raw threading.Thread sites outside the allowlist (or count "
            f"drift): {json.dumps(report['violations'], indent=2)[:2000]} — "
            "migrate per docs/plans/2026-09-20-execution-consolidation.md or "
            "edit scripts/raw_threads_allowlist.json with a reason"
        )
        assert report["site_count"] >= 45, (
            f"expected the grandfathered backlog to still exist, got {report['site_count']}"
        )
