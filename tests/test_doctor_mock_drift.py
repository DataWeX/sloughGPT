"""Tests for `scripts/test-doctor.py --mock-drift`.

The detector's dangerous failure mode is *silently reporting nothing*: a regex
that misses imports, or an empirical check that can't tell eager from lazy,
both yield a clean bill of health. These tests pin the two halves that the
verdict is built from, plus the CLI contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "test-doctor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("test_doctor", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    mod = _load_script()
    mod._md_setup_path()
    return mod


# ── Static half: who reads what ────────────────────────────────────────


def test_test_imports_matches_imports_beyond_first_line(tool, tmp_path):
    """Regression: `^` without re.MULTILINE only ever matched line 1, which
    made every relevance check return False and reported 0 confirmed."""
    f = tmp_path / "test_x.py"
    f.write_text(
        '"""docstring"""\n\nfrom __future__ import annotations\n\n'
        "import pytest\n\nfrom routers.kb import router\n"
    )
    assert tool._md_test_imports(f, "routers.kb") is True


def test_test_imports_matches_import_not_at_line_start(tool, tmp_path):
    f = tmp_path / "test_x.py"
    f.write_text("def go():\n    from routers.health import router\n    return router\n")
    assert tool._md_test_imports(f, "routers.health") is True


def test_test_imports_false_for_unrelated_module(tool, tmp_path):
    f = tmp_path / "test_x.py"
    f.write_text("from routers.kb import router\n")
    assert tool._md_test_imports(f, "routers.system") is False


def test_targets_only_collects_internal_definition_sites(tool):
    targets = tool._md_targets()
    assert targets, "expected some _internal patch targets in the suite"
    assert all("._internal." in t for t in targets)

    # The retarget must have removed the dead target from the file it lived in.
    # (Other files -- including this one, which quotes targets as strings --
    # may still name it, so assert on the patcher set, not on presence.)
    tgt = "domain.feedback._internal.model_health.get_health_monitor"
    patchers = targets.get(tgt, set())
    assert not any(p.name == "test_health_router.py" for p in patchers), (
        "test_health_router.py still patches a target production no longer reads"
    )


def test_path_module_maps_known_roots(tool):
    p = _REPO / "apps/api/server/routers/health.py"
    assert tool._md_path_module(p) == "routers.health"
    assert tool._md_path_module(_REPO / "domain/feedback/__init__.py") == "domain.feedback"


def test_relative_import_under_domain_keeps_its_prefix(tool):
    """`domain` must not be treated as a pythonpath root: dropping the prefix
    made `from .hf_dpo import X` resolve to an unimportable `feedback.*`."""
    f = _REPO / "domain/feedback/_internal/workflow.py"
    assert tool._md_path_module(f) == "domain.feedback._internal.workflow"
    assert tool._md_resolve(".hf_dpo", f) == "domain.feedback._internal.hf_dpo"


# ── Empirical half: does the patch actually move the reader? ───────────


def test_eager_reexport_does_not_follow_patch(tool):
    """domain/feedback/__init__.py binds eagerly and defines no __getattr__,
    so patching _internal can never move the package attribute."""
    reached = tool._md_reaches(
        "domain.feedback._internal.model_health.get_health_monitor",
        "domain.feedback",
        "get_health_monitor",
    )
    assert reached is False


def test_lazy_reexport_follows_patch(tool):
    """domain/training/__init__.py re-imports per access, so the patch lands."""
    reached = tool._md_reaches(
        "domain.training._internal.executor.get_training_executor",
        "domain.training",
        "get_training_executor",
    )
    assert reached is True


def test_reaches_is_none_when_reader_has_no_such_attr(tool):
    assert (
        tool._md_reaches(
            "domain.feedback._internal.model_health.get_health_monitor",
            "domain.feedback",
            "no_such_attr_anywhere",
        )
        is None
    )


# ── CLI contract ───────────────────────────────────────────────────────


def test_cmd_mock_drift_returns_status(tool, capsys):
    rc = tool.cmd_mock_drift(verbose=True)
    out = capsys.readouterr().out
    assert rc in (0, 1), f"unexpected exit code {rc}"
    assert "Mock drift" in out
    # must always state a verdict, never just go quiet
    assert "confirmed" in out or "no drift" in out


def test_cmd_mock_drift_reports_a_confirmed_hit(tool, monkeypatch, capsys):
    """End-to-end: fake the two collectors so a hit is guaranteed, and check
    the report carries both the reader module and a concrete fix."""
    target = "domain.feedback._internal.model_health.get_health_monitor"
    monkeypatch.setattr(
        tool,
        "_md_targets",
        lambda: {target: {_REPO / "tests/core-py/test_health_router.py"}},
    )
    monkeypatch.setattr(
        tool,
        "_md_readers",
        lambda: {"get_health_monitor": {"domain.feedback": {"routers.health"}}},
    )
    rc = tool.cmd_mock_drift()
    out = capsys.readouterr().out
    assert rc == 1
    assert target in out
    assert "domain.feedback" in out
    assert 'patch("domain.feedback.get_health_monitor")' in out


def test_reader_that_is_the_patched_module_is_not_a_hit(tool, monkeypatch, capsys):
    """If the only file reading via reader_mod IS the module being patched, the
    patch already replaces its module-global — so it is a mock that works, not
    a dead one. Reporting it would be a false positive (seen with
    `<pkg>._internal.training.MogDB` where training.py does
    `from mogdb import MogDB`)."""
    mod = "domain.feedback._internal.training"
    target = f"{mod}.MogDB"
    monkeypatch.setattr(
        tool,
        "_md_targets",
        lambda: {target: {_REPO / "tests/core-py/test_feedback_training.py"}},
    )
    # reader file == the patched module itself
    monkeypatch.setattr(tool, "_md_readers", lambda: {"MogDB": {"mogdb": {mod}}})
    rc = tool.cmd_mock_drift()
    out = capsys.readouterr().out
    assert rc == 0, f"false positive reported:\n{out}"
    assert "no drift" in out


# ── Unresolvable targets (patch raises at setup) ───────────────────────


def test_create_true_detected_across_lines(tool):
    txt = 'with patch(\n    "a.b.c",\n    return_value=1,\n    create=True,\n):'
    assert tool._md_create_true(txt, "a.b.c") is True


def test_create_true_absent(tool):
    assert tool._md_create_true('with patch("a.b.c", return_value=1):', "a.b.c") is False


def test_create_true_absent_when_target_not_found(tool):
    assert tool._md_create_true('with patch("x.y.z", create=True):', "a.b.c") is False


def test_unresolvable_reports_a_phantom_target(tool, tmp_path):
    f = tmp_path / "test_phantom.py"
    f.write_text(
        "from unittest.mock import patch\n\n"
        'def t():\n    with patch("domain.no.such.module.attr"):\n        pass\n'
    )
    res = tool._md_unresolvable({"domain.no.such.module.attr": {f}})
    assert len(res) == 1
    target, err, files = res[0]
    assert target == "domain.no.such.module.attr"
    assert "Error" in err
    assert files == [str(f)]


def test_unresolvable_skips_create_true_sites(tool, tmp_path):
    """create=True intentionally mocks a name that does not exist yet."""
    f = tmp_path / "test_create.py"
    f.write_text(
        "from unittest.mock import patch\n\n"
        "def t():\n"
        '    with patch("domain.shell._internal.repl.TuiRepl", return_value=1, create=True):\n'
        "        pass\n"
    )
    assert tool._md_unresolvable({"domain.shell._internal.repl.TuiRepl": {f}}) == []


def test_unresolvable_accepts_a_real_target(tool, tmp_path):
    f = tmp_path / "test_ok.py"
    f.write_text(
        "from unittest.mock import patch\n\n"
        'def t():\n    with patch("domain.feedback._internal.model_health.get_health_monitor"):\n'
        "        pass\n"
    )
    tgt = "domain.feedback._internal.model_health.get_health_monitor"
    assert tool._md_unresolvable({tgt: {f}}) == []


# ── Suspects (reader reachable within 2 imports) ───────────────────────


def test_direct_imports_reads_both_forms(tool, tmp_path):
    f = tmp_path / "t.py"
    f.write_text("from routers.kb import router\nimport os\n\nif True:\n    from a.b import c\n")
    imps = tool._md_direct_imports(f)
    assert {"routers.kb", "os", "a.b"} <= imps


def test_reachable_follows_edges_within_depth(tool):
    graph = {"a": {"b"}, "b": {"c"}, "c": {"d"}, "d": set()}
    assert tool._md_reachable({"a"}, graph, 1) == {"a", "b"}
    assert tool._md_reachable({"a"}, graph, 2) == {"a", "b", "c"}
    assert "d" not in tool._md_reachable({"a"}, graph, 2)


def test_reachable_tolerates_unknown_modules(tool):
    assert tool._md_reachable({"requests"}, {"requests": set()}, 2) == {"requests"}


def test_import_graph_pins_the_split_coverage_edge(tool):
    """routers/inference.py imports get_provider from domain.models at module
    level (line 45) — that edge is what makes the _internal patch dead there."""
    graph = tool._md_import_graph()
    assert "routers.inference" in graph
    assert "domain.models" in graph["routers.inference"]


def test_suspects_are_reported_but_never_block(tool, monkeypatch, capsys):
    """Import reach is weaker than runtime use, so a 2-hop hit must be
    printed as a suspect and must NOT flip the exit code."""
    target = "domain.x._internal.y.thing"
    monkeypatch.setattr(tool, "_md_targets", lambda: {target: {_REPO / "tests/test_absent.py"}})
    monkeypatch.setattr(tool, "_md_readers", lambda: {"thing": {"domain.x": {"routers.subject"}}})
    monkeypatch.setattr(tool, "_md_reaches", lambda t, r, a: False)
    monkeypatch.setattr(tool, "_md_unresolvable", lambda t: [])
    # no direct import (latent), but reachable in exactly 2 hops
    monkeypatch.setattr(
        tool,
        "_md_import_graph",
        lambda: {"routers.subject": {"routers.mid"}, "routers.mid": {"routers.leaf"}},
    )
    monkeypatch.setattr(tool, "_md_direct_imports", lambda p: {"routers.subject"})

    rc = tool.cmd_mock_drift()
    out = capsys.readouterr().out
    assert "Suspects" in out
    assert target in out
    assert "routers.subject" in out
    assert rc == 0, f"suspects must not block, got rc={rc}"


def test_suspects_absent_when_reader_not_reachable(tool, monkeypatch, capsys):
    target = "domain.x._internal.y.thing"
    monkeypatch.setattr(tool, "_md_targets", lambda: {target: {_REPO / "tests/test_absent.py"}})
    monkeypatch.setattr(tool, "_md_readers", lambda: {"thing": {"domain.x": {"routers.far_away"}}})
    monkeypatch.setattr(tool, "_md_reaches", lambda t, r, a: False)
    monkeypatch.setattr(tool, "_md_unresolvable", lambda t: [])
    monkeypatch.setattr(tool, "_md_import_graph", lambda: {"routers.subject": set()})
    monkeypatch.setattr(tool, "_md_direct_imports", lambda p: {"routers.subject"})

    out = ""
    rc = tool.cmd_mock_drift()
    out = capsys.readouterr().out
    assert "Suspects" not in out
    assert "latent" in out
    assert rc == 0
