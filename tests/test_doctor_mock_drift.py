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


# ── The second spelling: monkeypatch.setattr ─────────────────────────────


def _write_targets_root(tmp_path, body: str):
    root = tmp_path / "tests"
    root.mkdir()
    f = root / "test_scan.py"
    f.write_text(body)
    return f


def test_targets_pick_up_string_form_setattr(tool, monkeypatch, tmp_path):
    """`monkeypatch.setattr("pkg.mod.attr", v)` names a definition site exactly
    as `patch("pkg.mod.attr")` does, so it must be scanned too.  Before this the
    two spellings answered differently and 37 real targets were invisible."""
    tgt = "domain.learner._internal.knowledge_augmenter.get_knowledge_memory"
    f = _write_targets_root(
        tmp_path,
        f'def t(monkeypatch):\n    monkeypatch.setattr("{tgt}", 1)\n',
    )
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tool, "_MOCK_DRIFT_TEST_DIRS", ("tests",))
    targets = tool._md_targets()
    assert tgt in targets, "string-form setattr target was not collected"
    assert f in targets[tgt]


def test_targets_ignore_object_form_setattr(tool, monkeypatch, tmp_path):
    """The object form (`setattr(obj, "_save", v)`) names no dotted path, so
    there is nothing to resolve to a definition site.  It must contribute zero
    targets rather than a guessed one."""
    f = _write_targets_root(
        tmp_path,
        "def t(monkeypatch, svc):\n    monkeypatch.setattr(svc, '_save', 1)\n",
    )
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tool, "_MOCK_DRIFT_TEST_DIRS", ("tests",))
    assert tool._md_targets() == {}, f"object form leaked a target: {f}"


def test_targets_require_internal_definition_site(tool, monkeypatch, tmp_path):
    """The `._internal.` filter applies to both spellings — a setattr on a
    public package symbol is not a definition site."""
    _write_targets_root(tmp_path, 'def t(mp):\n    mp.setattr("domain.public.thing", 1)\n')
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tool, "_MOCK_DRIFT_TEST_DIRS", ("tests",))
    assert tool._md_targets() == {}


def test_real_repo_attrset_drift_is_now_visible(tool):
    """Regression: this exact drift existed while setattr was unscanned — the
    test patches domain.learner...knowledge_augmenter.get_knowledge_memory while
    production reads it from domain.knowledge / domain.memory."""
    tgt = "domain.learner._internal.knowledge_augmenter.get_knowledge_memory"
    targets = tool._md_targets()
    assert tgt in targets, "setattr-scan regression: drift is invisible again"
    assert any(p.name == "test_knowledge_augmenter.py" for p in targets[tgt])


# ── Boom-vs-baseline verdict ─────────────────────────────────────────────


def test_boom_verdict_dead_when_boom_never_fires(tool):
    """baseline pass + boom pass => production never executed the symbol, so
    the patch is a no-op and any assertion about it is vacuous."""
    assert tool._md_boom_verdict(0, 0, "armed") == "dead"


def test_boom_verdict_live_when_boom_fires(tool):
    """baseline pass + boom FAIL => the symbol really is on the executed path."""
    assert tool._md_boom_verdict(0, 1, "armed") == "live"


def test_boom_verdict_inconclusive_when_baseline_already_red(tool):
    """A baseline that already fails says nothing about the symbol."""
    assert tool._md_boom_verdict(1, 0, "armed") == "inconclusive"
    assert tool._md_boom_verdict(2, 1, "armed") == "inconclusive"


def test_boom_verdict_never_reads_unarmed_as_dead(tool):
    """The probe's dangerous failure mode is the opposite of the detector's:
    reporting a LIVE seam as dead would tell someone to delete a working mock.
    A symbol that could not be armed must be inconclusive, never dead."""
    for state in ("not-probeable", "unresolved", ""):
        assert tool._md_boom_verdict(0, 0, state) == "inconclusive", state


def test_boom_verdict_inconclusive_on_pytest_usage_errors(tool):
    """rc 4/5 (usage error / no tests collected) means the run never reached
    the symbol at all — a pass-by-accident, not evidence of deadness."""
    assert tool._md_boom_verdict(0, 4, "armed") == "inconclusive"
    assert tool._md_boom_verdict(0, 5, "armed") == "inconclusive"


# ── Static status: what --mock-drift already believed ────────────────────


def test_static_status_lands_for_lazy_reexport(tool):
    assert (
        tool._md_static_status(
            "domain.training._internal.executor.get_training_executor",
            {"get_training_executor": {"domain.training": set()}},
        )
        == "lands"
    )


def test_static_status_frozen_for_eager_reexport(tool):
    assert (
        tool._md_static_status(
            "domain.feedback._internal.model_health.get_health_monitor",
            {"get_health_monitor": {"domain.feedback": set()}},
        )
        == "frozen"
    )


def test_static_status_na_without_readers(tool):
    assert tool._md_static_status("domain.x._internal.y.thing", {}) == "n/a"


# ── The boom plugin ──────────────────────────────────────────────────────


def _load_plugin():
    path = _REPO / "scripts" / "dead_seam_plugin.py"
    spec = importlib.util.spec_from_file_location("dead_seam_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plugin_arms_callable_and_boom_raises(tool, monkeypatch, tmp_path):
    """The boom half must actually fire: armed means the symbol is replaced by
    something that raises when production reaches it."""
    import domain.feedback._internal.model_health as mh

    plugin = _load_plugin()
    marker = tmp_path / "marker.txt"
    monkeypatch.setenv("SLO_DEAD_SEAM_MARKER", str(marker))
    monkeypatch.setenv(
        "SLO_DEAD_SEAM_TARGET",
        "domain.feedback._internal.model_health.get_health_monitor",
    )
    # register the original so this module-scoped mutation cannot leak
    monkeypatch.setattr(mh, "get_health_monitor", mh.get_health_monitor)

    plugin.pytest_configure(None)

    assert "state=armed" in marker.read_text()
    with pytest.raises(plugin.DeadSeamHit):
        mh.get_health_monitor()


def test_plugin_declares_non_callable_unprobeable(tool, monkeypatch, tmp_path):
    """A constant cannot be booby-trapped by swapping in a function — a read
    would return it silently and look 'dead'. The plugin must say so, so the
    verdict lands on inconclusive instead of a false dead."""
    import domain.agents._internal.system as sysmod

    assert not callable(sysmod.AGENTS_DIR), "expected a non-callable constant"

    plugin = _load_plugin()
    marker = tmp_path / "marker.txt"
    monkeypatch.setenv("SLO_DEAD_SEAM_MARKER", str(marker))
    monkeypatch.setenv("SLO_DEAD_SEAM_TARGET", "domain.agents._internal.system.AGENTS_DIR")
    monkeypatch.setattr(sysmod, "AGENTS_DIR", sysmod.AGENTS_DIR)

    plugin.pytest_configure(None)

    txt = marker.read_text()
    assert "state=not-probeable" in txt
    # verdict must not be dead
    assert tool._md_boom_verdict(0, 0, "not-probeable") == "inconclusive"


def test_plugin_writes_unresolved_for_missing_symbol(tool, monkeypatch, tmp_path):
    plugin = _load_plugin()
    marker = tmp_path / "marker.txt"
    monkeypatch.setenv("SLO_DEAD_SEAM_MARKER", str(marker))
    monkeypatch.setenv("SLO_DEAD_SEAM_TARGET", "domain.no.such.module.nothing_here")

    plugin.pytest_configure(None)

    assert "state=unresolved" in marker.read_text()
    assert tool._md_boom_verdict(0, 0, "unresolved") == "inconclusive"


def test_plugin_is_a_noop_without_armed_env(tool, monkeypatch, tmp_path):
    """Bare `pytest -p dead_seam_plugin` must change nothing."""
    plugin = _load_plugin()
    marker = tmp_path / "marker.txt"
    monkeypatch.delenv("SLO_DEAD_SEAM_TARGET", raising=False)
    monkeypatch.setenv("SLO_DEAD_SEAM_MARKER", str(marker))

    plugin.pytest_configure(None)

    assert not marker.exists()


# ── cmd_dead_seam contract ───────────────────────────────────────────────


def test_dead_seam_missing_file_is_usage_error(tool, capsys):
    rc = tool.cmd_dead_seam("tests/definitely_absent_zzz.py")
    out = capsys.readouterr().out
    assert rc == 2
    assert "no such test" in out


def test_dead_seam_no_targets_is_clean(tool, monkeypatch, capsys):
    monkeypatch.setattr(tool, "_md_targets", lambda: {})
    rc = tool.cmd_dead_seam("tests/test_knowledge_augmenter.py")
    out = capsys.readouterr().out
    assert rc == 0
    assert "no patch targets" in out


def test_dead_seam_reports_dead_and_blocks(tool, monkeypatch, capsys):
    """A dead seam is the whole point of the probe, so it must set the exit
    code — a green run on a vacuous test is the failure being hunted."""
    tgt = "domain.x._internal.y.thing"
    tf = _REPO / "tests/test_knowledge_augmenter.py"
    monkeypatch.setattr(tool, "_md_targets", lambda: {tgt: {tf}})
    monkeypatch.setattr(tool, "_md_readers", lambda: {})
    monkeypatch.setattr(tool, "_run", lambda args, **kw: (0, "", ""))
    monkeypatch.setattr(tool, "_boom_run", lambda test, target, timeout=600: (0, "armed"))

    rc = tool.cmd_dead_seam("tests/test_knowledge_augmenter.py")
    out = capsys.readouterr().out
    assert rc == 1, "a dead seam must fail the run"
    assert "dead seam" in out
    assert tgt in out
    assert "never executed" in out


def test_dead_seam_live_seam_is_green(tool, monkeypatch, capsys):
    tgt = "domain.x._internal.y.thing"
    tf = _REPO / "tests/test_knowledge_augmenter.py"
    monkeypatch.setattr(tool, "_md_targets", lambda: {tgt: {tf}})
    monkeypatch.setattr(tool, "_md_readers", lambda: {})
    monkeypatch.setattr(tool, "_run", lambda args, **kw: (0, "", ""))
    monkeypatch.setattr(tool, "_boom_run", lambda test, target, timeout=600: (1, "armed"))

    rc = tool.cmd_dead_seam("tests/test_knowledge_augmenter.py")
    out = capsys.readouterr().out
    assert rc == 0, "a live seam must not fail the run"
    assert "no dead seam" in out


def test_dead_seam_unarmed_run_is_never_dead(tool, monkeypatch, capsys):
    """Marker lost/unwritten => inconclusive, not dead (never false-alarm)."""
    tgt = "domain.x._internal.y.thing"
    tf = _REPO / "tests/test_knowledge_augmenter.py"
    monkeypatch.setattr(tool, "_md_targets", lambda: {tgt: {tf}})
    monkeypatch.setattr(tool, "_md_readers", lambda: {})
    monkeypatch.setattr(tool, "_run", lambda args, **kw: (0, "", ""))
    monkeypatch.setattr(tool, "_boom_run", lambda test, target, timeout=600: (0, "unresolved"))

    rc = tool.cmd_dead_seam("tests/test_knowledge_augmenter.py")
    out = capsys.readouterr().out
    assert rc == 0
    assert "no dead seam" in out
    assert "inconclusive" in out


def test_dead_seam_skips_when_baseline_red(tool, monkeypatch, capsys):
    tf = _REPO / "tests/test_knowledge_augmenter.py"
    monkeypatch.setattr(tool, "_md_targets", lambda: {"domain.x._internal.y.thing": {tf}})
    monkeypatch.setattr(tool, "_run", lambda args, **kw: (1, "", ""))
    boomed = []
    monkeypatch.setattr(
        tool,
        "_boom_run",
        lambda test, target, timeout=600: boomed.append(target) or (0, "armed"),
    )

    rc = tool.cmd_dead_seam("tests/test_knowledge_augmenter.py")
    out = capsys.readouterr().out
    assert rc == 0
    assert "baseline already failing" in out
    assert boomed == [], "must not burn a second run on a red baseline"
