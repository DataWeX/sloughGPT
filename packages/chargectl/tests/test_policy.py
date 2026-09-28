"""Tests for chargectl.policy — pure band rules and the policy file."""

from __future__ import annotations

import json
from pathlib import Path

from chargectl.control import probe
from chargectl.policy import (
    CLEAR,
    SET_CEILING,
    SET_FLOOR,
    Policy,
    decide,
    default_policy_path,
    default_state_path,
    explain,
    load_policy,
    normalize,
    save_policy,
)
from chargectl.status import SimulatedBattery


class StubCapability:
    """Stand-in for control.Capability — policy stays free of control imports."""

    def __init__(self, supported: bool = True, start_supported: bool = True):
        self.supported = supported
        self.start_supported = start_supported


def status_at(level: int = 55):
    return SimulatedBattery().at(level).snapshot()


# ── Policy construction ──────────────────────────────────────────────────────


def test_defaults_are_off_and_sensible():
    p = Policy()
    assert p.enabled is False
    assert p.floor == 40
    assert p.ceiling == 80
    assert p.band == "40-80"
    assert p.controls_floor is True


def test_from_dict_parses_valid_policy():
    p = Policy.from_dict(
        {"enabled": True, "floor": 50, "ceiling": 75, "mode": "ceiling", "interval_seconds": 30}
    )
    assert (p.enabled, p.floor, p.ceiling, p.mode, p.interval_seconds) == (
        True,
        50,
        75,
        "ceiling",
        30.0,
    )
    assert p.controls_floor is False


def test_from_dict_tolerates_garbage():
    for raw in (
        None,
        [],
        "nope",
        {"floor": "x", "ceiling": None, "mode": "wild", "interval_seconds": "soon"},
    ):
        p = Policy.from_dict(raw)
        assert p.enabled is False
        assert p.floor == 40 and p.ceiling == 80
        assert p.mode == "band"


def test_from_dict_keeps_floor_below_ceiling():
    p = Policy.from_dict({"floor": 90, "ceiling": 50})
    assert p.floor < p.ceiling


def test_normalize_reports_errors():
    _, err = normalize(floor=80, ceiling=40)
    assert err and "floor" in err
    _, err = normalize(floor="40")
    assert err and "integer" in err
    _, err = normalize(mode="turbo")
    assert err and "mode" in err
    _, err = normalize(interval_seconds=0.5)
    assert err and "interval" in err


def test_normalize_applies_valid_values():
    p, err = normalize(floor=45, ceiling=70, mode="ceiling", enabled=True, interval_seconds=15)
    assert err is None
    assert (p.floor, p.ceiling, p.mode, p.enabled, p.interval_seconds) == (
        45,
        70,
        "ceiling",
        True,
        15.0,
    )


# ── decide() matrix ─────────────────────────────────────────────────────────


def test_disabled_policy_never_writes():
    assert decide(status_at(), StubCapability(), Policy(enabled=False), current_limit=80) is None


def test_disabled_policy_clears_only_what_it_owns():
    directive = decide(
        status_at(),
        StubCapability(),
        Policy(enabled=False),
        current_limit=80,
        owned=True,
    )
    assert directive is not None and directive.action == CLEAR and directive.value == 100


def test_disabled_policy_leaves_a_manual_cap_alone():
    assert (
        decide(
            status_at(),
            StubCapability(),
            Policy(enabled=False),
            current_limit=70,
            owned=False,
        )
        is None
    )


def test_disabled_policy_with_no_cap_in_force_is_a_noop():
    assert (
        decide(
            status_at(),
            StubCapability(),
            Policy(enabled=False),
            current_limit=100,
            owned=True,
        )
        is None
    )


def test_enabled_policy_is_a_noop_on_unsupported_machine():
    assert (
        decide(
            status_at(),
            StubCapability(supported=False),
            Policy(enabled=True),
            current_limit=None,
        )
        is None
    )


def test_drifted_ceiling_is_reasserted():
    directive = decide(
        status_at(),
        StubCapability(),
        Policy(enabled=True),
        current_limit=100,
        current_floor=40,
    )
    assert directive is not None
    assert directive.action == SET_CEILING
    assert directive.value == 80
    assert "re-asserting" in directive.reason


def test_matching_ceiling_and_floor_is_a_noop():
    assert (
        decide(
            status_at(),
            StubCapability(),
            Policy(enabled=True),
            current_limit=80,
            current_floor=40,
        )
        is None
    )


def test_matching_ceiling_but_missing_floor_sets_floor():
    directive = decide(
        status_at(),
        StubCapability(),
        Policy(enabled=True),
        current_limit=80,
        current_floor=None,
    )
    assert directive is not None and directive.action == SET_FLOOR and directive.value == 40


def test_ceiling_only_mode_ignores_the_floor():
    assert (
        decide(
            status_at(),
            StubCapability(),
            Policy(enabled=True, mode="ceiling"),
            current_limit=80,
            current_floor=None,
        )
        is None
    )


def test_band_mode_without_start_threshold_only_caps():
    directive = decide(
        status_at(),
        StubCapability(start_supported=False),
        Policy(enabled=True),
        current_limit=100,
        current_floor=None,
    )
    assert directive is not None and directive.action == SET_CEILING


def test_policy_is_level_agnostic():
    """The band is hardware hysteresis — level must not change the directive."""
    policy = Policy(enabled=True)
    at_10 = decide(status_at(10), StubCapability(), policy, current_limit=80, current_floor=40)
    at_95 = decide(status_at(95), StubCapability(), policy, current_limit=80, current_floor=40)
    assert at_10 is None and at_95 is None


# ── explain() ───────────────────────────────────────────────────────────────


def test_explain_lines():
    assert "off" in explain(Policy(enabled=False), StubCapability())
    assert "no charge threshold" in explain(Policy(enabled=True), StubCapability(supported=False))
    assert "ceiling only" in explain(Policy(enabled=True), StubCapability(start_supported=False))
    assert "40-80" in explain(Policy(enabled=True), StubCapability())


# ── policy file ─────────────────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "policy.json"
    saved = save_policy(Policy(enabled=True, floor=45, ceiling=75), path)
    assert saved == path
    loaded, err = load_policy(path)
    assert err is None
    assert (loaded.enabled, loaded.floor, loaded.ceiling) == (True, 45, 75)
    assert json.loads(path.read_text())["band"] == "45-75"


def test_missing_policy_file_returns_defaults(tmp_path: Path):
    policy, err = load_policy(tmp_path / "absent.json")
    assert err is None and policy == Policy()


def test_corrupt_policy_file_returns_defaults(tmp_path: Path):
    path = tmp_path / "policy.json"
    path.write_text("{ not json")
    policy, err = load_policy(path)
    assert err is not None and policy == Policy()


def test_save_policy_is_atomic_and_overwrites(tmp_path: Path):
    path = tmp_path / "nested" / "policy.json"
    save_policy(Policy(enabled=True), path)
    save_policy(Policy(enabled=False, ceiling=90), path)
    assert json.loads(path.read_text())["ceiling"] == 90
    assert not list(path.parent.glob(".chargectl-*"))


def test_default_paths_honour_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CHARGECTL_POLICY", str(tmp_path / "p.json"))
    monkeypatch.setenv("CHARGECTL_STATE", str(tmp_path / "s.json"))
    assert default_policy_path() == tmp_path / "p.json"
    assert default_state_path() == tmp_path / "s.json"


def test_decide_accepts_a_real_capability(tmp_path: Path):
    """The duck-typed stub and the real Capability must be interchangeable."""
    base = tmp_path / "power_supply"
    bat = base / "BAT0"
    bat.mkdir(parents=True)
    (bat / "type").write_text("Battery")
    (bat / "capacity").write_text("95")
    (bat / "charge_control_end_threshold").write_text("100")
    (bat / "charge_control_start_threshold").write_text("1")
    cap = probe(base)
    directive = decide(status_at(95), cap, Policy(enabled=True), current_limit=100, current_floor=1)
    assert directive is not None and directive.value == 80
