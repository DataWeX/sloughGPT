"""Tests for chargectl.daemon — the enforcement tick, state file, and --once loop."""

from __future__ import annotations

import json
import os
from pathlib import Path

from chargectl.daemon import apply_directive, read_state, run_daemon, tick, write_state
from chargectl.policy import Directive, Policy, save_policy

END = "charge_control_end_threshold"
START = "charge_control_start_threshold"


def make_sysfs(tmp_path: Path, *, end: str = "100\n", floor: str | None = "1\n") -> Path:
    base = tmp_path / "power_supply"
    bat = base / "BAT0"
    bat.mkdir(parents=True)
    (bat / "type").write_text("Battery")
    (bat / "capacity").write_text("55")
    (bat / "status").write_text("Charging")
    (bat / END).write_text(end)
    if floor is not None:
        (bat / START).write_text(floor)
    (base / "ADP1").mkdir()
    (base / "ADP1" / "online").write_text("1")
    return base


def thresholds(base: Path) -> tuple[str, str | None]:
    bat = base / "BAT0"
    floor = (bat / START).read_text().strip() if (bat / START).exists() else None
    return (bat / END).read_text().strip(), floor


def enabled_policy(**kwargs) -> Policy:
    return Policy(enabled=True, **kwargs)


# ── tick ────────────────────────────────────────────────────────────────────


def test_tick_applies_band_and_records_state(tmp_path: Path):
    base = make_sysfs(tmp_path)
    state_path = tmp_path / "state.json"
    state = tick(enabled_policy(), sys_base=base, state_path=state_path)

    assert thresholds(base) == ("80", "40")
    assert state["action"]["action"] == "set_ceiling"
    assert state["result"]["applied"] is True
    assert state["owned"] is True
    assert state["dry_run"] is False
    assert state["enabled"] is True
    assert state["explain"].startswith("policy on")
    assert state["status"]["source"] == "sysfs"
    assert state_path.is_file()
    assert read_state(state_path)["owned"] is True


def test_tick_is_idempotent(tmp_path: Path):
    base = make_sysfs(tmp_path)
    state_path = tmp_path / "state.json"
    tick(enabled_policy(), sys_base=base, state_path=state_path)
    second = tick(enabled_policy(), sys_base=base, state_path=state_path)

    assert second["action"] is None
    assert second["result"] is None
    assert second["owned"] is True  # ownership survives a do-nothing tick
    assert thresholds(base) == ("80", "40")


def test_tick_reasserts_after_kernel_drift(tmp_path: Path):
    base = make_sysfs(tmp_path)
    state_path = tmp_path / "state.json"
    tick(enabled_policy(), sys_base=base, state_path=state_path)

    # reboot / AC re-plug reset the thresholds
    (base / "BAT0" / END).write_text("100\n")
    (base / "BAT0" / START).write_text("0\n")

    state = tick(enabled_policy(), sys_base=base, state_path=state_path)
    assert thresholds(base) == ("80", "40")
    assert state["action"]["action"] == "set_ceiling"
    assert "re-asserting" in state["action"]["reason"]


def test_tick_ceiling_only_when_there_is_no_start_threshold(tmp_path: Path):
    base = make_sysfs(tmp_path, floor=None)
    state = tick(enabled_policy(), sys_base=base, state_path=tmp_path / "s.json")

    assert thresholds(base) == ("80", None)
    assert state["result"]["applied"] is True
    assert state["capability"]["start_supported"] is False


def test_tick_disabled_policy_does_not_touch_thresholds(tmp_path: Path):
    base = make_sysfs(tmp_path, end="70\n")
    state = tick(Policy(enabled=False), sys_base=base, state_path=tmp_path / "s.json")

    assert thresholds(base) == ("70", "1")
    assert state["action"] is None
    assert state["owned"] is False
    assert state["explain"].startswith("policy off")


def test_disabled_policy_lifts_only_its_own_cap(tmp_path: Path):
    base = make_sysfs(tmp_path)
    state_path = tmp_path / "s.json"
    tick(enabled_policy(), sys_base=base, state_path=state_path)
    assert thresholds(base) == ("80", "40")

    state = tick(Policy(enabled=False), sys_base=base, state_path=state_path)
    assert thresholds(base)[0] == "100"
    assert state["action"]["action"] == "clear"
    assert state["owned"] is False


def test_disabled_policy_leaves_a_manual_cap_alone(tmp_path: Path):
    base = make_sysfs(tmp_path, end="70\n", floor="20\n")
    state = tick(Policy(enabled=False), sys_base=base, state_path=tmp_path / "s.json")

    assert thresholds(base) == ("70", "20")
    assert state["owned"] is False


def test_tick_is_a_dry_run_without_a_battery(tmp_path: Path):
    missing = tmp_path / "no-such-sysfs"
    state = tick(enabled_policy(), sys_base=missing, state_path=tmp_path / "s.json")

    assert state["dry_run"] is True
    assert state["capability"]["supported"] is False
    assert state["action"] is None
    assert state["error"] is None
    assert state["status"]["source"] == "simulated"
    assert (tmp_path / "s.json").is_file()


def test_tick_never_raises_on_a_garbage_sysfs(tmp_path: Path):
    base = tmp_path / "power_supply"
    bat = base / "BAT0"
    bat.mkdir(parents=True)
    (bat / "type").write_text("Battery")
    (bat / "capacity").write_text("not-a-number")
    (bat / END).write_text("junk")

    state = tick(enabled_policy(), sys_base=base, state_path=tmp_path / "s.json")
    assert state["updated_at"] > 0


def test_tick_state_is_json_serialisable(tmp_path: Path):
    state = tick(enabled_policy(), sys_base=tmp_path / "absent", state_path=tmp_path / "s.json")
    json.dumps(state)
    assert json.loads((tmp_path / "s.json").read_text())["pid"] == os.getpid()


# ── state file ──────────────────────────────────────────────────────────────


def test_read_state_missing_and_corrupt(tmp_path: Path):
    assert read_state(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{{{")
    assert read_state(bad) == {}
    bad.write_text("[1,2,3]")
    assert read_state(bad) == {}


def test_write_state_is_atomic(tmp_path: Path):
    path = tmp_path / "nested" / "state.json"
    write_state({"pid": 1}, path)
    write_state({"pid": 2}, path)
    assert json.loads(path.read_text())["pid"] == 2
    assert not list(path.parent.glob(".chargectl-*"))


# ── apply_directive ─────────────────────────────────────────────────────────


def test_apply_directive_unknown_action():
    result = apply_directive(Directive("launch", 1, "nope"), Policy())
    assert result["applied"] is False and "unknown action" in result["reason"]


def test_apply_directive_clear_lifts_the_cap(tmp_path: Path):
    base = make_sysfs(tmp_path, end="80\n")
    result = apply_directive(Directive("clear", 100, "off"), Policy(), sys_base=base)
    assert result["applied"] is True and result["limit"] == 100


def test_apply_directive_floor_only(tmp_path: Path):
    base = make_sysfs(tmp_path, floor="1\n")
    result = apply_directive(Directive("set_floor", 40, "floor"), enabled_policy(), sys_base=base)
    assert result["applied"] is True
    assert thresholds(base)[1] == "40"


# ── run_daemon ──────────────────────────────────────────────────────────────


def test_run_daemon_once_writes_state_and_returns_zero(tmp_path: Path):
    base = make_sysfs(tmp_path)
    policy_path = tmp_path / "policy.json"
    state_path = tmp_path / "state.json"
    save_policy(enabled_policy(floor=45, ceiling=75), policy_path)

    code = run_daemon(once=True, policy_path=policy_path, state_path=state_path, sys_base=base)

    assert code == 0
    assert thresholds(base) == ("75", "45")
    assert read_state(state_path)["policy"]["ceiling"] == 75


def test_run_daemon_once_on_a_vm_exits_zero(tmp_path: Path):
    code = run_daemon(
        once=True,
        policy_path=tmp_path / "absent-policy.json",
        state_path=tmp_path / "s.json",
        sys_base=tmp_path / "no-sysfs",
    )
    assert code == 0
    assert read_state(tmp_path / "s.json")["dry_run"] is True


def test_run_daemon_reports_a_broken_policy_file(tmp_path: Path, caplog):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{{{")

    with caplog.at_level("WARNING", logger="chargectl.daemon"):
        code = run_daemon(
            once=True,
            policy_path=policy_path,
            state_path=tmp_path / "s.json",
            sys_base=tmp_path / "no-sysfs",
        )

    assert code == 0
    assert "policy unreadable" in caplog.text
