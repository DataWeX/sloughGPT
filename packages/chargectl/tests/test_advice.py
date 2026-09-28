"""Tests for chargectl.advice — the 80% longevity heuristic."""

from __future__ import annotations

import pytest
from chargectl.advice import optimize_hint
from chargectl.status import ChargeStatus


def status(level: int, charging: bool) -> ChargeStatus:
    return ChargeStatus(
        level=level,
        is_charging=charging,
        is_plugged=charging,
        health="Good",
        capacity=4000,
        voltage_mv=7400,
        current_ma=1200 if charging else -400,
        time_to_full_min=30 if charging else None,
        time_to_empty_min=None if charging else 120,
        source="simulated",
    )


def test_high_and_charging_says_unplug():
    hint = optimize_hint(status(95, True))
    assert hint["action"] == "unplug"
    assert hint["limit"] == 80
    assert "95%" in hint["reason"]


def test_at_or_above_limit_and_charging_says_cap():
    hint = optimize_hint(status(85, True))
    assert hint["action"] == "cap_at_80"
    assert "80%" in hint["reason"]


def test_low_and_discharging_says_plug_in():
    hint = optimize_hint(status(15, False))
    assert hint["action"] == "plug_in"
    assert hint["limit"] == 100


def test_mid_range_says_maintain():
    hint = optimize_hint(status(50, False))
    assert hint["action"] == "maintain"
    assert "on battery" in hint["reason"]


def test_low_but_charging_is_maintain_not_plug_in():
    hint = optimize_hint(status(10, True))
    assert hint["action"] == "maintain"


def test_custom_optimal_limit_flows_through():
    hint = optimize_hint(status(85, True), optimal_limit=60)
    assert hint["limit"] == 60
    assert "60%" in hint["reason"]


@pytest.mark.parametrize("level", [0, 20, 50, 80, 90, 100])
@pytest.mark.parametrize("charging", [True, False])
def test_every_state_returns_a_known_action(level: int, charging: bool):
    hint = optimize_hint(status(level, charging))
    assert hint["action"] in {"unplug", "cap_at_80", "plug_in", "maintain"}
    assert isinstance(hint["reason"], str) and hint["reason"]
