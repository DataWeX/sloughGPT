"""Tests for the world realm headless observability driver."""

import numpy as np
import pytest

from domains.shell.world_driver import (
    WorldDriver,
    _material_name,
    _parse_grid,
    main,
)
from domains.shell.simulation import (
    NUM_MATERIALS,
    MATERIAL_AIR,
    MATERIAL_EMBER,
    MATERIAL_ORGANIC,
    MATERIAL_STONE,
    MATERIAL_WATER,
    WorldParams,
)


def _small_params(**kw) -> WorldParams:
    return WorldParams(grid_size=(8, 6, 8), generate_world=True,
                       start_agents=2, **kw)


def test_material_names_all_materials():
    ids = {MATERIAL_AIR, MATERIAL_WATER, MATERIAL_STONE, MATERIAL_ORGANIC,
           MATERIAL_EMBER}
    names = {_material_name(m) for m in ids}
    assert "air" in names
    assert "stone" in names
    assert "ember" in names
    assert all(isinstance(n, str) and n.islower() for n in names)


def test_material_name_fallback_for_unknown_id():
    assert _material_name(999) == "material_999"


def test_parse_grid_valid():
    assert _parse_grid("4,5,6") == (4, 5, 6)


def test_parse_grid_invalid_part_count():
    with pytest.raises(ValueError):
        _parse_grid("4,5")


def test_parse_grid_non_positive():
    with pytest.raises(ValueError):
        _parse_grid("4,0,6")


def test_driver_builds_world_and_spawns_babies():
    driver = WorldDriver(_small_params(), seed=7)
    assert driver.scene.tick == 0
    assert len(driver.scene.babies) == 2
    assert len(driver.scene.alive_babies) == 2


def test_empty_world_all_air():
    params = WorldParams(grid_size=(4, 4, 4), generate_world=False,
                         start_agents=0)
    driver = WorldDriver(params, seed=1)
    pops = driver.material_populations()
    assert pops[MATERIAL_AIR] == 4 * 4 * 4
    assert sum(pops.values()) == 4 * 4 * 4


def test_generated_world_has_terrain():
    driver = WorldDriver(_small_params(), seed=7)
    pops = driver.material_populations()
    assert sum(pops.values()) == 8 * 6 * 8
    assert pops[MATERIAL_STONE] > 0
    assert pops[MATERIAL_WATER] > 0
    assert pops[MATERIAL_ORGANIC] > 0
    assert pops[MATERIAL_EMBER] > 0
    assert pops[MATERIAL_AIR] > 0


def test_snapshot_keys():
    driver = WorldDriver(_small_params(), seed=7)
    snap = driver.snapshot()
    assert snap["tick"] == 0
    assert set(snap) == {
        "tick", "alive_babies", "grid_energy", "entity_energy",
        "nest_energy", "nests", "total_energy", "total_signal",
        "mean_baby_energy", "materials",
    }
    assert snap["total_energy"] == pytest.approx(
        snap["grid_energy"] + snap["entity_energy"] + snap["nest_energy"])


def test_run_ticks_returns_one_snapshot_per_tick():
    driver = WorldDriver(_small_params(), seed=7)
    snaps = driver.run_ticks(5)
    assert len(snaps) == 5
    assert [s["tick"] for s in snaps] == [1, 2, 3, 4, 5]


def test_energy_ledger_consistent():
    driver = WorldDriver(_small_params(), seed=7)
    ledger = driver.energy_ledger()
    assert ledger["total"] == pytest.approx(
        ledger["grid"] + ledger["entities"] + ledger["nests"])
    per_mat = sum(ledger["per_material"].values())
    assert per_mat == pytest.approx(ledger["grid"])
    assert set(ledger["per_material"]) == set(range(NUM_MATERIALS))


def test_total_energy_never_increases():
    driver = WorldDriver(_small_params(), seed=7)
    snaps = driver.run_ticks(10)
    report = driver.conservation_report(snaps)
    assert report["monotonic"] is True
    assert report["start_total"] == pytest.approx(snaps[0]["total_energy"])
    assert report["end_total"] == pytest.approx(snaps[-1]["total_energy"])


def test_conservation_report_detects_violation():
    driver = WorldDriver(_small_params(), seed=7)
    fabricated = [
        {"total_energy": 100.0},
        {"total_energy": 90.0},
        {"total_energy": 95.0},
    ]
    report = driver.conservation_report(fabricated)
    assert report["monotonic"] is False
    assert report["violations"] == [(3, 90.0, 95.0)]


def test_conservation_report_empty():
    driver = WorldDriver(_small_params(), seed=7)
    report = driver.conservation_report([])
    assert report["monotonic"] is True
    assert report["start_total"] == 0.0
    assert report["end_total"] == 0.0


def test_deterministic_on_seed():
    a = WorldDriver(_small_params(world_seed=11), seed=11).snapshot()
    b = WorldDriver(_small_params(world_seed=11), seed=11).snapshot()
    assert a["materials"] == b["materials"]
    assert a["total_energy"] == pytest.approx(b["total_energy"])
    assert a["grid_energy"] == pytest.approx(b["grid_energy"])


def test_different_seed_differs():
    a = WorldDriver(_small_params(world_seed=11), seed=11).snapshot()
    b = WorldDriver(_small_params(world_seed=12), seed=12).snapshot()
    assert a["materials"] != b["materials"]


def test_mean_baby_energy_positive_at_start():
    driver = WorldDriver(_small_params(), seed=7)
    snap = driver.snapshot()
    assert snap["mean_baby_energy"] > 0.0
    assert snap["alive_babies"] == 2


def test_evolution_returns_summary():
    driver = WorldDriver(_small_params(), seed=3)
    result = driver.run_evolution(generations=2, population=4,
                                  ticks_per_generation=3, seed=3)
    assert result["generations"] == 2
    assert len(result["history"]) == 2
    for h in result["history"]:
        assert h["generation"] in (1, 2)
        assert h["best_fitness"] >= h["avg_fitness"]


def test_main_prints_tick_table(capsys):
    code = main(["--grid", "6,4,6", "--seed", "5", "--babies", "2",
                 "--ticks", "6", "--every", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "tick alive grid_energy" in out
    assert "material_populations" in out
    assert "energy_ledger" in out


def test_main_evolution_mode(capsys):
    code = main(["--grid", "6,4,6", "--seed", "5", "--babies", "2",
                 "--evolution", "--generations", "2", "--population", "3",
                 "--ticks-per-gen", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "generation best_fitness avg_fitness alive" in out
    assert "overall_best_fitness=" in out


def test_main_emergence_mode(capsys):
    code = main(["--grid", "6,4,6", "--seed", "5", "--babies", "2",
                 "--emergence", "--generations", "2", "--population", "3",
                 "--ticks-per-gen", "2", "--hidden-units", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "generation evolved_avg frozen_avg" in out
    assert "evolved_last_avg=" in out
    assert "frozen_last_avg=" in out
    assert "emergence=" in out


def test_main_rejects_bad_grid(capsys):
    code = main(["--grid", "4,5"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--grid" in captured.err
