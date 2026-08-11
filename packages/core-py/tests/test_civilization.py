"""
Tests for Stage 12 integrated world (the full program).

Each opt-in channel — structures, teaching, memory, messages, predation,
territoriality, lifecycle, specialization — is proven in isolation. Stage 12
turns them ALL on at once in a single evolution run and verifies four
invariants that only hold when the channels are genuinely composable:

  1. CONSERVATION under full load: over a live, fully-loaded generation the
     world total (grid + entity + nest energy) never increases. Each channel
     is individually transfer-safe; together they must still never create
     energy (``_conservation_sweep`` is the live-tick-loop tripwire).
  2. RNG ISOLATION under total load: the four behavior brains (cells, body,
     entity, move) drawn with the same seed are bit-identical whether every
     dedicated-stream channel is off or ALL on, so the locked selection
     proofs keep their exact genome layout and energy flow no matter how many
     channels coexist.
  3. CHANNEL LIVENESS: every opt-in channel demonstrably fires somewhere in
     the run — lessons (teaching), predations, defenses AND raids
     (territoriality), nests_built (structures), births (lifecycle), role
     deposits/raids (specialization), and a growing world reservoir (memory).
  4. SUSTAINABILITY: births > 0 with survivors at the final generation — the
     world grows its own population while predation and contest are taking
     lives.

The ``control`` arm (every channel off, learning off) is the negative
control: it produces zero channel activity by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    Genome,
    _conservation_sweep,
    benchmark_civilization,
)
from domains.shell.simulation import WorldParams


def _all_on_params(**kw) -> WorldParams:
    base = dict(
        grid_size=(16, 8, 16),
        learning_enabled=True,
        message_enabled=True,
        structure_enabled=True,
        teaching_enabled=True,
        memory_enabled=True,
        predation_enabled=True,
        territoriality_enabled=True,
        lifecycle_enabled=True,
        specialization_enabled=True,
        write_energy_scale=10.0,
        max_entities=32,
    )
    base.update(kw)
    return WorldParams(**base)


def _sweep_genomes(params: WorldParams, n: int, seed: int,
                   group_count: int = 2) -> list[Genome]:
    rng = np.random.default_rng(seed)
    return [Genome.random(params, rng, group_id=i % group_count)
            for i in range(n)]


class TestConservationSweep:
    """The live-tick-loop physics tripwire under full load."""

    def test_monotonic_with_all_channels_on(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 8, 7)
        result = _conservation_sweep(params, genomes, ticks=24)
        assert result["monotonic"] is True
        assert result["violations"] == []
        assert result["start_total"] > 0.0
        assert result["end_total"] <= result["start_total"] + 1e-6

    def test_end_total_never_exceeds_start_under_load(self):
        params = _all_on_params(generate_world=True, world_seed=3)
        genomes = _sweep_genomes(params, 8, 3)
        result = _conservation_sweep(params, genomes, ticks=16)
        assert result["start_total"] >= result["end_total"] - 1e-6
        assert isinstance(result["violations"], list)

    def test_sweep_deterministic(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 8, 7)
        a = _conservation_sweep(params, genomes, ticks=12)
        b = _conservation_sweep(params, genomes, ticks=12)
        assert a == b

    def test_sweep_reports_violation(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 8, 7)

        def _energy_injector(self):
            for b in self.scene.babies:
                b.entity.energy += 100.0

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("domains.shell.evolution.Simulation.step",
                       _energy_injector)
            result = _conservation_sweep(params, genomes, ticks=4)
        assert result["monotonic"] is False
        assert len(result["violations"]) >= 1
        assert all(v[2] > v[1] + 1e-6 for v in result["violations"])


class TestRNGIsolationUnderTotalLoad:
    """The strongest RNG claim: all dedicated streams on at once."""

    def test_four_behavior_brains_bit_identical_all_on(self):
        off = WorldParams(grid_size=(16, 8, 16))
        on = _all_on_params(generate_world=True, world_seed=7)
        for seed in (1, 3, 7):
            g_off = Genome.random(off, np.random.default_rng(seed), group_id=0)
            g_on = Genome.random(on, np.random.default_rng(seed), group_id=0)
            for name in ("cells", "body", "entity", "move"):
                for suf in ("W", "b"):
                    assert np.allclose(
                        g_off.tensors[f"{name}.{suf}"],
                        g_on.tensors[f"{name}.{suf}"])

    def test_all_channel_brains_present_when_on(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g = Genome.random(on, np.random.default_rng(7), group_id=0)
        for name in ("message", "teach", "predation", "territory",
                     "reproduce", "role"):
            assert f"{name}.W" in g.tensors
            assert f"{name}.b" in g.tensors

    def test_no_channel_brains_when_off(self):
        off = WorldParams(grid_size=(16, 8, 16))
        g = Genome.random(off, np.random.default_rng(7), group_id=0)
        for name in ("message", "teach", "predation", "territory",
                     "reproduce", "role"):
            assert f"{name}.W" not in g.tensors


class TestBenchmarkStructure:
    """Result shape, determinism, and the negative control arm."""

    def test_structure_and_verdict_keys(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert set(result) >= {
            "control", "civilization", "group_count", "group_weight",
            "control_last_avg", "civilization_last_avg",
            "conservation_monotonic", "conservation_violations",
            "conservation_start_total", "conservation_end_total",
            "brains_identical", "channels_live", "channels_live_all",
            "births", "alive_count", "population_size",
            "civilization_emerged",
        }
        assert len(result["control"]["history"]) == 3
        assert len(result["civilization"]["history"]) == 3
        assert result["births"] == \
            result["civilization"]["history"][-1]["births"]
        assert isinstance(result["channels_live"], dict)
        assert set(result["channels_live"]) >= {
            "lessons", "predations", "defenses", "raids", "nests_built",
            "births", "role_deposits", "role_raids", "memory",
        }
        assert isinstance(result["civilization_emerged"], bool)
        assert isinstance(result["conservation_monotonic"], bool)
        assert isinstance(result["brains_identical"], bool)

    def test_deterministic(self):
        a = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=2,
        )
        b = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=2,
        )
        assert a["civilization"]["history"] == \
            b["civilization"]["history"]
        assert a["control"]["history"] == b["control"]["history"]
        assert a["conservation_violations"] == b["conservation_violations"]
        assert a["conservation_start_total"] == \
            b["conservation_start_total"]

    def test_control_arm_produces_zero_channel_activity(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        quiet = ("lessons", "predations", "defenses", "raids",
                 "nests_built", "births", "role_deposits",
                 "role_raids", "memory_size")
        for entry in result["control"]["history"]:
            for key in quiet:
                assert entry.get(key, 0) == 0

    def test_both_arms_have_full_history(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        for arm in ("control", "civilization"):
            entry = result[arm]["history"][0]
            for key in ("cooperations", "contests", "avg_fitness",
                        "best_fitness"):
                assert key in entry


class TestIntegratedInvariants:
    """The hard claims that must hold regardless of seed."""

    def test_conservation_holds_across_seeds(self):
        for seed in (1, 3, 7):
            result = benchmark_civilization(
                population_size=8, generations=6, ticks_per_generation=24,
                organic_pools=3, seed=seed,
            )
            assert result["conservation_monotonic"] is True, seed
            assert result["conservation_violations"] == [], seed
            assert result["conservation_end_total"] <= \
                result["conservation_start_total"] + 1e-6, seed

    def test_rng_isolation_holds_across_seeds(self):
        for seed in (1, 3, 7):
            result = benchmark_civilization(
                population_size=8, generations=6, ticks_per_generation=24,
                organic_pools=3, seed=seed,
            )
            assert result["brains_identical"] is True, seed

    def test_conservation_and_rng_are_seed_independent(self):
        for seed in (1, 3, 7):
            result = benchmark_civilization(
                population_size=8, generations=6, ticks_per_generation=24,
                organic_pools=3, seed=seed,
            )
            assert result["conservation_monotonic"] == \
                result["brains_identical"] is True, seed


class TestFullProgram:
    """The demonstration seed: every channel fires and the world self-sustains."""

    def test_all_channels_fire(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert result["channels_live_all"] is True
        for name, fired in result["channels_live"].items():
            assert fired is True, name

    def test_world_self_sustains(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert result["births"] > 0
        assert result["alive_count"] > 0
        assert result["civilization_emerged"] is True

    def test_liveness_matches_derived_conditions(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        derived = (
            result["conservation_monotonic"]
            and result["brains_identical"]
            and result["channels_live_all"]
            and result["births"] > 0
        )
        assert result["civilization_emerged"] == derived
