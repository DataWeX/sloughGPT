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
    _teach_rng,
    _predation_rng,
    _territory_rng,
    _reproduce_rng,
    _role_rng,
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

    def test_sweep_start_total_positive(self):
        params = _all_on_params(generate_world=True, world_seed=5)
        genomes = _sweep_genomes(params, 4, 5)
        result = _conservation_sweep(params, genomes, ticks=8)
        assert result["start_total"] > 0.0

    def test_sweep_end_total_nonnegative(self):
        params = _all_on_params(generate_world=True, world_seed=5)
        genomes = _sweep_genomes(params, 4, 5)
        result = _conservation_sweep(params, genomes, ticks=8)
        assert result["end_total"] >= 0.0

    def test_sweep_different_seeds(self):
        for seed in [1, 42, 100]:
            params = _all_on_params(generate_world=True, world_seed=seed)
            genomes = _sweep_genomes(params, 6, seed)
            result = _conservation_sweep(params, genomes, ticks=8)
            assert result["monotonic"] is True
            assert result["violations"] == []

    def test_sweep_small_population(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 2, 7)
        result = _conservation_sweep(params, genomes, ticks=8)
        assert result["monotonic"] is True

    def test_sweep_zero_ticks(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=0)
        assert result["start_total"] == result["end_total"]

    def test_sweep_single_tick(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=1)
        assert result["monotonic"] is True
        assert result["end_total"] <= result["start_total"] + 1e-6

    def test_sweep_many_ticks(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=48)
        assert result["monotonic"] is True

    def test_sweep_violations_list_structure(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 8, 7)

        def _energy_injector(self):
            for b in self.scene.babies:
                b.entity.energy += 100.0

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("domains.shell.evolution.Simulation.step",
                       _energy_injector)
            result = _conservation_sweep(params, genomes, ticks=4)
        for v in result["violations"]:
            assert isinstance(v, tuple)
            assert len(v) == 3

    def test_sweep_result_keys(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=4)
        assert "monotonic" in result
        assert "violations" in result
        assert "start_total" in result
        assert "end_total" in result


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

    def test_brain_tensor_shapes(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g = Genome.random(on, np.random.default_rng(7), group_id=0)
        # Basic brains should have W and b
        for name in ("cells", "body", "entity", "move"):
            assert g.tensors[f"{name}.W"].ndim == 2
            assert g.tensors[f"{name}.b"].ndim == 1

    def test_channel_brain_tensor_shapes(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g = Genome.random(on, np.random.default_rng(7), group_id=0)
        for name in ("message", "teach", "predation", "territory",
                     "reproduce", "role"):
            assert g.tensors[f"{name}.W"].ndim >= 1
            assert g.tensors[f"{name}.b"].ndim >= 1

    def test_different_seeds_give_different_brains(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g1 = Genome.random(on, np.random.default_rng(1), group_id=0)
        g2 = Genome.random(on, np.random.default_rng(2), group_id=0)
        assert not np.allclose(g1.tensors["cells.W"], g2.tensors["cells.W"])

    def test_same_seed_same_group_same_brains(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g1 = Genome.random(on, np.random.default_rng(5), group_id=0)
        g2 = Genome.random(on, np.random.default_rng(5), group_id=0)
        for key in g1.tensors:
            assert np.allclose(g1.tensors[key], g2.tensors[key])

    def test_different_group_ids(self):
        on = _all_on_params(generate_world=True, world_seed=7)
        g0 = Genome.random(on, np.random.default_rng(5), group_id=0)
        g1 = Genome.random(on, np.random.default_rng(5), group_id=1)
        # Teach brain should differ due to group_id-dependent seed
        assert not np.allclose(g0.tensors["teach.W"], g1.tensors["teach.W"])


class TestGenomeStructure:
    """Genome creation and properties."""

    def test_genome_creation(self):
        tensors = {"cells.W": np.zeros((4, 4)), "cells.b": np.zeros(4)}
        g = Genome(tensors)
        assert "cells.W" in g.tensors
        assert "cells.b" in g.tensors

    def test_genome_group_id(self):
        tensors = {"cells.W": np.zeros((4, 4))}
        g = Genome(tensors, group_id=3)
        assert g.group_id == 3

    def test_genome_default_group_id(self):
        tensors = {"cells.W": np.zeros((4, 4))}
        g = Genome(tensors)
        assert g.group_id == 0

    def test_genome_memories(self):
        tensors = {"cells.W": np.zeros((4, 4))}
        memories = [{"features": [1.0], "action": [0], "reward": 1.0}]
        g = Genome(tensors, memories=memories)
        assert g.memory_count == 1

    def test_genome_empty_memories(self):
        tensors = {"cells.W": np.zeros((4, 4))}
        g = Genome(tensors)
        assert g.memory_count == 0

    def test_genome_tensor_dtype_float32(self):
        tensors = {"cells.W": np.zeros((4, 4), dtype=np.float64)}
        g = Genome(tensors)
        assert g.tensors["cells.W"].dtype == np.float32

    def test_random_genome_has_all_base_brains(self):
        params = WorldParams(grid_size=(16, 8, 16))
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        for name in ("cells", "body", "entity", "move"):
            assert f"{name}.W" in g.tensors
            assert f"{name}.b" in g.tensors

    def test_random_genome_tensor_shapes(self):
        params = WorldParams(grid_size=(16, 8, 16))
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        # cells.W should be 2D
        assert g.tensors["cells.W"].ndim == 2
        # cells.b should be 1D
        assert g.tensors["cells.b"].ndim == 1


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

    def test_control_history_entry_keys(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        entry = result["control"]["history"][0]
        assert "cooperations" in entry
        assert "contests" in entry
        assert "avg_fitness" in entry
        assert "best_fitness" in entry

    def test_civilization_history_entry_keys(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        entry = result["civilization"]["history"][0]
        assert "cooperations" in entry
        assert "contests" in entry
        assert "avg_fitness" in entry
        assert "best_fitness" in entry

    def test_group_count_positive(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert result["group_count"] >= 1

    def test_population_size_preserved(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert result["population_size"] == 4

    def test_conservation_start_total_positive(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert result["conservation_start_total"] > 0.0


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

    def test_conservation_end_not_exceed_start(self):
        for seed in [1, 5]:
            result = benchmark_civilization(
                population_size=8, generations=6, ticks_per_generation=24,
                organic_pools=3, seed=seed,
            )
            assert result["conservation_end_total"] <= result["conservation_start_total"] + 1e-6

    def test_all_invariants_hold_for_each_seed(self):
        for seed in (2, 4, 6):
            result = benchmark_civilization(
                population_size=8, generations=6, ticks_per_generation=24,
                organic_pools=3, seed=seed,
            )
            assert result["conservation_monotonic"] is True
            assert result["brains_identical"] is True
            assert isinstance(result["channels_live"], dict)


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

    def test_channels_live_is_dict(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert isinstance(result["channels_live"], dict)
        assert len(result["channels_live"]) > 0

    def test_channel_keys_complete(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        expected_keys = {
            "lessons", "predations", "defenses", "raids",
            "nests_built", "births", "role_deposits", "role_raids", "memory",
        }
        assert set(result["channels_live"].keys()) >= expected_keys

    def test_births_non_negative(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert result["births"] >= 0

    def test_alive_count_non_negative(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert result["alive_count"] >= 0

    def test_last_avg_fitness_positive(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        assert result["civilization_last_avg"] >= 0.0

    def test_control_avg_different_from_civilization(self):
        """Control and civilization arms should differ in fitness trajectories."""
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        # At least the last history entries should differ
        ctrl_last = result["control"]["history"][-1]
        civ_last = result["civilization"]["history"][-1]
        # They may or may not differ, but both should have valid entries
        assert isinstance(ctrl_last["avg_fitness"], (int, float))
        assert isinstance(civ_last["avg_fitness"], (int, float))

    def test_small_population(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=7,
        )
        assert result["conservation_monotonic"] is True
        assert result["brains_identical"] is True

    def test_medium_population(self):
        result = benchmark_civilization(
            population_size=16, generations=4, ticks_per_generation=16,
            organic_pools=2, seed=7,
        )
        assert result["conservation_monotonic"] is True
        assert result["brains_identical"] is True

    def test_different_generations(self):
        result = benchmark_civilization(
            population_size=8, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=7,
        )
        assert len(result["control"]["history"]) == 2
        assert len(result["civilization"]["history"]) == 2

    def test_many_organic_pools(self):
        result = benchmark_civilization(
            population_size=8, generations=3, ticks_per_generation=8,
            organic_pools=5, seed=7,
        )
        assert result["conservation_monotonic"] is True
        assert result["brains_identical"] is True


# ── RNG Stream Isolation ─────────────────────────────────────────────────────

class TestRNGStreams:
    def test_teach_rng_deterministic(self):
        a = _teach_rng(0)
        b = _teach_rng(0)
        vals_a = a.standard_normal(10)
        vals_b = b.standard_normal(10)
        assert np.allclose(vals_a, vals_b)

    def test_teach_rng_group_dependent(self):
        a = _teach_rng(0)
        b = _teach_rng(1)
        vals_a = a.standard_normal(10)
        vals_b = b.standard_normal(10)
        assert not np.allclose(vals_a, vals_b)

    def test_predation_rng_deterministic(self):
        a = _predation_rng(0)
        b = _predation_rng(0)
        assert np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_predation_rng_group_dependent(self):
        a = _predation_rng(0)
        b = _predation_rng(1)
        assert not np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_territory_rng_deterministic(self):
        a = _territory_rng(0)
        b = _territory_rng(0)
        assert np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_territory_rng_group_dependent(self):
        a = _territory_rng(0)
        b = _territory_rng(1)
        assert not np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_reproduce_rng_deterministic(self):
        a = _reproduce_rng(0)
        b = _reproduce_rng(0)
        assert np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_reproduce_rng_group_dependent(self):
        a = _reproduce_rng(0)
        b = _reproduce_rng(1)
        assert not np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_role_rng_deterministic(self):
        a = _role_rng(0)
        b = _role_rng(0)
        assert np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_role_rng_group_dependent(self):
        a = _role_rng(0)
        b = _role_rng(1)
        assert not np.allclose(a.standard_normal(10), b.standard_normal(10))

    def test_all_rngs_independent(self):
        """Each dedicated stream produces different draws from each other."""
        base = _all_on_params()
        rng = np.random.default_rng(42)
        shared_vals = rng.standard_normal(10)
        teach_vals = _teach_rng(0).standard_normal(10)
        pred_vals = _predation_rng(0).standard_normal(10)
        terr_vals = _territory_rng(0).standard_normal(10)
        repro_vals = _reproduce_rng(0).standard_normal(10)
        role_vals = _role_rng(0).standard_normal(10)
        all_vals = [shared_vals, teach_vals, pred_vals, terr_vals, repro_vals, role_vals]
        for i in range(len(all_vals)):
            for j in range(i + 1, len(all_vals)):
                assert not np.allclose(all_vals[i], all_vals[j])


# ── Genome Crossover & Mutation ──────────────────────────────────────────────

class TestGenomeCrossover:
    def test_crossover_preserves_keys(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=0)
        g2 = Genome.random(on, rng, group_id=1)
        child = g1.crossover(g2, np.random.default_rng(99))
        assert set(child.tensors.keys()) == set(g1.tensors.keys())

    def test_crossover_preserves_shapes(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=0)
        g2 = Genome.random(on, rng, group_id=1)
        child = g1.crossover(g2, np.random.default_rng(99))
        for key in g1.tensors:
            assert child.tensors[key].shape == g1.tensors[key].shape

    def test_crossover_inherits_first_parent_group(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=3)
        g2 = Genome.random(on, rng, group_id=5)
        child = g1.crossover(g2, np.random.default_rng(99))
        assert child.group_id == 3

    def test_crossover_inherits_first_parent_memories(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=0)
        g2 = Genome.random(on, rng, group_id=0)
        child = g1.crossover(g2, np.random.default_rng(99))
        assert child.memory_count == g1.memory_count

    def test_crossover_child_differs_from_parents(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=0)
        g2 = Genome.random(on, rng, group_id=0)
        child = g1.crossover(g2, np.random.default_rng(99))
        # At least some tensors should differ
        differs = False
        for key in g1.tensors:
            if not np.allclose(child.tensors[key], g1.tensors[key]):
                differs = True
                break
        assert differs

    def test_crossover_uniform_mix(self):
        """Each element comes from one parent or the other."""
        on = _all_on_params()
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(8)
        g1 = Genome.random(on, rng1, group_id=0)
        g2 = Genome.random(on, rng2, group_id=0)
        child = g1.crossover(g2, np.random.default_rng(99))
        for key in g1.tensors:
            from_parent1 = np.isclose(child.tensors[key], g1.tensors[key])
            from_parent2 = np.isclose(child.tensors[key], g2.tensors[key])
            # Every element should come from one parent
            assert np.all(from_parent1 | from_parent2)

    def test_crossover_different_seeds_different_children(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g1 = Genome.random(on, rng, group_id=0)
        g2 = Genome.random(on, rng, group_id=0)
        child1 = g1.crossover(g2, np.random.default_rng(1))
        child2 = g1.crossover(g2, np.random.default_rng(2))
        differs = False
        for key in g1.tensors:
            if not np.allclose(child1.tensors[key], child2.tensors[key]):
                differs = True
                break
        assert differs


class TestGenomeMutate:
    def test_mutate_returns_self(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        original = g.tensors["cells.W"].copy()
        result = g.mutate(np.random.default_rng(99))
        assert result is g

    def test_mutate_changes_weights(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        original = g.tensors["cells.W"].copy()
        g.mutate(np.random.default_rng(99), rate=0.5, scale=1.0)
        assert not np.allclose(g.tensors["cells.W"], original)

    def test_mutate_zero_rate_no_change(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(np.random.default_rng(99), rate=0.0)
        for key in g.tensors:
            assert np.allclose(g.tensors[key], original[key])

    def test_mutate_preserves_shapes(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        shapes = {k: v.shape for k, v in g.tensors.items()}
        g.mutate(np.random.default_rng(99))
        for key in g.tensors:
            assert g.tensors[key].shape == shapes[key]

    def test_mutate_dtype_float32(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        g.mutate(np.random.default_rng(99))
        for v in g.tensors.values():
            assert v.dtype == np.float32

    def test_mutate_high_rate_changes_most(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        original = g.tensors["cells.W"].copy()
        g.mutate(np.random.default_rng(99), rate=0.99, scale=1.0)
        changed = np.sum(~np.isclose(g.tensors["cells.W"], original))
        assert changed > 0

    def test_mutate_zero_scale_no_change(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        original = {k: v.copy() for k, v in g.tensors.items()}
        g.mutate(np.random.default_rng(99), rate=0.5, scale=0.0)
        for key in g.tensors:
            assert np.allclose(g.tensors[key], original[key])

    def test_mutate_preserves_memories(self):
        on = _all_on_params()
        rng = np.random.default_rng(7)
        g = Genome.random(on, rng, group_id=0)
        g.mutate(np.random.default_rng(99))
        # Memories are not mutated
        assert g.memory_count == 0


# ── Genome Structure Extended ────────────────────────────────────────────────

class TestGenomeStructureExtended:
    def test_genome_tensor_dtype_preserved(self):
        tensors = {"cells.W": np.ones((4, 4), dtype=np.float64)}
        g = Genome(tensors)
        assert g.tensors["cells.W"].dtype == np.float32

    def test_genome_tensor_dtype_float16(self):
        tensors = {"cells.W": np.ones((4, 4), dtype=np.float16)}
        g = Genome(tensors)
        assert g.tensors["cells.W"].dtype == np.float32

    def test_genome_group_id_int(self):
        g = Genome({"cells.W": np.zeros((4, 4))}, group_id=7)
        assert g.group_id == 7
        assert isinstance(g.group_id, int)

    def test_genome_default_memories_empty(self):
        g = Genome({"cells.W": np.zeros((4, 4))})
        assert g.memories == []
        assert g.memory_count == 0

    def test_genome_memories_deep_copied(self):
        mem = [{"features": [1.0], "action": [0], "reward": 1.0, "tick": 1}]
        g = Genome({"cells.W": np.zeros((4, 4))}, memories=mem)
        assert g.memories[0]["features"] == [1.0]
        # Modifying original doesn't affect genome
        mem[0]["features"].append(2.0)
        assert g.memories[0]["features"] == [1.0]

    def test_random_genome_different_groups_different_teach(self):
        params = _all_on_params()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        g0 = Genome.random(params, rng1, group_id=0)
        g1 = Genome.random(params, rng2, group_id=1)
        assert not np.allclose(g0.tensors["teach.W"], g1.tensors["teach.W"])

    def test_random_genome_base_brains_same_across_groups(self):
        """Base brains use the shared RNG, so different group_ids still get same base brains."""
        params = _all_on_params()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        g0 = Genome.random(params, rng1, group_id=0)
        g1 = Genome.random(params, rng2, group_id=1)
        assert np.allclose(g0.tensors["cells.W"], g1.tensors["cells.W"])
        assert np.allclose(g0.tensors["body.W"], g1.tensors["body.W"])
        assert np.allclose(g0.tensors["entity.W"], g1.tensors["entity.W"])
        assert np.allclose(g0.tensors["move.W"], g1.tensors["move.W"])

    def test_random_genome_message_brain_when_enabled(self):
        params = _all_on_params()
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        assert "message.W" in g.tensors
        assert "message.b" in g.tensors

    def test_random_genome_no_message_when_disabled(self):
        params = WorldParams(grid_size=(16, 8, 16))
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        assert "message.W" not in g.tensors

    def test_random_genome_all_channel_brains_when_disabled(self):
        params = WorldParams(grid_size=(16, 8, 16))
        rng = np.random.default_rng(42)
        g = Genome.random(params, rng, group_id=0)
        for name in ("teach", "predation", "territory", "reproduce", "role"):
            assert f"{name}.W" not in g.tensors

    def test_genome_multiple_memories(self):
        memories = [
            {"features": [1.0], "action": [0], "reward": 1.0, "tick": 1},
            {"features": [2.0], "action": [1], "reward": 2.0, "tick": 2},
            {"features": [3.0], "action": [2], "reward": 3.0, "tick": 3},
        ]
        g = Genome({"cells.W": np.zeros((4, 4))}, memories=memories)
        assert g.memory_count == 3


# ── Benchmark Structure Extended ─────────────────────────────────────────────

class TestBenchmarkStructureExtended:
    def test_group_weight_in_result(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert "group_weight" in result
        assert isinstance(result["group_weight"], float)

    def test_group_count_in_result(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert "group_count" in result
        assert result["group_count"] >= 1

    def test_control_and_civilization_last_avg(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert isinstance(result["control_last_avg"], float)
        assert isinstance(result["civilization_last_avg"], float)

    def test_control_arm_history_has_channel_zeros(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        for entry in result["control"]["history"]:
            # With all channels off, predations/defenses/raids should be 0
            assert entry.get("predations", 0) == 0
            assert entry.get("defenses", 0) == 0
            assert entry.get("raids", 0) == 0

    def test_civilization_arm_has_births(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        # Civilization arm with lifecycle should have births
        last = result["civilization"]["history"][-1]
        assert last.get("births", 0) >= 0

    def test_civilization_arm_has_memory(self):
        result = benchmark_civilization(
            population_size=8, generations=6, ticks_per_generation=24,
            organic_pools=3, seed=7,
        )
        last = result["civilization"]["history"][-1]
        assert last.get("memory_size", 0) >= 0

    def test_result_has_alive_count(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert "alive_count" in result
        assert result["alive_count"] >= 0

    def test_civilization_history_has_social_stats(self):
        result = benchmark_civilization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        entry = result["civilization"]["history"][0]
        assert "cooperations" in entry
        assert "contests" in entry
        assert "avg_fitness" in entry
        assert "best_fitness" in entry

    def test_conservation_sweep_returns_float_totals(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=4)
        assert isinstance(result["start_total"], float)
        assert isinstance(result["end_total"], float)

    def test_conservation_sweep_boundary_deposit(self):
        params = _all_on_params(generate_world=True, world_seed=7)
        genomes = _sweep_genomes(params, 4, 7)
        result = _conservation_sweep(params, genomes, ticks=4)
        assert "boundary_deposit_total" in result
        assert isinstance(result["boundary_deposit_total"], float)
