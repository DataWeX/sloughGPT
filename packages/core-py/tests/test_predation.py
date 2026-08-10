"""
Tests for Stage 8 predator-prey dynamics.

Predation is an opt-in channel (``predation_enabled``): when on, each baby
gains a ``perceptron_predation`` (entity-input -> 1 gate) that decides
whether to hunt the weakest nearby baby within range. A strike is lethal,
transfers the prey's full energy to the predator (conservation-safe — a
transfer, not creation), and costs ``predation_cost``. When off, no brain
exists and no RNG is drawn, so the locked selection proofs keep their exact
genome layout and energy flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    EvolutionEngine,
    Genome,
    benchmark_predation,
)
from domains.shell.simulation import (
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)


def _quiet_baby(params: WorldParams, energy: float, position,
                group_id: int = 0) -> SimBaby:
    """A baby whose every decision gate is pinned off except predation."""
    b = SimBaby(initial_energy=energy,
                position=np.array(position, dtype=np.float64),
                params=params, group_id=group_id)
    b.perceptron_cells.W[:] = 0.0
    b.perceptron_cells.b[:] = -10.0
    b.perceptron_body.W[:] = 0.0
    b.perceptron_body.b[:] = -10.0
    b.perceptron_move.W[:] = 0.0
    b.perceptron_move.b[:] = 0.0
    b.perceptron_entity.W[:] = 0.0
    b.perceptron_entity.b[:] = -10.0
    if b.perceptron_message is not None:
        b.perceptron_message.W[:] = 0.0
        b.perceptron_message.b[:] = -10.0
    if b.perceptron_teach is not None:
        b.perceptron_teach.W[:] = 0.0
        b.perceptron_teach.b[:] = -10.0
    if b.perceptron_predation is not None:
        b.perceptron_predation.W[:] = 0.0
        b.perceptron_predation.b[:] = -10.0
    return b


def _zero_predator(params: WorldParams, energy: float = 200.0,
                   position=(4.0, 1.0, 4.0)) -> SimBaby:
    """A predator whose predation gate stays at its zero init (sigmoid(0))."""
    b = SimBaby(initial_energy=energy,
                position=np.array(position, dtype=np.float64),
                params=params)
    b.perceptron_cells.W[:] = 0.0
    b.perceptron_cells.b[:] = -10.0
    b.perceptron_body.W[:] = 0.0
    b.perceptron_body.b[:] = -10.0
    b.perceptron_move.W[:] = 0.0
    b.perceptron_move.b[:] = 0.0
    b.perceptron_entity.W[:] = 0.0
    b.perceptron_entity.b[:] = -10.0
    if b.perceptron_message is not None:
        b.perceptron_message.W[:] = 0.0
        b.perceptron_message.b[:] = -10.0
    if b.perceptron_teach is not None:
        b.perceptron_teach.W[:] = 0.0
        b.perceptron_teach.b[:] = -10.0
    return b


def _params(**kw) -> WorldParams:
    base = dict(grid_size=(8, 4, 8), predation_enabled=True,
                social_enabled=False, message_enabled=False,
                teaching_enabled=False)
    base.update(kw)
    return WorldParams(**base)


def _open_predator(params: WorldParams, energy: float = 200.0,
                   position=(4.0, 1.0, 4.0)) -> SimBaby:
    """A predator with its predation gate forced open (sigmoid(10) ~ 1)."""
    b = _quiet_baby(params, energy, position)
    assert b.perceptron_predation is not None
    b.perceptron_predation.W[:] = 0.0
    b.perceptron_predation.b[:] = 10.0
    return b


class TestPredationBrain:
    def test_off_by_default_keeps_locked_proofs(self):
        # The channel is opt-in: default params create no predation brain, so
        # the benchmark genomes (and their RNG streams) are untouched.
        assert WorldParams().predation_enabled is False
        b = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        assert b.perceptron_predation is None
        on = SimBaby(params=_params())
        assert on.perceptron_predation is not None
        assert on.perceptron_predation.W.shape == (
            _params().entity_input_dim, 1,
        )

    def test_gate_below_threshold_does_not_hunt(self):
        params = _params()
        a = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        assert a.decide_predation(b) == 0.0
        assert a._last_predation_out is not None
        assert a._last_predation_out < params.predation_gate_threshold
        assert b.alive and b.energy == 20.0

    def test_zero_init_gate_hits_threshold(self):
        # sigmoid(0) == 0.5 == the default gate threshold, so a directly
        # constructed (non-genome) baby with predation enabled strikes the
        # neediest weaker neighbor — the gate is on the boundary.
        params = _params()
        a = _zero_predator(params, 200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        gate = a.decide_predation(b)
        assert gate == pytest.approx(0.5)
        assert gate >= params.predation_gate_threshold

    def test_open_gate_emits_full_strength(self):
        params = _params()
        a = _open_predator(params)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        gate = a.decide_predation(b)
        assert gate == pytest.approx(1.0 / (1.0 + np.exp(-10.0)))
        assert gate > params.predation_gate_threshold

    def test_hunt_transfers_energy_and_kills(self):
        params = _params()
        a = _open_predator(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        assert a.decide_predation(b) > 0.0
        gained = a.hunt(b)
        assert gained == 20.0
        assert a.energy == 220.0
        assert b.energy == 0.0
        assert not b.alive

    def test_hunt_is_conservation_safe(self):
        # The strike is a pure transfer: predator + prey energy is unchanged.
        params = _params()
        a = _open_predator(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        before = a.energy + b.energy
        a.hunt(b)
        assert a.energy + b.energy == pytest.approx(before)

    def test_hunt_noop_on_dead_prey(self):
        params = _params()
        a = _open_predator(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        a.hunt(b)
        assert a.energy == 220.0
        assert a.hunt(b) == 0.0
        assert a.energy == 220.0

    def test_hunt_noop_without_brain(self):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=200.0, params=params)
        b = SimBaby(initial_energy=20.0, params=params)
        assert a.perceptron_predation is None
        assert a.hunt(b) == 0.0
        assert b.alive


class TestPredationScene:
    def test_off_by_default_runs_no_predation(self):
        params = WorldParams(grid_size=(8, 4, 8))
        scene = SimScene(params=params)
        for pos, e in [((4.0, 1.0, 4.0), 200.0), ((4.0, 1.0, 5.0), 20.0)]:
            scene.add_baby(_quiet_baby(params, e, pos))
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        assert scene.params.predation_enabled is False
        assert sim.summary()["predations"] == 0
        assert sim.summary()["predation_energy_moved"] == 0.0
        assert len(scene.alive_babies) == 2

    def test_on_channel_strikes_neediest_weaker_prey(self):
        params = _params(predation_range=3.0)
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        weak = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        weaker = _quiet_baby(params, 10.0, (4.0, 1.0, 5.5))
        for b in (predator, weak, weaker):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        # The weakest nearby prey (10 energy) is eaten first.
        assert not weaker.alive
        assert weak.alive
        assert predator.alive
        assert predator.energy == pytest.approx(
            200.0 + 10.0 - params.predation_cost - params.see_cost
            - params.passive_drain,
            abs=1e-6,
        )

    def test_strike_requires_strictly_weaker_prey(self):
        params = _params()
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        richer = _quiet_baby(params, 300.0, (4.0, 1.0, 5.0))
        for b in (predator, richer):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        # The richer neighbor is not prey — nothing is hunted.
        assert sim.summary()["predations"] == 0
        assert len(scene.alive_babies) == 2

    def test_range_limits_strikes(self):
        params = _params(predation_range=1.0)
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        far = _quiet_baby(params, 20.0, (7.0, 3.0, 7.0))
        for b in (predator, far):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert sim.summary()["predations"] == 0
        assert len(scene.alive_babies) == 2

    def test_prey_removed_from_scene_next_step(self):
        params = _params()
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        prey = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        for b in (predator, prey):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert [b.entity.id for b in scene.alive_babies] == [predator.entity.id]
        assert sim.summary()["predations"] == 1

    def test_summary_totals_match_tick_log(self):
        params = _params()
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        prey = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        for b in (predator, prey):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        strikes = [r for r in sim._tick_log
                   if r.get("predation_energy", 0.0) > 0.0]
        assert sim.summary()["predations"] == len(strikes) == 1
        assert sim.summary()["predation_energy_moved"] == pytest.approx(
            sum(r["predation_energy"] for r in strikes),
        )
        row = strikes[0]
        assert row["prey_id"] == prey.entity.id
        assert row["predation_amplitude"] > 0.0

    def test_serialization_roundtrip_preserves_predation_brain(self):
        params = _params()
        predator = _open_predator(params, energy=200.0)
        prey = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        predator.decide_predation(prey)
        w_before = predator.perceptron_predation.W.copy()
        b_before = predator.perceptron_predation.b.copy()
        data = predator.to_dict()
        restored = SimBaby.from_dict(data, params=params)
        assert restored.perceptron_predation is not None
        assert np.allclose(restored.perceptron_predation.W, w_before)
        assert np.allclose(restored.perceptron_predation.b, b_before)

    def test_delta_rule_shapes_predation_gate_from_hunt(self):
        params = _params()
        scene = SimScene(params=params)
        predator = _open_predator(params, energy=200.0)
        prey = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
        for b in (predator, prey):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        # The successful hunt's reward flows through learn() and moves the
        # predation weights off their initialized zero vector.
        assert predator.perceptron_predation is not None
        assert not np.allclose(predator.perceptron_predation.W, 0.0)

    def test_determinism_with_predation_on(self):
        params = _params()
        outs = []
        for _ in range(2):
            np.random.seed(7)
            scene = SimScene(params=params)
            predator = _open_predator(params, energy=200.0)
            prey = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0))
            for b in (predator, prey):
                scene.add_baby(b)
            sim = Simulation(scene, max_ticks=2)
            sim.run()
            outs.append((sim.summary()["predations"],
                         sim.summary()["predation_energy_moved"]))
        assert outs[0] == outs[1]


class TestPredationEvolution:
    @staticmethod
    def _params(**kw) -> WorldParams:
        base = dict(grid_size=(16, 8, 16), predation_enabled=True,
                    teaching_enabled=False, memory_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    def test_genome_roundtrip(self):
        params = self._params()
        rng = np.random.default_rng(1)
        g = Genome.random(params, rng, group_id=2)
        assert "predation.W" in g.tensors
        assert g.tensors["predation.W"].shape == (params.entity_input_dim, 1)
        b = SimBaby(params=params)
        g.apply_to(b)
        assert b.perceptron_predation is not None
        assert np.allclose(b.perceptron_predation.W, g.tensors["predation.W"])

    def test_dedicated_stream_keeps_shared_draws_identical(self):
        # Locked-proof invariant: enabling predation must not perturb the four
        # behavior brains' RNG draws (or the perception-noise stream).
        off = WorldParams(grid_size=(16, 8, 16))
        on = self._params()
        g_off = Genome.random(off, np.random.default_rng(9), group_id=0)
        g_on = Genome.random(on, np.random.default_rng(9), group_id=0)
        for name in ("cells", "body", "entity", "move"):
            assert np.allclose(g_off.tensors[f"{name}.W"],
                               g_on.tensors[f"{name}.W"])
            assert np.allclose(g_off.tensors[f"{name}.b"],
                               g_on.tensors[f"{name}.b"])
        assert "predation.W" in g_on.tensors
        assert "predation.W" not in g_off.tensors

    def test_run_history_carries_predation_fields(self):
        eng = EvolutionEngine(
            params=self._params(),
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert "predations" in result["history"][0]
        assert "predation_rate" in result["history"][0]
        assert "predation_energy_moved" in result["history"][0]
        assert result["history"][0]["predation_rate"] >= 0.0

    def test_run_off_default_has_no_predation_brains(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert result["history"][0]["predations"] == 0

    def test_benchmark_structure_and_verdict_keys(self):
        result = benchmark_predation(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert set(result) >= {
            "control", "predation", "group_count", "group_weight",
            "control_last_avg", "predation_last_avg", "predation_rate",
            "predations", "predation_energy_moved", "predation_emerged",
        }
        assert len(result["control"]["history"]) == 3
        assert len(result["predation"]["history"]) == 3
        assert result["predations"] == result["predation"]["history"][-1]["predations"]

    def test_benchmark_deterministic(self):
        a = benchmark_predation(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        b = benchmark_predation(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        assert a["predation_last_avg"] == b["predation_last_avg"]
        assert a["predations"] == b["predations"]
        assert a["predation_emerged"] == b["predation_emerged"]
