"""
Tests for Stage 10 in-world life cycle (births and deaths inside the tick).

Lifecycle is an opt-in channel (``lifecycle_enabled``): when on, each baby
gains a ``perceptron_reproduce`` (body-input -> 1 gate) that decides whether
it is ready to breed. A baby whose gate clears while it stands above
``reproduce_energy_threshold`` spawns an offspring near itself: the child's
starting energy is ``birth_cost`` — up to ``birth_nest_fraction`` of it drawn
from the tribe's nearest nest bank (the tribe funds the child's start), the
rest from the parent — a pure transfer, never creation, so the world
conserves energy. The child inherits the parent's learned behavior weights
and best episodes (an asexual offspring), joins the parent's tribe, and is
seeded from the world reservoir like any newborn. Starvation still removes
babies every tick, so births and deaths now happen INSIDE the tick loop and a
scene's population can self-sustain without the evolution engine re-seeding
it. Population is bounded by ``max_entities`` and, ultimately, by the world's
conserved energy budget. When off, no brain exists and no RNG is drawn, so
the locked selection proofs keep their exact genome layout and energy flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    EvolutionEngine,
    Genome,
    benchmark_lifecycle,
)
from domains.shell.memory import WorldMemory
from domains.shell.simulation import (
    Nest,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)


def _quiet_baby(params: WorldParams, energy: float, position,
                group_id: int = 0) -> SimBaby:
    """A baby whose every decision gate is pinned off."""
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
    if b.perceptron_territory is not None:
        b.perceptron_territory.W[:] = 0.0
        b.perceptron_territory.b[:] = -10.0
    if b.perceptron_reproduce is not None:
        b.perceptron_reproduce.W[:] = 0.0
        b.perceptron_reproduce.b[:] = -10.0
    return b


def _zero_reproducer(params: WorldParams, energy: float = 200.0,
                     position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A parent whose reproduce gate stays at its zero init (sigmoid(0))."""
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
    if b.perceptron_territory is not None:
        b.perceptron_territory.W[:] = 0.0
        b.perceptron_territory.b[:] = -10.0
    return b


def _open_reproducer(params: WorldParams, energy: float = 200.0,
                     position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A parent with its reproduce gate forced open (sigmoid(10) ~ 1)."""
    b = _quiet_baby(params, energy, position, group_id=group_id)
    assert b.perceptron_reproduce is not None
    b.perceptron_reproduce.W[:] = 0.0
    b.perceptron_reproduce.b[:] = 10.0
    return b


def _params(**kw) -> WorldParams:
    base = dict(grid_size=(8, 4, 8), lifecycle_enabled=True,
                social_enabled=False, message_enabled=False,
                teaching_enabled=False, predation_enabled=False)
    base.update(kw)
    return WorldParams(**base)


def _nest(position, group_id: int = 0, stored_energy: float = 100.0) -> Nest:
    return Nest(id=1, position=np.array(position, dtype=np.float64),
                stored_energy=stored_energy, owner_group_id=group_id)


class TestReproduceBrain:
    def test_off_by_default_keeps_locked_proofs(self):
        # The channel is opt-in: default params create no reproduce brain, so
        # the benchmark genomes (and their RNG streams) are untouched.
        assert WorldParams().lifecycle_enabled is False
        b = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        assert b.perceptron_reproduce is None
        on = SimBaby(params=_params())
        assert on.perceptron_reproduce is not None
        assert on.perceptron_reproduce.W.shape == (
            _params().body_input_dim, 1,
        )

    def test_gate_below_threshold_does_not_breed(self):
        params = _params()
        a = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        assert a.decide_reproduce() == 0.0
        assert a._last_reproduce_out is not None
        assert a._last_reproduce_out < params.reproduce_gate_threshold

    def test_zero_init_gate_hits_threshold(self):
        # sigmoid(0) == 0.5 == the default gate threshold, so a directly
        # constructed (non-genome) baby with lifecycle enabled is on the
        # boundary and breeds once it stands above the energy threshold.
        params = _params()
        a = _zero_reproducer(params, 200.0)
        gate = a.decide_reproduce()
        assert gate == pytest.approx(0.5)
        assert gate >= params.reproduce_gate_threshold

    def test_open_gate_emits_full_strength(self):
        params = _params()
        a = _open_reproducer(params)
        gate = a.decide_reproduce()
        assert gate == pytest.approx(1.0 / (1.0 + np.exp(-10.0)))
        assert gate > params.reproduce_gate_threshold

    def test_body_input_reads_energy_and_position(self):
        params = _params()
        a = _zero_reproducer(params, 200.0, (4.0, 1.0, 4.0))
        a.decide_reproduce()
        body = a._last_reproduce_input
        assert body is not None
        assert body.shape == (params.body_input_dim,)
        assert body[0] == pytest.approx(200.0 / params.start_energy)
        assert body[1] == pytest.approx(4.0 / params.grid_size[0])
        assert body[2] == pytest.approx(1.0 / params.grid_size[1])

    def test_no_brain_returns_zero(self):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=200.0, params=params)
        assert a.perceptron_reproduce is None
        assert a.decide_reproduce() == 0.0


class TestBirthScene:
    def test_off_by_default_runs_no_births(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=True)
        scene = SimScene(params=params)
        parent = _open_reproducer(_params(), 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        assert scene.params.lifecycle_enabled is False
        assert scene.birth(parent) == (None, 0.0)
        assert len(scene.alive_babies) == 1
        assert scene.births == 0

    def test_birth_requires_energy_above_threshold(self):
        # The gate alone is not enough — the simulation loop also demands
        # genuine surplus (energy above reproduce_energy_threshold). The
        # scene method itself does not re-check the energy gate; the tick
        # loop gates on it. A birth attempted with a broke parent aborts.
        params = _params()
        scene = SimScene(params=params)
        parent = _quiet_baby(params, 5.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        assert parent.energy < params.birth_cost
        assert scene.birth(parent) == (None, 0.0)
        assert len(scene.alive_babies) == 1

    def test_birth_transfers_energy_conservation_safe(self):
        # A birth is a transfer, never creation: the child's birth_cost comes
        # out of the nest bank (birth_nest_fraction) and the parent's surplus.
        params = _params(structure_enabled=True)
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0, stored_energy=100.0)
        scene.nests.append(nest)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        before = parent.energy + nest.stored_energy
        child_id, moved = scene.birth(parent)
        assert moved == pytest.approx(params.birth_cost)
        assert child_id is not None
        # Nest funds half the birth cost, the parent funds the rest.
        assert nest.stored_energy == pytest.approx(
            100.0 - params.birth_cost * params.birth_nest_fraction)
        assert parent.energy == pytest.approx(
            200.0 - params.birth_cost * (1.0 - params.birth_nest_fraction))
        assert parent.energy + nest.stored_energy + params.birth_cost == pytest.approx(before)
        child = next(b for b in scene.babies if b.entity.id == child_id)
        assert child.energy == pytest.approx(params.birth_cost)
        assert scene.births == 1

    def test_birth_aborts_and_refunds_when_parent_cannot_fund(self):
        # If the parent cannot cover its share, the birth aborts and the nest
        # draw is refunded — breeding requires genuine surplus.
        params = _params(structure_enabled=True)
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0, stored_energy=100.0)
        scene.nests.append(nest)
        parent = _open_reproducer(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        assert parent.energy < params.birth_cost * (1.0 - params.birth_nest_fraction)
        child_id, moved = scene.birth(parent)
        assert (child_id, moved) == (None, 0.0)
        assert nest.stored_energy == pytest.approx(100.0)
        assert parent.energy == pytest.approx(20.0)
        assert len(scene.alive_babies) == 1

    def test_birth_respects_max_entities_cap(self):
        params = _params(max_entities=1)
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        assert scene.birth(parent) == (None, 0.0)
        assert len(scene.alive_babies) == 1

    def test_child_placed_within_birth_range_and_same_tribe(self):
        params = _params()
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=2)
        scene.add_baby(parent)
        child_id, _ = scene.birth(parent)
        child = next(b for b in scene.babies if b.entity.id == child_id)
        dx = abs(child.position[0] - parent.position[0])
        dz = abs(child.position[2] - parent.position[2])
        assert min(dx, params.grid_size[0] - dx) <= params.birth_range
        assert min(dz, params.grid_size[2] - dz) <= params.birth_range
        assert child.group_id == parent.group_id
        assert child.position[1] == pytest.approx(parent.position[1])

    def test_child_inherits_parent_weights_and_memotype(self):
        params = _params()
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        # Shape the parent's cells brain away from init so inheritance is visible.
        parent.perceptron_cells.W[:] = 0.25
        parent.perceptron_cells.b[:] = 0.5
        child_id, _ = scene.birth(parent)
        child = next(b for b in scene.babies if b.entity.id == child_id)
        assert np.allclose(child.perceptron_cells.W, parent.perceptron_cells.W)
        assert np.allclose(child.perceptron_cells.b, parent.perceptron_cells.b)
        assert child.perceptron_reproduce is not None
        assert np.allclose(child.perceptron_reproduce.W, parent.perceptron_reproduce.W)
        assert child.perceptron_reproduce.b[0] == pytest.approx(10.0)

    def test_child_seeded_from_world_reservoir(self):
        # add_baby seeds a newborn from the world memory when present, exactly
        # like any other newborn (see SimScene.add_baby): the child inherits
        # the reservoir's best episodes on top of its memotype, and
        # memory_seeds_given counts every newborn so seeded.
        params = _params(memory_enabled=True)
        world_memory = WorldMemory()
        for reward in (0.1, 0.5, 0.9):
            world_memory.record(np.array([1.0, 0.0], dtype=np.float32),
                                (0.0,), reward, tick=0, group_id=0, donor_id=0)
        scene = SimScene(params=params, world_memory=world_memory)
        parent = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        child_id, _ = scene.birth(parent)
        assert child_id is not None
        child = next(b for b in scene.babies if b.entity.id == child_id)
        parent_rewards = sorted([e.reward for e in parent.memory.recall(
            params.memory_seed, by_reward=True)], reverse=True)
        child_rewards = sorted([e.reward for e in child.memory.recall(
            100, by_reward=True)], reverse=True)
        assert parent_rewards == [0.9, 0.5, 0.1]
        # The child receives the reservoir's best episodes twice: consolidated
        # from the parent as memotype (spawn_child) and re-seeded at add_baby.
        assert child_rewards == [0.9, 0.9, 0.5, 0.5, 0.1, 0.1]
        assert scene.memory_seeds_given == 6  # parent + child, 3 each

    def test_no_nest_parent_funds_full_birth(self):
        # Without structures, the parent alone funds the whole birth cost —
        # still a transfer, never creation.
        params = _params(structure_enabled=False)
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        before = parent.energy
        child_id, moved = scene.birth(parent)
        assert moved == pytest.approx(params.birth_cost)
        assert parent.energy == pytest.approx(before - params.birth_cost)
        assert child_id is not None
        assert scene.births == 1


class TestLifecycleSimulation:
    def test_tick_loop_breeds_above_threshold(self):
        params = _params()
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        assert sim.summary()["births"] >= 1
        assert sim.summary()["birth_energy_moved"] >= params.birth_cost
        assert len(scene.babies) >= 2
        rows = [r for r in sim._tick_log if r.get("reproduced", False)]
        assert len(rows) == 1
        assert rows[0]["child_id"] is not None
        assert rows[0]["birth_energy"] == pytest.approx(params.birth_cost)

    def test_energy_gate_blocks_birth_below_threshold(self):
        # A rich gate but a starving parent: the loop checks the energy
        # threshold before attempting a birth.
        params = _params(reproduce_energy_threshold=300.0)
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert sim.summary()["births"] == 0
        assert len(scene.babies) == 1

    def test_starvation_deaths_remove_babies_every_tick(self):
        # With lifecycle on, starvation still removes dead babies every tick —
        # births and deaths share the same in-tick life cycle.
        params = _params()
        scene = SimScene(params=params)
        dying = _quiet_baby(params, 1.0, (1.0, 1.0, 1.0), group_id=0)
        scene.add_baby(dying)
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        assert sim.summary()["deaths"] >= 1
        # The dead baby leaves the living set (its device remains only as a
        # historical record, marked not alive).
        assert dying.entity.id not in [b.entity.id for b in scene.alive_babies]
        assert len(scene.alive_babies) == 0

    def test_result_row_carries_reproduction_fields(self):
        params = _params()
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        row = [r for r in sim._tick_log if r.get("reproduced", False)][0]
        assert {"reproduced", "birth_energy", "child_id"} <= set(row)
        assert row["reproduced"] is True
        assert row["birth_energy"] > 0.0

    def test_summary_counts_match_tick_log(self):
        params = _params()
        scene = SimScene(params=params)
        parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(parent)
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        births = [r for r in sim._tick_log if r.get("reproduced", False)]
        assert sim.summary()["births"] == len(births)
        assert sim.summary()["birth_energy_moved"] == pytest.approx(
            sum(r["birth_energy"] for r in births))

    def test_serialization_roundtrip_preserves_reproduce_brain(self):
        params = _params()
        parent = _open_reproducer(params, 200.0)
        parent.decide_reproduce()
        w_before = parent.perceptron_reproduce.W.copy()
        b_before = parent.perceptron_reproduce.b.copy()
        data = parent.to_dict()
        restored = SimBaby.from_dict(data, params=params)
        assert restored.perceptron_reproduce is not None
        assert np.allclose(restored.perceptron_reproduce.W, w_before)
        assert np.allclose(restored.perceptron_reproduce.b, b_before)

    def test_determinism_with_lifecycle_on(self):
        params = _params()
        outs = []
        for _ in range(2):
            np.random.seed(7)
            scene = SimScene(params=params)
            parent = _open_reproducer(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
            scene.add_baby(parent)
            sim = Simulation(scene, max_ticks=2)
            sim.run()
            outs.append((sim.summary()["births"],
                         sim.summary()["birth_energy_moved"]))
        assert outs[0] == outs[1]


class TestLifecycleEvolution:
    @staticmethod
    def _params(**kw) -> WorldParams:
        base = dict(grid_size=(16, 8, 16), lifecycle_enabled=True,
                    teaching_enabled=False, memory_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    def test_genome_roundtrip(self):
        params = self._params()
        rng = np.random.default_rng(1)
        g = Genome.random(params, rng, group_id=2)
        assert "reproduce.W" in g.tensors
        assert g.tensors["reproduce.W"].shape == (params.body_input_dim, 1)
        b = SimBaby(params=params)
        g.apply_to(b)
        assert b.perceptron_reproduce is not None
        assert np.allclose(b.perceptron_reproduce.W, g.tensors["reproduce.W"])

    def test_from_baby_extracts_reproduce_channel(self):
        params = self._params()
        b = SimBaby(params=params)
        g = Genome.from_baby(b, group_id=1)
        assert "reproduce.W" in g.tensors
        assert np.allclose(g.tensors["reproduce.W"], b.perceptron_reproduce.W)

    def test_dedicated_stream_keeps_shared_draws_identical(self):
        # Locked-proof invariant: enabling lifecycle must not perturb the four
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
        assert "reproduce.W" in g_on.tensors
        assert "reproduce.W" not in g_off.tensors

    def test_run_history_carries_lifecycle_fields(self):
        eng = EvolutionEngine(
            params=self._params(),
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        entry = result["history"][0]
        assert "births" in entry
        assert "birth_energy_moved" in entry
        assert "deaths" in entry
        assert "alive_count" in entry
        assert entry["births"] >= 0
        assert entry["birth_energy_moved"] >= 0.0
        assert entry["deaths"] >= 0

    def test_run_off_default_has_no_reproduce_brains(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert result["history"][0]["births"] == 0
        assert result["history"][0]["birth_energy_moved"] == 0.0

    def test_benchmark_structure_and_verdict_keys(self):
        result = benchmark_lifecycle(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert set(result) >= {
            "control", "lifecycle", "group_count", "group_weight",
            "control_last_avg", "lifecycle_last_avg", "births",
            "birth_energy_moved", "deaths", "alive_count",
            "population_size", "lifecycle_emerged",
        }
        assert len(result["control"]["history"]) == 3
        assert len(result["lifecycle"]["history"]) == 3
        assert result["births"] == result["lifecycle"]["history"][-1]["births"]
        assert result["deaths"] == result["lifecycle"]["history"][-1]["deaths"]
        assert result["alive_count"] == result["lifecycle"]["history"][-1]["alive_count"]

    def test_control_arm_never_breeds(self):
        result = benchmark_lifecycle(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        for entry in result["control"]["history"]:
            assert entry["births"] == 0
            assert entry["birth_energy_moved"] == 0.0

    def test_benchmark_deterministic(self):
        a = benchmark_lifecycle(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        b = benchmark_lifecycle(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        assert a["lifecycle_last_avg"] == b["lifecycle_last_avg"]
        assert a["births"] == b["births"]
        assert a["lifecycle_emerged"] == b["lifecycle_emerged"]
