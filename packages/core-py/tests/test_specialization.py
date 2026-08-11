"""
Tests for Stage 11 division of labor (specialization).

Specialization is an opt-in channel (``specialization_enabled``): when on,
each baby gains a ``perceptron_role`` (body-input -> 1 gate) whose value is
a heritable POSTURE. Below ``role_gate_threshold`` the baby is a BUILDER:
while it carries genuine surplus (above ``start_energy``) and stands within
``nest_use_radius`` of its own tribe's nearest nest, it banks
``role_deposit_fraction`` of that surplus into the bank — a deliberate
transfer that lifts the tribe's famine floor. At or above the threshold the
baby is a WARRIOR: standing within ``territory_radius`` of a FOREIGN
tribe's nearest nest it raids that bank even when not hungry, capped at
``role_raid_fraction`` of ``nest_draw_rate`` and the bank itself. Both acts
are pure transfers, never creation, so the world conserves energy, and both
land in the same tick's honest net reward (step 8b) so the posture is
selected for only where it pays. The role brain is constructed from fixed
zeros (no RNG draw) and its genome weights come from a DEDICATED role
stream, so enabling specialization leaves the four behavior brains'
draws — and the locked selection proofs — bit-identical. When off, no brain
exists, no RNG is drawn, and no role act can fire.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    EvolutionEngine,
    Genome,
    benchmark_specialization,
)
from domains.shell.simulation import (
    Nest,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)


def _baby(params: WorldParams, energy: float, position,
          group_id: int = 0, role_bias: float = -10.0) -> SimBaby:
    """A baby with all decision gates pinned off and a fixed role posture."""
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
    for name in ("message", "teach", "predation", "territory", "reproduce"):
        p = getattr(b, f"perceptron_{name}")
        if p is not None:
            p.W[:] = 0.0
            p.b[:] = -10.0
    assert b.perceptron_role is not None
    b.perceptron_role.W[:] = 0.0
    b.perceptron_role.b[:] = role_bias
    return b


def _builder(params: WorldParams, energy: float = 200.0,
             position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A baby whose role gate is pinned far below the threshold (Builder)."""
    return _baby(params, energy, position, group_id=group_id,
                 role_bias=-10.0)


def _warrior(params: WorldParams, energy: float = 200.0,
             position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A baby whose role gate is pinned far above the threshold (Warrior)."""
    return _baby(params, energy, position, group_id=group_id,
                 role_bias=10.0)


def _params(**kw) -> WorldParams:
    base = dict(grid_size=(8, 4, 8), specialization_enabled=True,
                structure_enabled=True, social_enabled=False,
                message_enabled=False, teaching_enabled=False,
                predation_enabled=False, territoriality_enabled=False,
                lifecycle_enabled=False)
    base.update(kw)
    return WorldParams(**base)


def _nest(position, group_id: int = 0, stored_energy: float = 100.0) -> Nest:
    return Nest(id=1, position=np.array(position, dtype=np.float64),
                stored_energy=stored_energy, owner_group_id=group_id)


class TestRoleBrain:
    def test_off_by_default_keeps_locked_proofs(self):
        # The channel is opt-in: default params create no role brain, so the
        # benchmark genomes (and their RNG streams) are untouched.
        assert WorldParams().specialization_enabled is False
        b = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        assert b.perceptron_role is None
        on = SimBaby(params=_params())
        assert on.perceptron_role is not None
        assert on.perceptron_role.W.shape == (
            _params().body_input_dim, 1,
        )

    def test_gate_below_threshold_is_builder(self):
        params = _params()
        a = _builder(params)
        gate = a.decide_role()
        assert gate < params.role_gate_threshold
        assert a._last_role_out is not None
        assert a._last_role_out < params.role_gate_threshold

    def test_open_gate_is_warrior(self):
        params = _params()
        a = _warrior(params)
        gate = a.decide_role()
        assert gate == pytest.approx(1.0 / (1.0 + np.exp(-10.0)))
        assert gate >= params.role_gate_threshold

    def test_zero_init_gate_hits_threshold(self):
        # sigmoid(0) == 0.5 == the default gate threshold, so a directly
        # constructed (non-genome) baby with specialization enabled sits on
        # the boundary and is classed a Warrior.
        params = _params()
        b = SimBaby(initial_energy=200.0, params=params)
        assert b.perceptron_role is not None
        gate = b.decide_role()
        assert gate == pytest.approx(0.5)
        assert gate >= params.role_gate_threshold

    def test_body_input_reads_energy_and_position(self):
        params = _params()
        a = _warrior(params, 200.0, (4.0, 1.0, 4.0))
        a.decide_role()
        body = a._last_role_input
        assert body is not None
        assert body.shape == (params.body_input_dim,)
        assert body[0] == pytest.approx(200.0 / params.start_energy)
        assert body[1] == pytest.approx(4.0 / params.grid_size[0])
        assert body[2] == pytest.approx(1.0 / params.grid_size[1])

    def test_no_brain_returns_zero(self):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=200.0, params=params)
        assert a.perceptron_role is None
        assert a.decide_role() == 0.0
        assert a._last_role_out is None


class TestDepositScene:
    def test_off_by_default_runs_no_deposit(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=True)
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0)
        scene.nests.append(nest)
        b = _builder(_params(), 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.params.specialization_enabled is False
        assert scene.deposit_nest(b) == 0.0
        assert nest.stored_energy == pytest.approx(100.0)

    def test_deposit_requires_surplus(self):
        # Below start_energy there is no surplus to bank — the act aborts
        # even with an own nest in range.
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0)
        scene.nests.append(nest)
        b = _builder(params, 90.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.deposit_nest(b) == 0.0
        assert nest.stored_energy == pytest.approx(100.0)
        assert b.energy == pytest.approx(90.0)

    def test_deposit_banks_fraction_of_surplus_conservation_safe(self):
        # The banked amount is role_deposit_fraction of the surplus, moved
        # out of the baby into the nest — a transfer, never creation.
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0, stored_energy=100.0)
        scene.nests.append(nest)
        b = _builder(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        before = b.energy + nest.stored_energy
        moved = scene.deposit_nest(b)
        assert moved == pytest.approx(
            (200.0 - params.start_energy) * params.role_deposit_fraction)
        assert nest.stored_energy == pytest.approx(110.0)
        assert b.energy == pytest.approx(190.0)
        assert b.energy + nest.stored_energy == pytest.approx(before)

    def test_deposit_requires_own_nest_in_range(self):
        # A builder banks only into its OWN tribe's nest and only when the
        # nest stands within nest_use_radius — a foreign nest in range is
        # never funded, and an own nest out of range is never reached.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1))
        scene.nests.append(_nest((7.0, 1.0, 7.0), group_id=0))
        b = _builder(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.deposit_nest(b) == 0.0
        assert all(n.stored_energy == pytest.approx(100.0)
                   for n in scene.nests)

    def test_deposit_keeps_start_energy_buffer(self):
        # The bank is capped at the surplus: a baby can never bank below its
        # start energy (its working buffer).
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=0, stored_energy=0.0)
        scene.nests.append(nest)
        b = _builder(params, 101.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        moved = scene.deposit_nest(b)
        assert moved == pytest.approx(1.0 * params.role_deposit_fraction)
        assert b.energy == pytest.approx(101.0 - moved)


class TestRaidScene:
    def test_off_by_default_runs_no_raid(self):
        params = WorldParams(grid_size=(8, 4, 8), territoriality_enabled=True)
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=1)
        scene.nests.append(nest)
        b = _warrior(_params(), 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.params.specialization_enabled is False
        assert scene.role_raid(b) == 0.0
        assert nest.stored_energy == pytest.approx(100.0)

    def test_raid_requires_foreign_nest_in_territory(self):
        # A warrior never raids its own tribe's bank, and a foreign nest
        # beyond territory_radius is out of reach.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        scene.nests.append(_nest((7.0, 1.0, 7.0), group_id=1))
        b = _warrior(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.role_raid(b) == 0.0
        assert all(n.stored_energy == pytest.approx(100.0)
                   for n in scene.nests)

    def test_raid_transfers_capped_by_rate_conservation_safe(self):
        # The steal is role_raid_fraction of the owner's draw rate, moved
        # from the foreign bank into the warrior — a transfer, never
        # creation, and it happens even though the warrior is not hungry.
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=1, stored_energy=100.0)
        scene.nests.append(nest)
        b = _warrior(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        before = b.energy + nest.stored_energy
        steal = scene.role_raid(b)
        assert steal == pytest.approx(
            params.nest_draw_rate * params.role_raid_fraction)
        assert nest.stored_energy == pytest.approx(100.0 - steal)
        assert b.energy == pytest.approx(200.0 + steal)
        assert b.energy + nest.stored_energy == pytest.approx(before)

    def test_raid_capped_by_bank(self):
        # A bank smaller than the rate cap is drained entirely — never more
        # than what exists.
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=1, stored_energy=0.2)
        scene.nests.append(nest)
        b = _warrior(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(b)
        assert scene.role_raid(b) == pytest.approx(0.2)
        assert nest.stored_energy == pytest.approx(0.0)


class TestSpecializationSimulation:
    def _scene_with_roles(self, params: WorldParams) -> SimScene:
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        scene.add_baby(_builder(params, 200.0, (4.0, 1.0, 4.0), group_id=0))
        scene.add_baby(_warrior(params, 200.0, (4.0, 1.0, 4.0), group_id=1))
        return scene

    def test_tick_loop_fields_both_acts(self):
        # One builder near its own nest and one warrior near a foreign nest:
        # in a single tick both role acts fire and the summary records them.
        params = _params()
        scene = self._scene_with_roles(params)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        assert sim.summary()["role_deposits"] >= 1
        assert sim.summary()["role_deposit_energy"] > 0.0
        assert sim.summary()["role_raids"] >= 1
        assert sim.summary()["role_raid_energy"] > 0.0

    def test_result_row_carries_role_fields(self):
        params = _params()
        scene = self._scene_with_roles(params)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        dep = [r for r in sim._tick_log if r.get("role_deposited", 0.0) > 0.0]
        raid = [r for r in sim._tick_log if r.get("role_raided", 0.0) > 0.0]
        assert len(dep) == 1
        assert len(raid) == 1
        assert {"role_deposited", "role_raided"} <= set(dep[0])
        assert dep[0]["role_deposited"] > 0.0
        assert raid[0]["role_raided"] > 0.0
        assert dep[0]["baby_id"] == scene.babies[0].entity.id
        assert raid[0]["baby_id"] == scene.babies[1].entity.id

    def test_summary_counts_match_tick_log(self):
        params = _params()
        scene = self._scene_with_roles(params)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        dep = [r for r in sim._tick_log if r.get("role_deposited", 0.0) > 0.0]
        raid = [r for r in sim._tick_log if r.get("role_raided", 0.0) > 0.0]
        assert sim.summary()["role_deposits"] == len(dep)
        assert sim.summary()["role_deposit_energy"] == pytest.approx(
            sum(r["role_deposited"] for r in dep))
        assert sim.summary()["role_raids"] == len(raid)
        assert sim.summary()["role_raid_energy"] == pytest.approx(
            sum(r["role_raided"] for r in raid))

    def test_serialization_roundtrip_preserves_role_brain(self):
        params = _params()
        b = _warrior(params, 200.0)
        b.decide_role()
        w_before = b.perceptron_role.W.copy()
        b_before = b.perceptron_role.b.copy()
        data = b.to_dict()
        restored = SimBaby.from_dict(data, params=params)
        assert restored.perceptron_role is not None
        assert np.allclose(restored.perceptron_role.W, w_before)
        assert np.allclose(restored.perceptron_role.b, b_before)
        assert restored._last_role_input is None
        assert restored._last_role_out is None

    def test_determinism_with_specialization_on(self):
        params = _params()
        outs = []
        for _ in range(2):
            np.random.seed(7)
            scene = self._scene_with_roles(params)
            sim = Simulation(scene, max_ticks=2)
            sim.run()
            s = sim.summary()
            outs.append((s["role_deposits"], s["role_deposit_energy"],
                         s["role_raids"], s["role_raid_energy"]))
        assert outs[0] == outs[1]


class TestSpecializationEvolution:
    @staticmethod
    def _params(**kw) -> WorldParams:
        base = dict(grid_size=(16, 8, 16), specialization_enabled=True,
                    teaching_enabled=False, memory_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    def test_genome_roundtrip(self):
        params = self._params()
        rng = np.random.default_rng(1)
        g = Genome.random(params, rng, group_id=2)
        assert "role.W" in g.tensors
        assert g.tensors["role.W"].shape == (params.body_input_dim, 1)
        b = SimBaby(params=params)
        g.apply_to(b)
        assert b.perceptron_role is not None
        assert np.allclose(b.perceptron_role.W, g.tensors["role.W"])

    def test_from_baby_extracts_role_channel(self):
        params = self._params()
        b = SimBaby(params=params)
        g = Genome.from_baby(b, group_id=1)
        assert "role.W" in g.tensors
        assert np.allclose(g.tensors["role.W"], b.perceptron_role.W)

    def test_dedicated_stream_keeps_shared_draws_identical(self):
        # Locked-proof invariant: enabling specialization must not perturb
        # the four behavior brains' RNG draws (or the perception-noise
        # stream).
        off = WorldParams(grid_size=(16, 8, 16))
        on = self._params()
        g_off = Genome.random(off, np.random.default_rng(9), group_id=0)
        g_on = Genome.random(on, np.random.default_rng(9), group_id=0)
        for name in ("cells", "body", "entity", "move"):
            assert np.allclose(g_off.tensors[f"{name}.W"],
                               g_on.tensors[f"{name}.W"])
            assert np.allclose(g_off.tensors[f"{name}.b"],
                               g_on.tensors[f"{name}.b"])
        assert "role.W" in g_on.tensors
        assert "role.W" not in g_off.tensors

    def test_run_history_carries_role_fields(self):
        eng = EvolutionEngine(
            params=self._params(),
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        entry = result["history"][0]
        for key in ("role_deposits", "role_deposit_rate",
                    "role_deposit_energy", "role_raids",
                    "role_raid_rate", "role_raid_energy"):
            assert key in entry
        assert entry["role_deposits"] >= 0
        assert entry["role_deposit_rate"] >= 0.0
        assert entry["role_deposit_energy"] >= 0.0
        assert entry["role_raids"] >= 0
        assert entry["role_raid_rate"] >= 0.0
        assert entry["role_raid_energy"] >= 0.0

    def test_run_off_default_has_no_role_brains(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert result["history"][0]["role_deposits"] == 0
        assert result["history"][0]["role_raids"] == 0

    def test_benchmark_structure_and_verdict_keys(self):
        result = benchmark_specialization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert set(result) >= {
            "control", "specialization", "group_count", "group_weight",
            "control_role_deposit_rate", "control_role_raid_rate",
            "specialization_role_deposit_rate", "specialization_role_raid_rate",
            "control_final_avg_fitness", "specialization_final_avg_fitness",
            "specialization_emerged",
        }
        assert len(result["control"]["history"]) == 3
        assert len(result["specialization"]["history"]) == 3
        assert result["specialization_role_deposit_rate"] == \
            result["specialization"]["history"][-1]["role_deposit_rate"]
        assert result["specialization_role_raid_rate"] == \
            result["specialization"]["history"][-1]["role_raid_rate"]

    def test_control_arm_never_roles(self):
        result = benchmark_specialization(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        for entry in result["control"]["history"]:
            assert entry["role_deposits"] == 0
            assert entry["role_raids"] == 0

    def test_benchmark_deterministic(self):
        a = benchmark_specialization(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        b = benchmark_specialization(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        assert a["specialization_final_avg_fitness"] == \
            b["specialization_final_avg_fitness"]
        assert a["specialization_role_deposit_rate"] == \
            b["specialization_role_deposit_rate"]
        assert a["specialization_role_raid_rate"] == \
            b["specialization_role_raid_rate"]
        assert a["specialization_emerged"] == b["specialization_emerged"]
