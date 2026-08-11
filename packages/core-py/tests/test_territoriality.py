"""
Tests for Stage 9 territoriality (claim / defend / raid).

Territory is CLAIMED by building nests (Stage 7): a tribe's region is the
ground within ``territory_radius`` of the nearest nest it owns. Territoriality
is an opt-in channel (``territoriality_enabled``): when on, each baby gains a
``perceptron_territory`` (entity-input -> 1 gate) that decides whether to
DEFEND the region. Standing on its own tribe's territory, a cleared gate
evicts the nearest foreign baby within ``defend_range`` — shoving it
``defend_push`` cells away (a pure relocation, never a kill) and transferring
``defend_take_fraction`` of its energy to the defender (a toll that scales
with what the trespasser carries, so evicting a rich foreigner pays), who
pays ``defend_cost`` (a transfer, not creation — the world conserves energy).
The territory is also a two-sided resource: a hungry baby standing on FOREIGN
ground (within ``territory_radius`` of a rival tribe's nearest nest) can RAID
that bank — draining ``nest_draw_rate`` per tick, a one-way transfer that is
exactly the shared value defending protects. When off, no brain exists and no
RNG is drawn, so the locked selection proofs keep their exact genome layout
and energy flow.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.evolution import (
    EvolutionEngine,
    Genome,
    benchmark_territoriality,
)
from domains.shell.simulation import (
    Nest,
    SimBaby,
    SimScene,
    Simulation,
    WorldParams,
)


def _quiet_baby(params: WorldParams, energy: float, position,
                group_id: int = 0) -> SimBaby:
    """A baby whose every decision gate is pinned off except territory."""
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


def _zero_defender(params: WorldParams, energy: float = 200.0,
                   position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A defender whose territory gate stays at its zero init (sigmoid(0))."""
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
    return b


def _params(**kw) -> WorldParams:
    base = dict(grid_size=(8, 4, 8), territoriality_enabled=True,
                social_enabled=False, message_enabled=False,
                teaching_enabled=False)
    base.update(kw)
    return WorldParams(**base)


def _open_defender(params: WorldParams, energy: float = 200.0,
                   position=(4.0, 1.0, 4.0), group_id: int = 0) -> SimBaby:
    """A defender with its territory gate forced open (sigmoid(10) ~ 1)."""
    b = _quiet_baby(params, energy, position, group_id=group_id)
    assert b.perceptron_territory is not None
    b.perceptron_territory.W[:] = 0.0
    b.perceptron_territory.b[:] = 10.0
    return b


def _nest(position, group_id: int = 0, stored_energy: float = 100.0) -> Nest:
    return Nest(id=1, position=np.array(position, dtype=np.float64),
                stored_energy=stored_energy, owner_group_id=group_id)


class TestTerritoryBrain:
    def test_off_by_default_keeps_locked_proofs(self):
        # The channel is opt-in: default params create no territory brain, so
        # the benchmark genomes (and their RNG streams) are untouched.
        assert WorldParams().territoriality_enabled is False
        b = SimBaby(params=WorldParams(grid_size=(8, 4, 8)))
        assert b.perceptron_territory is None
        on = SimBaby(params=_params())
        assert on.perceptron_territory is not None
        assert on.perceptron_territory.W.shape == (
            _params().entity_input_dim, 1,
        )

    def test_gate_below_threshold_does_not_defend(self):
        params = _params()
        a = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        assert a.decide_defend(b) == 0.0
        assert a._last_defend_out is not None
        assert a._last_defend_out < params.defend_gate_threshold
        assert b.alive and b.energy == 20.0

    def test_zero_init_gate_hits_threshold(self):
        # sigmoid(0) == 0.5 == the default gate threshold, so a directly
        # constructed (non-genome) baby with territoriality enabled defends —
        # the gate is on the boundary.
        params = _params()
        a = _zero_defender(params, 200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        gate = a.decide_defend(b)
        assert gate == pytest.approx(0.5)
        assert gate >= params.defend_gate_threshold

    def test_open_gate_emits_full_strength(self):
        params = _params()
        a = _open_defender(params)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        gate = a.decide_defend(b)
        assert gate == pytest.approx(1.0 / (1.0 + np.exp(-10.0)))
        assert gate > params.defend_gate_threshold

    def test_defend_transfers_energy_and_displaces(self):
        params = _params()
        a = _open_defender(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        anchor = np.array([4.0, 1.0, 4.0], dtype=np.float64)
        assert a.decide_defend(b) > 0.0
        taken = a.defend(b, anchor)
        # The toll scales with the trespasser's own energy.
        assert taken == pytest.approx(params.defend_take_fraction * 20.0)
        assert a.energy == 200.0 + taken
        assert b.energy == 20.0 - taken
        # A toll, not a strike — the trespasser is never killed.
        assert b.alive
        # The trespasser is softly shoved one cell away from the defender
        # (a small relocation, not a stranding at the territory edge).
        assert b.position[0] == pytest.approx(4.0)
        assert b.position[2] == pytest.approx(6.0)
        assert np.linalg.norm(
            b.position - np.array([4.0, 1.0, 5.0])) == pytest.approx(
                params.defend_push)

    def test_defend_is_conservation_safe(self):
        # The eviction is a pure transfer: defender + trespasser energy is
        # unchanged (defend_cost is charged by the simulation loop afterwards,
        # landing in the same tick's honest net reward).
        params = _params()
        a = _open_defender(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        before = a.energy + b.energy
        a.defend(b, np.array([4.0, 1.0, 4.0], dtype=np.float64))
        assert a.energy + b.energy == pytest.approx(before)

    def test_defend_noop_on_dead_trespasser(self):
        params = _params()
        a = _open_defender(params, energy=200.0)
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        b.entity.alive = False
        assert a.defend(b, np.array([4.0, 1.0, 4.0], dtype=np.float64)) == 0.0
        assert a.energy == 200.0

    def test_defend_noop_without_brain(self):
        params = WorldParams(grid_size=(8, 4, 8))
        a = SimBaby(initial_energy=200.0, params=params)
        b = SimBaby(initial_energy=20.0, params=params, group_id=1)
        assert a.perceptron_territory is None
        assert a.defend(b, np.array([4.0, 1.0, 4.0], dtype=np.float64)) == 0.0
        assert b.alive and b.energy == 20.0

    def test_push_away_shoves_one_cell_from_defender(self):
        params = _params()
        a = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0))
        b = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        out = a._push_away(b)
        # b moves one cell away from the defender, staying on the ground plane.
        assert out[0] == pytest.approx(4.0)
        assert out[1] == pytest.approx(1.0)
        assert out[2] == pytest.approx(6.0)
        assert np.linalg.norm(out - b.position) == pytest.approx(
            params.defend_push)


class TestTerritoryScene:
    def test_off_by_default_runs_no_defense(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=True)
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _quiet_baby(params, 200.0, (4.0, 1.0, 4.0), group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=3)
        sim.run()
        assert scene.params.territoriality_enabled is False
        assert sim.summary()["defenses"] == 0
        assert sim.summary()["defend_energy_moved"] == 0.0
        assert len(scene.alive_babies) == 2

    def test_on_channel_evicts_nearest_foreign_trespasser(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        near = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        farther = _quiet_baby(params, 30.0, (4.0, 1.0, 5.5), group_id=1)
        for b in (defender, near, farther):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        # The nearest foreign trespasser is evicted first (a toll, not a kill)
        # and softly shoved one cell away — still on the territory, not stranded.
        assert near.alive
        assert near.position[0] == pytest.approx(4.0)
        assert near.position[2] == pytest.approx(6.0)
        assert farther.position[2] == pytest.approx(5.5)
        assert sim.summary()["defenses"] == 1
        assert sim.summary()["defend_energy_moved"] == pytest.approx(
            params.defend_take_fraction * 20.0,
        )

    def test_defend_requires_standing_on_own_territory(self):
        params = _params()
        scene = SimScene(params=params)
        # The tribe's nest is far away — the defender is not on its own ground.
        scene.nests.append(_nest((7.0, 1.0, 7.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert sim.summary()["defenses"] == 0
        assert len(scene.alive_babies) == 2

    def test_defend_ignores_same_tribe_neighbor(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        kin = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=0)
        for b in (defender, kin):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        # A tribe-mate is never a trespasser — nothing is evicted.
        assert sim.summary()["defenses"] == 0
        assert len(scene.alive_babies) == 2

    def test_range_limits_evictions(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        far = _quiet_baby(params, 20.0, (7.0, 3.0, 7.0), group_id=1)
        for b in (defender, far):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert sim.summary()["defenses"] == 0
        assert len(scene.alive_babies) == 2

    def test_eviction_is_relocation_never_kill(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        # The evicted trespasser survives the defense (toll, not strike). The
        # soft shove keeps it in range for the second tick, so it is tolled
        # again — each toll is a fixed fraction of its remaining energy, so
        # the drain is geometric and never lethal.
        assert stranger.alive
        assert sim.summary()["defenses"] == 2
        toll1 = params.defend_take_fraction * 20.0
        after1 = (20.0 - toll1 - params.see_cost - params.passive_drain)
        toll2 = params.defend_take_fraction * after1
        expected = after1 - toll2 - params.see_cost - params.passive_drain
        assert stranger.energy == pytest.approx(expected, abs=1e-6)

    def test_defender_pays_defend_cost_for_eviction(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        # Defender: -see_cost + toll (a fraction of the trespasser's 20)
        # - defend_cost - passive_drain. The stranger is evicted before its
        # own tick, so its energy is still 20 at the moment of the toll.
        assert defender.energy == pytest.approx(
            200.0 - params.see_cost
            + params.defend_take_fraction * 20.0
            - params.defend_cost - params.passive_drain,
            abs=1e-6,
        )

    def test_summary_totals_match_tick_log(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        # Two ticks, two evictions: the soft shove keeps the trespasser in
        # range, so it is tolled again on the second tick.
        evictions = [r for r in sim._tick_log if r.get("defended", False)]
        assert sim.summary()["defenses"] == len(evictions) == 2
        assert sim.summary()["defend_energy_moved"] == pytest.approx(
            sum(r["defend_energy"] for r in evictions),
        )
        row = evictions[0]
        assert row["defended_id"] == stranger.entity.id
        assert row["defend_amplitude"] > 0.0

    def test_serialization_roundtrip_preserves_territory_brain(self):
        params = _params()
        defender = _open_defender(params, energy=200.0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        defender.decide_defend(stranger)
        w_before = defender.perceptron_territory.W.copy()
        b_before = defender.perceptron_territory.b.copy()
        data = defender.to_dict()
        restored = SimBaby.from_dict(data, params=params)
        assert restored.perceptron_territory is not None
        assert np.allclose(restored.perceptron_territory.W, w_before)
        assert np.allclose(restored.perceptron_territory.b, b_before)

    def test_delta_rule_shapes_territory_gate_from_defense(self):
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
        defender = _open_defender(params, energy=200.0, group_id=0)
        stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
        for b in (defender, stranger):
            scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        # The successful eviction's reward flows through learn() and moves the
        # territory weights off their initialized zero vector.
        assert defender.perceptron_territory is not None
        assert not np.allclose(defender.perceptron_territory.W, 0.0)

    def test_determinism_with_territoriality_on(self):
        params = _params()
        outs = []
        for _ in range(2):
            np.random.seed(7)
            scene = SimScene(params=params)
            scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0))
            defender = _open_defender(params, energy=200.0, group_id=0)
            stranger = _quiet_baby(params, 20.0, (4.0, 1.0, 5.0), group_id=1)
            for b in (defender, stranger):
                scene.add_baby(b)
            sim = Simulation(scene, max_ticks=2)
            sim.run()
            outs.append((sim.summary()["defenses"],
                         sim.summary()["defend_energy_moved"]))
        assert outs[0] == outs[1]


class TestTerritoryEvolution:
    @staticmethod
    def _params(**kw) -> WorldParams:
        base = dict(grid_size=(16, 8, 16), territoriality_enabled=True,
                    teaching_enabled=False, memory_enabled=False)
        base.update(kw)
        return WorldParams(**base)

    def test_genome_roundtrip(self):
        params = self._params()
        rng = np.random.default_rng(1)
        g = Genome.random(params, rng, group_id=2)
        assert "territory.W" in g.tensors
        assert g.tensors["territory.W"].shape == (params.entity_input_dim, 1)
        b = SimBaby(params=params)
        g.apply_to(b)
        assert b.perceptron_territory is not None
        assert np.allclose(b.perceptron_territory.W, g.tensors["territory.W"])

    def test_dedicated_stream_keeps_shared_draws_identical(self):
        # Locked-proof invariant: enabling territoriality must not perturb the
        # four behavior brains' RNG draws (or the perception-noise stream).
        off = WorldParams(grid_size=(16, 8, 16))
        on = self._params()
        g_off = Genome.random(off, np.random.default_rng(9), group_id=0)
        g_on = Genome.random(on, np.random.default_rng(9), group_id=0)
        for name in ("cells", "body", "entity", "move"):
            assert np.allclose(g_off.tensors[f"{name}.W"],
                               g_on.tensors[f"{name}.W"])
            assert np.allclose(g_off.tensors[f"{name}.b"],
                               g_on.tensors[f"{name}.b"])
        assert "territory.W" in g_on.tensors
        assert "territory.W" not in g_off.tensors

    def test_run_history_carries_territory_fields(self):
        eng = EvolutionEngine(
            params=self._params(),
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert "defenses" in result["history"][0]
        assert "defend_rate" in result["history"][0]
        assert "defend_energy_moved" in result["history"][0]
        assert result["history"][0]["defend_rate"] >= 0.0

    def test_run_off_default_has_no_territory_brains(self):
        eng = EvolutionEngine(
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert result["history"][0]["defenses"] == 0

    def test_benchmark_structure_and_verdict_keys(self):
        result = benchmark_territoriality(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert set(result) >= {
            "control", "territoriality", "group_count", "group_weight",
            "control_last_avg", "territoriality_last_avg", "defend_rate",
            "defenses", "defend_energy_moved", "territoriality_emerged",
        }
        assert len(result["control"]["history"]) == 3
        assert len(result["territoriality"]["history"]) == 3
        assert result["defenses"] == result["territoriality"]["history"][-1]["defenses"]

    def test_benchmark_deterministic(self):
        a = benchmark_territoriality(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        b = benchmark_territoriality(
            population_size=4, generations=2, ticks_per_generation=8,
            organic_pools=1, seed=5,
        )
        assert a["territoriality_last_avg"] == b["territoriality_last_avg"]
        assert a["defenses"] == b["defenses"]
        assert a["territoriality_emerged"] == b["territoriality_emerged"]

    def test_run_history_carries_raid_fields(self):
        eng = EvolutionEngine(
            params=self._params(),
            population_size=4, generations=2, ticks_per_generation=3,
            organic_pools=1, seed=3,
        )
        result = eng.run()
        assert "raids" in result["history"][0]
        assert "raid_energy_moved" in result["history"][0]
        assert result["history"][0]["raids"] >= 0
        assert result["history"][0]["raid_energy_moved"] >= 0.0

    def test_benchmark_structure_and_verdict_keys_carry_raids(self):
        result = benchmark_territoriality(
            population_size=4, generations=3, ticks_per_generation=8,
            organic_pools=1, seed=1,
        )
        assert "raids" in result
        assert "raid_energy_moved" in result
        assert result["raids"] == result["territoriality"]["history"][-1]["raids"]
        assert result["raid_energy_moved"] == result["territoriality"][
            "history"][-1]["raid_energy_moved"]


class TestRaid:
    def test_raid_requires_territoriality_channel(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=True)
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        assert scene.params.territoriality_enabled is False
        assert scene.raid_nest(raider) == 0.0
        assert raider.energy == 20.0
        assert scene.nests[0].stored_energy == 50.0

    def test_raid_requires_hunger(self):
        # A baby at or above start energy has no gap to fill — nothing is taken.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 120.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        assert scene.raid_nest(raider) == 0.0
        assert scene.nests[0].stored_energy == 50.0

    def test_raid_never_touches_own_tribe_bank(self):
        # A baby only drains a RIVAL's nest — its own tribe's bank is safe.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=0, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        assert scene.raid_nest(raider) == 0.0
        assert scene.nests[0].stored_energy == 50.0

    def test_raid_requires_standing_on_foreign_ground(self):
        # The nearest foreign nest is beyond territory_radius — no raid.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((7.0, 1.0, 7.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        assert scene.raid_nest(raider) == 0.0
        assert scene.nests[0].stored_energy == 50.0

    def test_raid_transfers_energy_from_foreign_nest(self):
        params = _params()
        scene = SimScene(params=params)
        nest = _nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0)
        scene.nests.append(nest)
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        stolen = scene.raid_nest(raider)
        # min(nest_draw_rate=1.0, gap=80, bank=50) = 1.0
        assert stolen == pytest.approx(params.nest_draw_rate)
        assert nest.stored_energy == pytest.approx(50.0 - stolen)
        assert raider.energy == pytest.approx(20.0 + stolen)
        # Conservation: a raid is a transfer, never creation.
        assert raider.energy + nest.stored_energy == pytest.approx(70.0)

    def test_raid_capped_by_gap_and_bank(self):
        # A nearly-full baby steals only the gap back to start energy.
        params = _params()
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 99.5, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        assert scene.raid_nest(raider) == pytest.approx(0.5)
        assert raider.energy == pytest.approx(100.0)
        # A small bank is drained only as far as it holds.
        scene2 = SimScene(params=params)
        scene2.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=0.4))
        raider2 = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene2.add_baby(raider2)
        assert scene2.raid_nest(raider2) == pytest.approx(0.4)
        assert scene2.nests[0].stored_energy == pytest.approx(0.0)

    def test_raid_targets_nearest_foreign_nest(self):
        params = _params()
        scene = SimScene(params=params)
        near = Nest(id=1, position=np.array([4.0, 1.0, 4.0], dtype=np.float64),
                    stored_energy=50.0, owner_group_id=1)
        far = Nest(id=2, position=np.array([6.0, 1.0, 4.0], dtype=np.float64),
                   stored_energy=50.0, owner_group_id=2)
        scene.nests.extend([near, far])
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        stolen = scene.raid_nest(raider)
        assert stolen == pytest.approx(params.nest_draw_rate)
        assert near.stored_energy == pytest.approx(50.0 - stolen)
        assert far.stored_energy == pytest.approx(50.0)

    def test_scene_records_raid_in_summary(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        # One raid per tick: the raider stays hungry (steal 1.0 balances the
        # see_cost + passive_drain of 1.0), so it drains the bank twice. The
        # bank also pays nest_decay upkeep each tick (before the raid).
        assert sim.summary()["raids"] == 2
        assert sim.summary()["raid_energy_moved"] == pytest.approx(
            2.0 * params.nest_draw_rate)
        row = [r for r in sim._tick_log if r.get("raided", 0.0) > 0.0][0]
        assert row["raided"] == pytest.approx(params.nest_draw_rate)
        assert scene.nests[0].stored_energy < 50.0 - 2.0 * params.nest_draw_rate

    def test_scene_off_channel_runs_no_raids(self):
        params = WorldParams(grid_size=(8, 4, 8), structure_enabled=True)
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        assert sim.summary()["raids"] == 0
        assert sim.summary()["raid_energy_moved"] == 0.0
        # Only nest upkeep applies when the channel is off — no theft.
        assert scene.nests[0].stored_energy == pytest.approx(
            50.0 * (1.0 - params.nest_decay) ** 2, abs=1e-6,
        )

    def test_raider_energy_flow_is_exact(self):
        # With an empty world, a single raid tick is exactly balanced: the
        # stolen 1.0 offsets see_cost (0.5) + passive_drain (0.5). The bank
        # pays nest_decay upkeep first (50 * 0.002 = 0.1), then the raid.
        params = _params(structure_enabled=True)
        scene = SimScene(params=params)
        scene.nests.append(_nest((4.0, 1.0, 4.0), group_id=1, stored_energy=50.0))
        raider = _quiet_baby(params, 20.0, (4.0, 1.0, 4.0), group_id=0)
        scene.add_baby(raider)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        assert raider.energy == pytest.approx(
            20.0 - params.see_cost - params.passive_drain
            + params.nest_draw_rate, abs=1e-6,
        )
        assert scene.nests[0].stored_energy == pytest.approx(
            50.0 * (1.0 - params.nest_decay) - params.nest_draw_rate, abs=1e-6,
        )
