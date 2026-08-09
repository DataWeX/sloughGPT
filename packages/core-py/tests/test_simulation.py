"""
Tests for the baby-in-the-world simulation model.

Covers WorldGrid, cell update functions, Entity, Perceptron,
SimBaby, SimScene, and the Simulation tick loop.
"""

import numpy as np
import pytest

from domains.shell.simulation import (
    WorldGrid, WorldCell, WorldParams,
    cell_update_diffusion, cell_update_waves,
    cell_update_energy_conservation, cell_update_default,
    Entity, EntityType, Perceptron, Perception, CellWrite, BabyAction,
    SimBaby, SimScene, Simulation,
    MATERIAL_AIR, MATERIAL_STONE, MATERIAL_ORGANIC, MATERIAL_SIGNAL,
    NUM_MATERIALS,
)


# ── WorldGrid ─────────────────────────────────────────────────────────────────

class TestWorldGrid:
    def test_creation(self):
        g = WorldGrid((16, 8, 16))
        assert g.size == (16, 8, 16)
        assert g.nx == 16
        assert g.ny == 8
        assert g.nz == 16
        assert g.total == 2048

    def test_reset_sets_defaults(self):
        g = WorldGrid((4, 4, 4))
        assert np.all(g.material == MATERIAL_AIR)
        assert np.all(g.energy == 0.0)
        assert np.all(g.temperature == 20.0)
        assert np.all(g.signal == 0.0)

    def test_idx(self):
        g = WorldGrid((4, 4, 4))
        assert g.idx(0, 0, 0) == 0
        assert g.idx(3, 3, 3) == 63

    def test_idx_wraps(self):
        g = WorldGrid((4, 4, 4))
        assert g.idx(4, 0, 0) == 0  # wraps
        assert g.idx(0, 4, 0) == 0  # wraps

    def test_coords_roundtrip(self):
        g = WorldGrid((8, 4, 8))
        for x, y, z in [(0, 0, 0), (7, 3, 7), (3, 1, 5)]:
            i = g.idx(x, y, z)
            cx, cy, cz = g.coords(i)
            assert (cx, cy, cz) == (x, y, z)

    def test_set_and_get_cell(self):
        g = WorldGrid((4, 4, 4))
        cell = WorldCell(material=MATERIAL_STONE, energy=50.0, temperature=100.0)
        g.set_cell(1, 2, 3, cell)
        result = g.get_cell(1, 2, 3)
        assert result.material == MATERIAL_STONE
        assert result.energy == 50.0
        assert result.temperature == 100.0

    def test_place_material(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(2, 1, 0, MATERIAL_ORGANIC, energy=30.0)
        cell = g.get_cell(2, 1, 0)
        assert cell.material == MATERIAL_ORGANIC
        assert cell.energy == 30.0

    def test_write_cell_succeeds(self):
        g = WorldGrid((4, 4, 4))
        assert g.write_cell(0, 0, 0, MATERIAL_STONE, 10.0)
        assert g.get_cell(0, 0, 0).material == MATERIAL_STONE

    def test_write_cell_out_of_bounds(self):
        g = WorldGrid((4, 4, 4))
        assert not g.write_cell(-1, 0, 0, MATERIAL_STONE, 0.0)
        assert not g.write_cell(4, 0, 0, MATERIAL_STONE, 0.0)

    def test_get_nearby_cells(self):
        g = WorldGrid((16, 8, 16))
        g.place_material(8, 4, 8, MATERIAL_ORGANIC, energy=100.0)
        cells = g.get_nearby_cells(8, 4, 8, radius=3.0)
        assert cells["count"] > 0
        organic_idx = np.where(cells["material"] == MATERIAL_ORGANIC)[0]
        assert len(organic_idx) == 1

    def test_nearby_cells_small_radius(self):
        g = WorldGrid((4, 4, 4))
        cells = g.get_nearby_cells(0, 0, 0, radius=0.1)
        assert cells["count"] == 1  # always includes origin cell

    def test_total_energy(self):
        g = WorldGrid((4, 4, 4))
        assert g.total_energy == 0.0
        g.place_material(0, 0, 0, MATERIAL_STONE, energy=50.0)
        assert g.total_energy == 50.0


# ── Cell Update Functions ─────────────────────────────────────────────────────

class TestCellUpdate:
    def test_diffusion_spreads_energy(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_STONE, energy=100.0)
        params = WorldParams(diffusion_rate=0.5, energy_loss=0.0)
        before = g.energy[g.idx(1, 1, 1)]
        cell_update_diffusion(g, params)
        after = g.energy[g.idx(1, 1, 1)]
        assert after < before  # energy spread out
        assert abs(g.total_energy - 100.0) < 1e-5  # conserved

    def test_waves_propagate_signal(self):
        g = WorldGrid((4, 4, 4))
        i = g.idx(1, 1, 1)
        g.signal[i] = 100.0
        params = WorldParams(wave_speed=0.5, signal_decay=0.1)
        before = g.signal[i]
        cell_update_waves(g, params)
        after = g.signal[i]
        assert after < before  # signal spread out

    def test_energy_conservation_reduces_energy(self):
        g = WorldGrid((4, 4, 4))
        np.copyto(g.energy, np.full(g.total, 100.0, dtype=np.float32))
        params = WorldParams(energy_loss=0.1)
        cell_update_energy_conservation(g, params)
        assert np.all(g.energy == 90.0)

    def test_energy_conservation_clamps_negative(self):
        g = WorldGrid((4, 4, 4))
        np.copyto(g.energy, np.full(g.total, -10.0, dtype=np.float32))
        params = WorldParams(energy_loss=0.0)
        cell_update_energy_conservation(g, params)
        assert np.all(g.energy >= 0.0)

    def test_default_update_runs_all(self):
        g = WorldGrid((4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_ORGANIC, energy=100.0)
        params = WorldParams()
        cell_update_default(g, params)
        assert g.total_energy < 100.0  # diffusion + loss
        assert g.total_energy > 0.0


# ── Entity ────────────────────────────────────────────────────────────────────

class TestEntity:
    def test_creation(self):
        e = Entity(id=1, position=np.array([1.0, 2.0, 3.0]), energy=50.0,
                   entity_type=EntityType.AGENT)
        assert e.id == 1
        assert np.allclose(e.position, [1.0, 2.0, 3.0])
        assert e.energy == 50.0
        assert e.entity_type == EntityType.AGENT
        assert e.alive

    def test_distance_to(self):
        a = Entity(position=np.array([0.0, 0.0, 0.0]))
        b = Entity(position=np.array([3.0, 4.0, 0.0]))
        assert a.distance_to(b) == 5.0

    def test_distance_to_point(self):
        e = Entity(position=np.array([0.0, 0.0, 0.0]))
        assert e.distance_to_point(np.array([1.0, 0.0, 0.0])) == 1.0


# ── Perceptron ────────────────────────────────────────────────────────────────

class TestPerceptron:
    def test_forward(self):
        p = Perceptron(4, 3)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = p.forward(x)
        assert out.shape == (3,)
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_forward_zeros(self):
        p = Perceptron(4, 3)
        x = np.zeros(4, dtype=np.float32)
        out = p.forward(x)
        assert out.shape == (3,)
        assert np.all(out >= 0.5 - 1e-5) and np.all(out <= 0.5 + 1e-5)

    def test_update_changes_weights(self):
        p = Perceptron(2, 1)
        x = np.array([1.0, 0.0], dtype=np.float32)
        before = p.W.copy()
        p.update(x, np.array([1.0]), lr=0.1)
        assert not np.allclose(p.W, before)

    def test_sigmoid_bounds(self):
        assert abs(Perceptron._sigmoid(100.0) - 1.0) < 1e-4
        assert abs(Perceptron._sigmoid(-100.0) - 0.0) < 1e-4
        assert abs(Perceptron._sigmoid(0.0) - 0.5) < 1e-5


# ── CellWrite / BabyAction ───────────────────────────────────────────────────

class TestBabyAction:
    def test_cell_write(self):
        w = CellWrite(x=1, y=2, z=3, material=MATERIAL_STONE, energy=10.0)
        assert w.x == 1
        assert w.y == 2
        assert w.z == 3
        assert w.material == MATERIAL_STONE
        assert w.energy == 10.0

    def test_baby_action_defaults_to_empty(self):
        a = BabyAction()
        assert a.writes == []


# ── SimBaby ───────────────────────────────────────────────────────────────────

class TestSimBaby:
    def test_creation(self):
        baby = SimBaby(initial_energy=100.0)
        assert baby.energy == 100.0
        assert baby.entity.entity_type == EntityType.AGENT
        assert baby.alive
        assert baby.tick_count == 0

    def test_position_through_entity(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.position = np.array([5.0, 3.0, 7.0])
        assert np.allclose(baby.position, [5.0, 3.0, 7.0])

    def test_perceive_returns_perception(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        assert p.nearby_cells["count"] > 0
        assert p.agent_body["energy"] == 100.0
        assert p.time_ms >= 0.0

    def test_feel_positive_delta(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.energy = 110.0
        delta = baby.feel(100.0)
        assert delta == 10.0

    def test_feel_negative_delta(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.energy = 80.0
        delta = baby.feel(100.0)
        assert delta == -20.0

    def test_react_returns_action(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        action = baby.react(p, 0.0)
        assert isinstance(action, BabyAction)
        assert len(action.writes) > 0

    def test_react_does_nothing_when_low_energy(self):
        baby = SimBaby(initial_energy=5.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        action = baby.react(p, -5.0)
        assert len(action.writes) == 0

    def test_react_gate_suppresses_writes(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        baby.perceptron_cells.b[0] = -20.0  # drive gate fully off
        action = baby.react(p, 0.0)
        assert len(action.writes) == 0

    def test_react_gate_maximizes_writes(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        baby.perceptron_cells.b[0] = 20.0   # drive gate fully on
        action = baby.react(p, 0.0)
        assert len(action.writes) == 6

    def test_react_material_gate_selects_material(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        baby.perceptron_cells.b[0] = 20.0
        baby.perceptron_cells.b[1] = -20.0  # material gate at 0
        action = baby.react(p, 0.0)
        assert all(w.material == MATERIAL_AIR for w in action.writes)

    def test_perception_features_normalized_range(self):
        baby = SimBaby(initial_energy=100.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        feat = baby._perception_features(p)
        assert feat.shape == (baby.params.cells_input_dim,)
        assert np.all(feat >= 0.0) and np.all(feat <= 1.0)

    def test_learn_updates_cells_perceptron(self):
        baby = SimBaby(initial_energy=100.0)
        baby.perceive(WorldGrid((16, 8, 16)))
        before = baby.perceptron_cells.W.copy()
        baby.learn(5.0)
        assert not np.allclose(baby.perceptron_cells.W, before)

    def test_apply_action_writes_cells(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([8.0, 4.0, 8.0]))
        world = WorldGrid((16, 8, 16))
        action = BabyAction(writes=[CellWrite(8, 4, 8, MATERIAL_STONE, 10.0)])
        n = baby.apply_action(action, world)
        assert n == 1
        assert world.get_cell(8, 4, 8).material == MATERIAL_STONE

    def test_absorb_energy_from_organic(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([5.0, 3.0, 5.0]))
        world = WorldGrid((16, 8, 16))
        world.place_material(5, 3, 5, MATERIAL_ORGANIC, energy=50.0)
        absorbed = baby.absorb_energy(world, radius=1)
        assert absorbed > 0
        assert world.get_cell(5, 3, 5).energy < 50.0

    def test_learn_updates_weights(self):
        baby = SimBaby(initial_energy=100.0)
        before = baby.perceptron_body.W.copy()
        baby.learn(5.0)
        assert not np.allclose(baby.perceptron_body.W, before)

    def test_learn_weakens_on_negative(self):
        baby = SimBaby(initial_energy=100.0)
        before = baby.perceptron_body.W.copy()
        baby.learn(-5.0)
        assert not np.allclose(baby.perceptron_body.W, before)

    def test_alive_false_when_zero_energy(self):
        baby = SimBaby(initial_energy=0.0)
        assert not baby.alive

    def test_alive_false_when_entity_dead(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.alive = False
        assert not baby.alive

    def test_info(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([3.0, 5.0, 7.0]))
        info = baby.info()
        assert info["id"] == baby.entity.id
        assert info["energy"] == 100.0
        assert info["alive"]


# ── SimScene ──────────────────────────────────────────────────────────────────

class TestSimScene:
    def test_creation(self):
        scene = SimScene()
        assert scene.world.size == (64, 32, 64)
        assert scene.tick == 0

    def test_add_baby(self):
        scene = SimScene()
        baby = SimBaby(initial_energy=100.0)
        scene.add_baby(baby)
        assert len(scene._devices) == 1
        assert baby.entity in scene.entities

    def test_spawn_babies(self):
        scene = SimScene()
        scene.spawn_babies(count=4)
        assert len(scene._devices) == 4
        assert len(scene.entities) == 4

    def test_spawn_babies_default_count(self):
        scene = SimScene(params=WorldParams(start_agents=3))
        scene.spawn_babies()
        assert len(scene._devices) == 3

    def test_place_material(self):
        scene = SimScene()
        scene.place_material(10, 5, 10, MATERIAL_ORGANIC, energy=100.0)
        cell = scene.world.get_cell(10, 5, 10)
        assert cell.material == MATERIAL_ORGANIC
        assert cell.energy == 100.0

    def test_update_cells(self):
        scene = SimScene()
        scene.place_material(10, 5, 10, MATERIAL_ORGANIC, energy=100.0)
        scene.update_cells()
        assert scene.world.total_energy < 100.0  # diffusion + loss

    def test_get_baby(self):
        scene = SimScene()
        baby = SimBaby(initial_energy=100.0)
        scene.add_baby(baby)
        assert scene.get_baby(baby.entity.id) is baby
        assert scene.get_baby(-1) is None

    def test_alive_babies(self):
        scene = SimScene()
        b1 = SimBaby(initial_energy=100.0)
        b2 = SimBaby(initial_energy=0.0)  # dead
        scene.add_baby(b1)
        scene.add_baby(b2)
        alive = scene.alive_babies
        assert len(alive) == 1
        assert alive[0] is b1

    def test_info(self):
        scene = SimScene()
        scene.spawn_babies(count=2)
        info = scene.info()
        assert info["grid_size"] == (64, 32, 64)
        assert info["alive_babies"] == 2


# ── Simulation ────────────────────────────────────────────────────────────────

_SIM_PARAMS = WorldParams(grid_size=(8, 4, 8), start_agents=1)


class TestSimulation:
    def test_creation(self):
        scene = SimScene(params=_SIM_PARAMS)
        sim = Simulation(scene, max_ticks=10)
        assert sim.max_ticks == 10
        assert not sim._running

    def test_step_returns_results(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        results = sim.step()
        assert len(results) == 1
        assert results[0]["baby_id"] > 0
        assert results[0]["tick"] == 1
        assert results[0]["energy"] < 100.0  # drain + costs

    def test_run_executes_max_ticks(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        results = sim.run()
        assert scene.tick == 10
        assert len(results) == 10

    def test_stop_stops_simulation(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=100)
        sim._running = True
        sim.stop()
        assert not sim._running

    def test_summary(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=5)
        sim.run()
        summary = sim.summary()
        assert summary["total_ticks"] == 5
        assert summary["total_baby_ticks"] == 5
        assert summary["total_cells_written"] >= 0

    def test_step_reduces_energy_over_time(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=20)
        first = sim.step()[0]["energy"]
        for _ in range(10):
            sim.step()
        last = sim.step()[0]["energy"]
        assert last < first  # energy decreases over time

    def test_baby_dies_at_zero_energy(self):
        p = WorldParams(grid_size=(8, 4, 8), start_agents=1, passive_drain=5.0, see_cost=0.0)
        scene = SimScene(params=p)
        baby = SimBaby(initial_energy=1.0, params=p)  # almost dead
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=20)
        results = sim.run()
        energies = [r["energy"] for r in results]
        assert any(e <= 0 for e in energies)  # died at some point

    def test_multiple_babies(self):
        p = WorldParams(grid_size=(8, 4, 8), start_agents=4)
        scene = SimScene(params=p)
        scene.spawn_babies(count=4)
        sim = Simulation(scene, max_ticks=10)
        results = sim.run()
        assert len(results) >= 4
        assert scene.tick == 10

    def test_tick_log(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=5)
        assert len(sim.tick_log) == 0
        sim.run()
        assert len(sim.tick_log) == 5

    def test_summary_empty_tick_log(self):
        sim = Simulation(SimScene(params=_SIM_PARAMS))
        summary = sim.summary()
        assert summary == {"total_ticks": 0, "total_actions": 0}

    def test_nearby_cells_negative_radius_returns_empty(self):
        scene = SimScene(params=_SIM_PARAMS)
        out = scene.world.get_nearby_cells(3, 2, 3, radius=-0.5)
        assert out["count"] == 0
        assert out["material"].shape == (0,)

    def test_step_verbose_logs(self, caplog):
        import logging

        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10, verbose=True)
        with caplog.at_level(logging.INFO, logger="slo.sim"):
            sim.step()
        assert any("tick=" in r.message for r in caplog.records)

    def test_step_skips_dead_baby_in_device_list(self):
        class _UnfilteredScene(SimScene):
            @property
            def alive_babies(self):
                return list(self._devices)

        scene = _UnfilteredScene(params=_SIM_PARAMS)
        baby = SimBaby(initial_energy=1.0, params=_SIM_PARAMS)
        baby.entity.energy = 0.0
        baby.entity.alive = False
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=10)
        results = sim.step()
        assert results == []
        assert all(not e.alive for e in scene.entities)

    def test_run_breaks_when_stopped(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=50)

        def _step_once_then_stop():
            sim.stop()
            return []

        sim.step = _step_once_then_stop
        results = sim.run()
        assert results == []
        assert not sim._running


# ── Additional Coverage ──────────────────────────────────────────────────────


class TestEntityExtra:
    def test_distance_to_same_point(self):
        e1 = Entity(position=np.array([1.0, 2.0, 3.0]))
        e2 = Entity(position=np.array([1.0, 2.0, 3.0]))
        assert e1.distance_to(e2) == 0.0

    def test_distance_to_different_point(self):
        e1 = Entity(position=np.array([0.0, 0.0, 0.0]))
        e2 = Entity(position=np.array([3.0, 4.0, 0.0]))
        assert abs(e1.distance_to(e2) - 5.0) < 1e-6

    def test_distance_to_point(self):
        e = Entity(position=np.array([1.0, 0.0, 0.0]))
        assert abs(e.distance_to_point(np.array([4.0, 0.0, 0.0])) - 3.0) < 1e-6

    def test_distance_to_point_zero(self):
        e = Entity(position=np.array([5.0, 5.0, 5.0]))
        assert e.distance_to_point(np.array([5.0, 5.0, 5.0])) == 0.0


class TestPerceptron:
    def test_output_shape(self):
        p = Perceptron(4, 3)
        out = p.forward(np.array([1.0, 0.5, 0.2, 0.8]))
        assert out.shape == (3,)

    def test_output_in_01_range(self):
        p = Perceptron(4, 3)
        out = p.forward(np.array([1.0, 0.5, 0.2, 0.8]))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_update_changes_weights(self):
        p = Perceptron(4, 3)
        w_before = p.W.copy()
        x = np.array([1.0, 0.5, 0.2, 0.8])
        p.update(x, np.array([1.0, -1.0, 0.5]), lr=0.1)
        assert not np.array_equal(p.W, w_before)

    def test_sigmoid_bounds(self):
        assert Perceptron._sigmoid(np.array([100.0]))[0] > 0.999
        assert Perceptron._sigmoid(np.array([-100.0]))[0] < 0.001


class TestSimBabyExtra:
    def test_perceive_reduces_energy(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([4.0, 2.0, 4.0]))
        grid = WorldGrid((8, 4, 8))
        before = baby.energy
        baby.perceive(grid)
        assert baby.energy <= before

    def test_feel_positive(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.energy = 110.0
        delta = baby.feel(100.0)
        assert delta == 10.0

    def test_feel_negative(self):
        baby = SimBaby(initial_energy=100.0)
        baby.entity.energy = 90.0
        delta = baby.feel(100.0)
        assert delta == -10.0

    def test_react_writes_when_energy_high(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([4.0, 2.0, 4.0]))
        p = Perception()
        action = baby.react(p, 0.0)
        assert len(action.writes) == 3

    def test_react_no_writes_when_energy_low(self):
        baby = SimBaby(initial_energy=5.0,
                       position=np.array([4.0, 2.0, 4.0]))
        p = Perception()
        action = baby.react(p, 0.0)
        assert len(action.writes) == 0

    def test_apply_action_returns_count(self):
        baby = SimBaby(initial_energy=100.0,
                       position=np.array([4.0, 2.0, 4.0]))
        grid = WorldGrid((8, 4, 8))
        action = BabyAction(writes=[
            CellWrite(4, 2, 4, MATERIAL_STONE, 1.0),
            CellWrite(0, 0, 0, MATERIAL_ORGANIC, 5.0),
        ])
        count = baby.apply_action(action, grid)
        assert count == 2

    def test_absorb_energy_from_organic(self):
        baby = SimBaby(initial_energy=50.0,
                       position=np.array([4.0, 2.0, 4.0]))
        grid = WorldGrid((8, 4, 8))
        grid.place_material(4, 2, 4, MATERIAL_ORGANIC, energy=10.0)
        absorbed = baby.absorb_energy(grid)
        assert absorbed > 0.0
        assert absorbed <= 10.0

    def test_absorb_energy_no_organic(self):
        baby = SimBaby(initial_energy=50.0,
                       position=np.array([4.0, 2.0, 4.0]))
        grid = WorldGrid((8, 4, 8))
        absorbed = baby.absorb_energy(grid)
        assert absorbed == 0.0

    def test_learn_positive_delta(self):
        baby = SimBaby(initial_energy=100.0)
        w_before = baby.perceptron_body.W.copy()
        baby.learn(10.0)
        assert not np.array_equal(baby.perceptron_body.W, w_before)

    def test_tick_count(self):
        baby = SimBaby(initial_energy=100.0)
        assert baby.tick_count == 0
        baby._total_ticks = 5
        assert baby.tick_count == 5


class TestCellUpdateWaves:
    def test_signal_propagates(self):
        g = WorldGrid((8, 4, 8))
        g.signal[g.idx(4, 2, 4)] = 10.0
        cell_update_waves(g, WorldParams(grid_size=(8, 4, 8)))
        center = g.signal[g.idx(4, 2, 4)]
        neighbor = g.signal[g.idx(5, 2, 4)]
        assert center < 10.0
        assert neighbor > 0.0


class TestCellUpdateDefault:
    def test_runs_without_error(self):
        g = WorldGrid((8, 4, 8))
        g.signal[g.idx(4, 2, 4)] = 10.0
        cell_update_default(g, WorldParams(grid_size=(8, 4, 8)))
        assert g.total_signal >= 0.0


class TestPerception:
    def test_default_fields(self):
        p = Perception()
        assert p.nearby_cells == {}
        assert p.nearby_entities == []
        assert p.agent_body == {}
        assert p.time_ms == 0.0


class TestCellWrite:
    def test_defaults(self):
        cw = CellWrite()
        assert cw.x == 0
        assert cw.y == 0
        assert cw.z == 0
        assert cw.material == MATERIAL_AIR
        assert cw.energy == 0.0


class TestBabyAction:
    def test_default_empty(self):
        ba = BabyAction()
        assert ba.writes == []

    def test_with_writes(self):
        cw = CellWrite(x=1, y=2, z=3, material=2, energy=5.0)
        ba = BabyAction(writes=[cw])
        assert len(ba.writes) == 1
        assert ba.writes[0].x == 1


class TestSimulationExtra:
    def test_multiple_steps(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        for _ in range(5):
            sim.step()
        assert scene.tick == 5
        assert len(sim.tick_log) == 5

    def test_dead_baby_removed_after_step(self):
        scene = SimScene(params=_SIM_PARAMS)
        baby = SimBaby(initial_energy=1.0, params=_SIM_PARAMS)
        scene.add_baby(baby)
        baby.entity.energy = 0.0
        baby.entity.alive = False
        sim = Simulation(scene, max_ticks=10)
        sim.step()
        assert len(scene.entities) == 0

    def test_tick_log_property(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        assert sim.tick_log == []
        sim.step()
        assert len(sim.tick_log) == 1
        assert sim.tick_log[0]["tick"] == 1

    def test_summary_with_ticks(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        sim.step()
        s = sim.summary()
        assert s["total_ticks"] == 1
        assert s["total_baby_ticks"] == 1
        assert "alive_at_end" in s
        assert "deaths" in s


# ── Multi-Agent (Social) ─────────────────────────────────────────────────────

_POS_A = np.array([4.0, 2.0, 4.0], dtype=np.float64)
_POS_B = np.array([4.0, 2.0, 5.0], dtype=np.float64)
_POS_FAR = np.array([7.0, 3.0, 7.0], dtype=np.float64)


class TestSceneNearbyBabies:
    def test_returns_neighbors_in_radius(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(position=_POS_B, params=_SIM_PARAMS)
        scene.add_baby(a)
        scene.add_baby(b)
        n = scene.nearby_babies(_POS_A, radius=3.0)
        assert len(n) == 2
        assert n[0].entity.id == a.entity.id  # self first (distance 0)
        assert n[1].entity.id == b.entity.id

    def test_includes_self_without_exclude_id(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        scene.add_baby(a)
        n = scene.nearby_babies(_POS_A, radius=3.0)
        assert len(n) == 1
        assert n[0].entity.id == a.entity.id

    def test_excludes_self(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(position=_POS_B, params=_SIM_PARAMS)
        scene.add_baby(a)
        scene.add_baby(b)
        n = scene.nearby_babies(_POS_A, radius=3.0, exclude_id=a.entity.id)
        assert len(n) == 1
        assert n[0].entity.id == b.entity.id

    def test_outside_radius_excluded(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        far = SimBaby(position=_POS_FAR, params=_SIM_PARAMS)
        scene.add_baby(a)
        scene.add_baby(far)
        n = scene.nearby_babies(_POS_A, radius=3.0)
        assert len(n) == 1
        assert n[0].entity.id == a.entity.id

    def test_sorted_by_distance(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        close = SimBaby(position=np.array([4.0, 2.0, 4.5]), params=_SIM_PARAMS)
        farther = SimBaby(position=_POS_B, params=_SIM_PARAMS)
        scene.add_baby(a)
        scene.add_baby(close)
        scene.add_baby(farther)
        n = scene.nearby_babies(_POS_A, radius=5.0)
        assert n[0].entity.id == a.entity.id
        assert n[1].entity.id == close.entity.id
        assert n[2].entity.id == farther.entity.id

    def test_skips_dead_babies(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        dead = SimBaby(position=_POS_B, params=_SIM_PARAMS)
        dead.entity.alive = False
        dead.entity.energy = 0.0
        scene.add_baby(a)
        scene.add_baby(dead)
        n = scene.nearby_babies(_POS_A, radius=3.0)
        assert len(n) == 1
        assert n[0].entity.id == a.entity.id


class TestSocialPerception:
    def test_perceive_detects_nearby_agents(self):
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(position=_POS_B, params=_SIM_PARAMS)
        p = a.perceive(WorldGrid(_SIM_PARAMS.grid_size), babies=[a, b])
        assert len(p.nearby_entities) == 1
        ent = p.nearby_entities[0]
        assert ent["id"] == b.entity.id
        assert ent["type"] == int(EntityType.AGENT)
        assert ent["distance"] == pytest.approx(1.0, abs=1e-6)

    def test_perceive_no_babies_arg_returns_empty(self):
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        p = a.perceive(WorldGrid(_SIM_PARAMS.grid_size))
        assert p.nearby_entities == []

    def test_perceive_excludes_self(self):
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        p = a.perceive(WorldGrid(_SIM_PARAMS.grid_size), babies=[a])
        assert p.nearby_entities == []

    def test_perceive_excludes_far_agents(self):
        a = SimBaby(position=_POS_A, params=_SIM_PARAMS)
        far = SimBaby(position=np.array([0.0, 0.0, 0.0]), params=_SIM_PARAMS)
        p = a.perceive(WorldGrid(_SIM_PARAMS.grid_size), babies=[a, far])
        assert p.nearby_entities == []


class TestSocialMechanics:
    def test_share_energy_transfers(self):
        a = SimBaby(initial_energy=200.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        moved = a.share_energy(b)
        assert moved > 0.0
        assert a.energy == pytest.approx(200.0 - moved, abs=1e-6)
        assert b.energy == pytest.approx(50.0 + moved, abs=1e-6)

    def test_share_energy_conserved(self):
        a = SimBaby(initial_energy=200.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        total_before = a.energy + b.energy
        a.share_energy(b)
        assert a.energy + b.energy == pytest.approx(total_before, abs=1e-6)

    def test_share_energy_dead_target_zero(self):
        a = SimBaby(initial_energy=200.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        b.entity.alive = False
        b.entity.energy = 0.0
        assert a.share_energy(b) == 0.0

    def test_contest_energy_takes_from_weaker(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        taken = a.contest_energy(b)
        assert taken > 0.0
        assert a.energy == pytest.approx(100.0 + taken, abs=1e-6)
        assert b.energy == pytest.approx(50.0 - taken, abs=1e-6)

    def test_contest_energy_no_take_from_stronger(self):
        a = SimBaby(initial_energy=50.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=100.0, position=_POS_B, params=_SIM_PARAMS)
        assert a.contest_energy(b) == 0.0
        assert b.energy == pytest.approx(100.0, abs=1e-6)

    def test_contest_energy_conserved(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        total_before = a.energy + b.energy
        a.contest_energy(b)
        assert a.energy + b.energy == pytest.approx(total_before, abs=1e-6)

    def test_contest_energy_capped_by_target_energy(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=1.0, position=_POS_B, params=_SIM_PARAMS)
        taken = a.contest_energy(b)
        assert taken <= 1.0
        assert b.energy == 0.0


class TestSocialStep:
    def _force_perceptron(self, baby, compete, cooperate):
        baby.perceptron_entity.W[:] = 0.0
        baby.perceptron_entity.b[0] = compete
        baby.perceptron_entity.b[1] = cooperate

    def test_cooperates_when_gate_high_and_surplus(self):
        a = SimBaby(initial_energy=200.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        self._force_perceptron(a, compete=-10.0, cooperate=10.0)
        result = a.social_step(b)
        assert result["act"] == "cooperate"
        assert result["energy_moved"] > 0.0

    def test_contests_when_gate_high(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        self._force_perceptron(a, compete=10.0, cooperate=-10.0)
        result = a.social_step(b)
        assert result["act"] == "contest"
        assert result["energy_moved"] > 0.0

    def test_no_action_when_gates_low(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        self._force_perceptron(a, compete=-10.0, cooperate=-10.0)
        result = a.social_step(b)
        assert result["act"] == "none"
        assert result["energy_moved"] == 0.0

    def test_no_contest_against_stronger(self):
        a = SimBaby(initial_energy=50.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=100.0, position=_POS_B, params=_SIM_PARAMS)
        self._force_perceptron(a, compete=10.0, cooperate=-10.0)
        result = a.social_step(b)
        assert result["act"] == "none"
        assert result["energy_moved"] == 0.0

    def test_no_cooperate_without_surplus(self):
        a = SimBaby(initial_energy=10.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=5.0, position=_POS_B, params=_SIM_PARAMS)
        self._force_perceptron(a, compete=-10.0, cooperate=10.0)
        result = a.social_step(b)
        assert result["act"] == "none"

    def test_dead_neighbor_no_action(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        b.entity.alive = False
        b.entity.energy = 0.0
        self._force_perceptron(a, compete=10.0, cooperate=10.0)
        result = a.social_step(b)
        assert result["act"] == "none"


class TestSimulationSocial:
    def test_step_includes_social_fields(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        results = sim.step()
        assert results[0]["social_act"] == "none"
        assert results[0]["social_energy"] == 0.0

    def test_summary_has_social_stats(self):
        scene = SimScene(params=_SIM_PARAMS)
        scene.spawn_babies(count=1)
        sim = Simulation(scene, max_ticks=10)
        sim.step()
        s = sim.summary()
        assert s["cooperations"] == 0
        assert s["contests"] == 0
        assert s["social_energy_moved"] == 0.0

    def test_two_close_babies_interact(self):
        scene = SimScene(params=_SIM_PARAMS)
        a = SimBaby(initial_energy=200.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        a.perceptron_entity.W[:] = 0.0
        a.perceptron_entity.b[:] = 0.0
        scene.add_baby(a)
        scene.add_baby(b)
        sim = Simulation(scene, max_ticks=5)
        results = sim.run()
        social_acts = {r["social_act"] for r in results}
        assert "cooperate" in social_acts or "contest" in social_acts

    def test_learn_updates_entity_perceptron(self):
        a = SimBaby(initial_energy=100.0, position=_POS_A, params=_SIM_PARAMS)
        b = SimBaby(initial_energy=50.0, position=_POS_B, params=_SIM_PARAMS)
        a.perceive(WorldGrid(_SIM_PARAMS.grid_size), babies=[a, b])
        w_before = a.perceptron_entity.W.copy()
        a.learn(5.0)
        assert not np.array_equal(a.perceptron_entity.W, w_before)
