"""
Comprehensive tests for signal communication — the broadcast channel of the world.

Covers: WorldGrid, WorldParams, SimBaby, SimScene, Simulation,
cell_update_default, Entity, Nest, Perceptron, EpisodicMemory.
"""

from __future__ import annotations

import numpy as np
import pytest

from domains.shell.simulation import (
    BabyAction,
    CellWrite,
    Entity,
    EntityType,
    MATERIAL_AIR,
    MATERIAL_LIVING,
    MATERIAL_ORGANIC,
    MATERIAL_SIGNAL,
    MATERIAL_STONE,
    MATERIAL_WATER,
    Nest,
    Perception,
    Perceptron,
    SimBaby,
    SimScene,
    Simulation,
    WorldGrid,
    WorldParams,
    cell_update_combustion,
    cell_update_conduction,
    cell_update_default,
    cell_update_diffusion,
    cell_update_ember,
    cell_update_energy_conservation,
    cell_update_living,
    cell_update_materials,
    cell_update_metabolism,
    cell_update_temperature,
    cell_update_waves,
    cell_update_water,
    generate_world,
)
from domains.shell.memory import EpisodicMemory, WorldMemory

SIGNAL_IDX = 4  # 5th feature — broadcast strength


def _params(**kw) -> WorldParams:
    defaults = dict(grid_size=(16, 8, 16), wave_speed=0.5, signal_decay=0.1)
    defaults.update(kw)
    return WorldParams(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# WorldParams
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldParams:
    def test_default_values(self):
        p = WorldParams()
        assert p.grid_size == (64, 32, 64)
        assert p.see_radius == 5.0
        assert p.start_energy == 100.0

    def test_cells_input_dim(self):
        assert WorldParams().cells_input_dim == 5

    def test_custom_params(self):
        p = WorldParams(grid_size=(8, 4, 8), see_radius=3.0)
        assert p.grid_size == (8, 4, 8)
        assert p.see_radius == 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# WorldGrid
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldGridBasics:
    def test_idx_roundtrip(self):
        g = WorldGrid((4, 3, 5))
        for x in range(4):
            for y in range(3):
                for z in range(5):
                    i = g.idx(x, y, z)
                    assert g.coords(i) == (x, y, z)

    def test_total_count(self):
        g = WorldGrid((4, 3, 5))
        assert g.total == 60

    def test_reset(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(1, 1, 1, MATERIAL_STONE, energy=10.0)
        g.reset()
        assert g.energy[g.idx(1, 1, 1)] == 0.0
        assert g.material[g.idx(1, 1, 1)] == MATERIAL_AIR

    def test_set_cell(self):
        g = WorldGrid((4, 3, 5))
        cell = g.get_cell(1, 1, 1)
        assert cell.material == MATERIAL_AIR
        g.set_cell(1, 1, 1, cell.__class__(material=MATERIAL_STONE, energy=5.0, temperature=30.0))
        cell2 = g.get_cell(1, 1, 1)
        assert cell2.material == MATERIAL_STONE
        assert cell2.energy == 5.0

    def test_total_energy(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(0, 0, 0, MATERIAL_ORGANIC, energy=10.0)
        assert g.total_energy == pytest.approx(10.0)

    def test_total_signal(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(0, 0, 0, MATERIAL_SIGNAL, energy=5.0)
        assert g.total_signal == pytest.approx(5.0)

    def test_to_dict_and_from_dict(self):
        g = WorldGrid((4, 3, 5))
        g.place_material(1, 1, 1, MATERIAL_STONE, energy=3.0)
        d = g.to_dict()
        g2 = WorldGrid.from_dict(d)
        assert g2.size == (4, 3, 5)
        assert g2.material[g2.idx(1, 1, 1)] == MATERIAL_STONE
        assert g2.energy[g2.idx(1, 1, 1)] == pytest.approx(3.0)

    def test_from_dict_mismatched_size(self):
        g = WorldGrid((4, 3, 5))
        d = g.to_dict()
        d["size"] = [4, 3, 6]  # mismatch
        with pytest.raises(ValueError):
            WorldGrid.from_dict(d)

    def test_wrap_around_idx(self):
        """idx() wraps coordinates, write_cell() does not."""
        g = WorldGrid((4, 3, 5))
        # idx wraps: (4,0,0) on a size-4 grid → (0,0,0)
        assert g.idx(4, 0, 0) == g.idx(0, 0, 0)
        # write_cell checks bounds, so x=4 is out of range
        assert g.write_cell(4, 0, 0, MATERIAL_STONE) is False
        # Directly writing at (0,0,0) works
        g.write_cell(0, 0, 0, MATERIAL_STONE)
        assert g.material[g.idx(0, 0, 0)] == MATERIAL_STONE


class TestSignalEmission:
    def test_write_signal_cell_emits_signal(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=2.5)
        assert g.signal[g.idx(8, 4, 8)] == 2.5

    def test_write_signal_cell_sets_material_and_energy(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=2.5)
        cell = g.get_cell(8, 4, 8)
        assert cell.material == MATERIAL_SIGNAL
        assert cell.energy == 2.5

    def test_write_non_signal_cell_does_not_emit(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_STONE, energy=2.5)
        assert g.signal[g.idx(8, 4, 8)] == 0.0

    def test_place_signal_material_emits_signal(self):
        g = WorldGrid((16, 8, 16))
        g.place_material(4, 2, 4, MATERIAL_SIGNAL, energy=3.0)
        assert g.signal[g.idx(4, 2, 4)] == 3.0

    def test_out_of_bounds_signal_write_ignored(self):
        g = WorldGrid((16, 8, 16))
        assert g.write_cell(-1, 0, 0, MATERIAL_SIGNAL, energy=1.0) is False
        assert g.total_signal == 0.0

    def test_write_cell_returns_true(self):
        g = WorldGrid((16, 8, 16))
        assert g.write_cell(5, 3, 5, MATERIAL_AIR) is True

    def test_signal_accumulates(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=1.0)
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=2.0)
        assert g.signal[g.idx(8, 4, 8)] == 3.0


class TestSignalPerception:
    def test_nearby_cells_include_signal(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=2.0)
        cells = g.get_nearby_cells(8, 4, 8, radius=3.0)
        assert "signal" in cells
        assert cells["signal"].shape == (cells["count"],)
        center = np.where(cells["material"] == MATERIAL_SIGNAL)[0]
        assert cells["signal"][center].max() == 2.0

    def test_empty_read_has_signal_key(self):
        g = WorldGrid((4, 4, 4))
        cells = g.get_nearby_cells(0, 0, 0, radius=-0.5)
        assert cells["count"] == 0
        assert cells["signal"].size == 0

    def test_nearby_cells_all_keys(self):
        g = WorldGrid((8, 8, 8))
        cells = g.get_nearby_cells(4, 4, 4, radius=2.0)
        for key in ["material", "energy", "temperature", "signal", "distance", "count"]:
            assert key in cells

    def test_signal_feature_zero_without_signal(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        feat = baby._perception_features(p)
        assert feat.shape == (baby.params.cells_input_dim,)
        assert feat[SIGNAL_IDX] == 0.0

    def test_signal_feature_positive_near_source(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        world.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=1.0)
        p = baby.perceive(world)
        feat = baby._perception_features(p)
        assert 0.0 < feat[SIGNAL_IDX] <= 1.0

    def test_signal_feature_learning_updates_signal_row(self):
        params = _params()
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        world = WorldGrid((16, 8, 16))
        world.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=1.0)
        baby.perceive(world)
        before = baby.perceptron_cells.W.copy()
        baby.learn(5.0)
        assert not np.array_equal(baby.perceptron_cells.W[SIGNAL_IDX],
                                  before[SIGNAL_IDX])

    def test_signal_row_untouched_without_signal(self):
        params = _params()
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        world = WorldGrid((16, 8, 16))
        baby.perceive(world)
        before = baby.perceptron_cells.W.copy()
        baby.learn(5.0)
        assert np.array_equal(baby.perceptron_cells.W[SIGNAL_IDX],
                              before[SIGNAL_IDX])


class TestSignalPropagation:
    def test_waves_carry_written_signal_to_neighbors(self):
        g = WorldGrid((16, 8, 16))
        params = _params()
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=10.0)
        cell_update_default(g, params)
        neighbor = g.signal[g.idx(9, 4, 8)]
        assert neighbor > 0.0
        assert neighbor < 10.0  # partial transfer, not full jump

    def test_signal_reaches_distance_over_ticks(self):
        g = WorldGrid((16, 8, 16))
        params = _params()
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=10.0)
        for _ in range(6):
            cell_update_default(g, params)
        assert g.signal[g.idx(12, 4, 8)] > 0.0  # 4 cells away after several ticks

    def test_signal_does_not_emit_from_air_writes(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_AIR, energy=1.0)
        assert g.total_signal == 0.0

    def test_wave_speed_zero_no_propagation(self):
        g = WorldGrid((4, 4, 4))
        params = _params(wave_speed=0.0)
        g.write_cell(2, 2, 2, MATERIAL_SIGNAL, energy=5.0)
        cell_update_waves(g, params)
        # Only the source cell has signal
        assert g.signal[g.idx(2, 2, 2)] == 5.0


class TestCellUpdates:
    def test_diffusion_spreads_energy(self):
        g = WorldGrid((8, 8, 8))
        g.energy[g.idx(4, 4, 4)] = 100.0
        params = _params(diffusion_rate=0.5)
        total_before = g.total_energy
        cell_update_diffusion(g, params)
        # Total energy should be conserved
        assert g.total_energy == pytest.approx(total_before)
        # Source cell should have less energy
        assert g.energy[g.idx(4, 4, 4)] < 100.0

    def test_combustion_ignites_organic(self):
        g = WorldGrid((4, 4, 4))
        params = _params(ignition_temp=100.0)
        g.material[g.idx(2, 2, 2)] = MATERIAL_ORGANIC
        g.temperature[g.idx(2, 2, 2)] = 150.0
        cell_update_combustion(g, params)
        from domains.shell.simulation import MATERIAL_EMBER
        assert g.material[g.idx(2, 2, 2)] == MATERIAL_EMBER

    def test_combustion_no_ignite_below_temp(self):
        g = WorldGrid((4, 4, 4))
        params = _params(ignition_temp=100.0)
        g.material[g.idx(2, 2, 2)] = MATERIAL_ORGANIC
        g.temperature[g.idx(2, 2, 2)] = 50.0
        cell_update_combustion(g, params)
        assert g.material[g.idx(2, 2, 2)] == MATERIAL_ORGANIC

    def test_metabolism_reduces_energy(self):
        g = WorldGrid((4, 4, 4))
        params = _params(organic_metabolism=0.1)
        g.material[g.idx(2, 2, 2)] = MATERIAL_ORGANIC
        g.energy[g.idx(2, 2, 2)] = 100.0
        cell_update_metabolism(g, params)
        assert g.energy[g.idx(2, 2, 2)] < 100.0

    def test_metabolism_zero_rate_noop(self):
        g = WorldGrid((4, 4, 4))
        params = _params(organic_metabolism=0.0)
        g.material[g.idx(2, 2, 2)] = MATERIAL_ORGANIC
        g.energy[g.idx(2, 2, 2)] = 100.0
        cell_update_metabolism(g, params)
        assert g.energy[g.idx(2, 2, 2)] == 100.0

    def test_water_dampens_signal(self):
        g = WorldGrid((4, 4, 4))
        params = _params(water_signal_dampen=0.5)
        g.material[g.idx(2, 2, 2)] = MATERIAL_WATER
        g.signal[g.idx(2, 2, 2)] = 10.0
        cell_update_water(g, params)
        assert g.signal[g.idx(2, 2, 2)] == 5.0

    def test_temperature_relaxes_to_ambient(self):
        g = WorldGrid((4, 4, 4))
        params = _params(ambient_temp=20.0, ambient_cooling=0.1)
        g.temperature[:] = 100.0
        cell_update_temperature(g, params)
        assert g.temperature[0] < 100.0

    def test_energy_conservation(self):
        g = WorldGrid((4, 4, 4))
        params = _params(energy_loss=0.1)
        g.energy[:] = 100.0
        cell_update_energy_conservation(g, params)
        assert g.energy[0] == pytest.approx(90.0)

    def test_energy_clamps_to_zero(self):
        g = WorldGrid((4, 4, 4))
        params = _params(energy_loss=1.0)
        g.energy[:] = 10.0
        cell_update_energy_conservation(g, params)
        assert g.energy[0] >= 0.0

    def test_material_update_chain(self):
        g = WorldGrid((4, 4, 4))
        params = _params()
        cell_update_materials(g, params)  # Should not crash

    def test_no_water_noop(self):
        g = WorldGrid((4, 4, 4))
        params = _params()
        g.signal[:] = 5.0
        cell_update_water(g, params)
        # No water cells, so signal should be unchanged
        assert g.signal[0] == 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# Entity
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntity:
    def test_distance_to(self):
        a = Entity(position=np.array([0.0, 0.0, 0.0]))
        b = Entity(position=np.array([3.0, 0.0, 0.0]))
        assert a.distance_to(b) == pytest.approx(3.0)

    def test_distance_to_point(self):
        e = Entity(position=np.array([0.0, 0.0, 0.0]))
        assert e.distance_to_point(np.array([4.0, 0.0, 0.0])) == pytest.approx(4.0)

    def test_to_dict_and_from_dict(self):
        e = Entity(id=1, position=np.array([1.0, 2.0, 3.0]), energy=50.0,
                    entity_type=EntityType.AGENT, alive=True)
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.id == 1
        assert np.allclose(e2.position, [1.0, 2.0, 3.0])
        assert e2.energy == 50.0
        assert e2.entity_type == EntityType.AGENT
        assert e2.alive is True


# ═══════════════════════════════════════════════════════════════════════════════
# Nest
# ═══════════════════════════════════════════════════════════════════════════════

class TestNest:
    def test_distance_to_point(self):
        n = Nest(id=1, position=np.array([5.0, 0.0, 5.0]),
                 stored_energy=100.0, owner_group_id=0)
        assert n.distance_to_point(np.array([8.0, 0.0, 5.0])) == pytest.approx(3.0)

    def test_to_dict_and_from_dict(self):
        n = Nest(id=1, position=np.array([2.0, 1.0, 3.0]),
                 stored_energy=50.0, owner_group_id=2, alive=True)
        d = n.to_dict()
        n2 = Nest.from_dict(d)
        assert n2.id == 1
        assert np.allclose(n2.position, [2.0, 1.0, 3.0])
        assert n2.stored_energy == 50.0
        assert n2.owner_group_id == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Perceptron
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerceptron:
    def test_forward_shape(self):
        p = Perceptron(5, 3)
        out = p.forward(np.zeros(5))
        assert out.shape == (3,)

    def test_forward_range(self):
        p = Perceptron(5, 3)
        out = p.forward(np.random.randn(5))
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_update_changes_weights(self):
        p = Perceptron(5, 3)
        old_W = p.W.copy()
        old_b = p.b.copy()
        p.update(np.ones(5), np.ones(3), lr=0.1)
        assert not np.array_equal(p.W, old_W) or not np.array_equal(p.b, old_b)

    def test_to_dict_and_from_dict(self):
        p = Perceptron(5, 3)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        assert np.allclose(p.W, p2.W)
        assert np.allclose(p.b, p2.b)

    def test_hidden_layer(self):
        p = Perceptron(5, 3, hidden_units=4)
        assert p.hidden_units == 4
        assert p.H is not None
        out = p.forward(np.zeros(5))
        assert out.shape == (3,)

    def test_hidden_to_dict_roundtrip(self):
        p = Perceptron(5, 3, hidden_units=4)
        d = p.to_dict()
        assert "H" in d
        p2 = Perceptron.from_dict(d)
        assert p2.hidden_units == 4
        assert np.allclose(p.H, p2.H)

    def test_sigmoid_clipping(self):
        """Forward pass should not overflow."""
        p = Perceptron(5, 3)
        out = p.forward(np.array([1000.0, -1000.0, 0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(out))


# ═══════════════════════════════════════════════════════════════════════════════
# SimBaby
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimBaby:
    def test_initial_state(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        assert baby.energy == 100.0
        assert baby.alive
        assert baby.group_id == 0

    def test_feel_positive(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        delta = baby.feel(90.0)
        assert delta == pytest.approx(10.0)

    def test_feel_negative(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        delta = baby.feel(110.0)
        assert delta == pytest.approx(-10.0)

    def test_perception_features_shape(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        feat = baby._perception_features(p)
        assert feat.shape == (baby.params.cells_input_dim,)

    def test_perception_features_clipped(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        world.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=100.0)
        p = baby.perceive(world)
        feat = baby._perception_features(p)
        assert np.all(feat >= 0.0) and np.all(feat <= 1.0)

    def test_react_still_when_low_energy(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                       initial_energy=5.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        action = baby.react(p, 0.0)
        assert len(action.writes) == 0

    def test_react_writes_when_energy(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        action = baby.react(p, 0.0)
        assert isinstance(action, BabyAction)

    def test_apply_action(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        action = BabyAction(writes=[
            CellWrite(8, 4, 8, MATERIAL_STONE, 1.0),
            CellWrite(9, 4, 8, MATERIAL_STONE, 2.0),
        ])
        count = baby.apply_action(action, world)
        assert count == 2
        assert world.energy[world.idx(8, 4, 8)] == 1.0

    def test_apply_action_out_of_bounds(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        action = BabyAction(writes=[CellWrite(-1, 0, 0, MATERIAL_STONE)])
        count = baby.apply_action(action, world)
        assert count == 0

    def test_decide_move_low_energy(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                       initial_energy=5.0)
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        direction = baby.decide_move(p)
        assert direction is None

    def test_decide_move_high_energy(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        p = baby.perceive(world)
        direction = baby.decide_move(p)
        assert direction is not None
        assert direction.shape == (3,)

    def test_share_energy(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=200.0)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=50.0)
        transfer = a.share_energy(b)
        assert transfer > 0
        assert a.energy < 200.0
        assert b.energy > 50.0

    def test_share_energy_dead_target(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        b.entity.alive = False
        assert a.share_energy(b) == 0.0

    def test_contest_energy(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=200.0)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=50.0)
        taken = a.contest_energy(b)
        assert taken > 0
        assert a.energy > 200.0
        assert b.energy < 50.0

    def test_contest_energy_no_take_when_weaker(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=50.0)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=200.0)
        taken = a.contest_energy(b)
        assert taken == 0.0

    def test_contest_energy_dead_target(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        b.entity.alive = False
        assert a.contest_energy(b) == 0.0

    def test_absorb_energy(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        world.place_material(8, 4, 8, MATERIAL_ORGANIC, energy=10.0)
        absorbed = baby.absorb_energy(world)
        assert absorbed > 0

    def test_absorb_energy_no_organic(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        absorbed = baby.absorb_energy(world)
        assert absorbed == 0.0

    def test_alive_property(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        assert baby.alive
        baby.entity.alive = False
        assert not baby.alive

    def test_info(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        info = baby.info()
        assert "id" in info
        assert "energy" in info
        assert "alive" in info

    def test_to_dict_and_from_dict(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        d = baby.to_dict()
        baby2 = SimBaby.from_dict(d, params=_params())
        assert baby2.entity.energy == baby.entity.energy
        assert baby2._total_ticks == baby._total_ticks

    def test_entity_features(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        entity = {"type": 0, "energy": 100.0, "distance": 2.0, "angle": 0.5,
                  "group_id": 0, "message": 0.0}
        feat = baby._entity_features(entity)
        assert feat.shape == (baby.params.entity_input_dim,)

    def test_distance_to_point(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        d = baby.distance_to_point(np.array([11.0, 4.0, 8.0]))
        assert d == pytest.approx(3.0)

    def test_position_property(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        assert np.allclose(baby.position, [8.0, 4.0, 8.0])

    def test_energy_property(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        assert baby.energy == 100.0

    def test_tick_count(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        assert baby.tick_count == 0


class TestSimBabyLearn:
    def test_learn_positive(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        baby.perceive(world)
        baby.react(baby._last_perception, 0.0)
        baby.learn(5.0)
        assert baby._last_learning is not None

    def test_learn_negative(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        baby.perceive(world)
        baby.react(baby._last_perception, 0.0)
        baby.learn(-5.0)
        assert baby._last_learning is not None

    def test_learn_records_memory(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        baby.perceive(world)
        baby.react(baby._last_perception, 0.0)
        baby.learn(2.0)
        assert len(baby.memory) > 0

    def test_recall_memories(self):
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params())
        world = WorldGrid((16, 8, 16))
        baby.perceive(world)
        baby.react(baby._last_perception, 0.0)
        baby.learn(2.0)
        memories = baby.recall_memories(k=1)
        assert len(memories) >= 1
        assert "features" in memories[0]


# ═══════════════════════════════════════════════════════════════════════════════
# SimScene
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimScene:
    def test_add_baby(self):
        params = _params()
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        assert len(scene.babies) == 1

    def test_update_cells(self):
        params = _params()
        scene = SimScene(params)
        scene.update_cells()  # Should not crash

    def test_info(self):
        params = _params()
        scene = SimScene(params)
        info = scene.info()
        assert "tick" in info
        assert "total_energy" in info

    def test_to_dict_and_from_dict(self):
        params = _params()
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        scene._tick = 5
        d = scene.to_dict()
        scene2 = SimScene.from_dict(d)
        assert scene2.tick == 5
        assert len(scene2.babies) == 1

    def test_get_baby(self):
        params = _params()
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        assert scene.get_baby(baby.entity.id) is baby

    def test_get_baby_missing(self):
        params = _params()
        scene = SimScene(params)
        assert scene.get_baby(9999) is None

    def test_nearby_babies(self):
        params = _params()
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([9.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)
        nearby = scene.nearby_babies(np.array([8.0, 4.0, 8.0]), radius=5.0)
        assert len(nearby) == 2

    def test_nearby_babies_excludes_dead(self):
        params = _params()
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b.entity.alive = False
        scene.add_baby(a)
        scene.add_baby(b)
        nearby = scene.nearby_babies(np.array([8.0, 4.0, 8.0]), radius=5.0,
                                     exclude_id=a.entity.id)
        assert len(nearby) == 0

    def test_alive_babies(self):
        params = _params()
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        assert len(scene.alive_babies) == 1
        a.entity.alive = False
        assert len(scene.alive_babies) == 0

    def test_nearest_nest(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n = Nest(id=1, position=np.array([8.0, 4.0, 8.0]),
                 stored_energy=50.0, owner_group_id=0)
        scene.nests.append(n)
        found = scene.nearest_nest(np.array([8.0, 4.0, 8.0]), radius=5.0)
        assert found is n

    def test_nearest_nest_out_of_range(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n = Nest(id=1, position=np.array([0.0, 0.0, 0.0]),
                 stored_energy=50.0, owner_group_id=0)
        scene.nests.append(n)
        found = scene.nearest_nest(np.array([15.0, 7.0, 15.0]), radius=1.0)
        assert found is None


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulation:
    def test_run_returns_results(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params,
                        initial_energy=100.0)
        scene.add_baby(baby)
        results = Simulation(scene, max_ticks=3).run()
        assert results
        assert scene.tick == 3

    def test_summary_empty(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        sim = Simulation(scene, max_ticks=0)
        sim.run()
        summary = sim.summary()
        assert summary["total_ticks"] == 0

    def test_summary_with_data(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params,
                        initial_energy=100.0)
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        summary = sim.summary()
        assert summary["total_ticks"] == 2

    def test_stop(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=100)
        sim._running = True
        sim.stop()
        assert not sim._running

    def test_tick_log(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=1)
        sim.run()
        assert len(sim.tick_log) == 1

    def test_step_returns_per_baby(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([9.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)
        sim = Simulation(scene, max_ticks=1)
        results = sim.step()
        assert len(results) == 2

    def test_single_baby_no_crash(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        results = Simulation(scene, max_ticks=1).run()
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# EpisodicMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpisodicMemory:
    def test_record_and_recall(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(5), (0.5, 0.5, 0.5), 1.0, tick=1)
        episodes = em.recall(1)
        assert len(episodes) == 1
        assert episodes[0].reward == 1.0

    def test_capacity_overflow(self):
        em = EpisodicMemory(capacity=3)
        for i in range(5):
            em.record(np.zeros(3), (0.5,), float(i), tick=i)
        assert len(em) == 3

    def test_mean_reward(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 2.0)
        em.record(np.zeros(3), (0.5,), 4.0)
        assert em.mean_reward(2) == pytest.approx(3.0)

    def test_mean_reward_empty(self):
        em = EpisodicMemory(capacity=10)
        assert em.mean_reward() == 0.0

    def test_is_full(self):
        em = EpisodicMemory(capacity=2)
        assert not em.is_full
        em.record(np.zeros(3), (0.5,), 1.0)
        em.record(np.zeros(3), (0.5,), 2.0)
        assert em.is_full

    def test_stats(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 1.0, tick=5)
        s = em.stats()
        assert s["capacity"] == 10
        assert s["size"] == 1

    def test_to_dict_and_from_dict(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.ones(3), (0.5, 0.5), 3.0, tick=7)
        d = em.to_dict()
        em2 = EpisodicMemory.from_dict(d)
        assert em2.capacity == 10
        episodes = em2.recall(1)
        assert episodes[0].reward == 3.0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            EpisodicMemory(capacity=0)

    def test_recall_by_reward(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 1.0)
        em.record(np.zeros(3), (0.5,), 5.0)
        em.record(np.zeros(3), (0.5,), 3.0)
        best = em.recall(1, by_reward=True)
        assert best[0].reward == 5.0

    def test_recall_negative_k(self):
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 1.0)
        result = em.recall(-1)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Communication Loop (integration)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunicationLoop:
    def test_baby_b_perceives_baby_a_broadcast(self):
        params = _params()
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)

        action = BabyAction(writes=[
            CellWrite(8, 4, 8, MATERIAL_SIGNAL, 1.0),
            CellWrite(9, 4, 8, MATERIAL_SIGNAL, 1.0),
        ])
        a.apply_action(action, scene.world)
        scene.update_cells()

        feat = b._perception_features(b.perceive(scene.world))
        assert feat[SIGNAL_IDX] > 0.0

    def test_communication_survives_scene_snapshot(self):
        params = _params()
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)
        a.apply_action(BabyAction(writes=[
            CellWrite(8, 4, 8, MATERIAL_SIGNAL, 1.0),
        ]), scene.world)
        scene.update_cells()

        restored = SimScene.from_dict(scene.to_dict())
        assert np.array_equal(restored.world.signal, scene.world.signal)
        feat = restored.babies[1]._perception_features(
            restored.babies[1].perceive(restored.world)
        )
        assert feat[SIGNAL_IDX] > 0.0

    def test_full_tick_loop_emits_and_senses(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params,
                        initial_energy=100.0)
        scene.add_baby(baby)
        scene.world.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=2.0)
        results = Simulation(scene, max_ticks=3).run()
        assert results
        assert scene.world.total_signal > 0.0
        assert scene.tick == 3


# ═══════════════════════════════════════════════════════════════════════════════
# World Generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldGeneration:
    def test_generate_world_has_floor(self):
        params = _params(grid_size=(16, 4, 16), generate_world=True)
        g = WorldGrid(params.grid_size)
        generate_world(g, params, seed=42)
        # y=0 should be stone or ember (ember vents are buried in the floor)
        from domains.shell.simulation import MATERIAL_EMBER
        for x in range(16):
            for z in range(16):
                mat = g.material[g.idx(x, 0, z)]
                assert mat in (MATERIAL_STONE, MATERIAL_EMBER)

    def test_generate_world_deterministic(self):
        params = _params(grid_size=(8, 4, 8))
        g1 = WorldGrid((8, 4, 8))
        g2 = WorldGrid((8, 4, 8))
        generate_world(g1, params, seed=42)
        generate_world(g2, params, seed=42)
        assert np.array_equal(g1.material, g2.material)

    def test_generate_world_flat_grid(self):
        params = _params(grid_size=(8, 1, 8))
        g = WorldGrid((8, 1, 8))
        generate_world(g, params, seed=42)
        # Only 1 y-layer, should be stone floor
        assert g.material[g.idx(0, 0, 0)] == MATERIAL_STONE


# ═══════════════════════════════════════════════════════════════════════════════
# WorldMemory
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorldMemory:
    def test_record_and_recall(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0, tick=1, group_id=0, donor_id=1)
        episodes = wm.recall(1)
        assert len(episodes) == 1
        assert episodes[0].reward == 1.0

    def test_recall_by_reward(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0)
        wm.record(np.zeros(3), (0.5,), 5.0)
        wm.record(np.zeros(3), (0.5,), 3.0)
        best = wm.recall(1, by_reward=True)
        assert best[0].reward == 5.0

    def test_recall_chronological(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0, tick=1)
        wm.record(np.zeros(3), (0.5,), 2.0, tick=2)
        wm.record(np.zeros(3), (0.5,), 3.0, tick=3)
        recent = wm.recall(2, by_reward=False)
        assert len(recent) == 2
        assert recent[-1].tick == 3

    def test_recall_by_group_id(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0, group_id=0)
        wm.record(np.zeros(3), (0.5,), 2.0, group_id=1)
        wm.record(np.zeros(3), (0.5,), 3.0, group_id=0)
        group0 = wm.recall(10, group_id=0)
        assert len(group0) == 2
        assert all(e.group_id == 0 for e in group0)

    def test_recall_empty(self):
        wm = WorldMemory()
        assert wm.recall(5) == []

    def test_recall_negative_k(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0)
        assert wm.recall(-1) == []

    def test_consolidate(self):
        wm = WorldMemory()
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 1.0, tick=1)
        em.record(np.zeros(3), (0.5,), 5.0, tick=2)
        count = wm.consolidate(em, k=2, group_id=1, donor_id=99)
        assert count == 2
        assert len(wm) == 2

    def test_consolidate_zero_k(self):
        wm = WorldMemory()
        em = EpisodicMemory(capacity=10)
        em.record(np.zeros(3), (0.5,), 1.0)
        count = wm.consolidate(em, k=0)
        assert count == 0

    def test_mean_reward(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 2.0)
        wm.record(np.zeros(3), (0.5,), 4.0)
        assert wm.mean_reward(2) == pytest.approx(3.0)

    def test_mean_reward_empty(self):
        wm = WorldMemory()
        assert wm.mean_reward() == 0.0

    def test_len(self):
        wm = WorldMemory()
        assert len(wm) == 0
        wm.record(np.zeros(3), (0.5,), 1.0)
        assert len(wm) == 1

    def test_stats(self):
        wm = WorldMemory()
        wm.record(np.zeros(3), (0.5,), 1.0, tick=5, group_id=2)
        s = wm.stats()
        assert s["size"] == 1
        assert s["oldest_tick"] == 5
        assert s["newest_tick"] == 5
        assert 2 in s["groups"]

    def test_stats_empty(self):
        wm = WorldMemory()
        s = wm.stats()
        assert s["size"] == 0
        assert s["oldest_tick"] == 0

    def test_to_dict_and_from_dict(self):
        wm = WorldMemory()
        wm.record(np.ones(3), (0.5, 0.5), 3.0, tick=7, group_id=1, donor_id=2)
        d = wm.to_dict()
        wm2 = WorldMemory.from_dict(d)
        assert len(wm2) == 1
        episodes = wm2.recall(1)
        assert episodes[0].reward == 3.0
        assert episodes[0].group_id == 1

    def test_from_dict_empty(self):
        wm = WorldMemory.from_dict({"episodes": []})
        assert len(wm) == 0

    def test_initial_with_episodes(self):
        from domains.shell.memory import WorldEpisode
        ep = WorldEpisode(features=np.zeros(3), action=(0.5,), reward=1.0, tick=0)
        wm = WorldMemory(episodes=[ep])
        assert len(wm) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SimBaby advanced
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimBabyAdvanced:
    def test_share_energy_self_has_no_surplus(self):
        """share_energy uses share_fraction of self.energy, not surplus check."""
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=50.0)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=50.0)
        transfer = a.share_energy(b)
        # transfer = min(50 * 0.1, 100) = 5.0
        assert transfer == pytest.approx(5.0)

    def test_contest_energy_equal(self):
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=100.0)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=_params(),
                     initial_energy=100.0)
        taken = a.contest_energy(b)
        assert taken == 0.0

    def test_spawn_child_copies_weights(self):
        params = _params()
        parent = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        child = parent.spawn_child(np.array([9.0, 4.0, 8.0]))
        assert np.allclose(parent.perceptron_cells.W, child.perceptron_cells.W)

    def test_spawn_child_inherits_group(self):
        params = _params()
        parent = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params, group_id=3)
        child = parent.spawn_child(np.array([9.0, 4.0, 8.0]))
        assert child.group_id == 3

    def test_social_step_none_act(self):
        params = _params()
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b.entity.alive = False
        result = a.social_step(b)
        assert result["act"] == "none"
        assert result["energy_moved"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SimScene advanced
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimSceneAdvanced:
    def test_deliver_messages(self):
        params = _params(message_enabled=True)
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)
        scene._pending_messages.append((a.entity.id, b.entity.id, 0.8))
        scene.deliver_messages()
        assert a.entity.id in b._inbox
        assert b._inbox[a.entity.id] == 0.8

    def test_deliver_messages_dead_target(self):
        params = _params(message_enabled=True)
        scene = SimScene(params)
        a = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        b = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(a)
        scene.add_baby(b)
        b.entity.alive = False
        scene._pending_messages.append((a.entity.id, b.entity.id, 0.5))
        scene.deliver_messages()
        assert a.entity.id not in b._inbox

    def test_nearest_nest_with_group_filter(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n1 = Nest(id=1, position=np.array([8.0, 4.0, 8.0]),
                   stored_energy=50.0, owner_group_id=0)
        n2 = Nest(id=2, position=np.array([8.0, 4.0, 8.0]),
                   stored_energy=50.0, owner_group_id=1)
        scene.nests.extend([n1, n2])
        found = scene.nearest_nest(np.array([8.0, 4.0, 8.0]), radius=5.0, group_id=0)
        assert found is n1

    def test_update_nests_decay(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n = Nest(id=1, position=np.array([8.0, 4.0, 8.0]),
                 stored_energy=100.0, owner_group_id=0)
        scene.nests.append(n)
        scene.update_nests()
        assert n.stored_energy < 100.0

    def test_update_nests_removes_empty(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n = Nest(id=1, position=np.array([8.0, 4.0, 8.0]),
                 stored_energy=1e-12, owner_group_id=0)
        scene.nests.append(n)
        scene.update_nests()
        assert len(scene.nests) == 0

    def test_draw_nest_disabled(self):
        params = _params(structure_enabled=False)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        assert scene.draw_nest(baby) == 0.0

    def test_draw_nest_no_nearby(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        baby.entity.energy = 50.0
        assert scene.draw_nest(baby) == 0.0

    def test_draw_nest_has_enough_energy(self):
        params = _params(structure_enabled=True)
        scene = SimScene(params)
        n = Nest(id=1, position=np.array([8.0, 4.0, 8.0]),
                 stored_energy=100.0, owner_group_id=0)
        scene.nests.append(n)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        baby.entity.energy = 150.0  # above start_energy
        assert scene.draw_nest(baby) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation advanced
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulationAdvanced:
    def test_summary_has_all_keys(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=2)
        sim.run()
        summary = sim.summary()
        for key in ["total_ticks", "total_baby_ticks", "avg_energy",
                     "total_cells_written", "cooperations", "contests",
                     "deaths", "alive_count"]:
            assert key in summary

    def test_run_multiple_ticks(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params,
                        initial_energy=200.0)
        scene.add_baby(baby)
        results = Simulation(scene, max_ticks=5).run()
        assert len(results) == 5

    def test_step_returns_per_baby_dict_keys(self):
        params = _params(start_agents=0)
        scene = SimScene(params)
        baby = SimBaby(position=np.array([8.0, 4.0, 8.0]), params=params)
        scene.add_baby(baby)
        sim = Simulation(scene, max_ticks=1)
        results = sim.step()
        assert len(results) == 1
        r = results[0]
        for key in ["baby_id", "tick", "energy", "energy_delta",
                     "cells_written", "moved", "alive"]:
            assert key in r
