"""Tests for domains.shell.simulation — EntityType, WorldParams, Perception, CellWrite, BabyAction, Perceptron, WorldCell, WorldGrid, Entity, Nest."""

import numpy as np
import pytest
from domains.shell.simulation import (
    EntityType, WorldParams, Perception, CellWrite, BabyAction, Perceptron,
    WorldCell, WorldGrid, Entity, Nest, MATERIAL_AIR, MATERIAL_WATER,
    MATERIAL_STONE, MATERIAL_ORGANIC, MATERIAL_METAL, MATERIAL_EMBER,
    MATERIAL_LIVING, MATERIAL_SIGNAL, NUM_MATERIALS,
)


# ── EntityType ────────────────────────────────────────────────────────────────

class TestEntityType:
    def test_all_members(self):
        assert len(EntityType) == 5

    def test_values(self):
        assert EntityType.AGENT.value == 0
        assert EntityType.OBJECT.value == 1
        assert EntityType.LIGHT.value == 2
        assert EntityType.EFFECTOR.value == 3
        assert EntityType.SENSOR.value == 4

    def test_member_names(self):
        names = [e.name for e in EntityType]
        assert "AGENT" in names
        assert "OBJECT" in names
        assert "LIGHT" in names
        assert "EFFECTOR" in names
        assert "SENSOR" in names

    def test_is_int_enum(self):
        assert isinstance(EntityType.AGENT, int)

    def test_iteration(self):
        members = list(EntityType)
        assert len(members) == 5

    def test_comparison(self):
        assert EntityType.AGENT == 0
        assert EntityType.OBJECT != 0

    def test_unique_values(self):
        values = [e.value for e in EntityType]
        assert len(values) == len(set(values))


# ── Material Constants ────────────────────────────────────────────────────────

class TestMaterialConstants:
    def test_num_materials(self):
        assert NUM_MATERIALS == 8

    def test_material_values_distinct(self):
        vals = [MATERIAL_AIR, MATERIAL_WATER, MATERIAL_STONE, MATERIAL_ORGANIC,
                MATERIAL_METAL, MATERIAL_EMBER, MATERIAL_LIVING, MATERIAL_SIGNAL]
        assert len(vals) == len(set(vals))

    def test_material_air_is_zero(self):
        assert MATERIAL_AIR == 0

    def test_material_range(self):
        assert 0 <= MATERIAL_AIR < NUM_MATERIALS
        assert 0 <= MATERIAL_SIGNAL < NUM_MATERIALS


# ── WorldParams ───────────────────────────────────────────────────────────────

class TestWorldParams:
    def test_defaults(self):
        wp = WorldParams()
        assert wp.grid_size == (64, 32, 64)
        assert wp.start_energy == 100.0
        assert wp.start_agents == 4
        assert wp.social_enabled is True
        assert wp.message_enabled is False

    def test_custom(self):
        wp = WorldParams(grid_size=(32, 16, 32), start_energy=50.0)
        assert wp.grid_size == (32, 16, 32)
        assert wp.start_energy == 50.0

    def test_tick_rate_default(self):
        wp = WorldParams()
        assert wp.tick_rate == 0.1

    def test_see_radius_default(self):
        wp = WorldParams()
        assert wp.see_radius == 5.0

    def test_diffusion_rate_default(self):
        wp = WorldParams()
        assert wp.diffusion_rate == 0.1

    def test_structure_disabled_by_default(self):
        wp = WorldParams()
        assert wp.structure_enabled is False

    def test_lifecycle_disabled_by_default(self):
        wp = WorldParams()
        assert wp.lifecycle_enabled is False

    def test_solar_disabled_by_default(self):
        wp = WorldParams()
        assert wp.solar_enabled is False

    def test_memory_capacity_default(self):
        wp = WorldParams()
        assert wp.memory_capacity == 64

    def test_brain_hidden_units_default(self):
        wp = WorldParams()
        assert wp.brain_hidden_units == 0

    def test_social_params(self):
        wp = WorldParams()
        assert wp.social_radius == 3.0
        assert wp.share_fraction == 0.1
        assert wp.contest_take == 2.0

    def test_message_params(self):
        wp = WorldParams()
        assert wp.message_cost == 0.5
        assert wp.message_range == 5.0

    def test_nest_params(self):
        wp = WorldParams()
        assert wp.nest_radius == 2.0
        assert wp.nest_seed_energy == 3.0
        assert wp.max_nests == 8

    def test_temperature_params(self):
        wp = WorldParams()
        assert wp.ambient_temp == 20.0
        assert wp.ignition_temp == 100.0
        assert wp.burn_temp == 150.0

    def test_perception_dims(self):
        wp = WorldParams()
        assert wp.cells_input_dim == 5
        assert wp.body_input_dim == 3
        assert wp.entity_input_dim == 5

    def test_custom_energy_loss(self):
        wp = WorldParams(energy_loss=0.05)
        assert wp.energy_loss == 0.05

    def test_custom_solar(self):
        wp = WorldParams(solar_enabled=True, solar_day_ticks=48)
        assert wp.solar_enabled is True
        assert wp.solar_day_ticks == 48

    def test_specialization_disabled_by_default(self):
        wp = WorldParams()
        assert wp.specialization_enabled is False


# ── WorldCell ─────────────────────────────────────────────────────────────────

class TestWorldCell:
    def test_defaults(self):
        c = WorldCell()
        assert c.material == MATERIAL_AIR
        assert c.energy == 0.0
        assert c.temperature == 20.0

    def test_custom(self):
        c = WorldCell(material=MATERIAL_WATER, energy=50.0, temperature=80.0)
        assert c.material == MATERIAL_WATER
        assert c.energy == 50.0
        assert c.temperature == 80.0

    def test_from_world_grid_get_cell(self):
        grid = WorldGrid(size=(4, 4, 4))
        grid.place_material(1, 2, 3, MATERIAL_STONE, energy=10.0, temperature=50.0)
        c = grid.get_cell(1, 2, 3)
        assert c.material == MATERIAL_STONE
        assert c.energy == 10.0
        assert c.temperature == 50.0


# ── WorldGrid ─────────────────────────────────────────────────────────────────

class TestWorldGrid:
    def test_init_default(self):
        g = WorldGrid()
        assert g.size == (64, 32, 64)
        assert g.total == 64 * 32 * 64

    def test_init_custom(self):
        g = WorldGrid(size=(4, 4, 4))
        assert g.size == (4, 4, 4)
        assert g.total == 64

    def test_reset(self):
        g = WorldGrid(size=(4, 4, 4))
        g.place_material(0, 0, 0, MATERIAL_STONE, energy=100.0)
        g.reset()
        assert g.energy[0] == 0.0
        assert g.material[0] == MATERIAL_AIR

    def test_idx_wraparound(self):
        g = WorldGrid(size=(4, 4, 4))
        i1 = g.idx(0, 0, 0)
        i2 = g.idx(4, 0, 0)
        assert i1 == i2

    def test_idx_negative_wraparound(self):
        g = WorldGrid(size=(4, 4, 4))
        i1 = g.idx(0, 0, 0)
        i2 = g.idx(-4, 0, 0)
        assert i1 == i2

    def test_coords_roundtrip(self):
        g = WorldGrid(size=(4, 4, 4))
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    i = g.idx(x, y, z)
                    cx, cy, cz = g.coords(i)
                    assert (cx, cy, cz) == (x, y, z)

    def test_get_set_cell(self):
        g = WorldGrid(size=(4, 4, 4))
        cell = WorldCell(material=MATERIAL_WATER, energy=25.0, temperature=30.0)
        g.set_cell(1, 2, 3, cell)
        c = g.get_cell(1, 2, 3)
        assert c.material == MATERIAL_WATER
        assert c.energy == 25.0

    def test_place_material(self):
        g = WorldGrid(size=(4, 4, 4))
        g.place_material(2, 1, 0, MATERIAL_ORGANIC, energy=50.0, temperature=90.0)
        c = g.get_cell(2, 1, 0)
        assert c.material == MATERIAL_ORGANIC
        assert c.energy == 50.0
        assert c.temperature == 90.0

    def test_write_cell_success(self):
        g = WorldGrid(size=(4, 4, 4))
        ok = g.write_cell(1, 1, 1, MATERIAL_METAL, energy=10.0)
        assert ok is True
        assert g.get_cell(1, 1, 1).material == MATERIAL_METAL

    def test_write_cell_out_of_bounds(self):
        g = WorldGrid(size=(4, 4, 4))
        ok = g.write_cell(10, 0, 0, MATERIAL_METAL)
        assert ok is False

    def test_write_cell_signal_adds_amplitude(self):
        g = WorldGrid(size=(4, 4, 4))
        g.write_cell(0, 0, 0, MATERIAL_SIGNAL, energy=5.0)
        assert g.signal[0] == 5.0

    def test_place_material_signal_adds_amplitude(self):
        g = WorldGrid(size=(4, 4, 4))
        g.place_material(0, 0, 0, MATERIAL_SIGNAL, energy=3.0)
        assert g.signal[0] == 3.0

    def test_total_energy(self):
        g = WorldGrid(size=(4, 4, 4))
        g.place_material(0, 0, 0, MATERIAL_STONE, energy=10.0)
        g.place_material(1, 0, 0, MATERIAL_STONE, energy=20.0)
        assert g.total_energy == pytest.approx(30.0)

    def test_total_signal(self):
        g = WorldGrid(size=(4, 4, 4))
        g.signal[0] = 5.0
        g.signal[10] = 3.0
        assert g.total_signal == pytest.approx(8.0)

    def test_get_nearby_cells_empty_radius(self):
        g = WorldGrid(size=(4, 4, 4))
        result = g.get_nearby_cells(2, 2, 2, radius=0.0)
        assert result["count"] == 1
        assert len(result["material"]) == 1

    def test_get_nearby_cells_returns_arrays(self):
        g = WorldGrid(size=(8, 8, 8))
        g.place_material(4, 4, 4, MATERIAL_WATER, energy=10.0)
        result = g.get_nearby_cells(4, 4, 4, radius=1.5)
        assert result["count"] > 0
        assert "material" in result
        assert "energy" in result
        assert "temperature" in result
        assert "distance" in result

    def test_to_dict_roundtrip(self):
        g = WorldGrid(size=(4, 4, 4))
        g.place_material(1, 1, 1, MATERIAL_STONE, energy=42.0, temperature=77.0)
        d = g.to_dict()
        g2 = WorldGrid.from_dict(d)
        assert g2.size == g.size
        assert g2.get_cell(1, 1, 1).energy == pytest.approx(42.0)
        assert g2.get_cell(1, 1, 1).temperature == pytest.approx(77.0)

    def test_from_dict_mismatched_size_raises(self):
        g = WorldGrid(size=(4, 4, 4))
        d = g.to_dict()
        d["material"] = [0] * 10
        with pytest.raises(ValueError, match="does not match"):
            WorldGrid.from_dict(d)

    def test_default_temperature(self):
        g = WorldGrid(size=(4, 4, 4))
        assert np.all(g.temperature == 20.0)

    def test_default_energy_zero(self):
        g = WorldGrid(size=(4, 4, 4))
        assert np.all(g.energy == 0.0)


# ── Entity ────────────────────────────────────────────────────────────────────

class TestEntity:
    def test_defaults(self):
        e = Entity()
        assert e.id == 0
        assert e.energy == 100.0
        assert e.entity_type == EntityType.OBJECT
        assert e.alive is True

    def test_distance_to(self):
        a = Entity(position=np.array([0.0, 0.0, 0.0]))
        b = Entity(position=np.array([3.0, 4.0, 0.0]))
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_to_point(self):
        e = Entity(position=np.array([1.0, 2.0, 0.0]))
        d = e.distance_to_point(np.array([4.0, 6.0, 0.0]))
        assert d == pytest.approx(5.0)

    def test_to_dict_roundtrip(self):
        e = Entity(id=7, position=np.array([1.0, 2.0, 3.0]),
                    energy=55.0, entity_type=EntityType.AGENT, alive=False)
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.id == 7
        assert np.allclose(e2.position, [1.0, 2.0, 3.0])
        assert e2.energy == 55.0
        assert e2.entity_type == EntityType.AGENT
        assert e2.alive is False

    def test_to_dict_types(self):
        e = Entity(id=1, position=np.array([0.0, 0.0, 0.0]))
        d = e.to_dict()
        assert isinstance(d["id"], int)
        assert isinstance(d["position"], list)
        assert isinstance(d["energy"], float)
        assert isinstance(d["entity_type"], int)
        assert isinstance(d["alive"], bool)

    def test_same_position_distance_zero(self):
        e = Entity(position=np.array([5.0, 5.0, 5.0]))
        assert e.distance_to_point(np.array([5.0, 5.0, 5.0])) == pytest.approx(0.0)

    def test_custom_entity_type(self):
        e = Entity(entity_type=EntityType.LIGHT)
        assert e.entity_type == EntityType.LIGHT

    def test_dead_entity(self):
        e = Entity(alive=False)
        assert e.alive is False


# ── Nest ──────────────────────────────────────────────────────────────────────

class TestNest:
    def test_init(self):
        n = Nest(id=1, position=np.array([1.0, 2.0, 3.0]),
                 stored_energy=50.0, owner_group_id=0)
        assert n.id == 1
        assert n.stored_energy == 50.0
        assert n.owner_group_id == 0
        assert n.alive is True

    def test_distance_to_point(self):
        n = Nest(id=1, position=np.array([0.0, 0.0, 0.0]),
                 stored_energy=10.0, owner_group_id=0)
        d = n.distance_to_point(np.array([3.0, 4.0, 0.0]))
        assert d == pytest.approx(5.0)

    def test_to_dict_roundtrip(self):
        n = Nest(id=5, position=np.array([2.0, 3.0, 4.0]),
                 stored_energy=75.0, owner_group_id=1, alive=False)
        d = n.to_dict()
        n2 = Nest.from_dict(d)
        assert n2.id == 5
        assert np.allclose(n2.position, [2.0, 3.0, 4.0])
        assert n2.stored_energy == 75.0
        assert n2.owner_group_id == 1
        assert n2.alive is False

    def test_to_dict_types(self):
        n = Nest(id=1, position=np.array([0.0, 0.0, 0.0]),
                 stored_energy=0.0, owner_group_id=0)
        d = n.to_dict()
        assert isinstance(d["id"], int)
        assert isinstance(d["position"], list)
        assert isinstance(d["stored_energy"], float)
        assert isinstance(d["owner_group_id"], int)
        assert isinstance(d["alive"], bool)


# ── Perception ────────────────────────────────────────────────────────────────

class TestPerception:
    def test_defaults(self):
        p = Perception()
        assert p.nearby_cells == {}
        assert p.nearby_entities == []
        assert p.agent_body == {}
        assert p.time_ms == 0.0

    def test_custom(self):
        p = Perception(
            nearby_cells={"material": np.array([1, 2])},
            nearby_entities=[{"id": 1}],
            agent_body={"energy": 100},
            time_ms=42.0,
        )
        assert p.nearby_cells["material"][0] == 1
        assert len(p.nearby_entities) == 1
        assert p.agent_body["energy"] == 100
        assert p.time_ms == 42.0


# ── CellWrite ─────────────────────────────────────────────────────────────────

class TestCellWrite:
    def test_defaults(self):
        cw = CellWrite()
        assert cw.x == 0
        assert cw.y == 0
        assert cw.z == 0
        assert cw.energy == 0.0

    def test_custom(self):
        cw = CellWrite(x=5, y=10, z=2, energy=3.5)
        assert cw.x == 5
        assert cw.y == 10
        assert cw.z == 2
        assert cw.energy == 3.5

    def test_material_default(self):
        cw = CellWrite()
        assert cw.material == MATERIAL_AIR

    def test_custom_material(self):
        cw = CellWrite(material=MATERIAL_WATER)
        assert cw.material == MATERIAL_WATER

    def test_negative_coordinates(self):
        cw = CellWrite(x=-1, y=-2, z=-3)
        assert cw.x == -1

    def test_zero_energy(self):
        cw = CellWrite(energy=0.0)
        assert cw.energy == 0.0


# ── BabyAction ────────────────────────────────────────────────────────────────

class TestBabyAction:
    def test_defaults(self):
        ba = BabyAction()
        assert ba.writes == []

    def test_with_writes(self):
        ba = BabyAction(writes=[CellWrite(x=1, y=2)])
        assert len(ba.writes) == 1

    def test_multiple_writes(self):
        ba = BabyAction(writes=[CellWrite(x=1), CellWrite(x=2), CellWrite(x=3)])
        assert len(ba.writes) == 3

    def test_write_materials(self):
        ba = BabyAction(writes=[
            CellWrite(x=0, material=MATERIAL_WATER),
            CellWrite(x=1, material=MATERIAL_STONE),
        ])
        assert ba.writes[0].material == MATERIAL_WATER
        assert ba.writes[1].material == MATERIAL_STONE


# ── Perceptron ────────────────────────────────────────────────────────────────

class TestPerceptron:
    def test_init(self):
        p = Perceptron(input_dim=4, output_dim=2)
        assert p.W.shape == (4, 2)
        assert p.b.shape == (2,)

    def test_hidden(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        assert p.H is not None
        assert p.H.shape == (4, 3)

    def test_forward(self):
        p = Perceptron(input_dim=4, output_dim=2)
        x = np.random.randn(4).astype(np.float32)
        out = p.forward(x)
        assert out.shape == (2,)

    def test_hidden_forward(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        x = np.random.randn(4).astype(np.float32)
        out = p.forward(x)
        assert out.shape == (2,)

    def test_forward_output_range(self):
        p = Perceptron(input_dim=4, output_dim=2)
        x = np.random.randn(4).astype(np.float32)
        out = p.forward(x)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_update_modifies_weights(self):
        p = Perceptron(input_dim=4, output_dim=2)
        w_before = p.W.copy()
        x = np.random.randn(4).astype(np.float32)
        p.update(x, error=np.array([0.1, -0.1]), lr=0.01)
        assert not np.allclose(p.W, w_before)

    def test_update_no_change_with_zero_lr(self):
        p = Perceptron(input_dim=4, output_dim=2)
        w_before = p.W.copy()
        x = np.random.randn(4).astype(np.float32)
        p.update(x, error=np.array([1.0, 1.0]), lr=0.0)
        assert np.allclose(p.W, w_before)

    def test_to_dict_roundtrip(self):
        p = Perceptron(input_dim=4, output_dim=2)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        np.testing.assert_array_equal(p.W, p2.W)
        np.testing.assert_array_equal(p.b, p2.b)

    def test_to_dict_with_hidden(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        d = p.to_dict()
        assert "H" in d
        assert "bh" in d
        p2 = Perceptron.from_dict(d)
        np.testing.assert_array_equal(p.H, p2.H)
        np.testing.assert_array_equal(p.bh, p2.bh)

    def test_to_dict_without_hidden(self):
        p = Perceptron(input_dim=4, output_dim=2)
        d = p.to_dict()
        assert "H" not in d
        assert "bh" not in d

    def test_random_weights_not_all_same(self):
        p = Perceptron(input_dim=10, output_dim=5)
        assert not np.all(p.W == p.W[0, 0])

    def test_sigmoid_range(self):
        x = np.array([-100.0, 0.0, 100.0])
        s = Perceptron._sigmoid(x)
        assert s[0] >= 0.0 and s[0] <= 1.0
        assert s[1] == pytest.approx(0.5, abs=0.01)
        assert s[2] >= 0.0 and s[2] <= 1.0

    def test_features_without_hidden(self):
        p = Perceptron(input_dim=4, output_dim=2)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        f = p._features(x)
        np.testing.assert_array_equal(f, x)

    def test_features_with_hidden(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        f = p._features(x)
        assert f.shape == (4 + 3,)

    def test_from_dict_hidden_units_count(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        assert p2.hidden_units == 3

    def test_from_dict_no_hidden_units(self):
        p = Perceptron(input_dim=4, output_dim=2)
        d = p.to_dict()
        p2 = Perceptron.from_dict(d)
        assert p2.hidden_units == 0
        assert p2.H is None

    def test_update_with_hidden_preserves_H(self):
        p = Perceptron(input_dim=4, output_dim=2, hidden_units=3)
        H_before = p.H.copy()
        x = np.random.randn(4).astype(np.float32)
        p.update(x, error=np.array([0.1, -0.1]), lr=0.01)
        np.testing.assert_array_equal(p.H, H_before)

    def test_multiple_updates_accumulate(self):
        p = Perceptron(input_dim=4, output_dim=2)
        w_after_one = None
        x = np.random.randn(4).astype(np.float32)
        p.update(x, error=np.array([0.1, 0.1]), lr=0.01)
        w_after_one = p.W.copy()
        p.update(x, error=np.array([0.1, 0.1]), lr=0.01)
        assert not np.allclose(p.W, w_after_one)
