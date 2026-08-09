"""
Tests for signal communication — the broadcast channel of the world.

Writing SIGNAL material must emit wave amplitude into the signal field;
babies must perceive that amplitude as a learnable feature; waves must carry
it across the grid; and the whole loop must survive a scene snapshot.
"""

from __future__ import annotations

import numpy as np

from domains.shell.simulation import (
    BabyAction,
    CellWrite,
    MATERIAL_AIR,
    MATERIAL_SIGNAL,
    MATERIAL_STONE,
    SimBaby,
    SimScene,
    Simulation,
    WorldGrid,
    WorldParams,
)

SIGNAL_IDX = 4  # 5th feature — broadcast strength


def _params(**kw) -> WorldParams:
    defaults = dict(grid_size=(16, 8, 16), wave_speed=0.5, signal_decay=0.1)
    defaults.update(kw)
    return WorldParams(**defaults)


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

    def test_default_cells_input_dim_is_five(self):
        assert WorldParams().cells_input_dim == 5

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
        from domains.shell.simulation import cell_update_default
        cell_update_default(g, params)
        neighbor = g.signal[g.idx(9, 4, 8)]
        assert neighbor > 0.0
        assert neighbor < 10.0  # partial transfer, not full jump

    def test_signal_reaches_distance_over_ticks(self):
        g = WorldGrid((16, 8, 16))
        params = _params()
        g.write_cell(8, 4, 8, MATERIAL_SIGNAL, energy=10.0)
        from domains.shell.simulation import cell_update_default
        for _ in range(6):
            cell_update_default(g, params)
        assert g.signal[g.idx(12, 4, 8)] > 0.0  # 4 cells away after several ticks

    def test_signal_does_not_emit_from_air_writes(self):
        g = WorldGrid((16, 8, 16))
        g.write_cell(8, 4, 8, MATERIAL_AIR, energy=1.0)
        assert g.total_signal == 0.0


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
