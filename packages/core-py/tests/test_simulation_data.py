"""Tests for domains.shell.simulation — EntityType, WorldParams, Perception, CellWrite, BabyAction, Perceptron."""

import numpy as np
from domains.shell.simulation import (
    EntityType, WorldParams, Perception, CellWrite, BabyAction, Perceptron,
)


class TestEntityType:
    def test_all_members(self):
        assert len(EntityType) == 5
    def test_values(self):
        assert EntityType.AGENT.value == 0
        assert EntityType.OBJECT.value == 1
        assert EntityType.LIGHT.value == 2


class TestWorldParams:
    def test_defaults(self):
        wp = WorldParams()
        assert wp.grid_size == (64, 32, 64)
        assert wp.start_energy == 100.0
        assert wp.start_agents == 4
        assert wp.social_enabled is True
        assert wp.message_enabled is False
        assert wp.nest_enabled is False

    def test_custom(self):
        wp = WorldParams(grid_size=(32, 16, 32), start_energy=50.0)
        assert wp.grid_size == (32, 16, 32)
        assert wp.start_energy == 50.0


class TestPerception:
    def test_defaults(self):
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
        assert cw.energy == 0.0

    def test_custom(self):
        cw = CellWrite(x=5, y=10, z=2, energy=3.5)
        assert cw.x == 5
        assert cw.energy == 3.5


class TestBabyAction:
    def test_defaults(self):
        ba = BabyAction()
        assert ba.writes == []

    def test_with_writes(self):
        ba = BabyAction(writes=[CellWrite(x=1, y=2)])
        assert len(ba.writes) == 1


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
