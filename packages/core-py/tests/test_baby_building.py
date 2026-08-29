import numpy as np
import pytest

from domains.collections.baby_building import (
    StructureType, ToolType, BuildingConfig, Structure, Tool,
    BabyBuilding, BuildingRegistry,
)


class FakeBaby:
    def __init__(self, id=1, energy=100.0):
        self.entity = type('Entity', (), {'id': id, 'entity_type': 1})()
        self.position = np.array([32, 0, 32], dtype=np.float64)
        self.energy = energy
        self.alive = True


class TestStructure:
    def test_create(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1))
        assert s.health == 100.0
        assert s.is_alive()

    def test_take_damage(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1))
        s.take_damage(50.0)
        assert s.health == 50.0

    def test_destroy(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1))
        s.take_damage(100.0)
        assert not s.is_alive()

    def test_repair(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1))
        s.take_damage(50.0)
        s.repair(30.0)
        assert s.health == 80.0

    def test_upgrade(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1))
        s.take_damage(50.0)
        success = s.upgrade()
        assert success
        assert s.level == 2
        assert s.health == 100.0

    def test_upgrade_max(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 0, 1), level=5)
        success = s.upgrade()
        assert not success


class TestTool:
    def test_create(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        assert t.durability == 100.0
        assert t.is_usable()

    def test_use(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(10.0)
        assert t.durability == 90.0

    def test_broken(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(100.0)
        assert not t.is_usable()

    def test_repair(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(50.0)
        t.repair(30.0)
        assert t.durability == 80.0

    def test_upgrade(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        old_efficiency = t.efficiency
        success = t.upgrade()
        assert success
        assert t.level == 2
        assert t.efficiency > old_efficiency


class TestBabyBuilding:
    def test_build(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.WALL, (1, 0, 1))
        assert structure is not None
        assert structure.structure_type == StructureType.WALL
        assert baby.energy < 100.0

    def test_build_insufficient_energy(self):
        baby = FakeBaby(energy=5.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.WALL, (1, 0, 1))
        assert structure is None

    def test_craft(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        tool = building.craft(ToolType.PICKAXE)
        assert tool is not None
        assert tool.tool_type == ToolType.PICKAXE

    def test_craft_insufficient_energy(self):
        baby = FakeBaby(energy=5.0)
        building = BabyBuilding(baby)
        tool = building.craft(ToolType.PICKAXE)
        assert tool is None

    def test_get_structure(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        building.build(StructureType.WALL, (1, 0, 1))
        structure = building.get_structure(StructureType.WALL)
        assert structure is not None

    def test_get_tool(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        building.craft(ToolType.PICKAXE)
        tool = building.get_tool(ToolType.PICKAXE)
        assert tool is not None

    def test_repair_structure(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.WALL, (1, 0, 1))
        structure.take_damage(50.0)
        success = building.repair_structure(structure)
        assert success
        assert structure.health > 50.0

    def test_use_tool(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        tool = building.craft(ToolType.PICKAXE)
        result = building.use_tool(tool, "mine")
        assert result["success"]
        assert result["action"] == "mine"

    def test_store_item(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.STORAGE, (1, 0, 1))
        success = building.store_item(structure, 5)
        assert success
        assert structure.stored_items == 5

    def test_retrieve_item(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.STORAGE, (1, 0, 1))
        building.store_item(structure, 5)
        success = building.retrieve_item(structure, 3)
        assert success
        assert structure.stored_items == 2

    def test_retrieve_too_many(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        structure = building.build(StructureType.STORAGE, (1, 0, 1))
        building.store_item(structure, 2)
        success = building.retrieve_item(structure, 5)
        assert not success

    def test_tick(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        building.build(StructureType.WALL, (1, 0, 1))
        building.tick()
        assert len(building._structures) == 1

    def test_summary(self):
        baby = FakeBaby(energy=100.0)
        building = BabyBuilding(baby)
        building.build(StructureType.WALL, (1, 0, 1))
        building.craft(ToolType.PICKAXE)
        s = building.summary()
        assert s["structures"] == 1
        assert s["tools"] == 1


class TestBuildingRegistry:
    def test_register(self):
        registry = BuildingRegistry()
        baby = FakeBaby()
        building = BabyBuilding(baby)
        registry.register(baby.entity.id, building)
        assert baby.entity.id in registry._buildings

    def test_unregister(self):
        registry = BuildingRegistry()
        baby = FakeBaby()
        building = BabyBuilding(baby)
        registry.register(baby.entity.id, building)
        registry.unregister(baby.entity.id)
        assert baby.entity.id not in registry._buildings

    def test_get(self):
        registry = BuildingRegistry()
        baby = FakeBaby()
        building = BabyBuilding(baby)
        registry.register(baby.entity.id, building)
        result = registry.get(baby.entity.id)
        assert result is not None

    def test_find_structures(self):
        registry = BuildingRegistry()
        baby = FakeBaby()
        building = BabyBuilding(baby)
        building.build(StructureType.WALL, (1, 0, 1))
        building.build(StructureType.WALL, (2, 0, 2))
        registry.register(baby.entity.id, building)
        registry.tick()
        results = registry.find_structures(StructureType.WALL, radius=5, center=(1, 0, 1))
        assert len(results) >= 1

    def test_summary(self):
        registry = BuildingRegistry()
        baby = FakeBaby()
        building = BabyBuilding(baby)
        building.build(StructureType.WALL, (1, 0, 1))
        registry.register(baby.entity.id, building)
        s = registry.summary()
        assert s["registered_builders"] == 1
        assert s["total_structures"] == 1
