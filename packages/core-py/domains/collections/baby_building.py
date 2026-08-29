import numpy as np
from dataclasses import dataclass, field
from typing import Any
from enum import IntEnum


class StructureType(IntEnum):
    WALL = 1
    FLOOR = 2
    STORAGE = 3
    WORKSHOP = 4
    SHELTER = 5
    BEACON = 6
    FARM = 7
    LIBRARY = 8


class ToolType(IntEnum):
    PICKAXE = 1
    HAMMER = 2
    SAW = 3
    LANTERN = 4
    COMPASS = 5
    GRAPPLING_HOOK = 6
    SHOVEL = 7
    TOOLBOX = 8


@dataclass
class BuildingConfig:
    build_range: int = 2
    build_cost_energy: float = 10.0
    build_cost_material: int = 5
    tool_craft_cost_energy: float = 15.0
    tool_craft_cost_material: int = 8
    max_structures: int = 50
    max_tools: int = 10
    structure_health_decay: float = 0.01
    repair_cost: float = 5.0


@dataclass
class Structure:
    structure_type: StructureType
    position: tuple[int, int, int]
    health: float = 100.0
    owner_id: int = 0
    level: int = 1
    stored_items: int = 0
    metadata: dict = field(default_factory=dict)

    def is_alive(self) -> bool:
        return self.health > 0

    def take_damage(self, amount: float):
        self.health = max(0.0, self.health - amount)

    def repair(self, amount: float):
        self.health = min(100.0, self.health + amount)

    def upgrade(self) -> bool:
        if self.level < 5:
            self.level += 1
            self.health = 100.0
            return True
        return False


@dataclass
class Tool:
    tool_type: ToolType
    durability: float = 100.0
    efficiency: float = 1.0
    owner_id: int = 0
    level: int = 1
    metadata: dict = field(default_factory=dict)

    def is_usable(self) -> bool:
        return self.durability > 0

    def use(self, amount: float = 1.0):
        self.durability = max(0.0, self.durability - amount)

    def repair(self, amount: float):
        self.durability = min(100.0, self.durability + amount)

    def upgrade(self) -> bool:
        if self.level < 5:
            self.level += 1
            self.efficiency += 0.2
            self.durability = 100.0
            return True
        return False


class BabyBuilding:
    def __init__(self, baby, config: BuildingConfig | None = None):
        self.baby = baby
        self.config = config or BuildingConfig()
        self._structures: list[Structure] = []
        self._tools: list[Tool] = []
        self._build_count = 0
        self._craft_count = 0

    def can_build(self, structure_type: StructureType) -> dict:
        if len(self._structures) >= self.config.max_structures:
            return {"can_build": False, "reason": "max_structures"}
        if self.baby.energy < self.config.build_cost_energy:
            return {"can_build": False, "reason": "insufficient_energy"}
        return {"can_build": True, "reason": "ok"}

    def build(self, structure_type: StructureType, position: tuple[int, int, int]) -> Structure | None:
        check = self.can_build(structure_type)
        if not check["can_build"]:
            return None

        self.baby.energy -= self.config.build_cost_energy
        structure = Structure(
            structure_type=structure_type,
            position=position,
            owner_id=self.baby.entity.id,
        )
        self._structures.append(structure)
        self._build_count += 1
        return structure

    def can_craft(self, tool_type: ToolType) -> dict:
        if len(self._tools) >= self.config.max_tools:
            return {"can_craft": False, "reason": "max_tools"}
        if self.baby.energy < self.config.tool_craft_cost_energy:
            return {"can_craft": False, "reason": "insufficient_energy"}
        return {"can_craft": True, "reason": "ok"}

    def craft(self, tool_type: ToolType) -> Tool | None:
        check = self.can_craft(tool_type)
        if not check["can_craft"]:
            return None

        self.baby.energy -= self.config.tool_craft_cost_energy
        tool = Tool(
            tool_type=tool_type,
            owner_id=self.baby.entity.id,
        )
        self._tools.append(tool)
        self._craft_count += 1
        return tool

    def get_structure(self, structure_type: StructureType) -> Structure | None:
        for s in self._structures:
            if s.structure_type == structure_type and s.is_alive():
                return s
        return None

    def get_tool(self, tool_type: ToolType) -> Tool | None:
        for t in self._tools:
            if t.tool_type == tool_type and t.is_usable():
                return t
        return None

    def repair_structure(self, structure: Structure) -> bool:
        if self.baby.energy < self.config.repair_cost:
            return False
        tool = self.get_tool(ToolType.HAMMER)
        repair_amount = 20.0 * (tool.efficiency if tool else 1.0)
        structure.repair(repair_amount)
        self.baby.energy -= self.config.repair_cost
        if tool:
            tool.use(1.0)
        return True

    def use_tool(self, tool: Tool, action: str = "use") -> dict:
        if not tool.is_usable():
            return {"success": False, "reason": "broken"}
        tool.use(1.0)
        return {"success": True, "action": action, "tool": tool.tool_type.name}

    def store_item(self, structure: Structure, count: int = 1) -> bool:
        if structure.structure_type != StructureType.STORAGE:
            return False
        structure.stored_items += count
        return True

    def retrieve_item(self, structure: Structure, count: int = 1) -> bool:
        if structure.structure_type != StructureType.STORAGE:
            return False
        if structure.stored_items < count:
            return False
        structure.stored_items -= count
        return True

    def tick(self):
        for structure in self._structures:
            structure.take_damage(self.config.structure_health_decay)
        self._structures = [s for s in self._structures if s.is_alive()]

    def summary(self) -> dict:
        structure_counts = {}
        for s in self._structures:
            name = s.structure_type.name
            structure_counts[name] = structure_counts.get(name, 0) + 1

        tool_counts = {}
        for t in self._tools:
            name = t.tool_type.name
            tool_counts[name] = tool_counts.get(name, 0) + 1

        return {
            "structures": len(self._structures),
            "tools": len(self._tools),
            "structure_counts": structure_counts,
            "tool_counts": tool_counts,
            "build_count": self._build_count,
            "craft_count": self._craft_count,
        }


class BuildingRegistry:
    def __init__(self):
        self._buildings: dict[int, BabyBuilding] = {}
        self._global_structures: list[Structure] = []

    def register(self, baby_id: int, building: BabyBuilding):
        self._buildings[baby_id] = building

    def unregister(self, baby_id: int):
        self._buildings.pop(baby_id, None)

    def get(self, baby_id: int) -> BabyBuilding | None:
        return self._buildings.get(baby_id)

    def tick(self):
        for building in self._buildings.values():
            building.tick()
        self._global_structures = [
            s for b in self._buildings.values()
            for s in b._structures if s.is_alive()
        ]

    def find_structures(self, structure_type: StructureType, radius: int = 10,
                        center: tuple[int, int, int] | None = None) -> list[Structure]:
        results = []
        for structure in self._global_structures:
            if structure.structure_type != structure_type:
                continue
            if center is not None:
                dist = sum((a - b) ** 2 for a, b in zip(structure.position, center)) ** 0.5
                if dist > radius:
                    continue
            results.append(structure)
        return results

    def summary(self) -> dict:
        total_structures = sum(b.summary()["structures"] for b in self._buildings.values())
        total_tools = sum(b.summary()["tools"] for b in self._buildings.values())
        return {
            "registered_builders": len(self._buildings),
            "total_structures": total_structures,
            "total_tools": total_tools,
        }
