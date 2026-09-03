from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .perception import WorldPerception, PerceptionEvent


@dataclass
class BabyPerceptionConfig:
    see_radius: int = 5
    material_weight: float = 1.0
    energy_weight: float = 0.5
    novelty_bonus: float = 2.0
    memory_size: int = 50
    learning_rate: float = 0.1


class BabyPerception:
    def __init__(self, baby, config: BabyPerceptionConfig | None = None):
        self.baby = baby
        self.config = config or BabyPerceptionConfig()
        self._perceived_events: list[PerceptionEvent] = []
        self._material_counts: dict[int, int] = {}
        self._novelty_scores: dict[int, float] = {}
        self._total_perceived = 0
        self._memory: list[dict] = []

    def perceive_world(self, world, perception: WorldPerception) -> dict:
        gx, gy, gz = int(self.baby.position[0]), int(self.baby.position[1]), int(self.baby.position[2])
        radius = self.config.see_radius

        nearby_events = []
        for event in perception.events:
            ex, ey, ez = event.grid_pos
            dist = np.sqrt((gx - ex) ** 2 + (gz - ez) ** 2)
            if dist <= radius:
                nearby_events.append(event)

        material_features = np.zeros(8, dtype=np.float32)
        energy_sum = 0.0
        novelty_sum = 0.0

        for event in nearby_events:
            mat_id = event.material_type
            if 1 <= mat_id <= 8:
                material_features[mat_id - 1] += 1.0
            energy_sum += event.energy

            count = self._material_counts.get(mat_id, 0)
            novelty = 1.0 / (1.0 + count)
            novelty_sum += novelty
            self._material_counts[mat_id] = count + 1

        if material_features.sum() > 0:
            material_features /= material_features.sum()

        self._perceived_events.extend(nearby_events)
        if len(self._perceived_events) > self.config.memory_size:
            self._perceived_events = self._perceived_events[-self.config.memory_size:]

        result = {
            "material_features": material_features,
            "energy_sum": energy_sum,
            "novelty_sum": novelty_sum,
            "event_count": len(nearby_events),
            "total_perceived": self._total_perceived + len(nearby_events),
        }
        self._total_perceived += len(nearby_events)

        self._memory.append(result)
        if len(self._memory) > self.config.memory_size:
            self._memory = self._memory[-self.config.memory_size:]

        return result

    def get_material_preference(self) -> int:
        if not self._material_counts:
            return 1
        return max(self._material_counts, key=self._material_counts.get)

    def get_novelty_score(self, material_id: int) -> float:
        count = self._material_counts.get(material_id, 0)
        return 1.0 / (1.0 + count)

    def summary(self) -> dict:
        return {
            "total_perceived": self._total_perceived,
            "unique_materials": len(self._material_counts),
            "material_counts": dict(self._material_counts),
            "memory_size": len(self._memory),
            "preferred_material": self.get_material_preference(),
        }


class BabyAction:
    def __init__(self, baby, config: BabyPerceptionConfig | None = None):
        self.baby = baby
        self.config = config or BabyPerceptionConfig()
        self._last_action = None
        self._action_counts: dict[str, int] = {}

    def move_toward(self, target_pos: np.ndarray, world) -> bool:
        current = self.baby.position.copy()
        direction = target_pos - current
        dist = np.linalg.norm(direction)
        if dist < 1.0:
            return False

        direction = direction / dist
        new_pos = current + direction

        gx, gy, gz = int(new_pos[0]), int(new_pos[1]), int(new_pos[2])
        nx, ny, nz = world.size
        if 0 <= gx < nx and 0 <= gy < ny and 0 <= gz < nz:
            self.baby.position[:] = new_pos
            self._last_action = "move"
            self._action_counts["move"] = self._action_counts.get("move", 0) + 1
            return True
        return False

    def consume_energy(self, amount: float) -> bool:
        if self.baby.energy >= amount:
            self.baby.energy -= amount
            self._last_action = "consume"
            self._action_counts["consume"] = self._action_counts.get("consume", 0) + 1
            return True
        return False

    def write_to_grid(self, world, material_id: int, energy: float) -> bool:
        gx, gy, gz = int(self.baby.position[0]), int(self.baby.position[1]), int(self.baby.position[2])
        nx, ny, nz = world.size
        if 0 <= gx < nx and 0 <= gy < ny and 0 <= gz < nz:
            idx = world.idx(gx, gy, gz)
            world.material[idx] = material_id
            world.energy[idx] = energy
            self._last_action = "write"
            self._action_counts["write"] = self._action_counts.get("write", 0) + 1
            return True
        return False

    def interact_with_event(self, event: PerceptionEvent, world) -> dict:
        result = {"success": False, "action": None}

        dist = np.linalg.norm(self.baby.position - np.array(event.grid_pos, dtype=np.float64))
        if dist > self.config.see_radius:
            return result

        if self.baby.energy > 20:
            success = self.move_toward(np.array(event.grid_pos, dtype=np.float64), world)
            if success:
                result["success"] = True
                result["action"] = "approach"

        if self.baby.energy > 10:
            self.baby.energy += event.energy * 0.1
            result["success"] = True
            result["action"] = "absorb"

        return result

    def summary(self) -> dict:
        return {
            "last_action": self._last_action,
            "action_counts": dict(self._action_counts),
        }


class BabyLearning:
    def __init__(self, baby, config: BabyPerceptionConfig | None = None):
        self.baby = baby
        self.config = config or BabyPerceptionConfig()
        self._experiences: list[dict] = []
        self._material_rewards: dict[int, float] = {}
        self._total_learning = 0

    def record_experience(self, perception_result: dict, action_result: dict):
        experience = {
            "perception": perception_result,
            "action": action_result,
            "energy": self.baby.energy,
        }
        self._experiences.append(experience)
        if len(self._experiences) > self.config.memory_size:
            self._experiences = self._experiences[-self.config.memory_size:]

        if action_result.get("success"):
            material_id = perception_result.get("preferred_material", 1)
            reward = self._material_rewards.get(material_id, 0.0)
            self._material_rewards[material_id] = reward + 1.0

        self._total_learning += 1

    def get_material_value(self, material_id: int) -> float:
        return self._material_rewards.get(material_id, 0.0)

    def get_experience_buffer(self) -> list[dict]:
        return list(self._experiences)

    def summary(self) -> dict:
        return {
            "total_experiences": len(self._experiences),
            "total_learning": self._total_learning,
            "material_rewards": dict(self._material_rewards),
            "preferred_material": self.get_preferred_material(),
        }

    def get_preferred_material(self) -> int:
        if not self._material_rewards:
            return 1
        return max(self._material_rewards, key=self._material_rewards.get)


class BabyPerceptionSystem:
    def __init__(self, config: BabyPerceptionConfig | None = None):
        self.config = config or BabyPerceptionConfig()
        self._babies: dict[int, dict] = {}

    def register_baby(self, baby):
        baby_id = baby.entity.id
        self._babies[baby_id] = {
            "perception": BabyPerception(baby, self.config),
            "action": BabyAction(baby, self.config),
            "learning": BabyLearning(baby, self.config),
        }

    def unregister_baby(self, baby):
        baby_id = baby.entity.id
        self._babies.pop(baby_id, None)

    def get_modules(self, baby) -> dict | None:
        baby_id = baby.entity.id
        return self._babies.get(baby_id)

    def tick(self, world, perception: WorldPerception) -> list[dict]:
        results = []
        for baby_id, modules in self._babies.items():
            baby = modules["perception"].baby
            if not baby.alive:
                continue

            perc_result = modules["perception"].perceive_world(world, perception)
            action_result = {"success": False, "action": None}

            if perc_result["event_count"] > 0:
                best_event = max(
                    perception.events,
                    key=lambda e: e.energy * self.config.energy_weight +
                                  modules["perception"].get_novelty_score(e.material_type) * self.config.novelty_bonus,
                    default=None,
                )
                if best_event:
                    action_result = modules["action"].interact_with_event(best_event, world)

            modules["learning"].record_experience(perc_result, action_result)
            results.append({
                "baby_id": baby_id,
                "perception": perc_result,
                "action": action_result,
            })

        return results

    def summary(self) -> dict:
        baby_summaries = {}
        for baby_id, modules in self._babies.items():
            baby_summaries[baby_id] = {
                "perception": modules["perception"].summary(),
                "action": modules["action"].summary(),
                "learning": modules["learning"].summary(),
            }
        return {
            "total_babies": len(self._babies),
            "babies": baby_summaries,
        }
