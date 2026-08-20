import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SocialConfig:
    communication_range: float = 5.0
    teaching_range: float = 3.0
    max_message_buffer: int = 10
    trust_decay: float = 0.95
    social_learning_rate: float = 0.1
    imitation_chance: float = 0.3
    teaching_cost: float = 5.0


@dataclass
class Message:
    sender_id: int
    receiver_id: int
    content: dict
    timestamp: float
    message_type: str = "info"


class BabySocial:
    def __init__(self, baby, config: SocialConfig | None = None):
        self.baby = baby
        self.config = config or SocialConfig()
        self._inbox: list[Message] = []
        self._outbox: list[Message] = []
        self._trust_scores: dict[int, float] = {}
        self._social_memory: list[dict] = []
        self._teach_count = 0
        self._learn_count = 0

    def send_message(self, receiver_id: int, content: dict, message_type: str = "info") -> Message:
        msg = Message(
            sender_id=self.baby.entity.id,
            receiver_id=receiver_id,
            content=content,
            timestamp=0.0,
            message_type=message_type,
        )
        self._outbox.append(msg)
        return msg

    def receive_message(self, message: Message):
        self._inbox.append(message)
        if len(self._inbox) > self.config.max_message_buffer:
            self._inbox = self._inbox[-self.config.max_message_buffer:]

    def get_messages(self) -> list[Message]:
        messages = list(self._inbox)
        self._inbox.clear()
        return messages

    def update_trust(self, other_id: int, delta: float):
        current = self._trust_scores.get(other_id, 0.5)
        self._trust_scores[other_id] = max(0.0, min(1.0, current + delta))

    def get_trust(self, other_id: int) -> float:
        return self._trust_scores.get(other_id, 0.5)

    def observe_action(self, other_baby, action_type: str, result: dict):
        if result.get("success"):
            observation = {
                "other_id": other_baby.entity.id,
                "action": action_type,
                "result": result,
                "other_energy": other_baby.energy,
            }
            self._social_memory.append(observation)
            if len(self._social_memory) > self.config.max_message_buffer:
                self._social_memory = self._social_memory[-self.config.max_message_buffer:]

    def try_imitate(self, other_baby, world) -> dict:
        if np.random.random() > self.config.imitation_chance:
            return {"success": False, "reason": "chance_failed"}

        other_id = other_baby.entity.id
        trust = self.get_trust(other_id)
        if trust < 0.3:
            return {"success": False, "reason": "low_trust"}

        dist = np.linalg.norm(self.baby.position - other_baby.position)
        if dist > self.config.communication_range:
            return {"success": False, "reason": "out_of_range"}

        self.baby.energy -= self.config.teaching_cost * 0.1
        self._learn_count += 1

        return {
            "success": True,
            "action": "imitate",
            "other_id": other_id,
        }

    def teach(self, other_baby, lesson: dict) -> dict:
        dist = np.linalg.norm(self.baby.position - other_baby.position)
        if dist > self.config.teaching_range:
            return {"success": False, "reason": "out_of_range"}

        if self.baby.energy < self.config.teaching_cost:
            return {"success": False, "reason": "insufficient_energy"}

        self.baby.energy -= self.config.teaching_cost
        self._teach_count += 1

        msg = self.send_message(other_baby.entity.id, lesson, message_type="teach")
        other_baby_social = getattr(other_baby, '_social', None)
        if other_baby_social:
            other_baby_social.receive_message(msg)

        return {
            "success": True,
            "action": "teach",
            "other_id": other_baby.entity.id,
            "energy_cost": self.config.teaching_cost,
        }

    def share_knowledge(self, other_baby) -> dict:
        knowledge = {
            "material_preference": getattr(self.baby, '_learning', {}).get("preferred_material", 1),
            "total_experiences": getattr(self.baby, '_learning', {}).get("total_experiences", 0),
            "energy": self.baby.energy,
        }
        return self.teach(other_baby, knowledge)

    def summary(self) -> dict:
        return {
            "inbox_size": len(self._inbox),
            "outbox_size": len(self._outbox),
            "trust_scores": dict(self._trust_scores),
            "social_memory_size": len(self._social_memory),
            "teach_count": self._teach_count,
            "learn_count": self._learn_count,
        }


class BabyCultural:
    def __init__(self, config: SocialConfig | None = None):
        self.config = config or SocialConfig()
        self._cultural_memory: list[dict] = []
        self._generations: list[list[dict]] = []
        self._current_generation: list[dict] = []

    def record_culture(self, baby, perception_summary: dict, learning_summary: dict):
        culture_record = {
            "baby_id": baby.entity.id,
            "perception": perception_summary,
            "learning": learning_summary,
            "energy": baby.energy,
            "position": baby.position.tolist(),
        }
        self._cultural_memory.append(culture_record)
        self._current_generation.append(culture_record)

    def end_generation(self):
        if self._current_generation:
            self._generations.append(self._current_generation)
            self._current_generation = []

    def get_generation_stats(self, generation_idx: int = -1) -> dict:
        if not self._generations:
            return {}
        gen = self._generations[generation_idx]
        energies = [r["energy"] for r in gen]
        return {
            "size": len(gen),
            "avg_energy": float(np.mean(energies)) if energies else 0.0,
            "max_energy": float(np.max(energies)) if energies else 0.0,
            "min_energy": float(np.min(energies)) if energies else 0.0,
        }

    def get_cultural_trend(self) -> dict:
        if not self._generations:
            return {"generations": 0}
        gen_sizes = [len(g) for g in self._generations]
        gen_energies = [np.mean([r["energy"] for r in g]) for g in self._generations if g]
        return {
            "generations": len(self._generations),
            "total_records": len(self._cultural_memory),
            "generation_sizes": gen_sizes,
            "energy_trend": [float(e) for e in gen_energies],
        }

    def get_top_performers(self, n: int = 5) -> list[dict]:
        sorted_records = sorted(
            self._cultural_memory,
            key=lambda r: r["energy"],
            reverse=True,
        )
        return sorted_records[:n]

    def summary(self) -> dict:
        return {
            "total_records": len(self._cultural_memory),
            "generations": len(self._generations),
            "current_generation_size": len(self._current_generation),
            "trend": self.get_cultural_trend(),
        }


class BabySocialSystem:
    def __init__(self, config: SocialConfig | None = None):
        self.config = config or SocialConfig()
        self._social_modules: dict[int, BabySocial] = {}
        self._cultural = BabyCultural(config)
        self._interaction_log: list[dict] = []

    def register_baby(self, baby):
        social = BabySocial(baby, self.config)
        baby._social = social
        self._social_modules[baby.entity.id] = social

    def unregister_baby(self, baby):
        self._social_modules.pop(baby.entity.id, None)

    def get_social(self, baby) -> BabySocial | None:
        return self._social_modules.get(baby.entity.id)

    def tick(self, babies: list, world) -> list[dict]:
        results = []
        alive = [b for b in babies if b.alive]

        for baby in alive:
            social = self._social_modules.get(baby.entity.id)
            if not social:
                continue

            messages = social.get_messages()
            for msg in messages:
                social.update_trust(msg.sender_id, 0.05)

            nearby = []
            for other in alive:
                if other.entity.id == baby.entity.id:
                    continue
                dist = np.linalg.norm(baby.position - other.position)
                if dist <= self.config.communication_range:
                    nearby.append((other, dist))

            interaction = {"baby_id": baby.entity.id, "nearby": len(nearby), "messages": len(messages)}

            if nearby and baby.energy > 20:
                other, dist = min(nearby, key=lambda x: x[1])
                if dist <= self.config.teaching_range:
                    result = social.share_knowledge(other)
                    interaction["teach"] = result

            for other, dist in nearby:
                social.observe_action(other, "presence", {"success": True})

            results.append(interaction)
            self._interaction_log.append(interaction)

        return results

    def end_generation(self):
        self._cultural.end_generation()

    def record_culture(self, babies: list, perception_system=None, learning_system=None):
        for baby in babies:
            if not baby.alive:
                continue
            perc_summary = {}
            learn_summary = {}
            if perception_system:
                modules = perception_system.get_modules(baby)
                if modules:
                    perc_summary = modules["perception"].summary()
            if learning_system:
                modules = learning_system.get_modules(baby)
                if modules:
                    learn_summary = modules["learning"].summary()
            self._cultural.record_culture(baby, perc_summary, learn_summary)

    def summary(self) -> dict:
        return {
            "social_modules": len(self._social_modules),
            "interactions": len(self._interaction_log),
            "cultural": self._cultural.summary(),
        }
