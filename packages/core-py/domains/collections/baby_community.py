import numpy as np
from dataclasses import dataclass, field
from typing import Any
from enum import IntEnum


class CommunityRole(IntEnum):
    LEADER = 1
    ELDER = 2
    BUILDER = 3
    HUNTER = 4
    TEACHER = 5
    SCOUT = 6
    MEMBER = 7


class RelationType(IntEnum):
    FRIEND = 1
    FAMILY = 2
    MENTOR = 3
    RIVAL = 4
    ALLY = 5
    STRANGER = 6


@dataclass
class CommunityConfig:
    max_community_size: int = 20
    min_community_size: int = 2
    formation_threshold: float = 0.6
    loyalty_decay: float = 0.98
    reputation_gain: float = 0.1
    reputation_loss: float = 0.05
    leadership_energy_threshold: float = 80.0


@dataclass
class Relation:
    baby_a: int
    baby_b: int
    relation_type: RelationType
    strength: float = 0.5
    interactions: int = 0
    trust: float = 0.5
    last_interaction: float = 0.0

    def strengthen(self, amount: float):
        self.strength = min(1.0, self.strength + amount)
        self.interactions += 1

    def weaken(self, amount: float):
        self.strength = max(0.0, self.strength - amount)
        self.interactions += 1


@dataclass
class Member:
    baby_id: int
    role: CommunityRole = CommunityRole.MEMBER
    loyalty: float = 0.5
    reputation: float = 0.5
    contributions: int = 0
    joined_tick: int = 0


class BabyCommunity:
    def __init__(self, community_id: int, config: CommunityConfig | None = None):
        self.id = community_id
        self.config = config or CommunityConfig()
        self._members: dict[int, Member] = {}
        self._relations: dict[tuple[int, int], Relation] = {}
        self._shared_resources: dict[str, Any] = {}
        self._history: list[dict] = []
        self._name = f"Community_{community_id}"

    def add_member(self, baby_id: int, role: CommunityRole = CommunityRole.MEMBER) -> bool:
        if len(self._members) >= self.config.max_community_size:
            return False
        if baby_id in self._members:
            return False
        member = Member(baby_id=baby_id, role=role)
        self._members[baby_id] = member
        for existing_id in self._members:
            if existing_id != baby_id:
                self._add_relation(baby_id, existing_id, RelationType.STRANGER)
        return True

    def remove_member(self, baby_id: int) -> bool:
        if baby_id not in self._members:
            return False
        del self._members[baby_id]
        to_remove = [k for k in self._relations if baby_id in k]
        for k in to_remove:
            del self._relations[k]
        return True

    def _add_relation(self, baby_a: int, baby_b: int, relation_type: RelationType):
        key = (min(baby_a, baby_b), max(baby_a, baby_b))
        if key not in self._relations:
            self._relations[key] = Relation(
                baby_a=key[0], baby_b=key[1],
                relation_type=relation_type,
            )

    def get_relation(self, baby_a: int, baby_b: int) -> Relation | None:
        key = (min(baby_a, baby_b), max(baby_a, baby_b))
        return self._relations.get(key)

    def update_relation(self, baby_a: int, baby_b: int, relation_type: RelationType,
                        strength_delta: float = 0.1):
        relation = self.get_relation(baby_a, baby_b)
        if relation:
            relation.relation_type = relation_type
            if strength_delta > 0:
                relation.strengthen(strength_delta)
            else:
                relation.weaken(abs(strength_delta))
        else:
            self._add_relation(baby_a, baby_b, relation_type)

    def set_role(self, baby_id: int, role: CommunityRole) -> bool:
        if baby_id not in self._members:
            return False
        self._members[baby_id].role = role
        return True

    def get_members_by_role(self, role: CommunityRole) -> list[int]:
        return [m.baby_id for m in self._members.values() if m.role == role]

    def get_leader(self) -> int | None:
        leaders = self.get_members_by_role(CommunityRole.LEADER)
        return leaders[0] if leaders else None

    def elect_leader(self, fitness_scores: dict[int, float]) -> int | None:
        candidates = [
            (mid, fitness_scores.get(mid, 0.0))
            for mid in self._members
            if fitness_scores.get(mid, 0.0) > self.config.leadership_energy_threshold * 0.01
        ]
        if not candidates:
            return None
        winner = max(candidates, key=lambda x: x[1])
        self.set_role(winner[0], CommunityRole.LEADER)
        return winner[0]

    def add_contribution(self, baby_id: int, amount: int = 1):
        if baby_id in self._members:
            self._members[baby_id].contributions += amount
            self._members[baby_id].reputation += self.config.reputation_gain

    def contribute_resource(self, key: str, value: Any):
        self._shared_resources[key] = value

    def get_resource(self, key: str) -> Any | None:
        return self._shared_resources.get(key)

    def tick(self):
        for member in self._members.values():
            member.loyalty *= self.config.loyalty_decay

    def get_top_contributors(self, n: int = 5) -> list[int]:
        sorted_members = sorted(
            self._members.values(),
            key=lambda m: m.contributions,
            reverse=True,
        )
        return [m.baby_id for m in sorted_members[:n]]

    def get_cohesion(self) -> float:
        if len(self._members) < 2:
            return 0.0
        total_trust = 0.0
        count = 0
        for relation in self._relations.values():
            total_trust += relation.trust
            count += 1
        return total_trust / max(count, 1)

    def summary(self) -> dict:
        role_counts = {}
        for member in self._members.values():
            name = member.role.name
            role_counts[name] = role_counts.get(name, 0) + 1
        return {
            "id": self.id,
            "name": self._name,
            "size": len(self._members),
            "roles": role_counts,
            "relations": len(self._relations),
            "cohesion": self.get_cohesion(),
            "resources": len(self._shared_resources),
        }


class BabyCommunitySystem:
    def __init__(self, config: CommunityConfig | None = None):
        self.config = config or CommunityConfig()
        self._communities: dict[int, BabyCommunity] = {}
        self._baby_community: dict[int, int] = {}
        self._next_id = 0

    def create_community(self) -> BabyCommunity:
        community = BabyCommunity(self._next_id, self.config)
        self._communities[self._next_id] = community
        self._next_id += 1
        return community

    def disband_community(self, community_id: int) -> bool:
        if community_id not in self._communities:
            return False
        community = self._communities[community_id]
        for baby_id in list(community._members.keys()):
            self._baby_community.pop(baby_id, None)
        del self._communities[community_id]
        return True

    def add_baby_to_community(self, baby_id: int, community_id: int,
                              role: CommunityRole = CommunityRole.MEMBER) -> bool:
        if community_id not in self._communities:
            return False
        if baby_id in self._baby_community:
            return False
        community = self._communities[community_id]
        success = community.add_member(baby_id, role)
        if success:
            self._baby_community[baby_id] = community_id
        return success

    def remove_baby_from_community(self, baby_id: int) -> bool:
        community_id = self._baby_community.get(baby_id)
        if community_id is None:
            return False
        community = self._communities[community_id]
        success = community.remove_member(baby_id)
        if success:
            self._baby_community.pop(baby_id, None)
        return success

    def get_baby_community(self, baby_id: int) -> BabyCommunity | None:
        community_id = self._baby_community.get(baby_id)
        if community_id is not None:
            return self._communities.get(community_id)
        return None

    def find_nearest_community(self, position: np.ndarray,
                               max_distance: float = 10.0) -> BabyCommunity | None:
        best_community = None
        best_distance = max_distance
        for community in self._communities.values():
            if len(community._members) >= self.config.max_community_size:
                continue
            avg_pos = np.zeros(3)
            count = 0
            for member_id in community._members:
                avg_pos += np.array([32, 0, 32], dtype=np.float64)
                count += 1
            if count > 0:
                avg_pos /= count
                dist = np.linalg.norm(position - avg_pos)
                if dist < best_distance:
                    best_distance = dist
                    best_community = community
        return best_community

    def tick(self):
        for community in self._communities.values():
            community.tick()

    def get_all_communities(self) -> list[BabyCommunity]:
        return list(self._communities.values())

    def summary(self) -> dict:
        total_members = sum(c.summary()["size"] for c in self._communities.values())
        return {
            "communities": len(self._communities),
            "total_members": total_members,
            "community_sizes": [c.summary()["size"] for c in self._communities.values()],
        }
