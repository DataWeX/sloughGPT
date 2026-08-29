import numpy as np
import pytest

from domains.collections.baby_community import (
    CommunityRole, RelationType, CommunityConfig, Relation, Member,
    BabyCommunity, BabyCommunitySystem,
)


class TestRelation:
    def test_strengthen(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND)
        r.strengthen(0.2)
        assert r.strength == 0.7
        assert r.interactions == 1

    def test_strengthen_max(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND, strength=0.9)
        r.strengthen(0.2)
        assert r.strength == 1.0

    def test_weaken(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND, strength=0.8)
        r.weaken(0.3)
        assert abs(r.strength - 0.5) < 1e-6

    def test_weaken_min(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND, strength=0.1)
        r.weaken(0.5)
        assert r.strength == 0.0


class TestBabyCommunity:
    def test_add_member(self):
        community = BabyCommunity(0)
        success = community.add_member(1)
        assert success
        assert 1 in community._members

    def test_add_member_duplicate(self):
        community = BabyCommunity(0)
        community.add_member(1)
        success = community.add_member(1)
        assert not success

    def test_add_member_max_size(self):
        config = CommunityConfig(max_community_size=2)
        community = BabyCommunity(0, config)
        community.add_member(1)
        community.add_member(2)
        success = community.add_member(3)
        assert not success

    def test_remove_member(self):
        community = BabyCommunity(0)
        community.add_member(1)
        success = community.remove_member(1)
        assert success
        assert 1 not in community._members

    def test_remove_member_not_found(self):
        community = BabyCommunity(0)
        success = community.remove_member(999)
        assert not success

    def test_update_relation(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        community.update_relation(1, 2, RelationType.FRIEND, 0.2)
        relation = community.get_relation(1, 2)
        assert relation is not None
        assert relation.relation_type == RelationType.FRIEND

    def test_get_relation(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        relation = community.get_relation(1, 2)
        assert relation is not None
        assert relation.relation_type == RelationType.STRANGER

    def test_set_role(self):
        community = BabyCommunity(0)
        community.add_member(1)
        success = community.set_role(1, CommunityRole.LEADER)
        assert success
        assert community._members[1].role == CommunityRole.LEADER

    def test_get_members_by_role(self):
        community = BabyCommunity(0)
        community.add_member(1, CommunityRole.BUILDER)
        community.add_member(2, CommunityRole.BUILDER)
        community.add_member(3, CommunityRole.HUNTER)
        builders = community.get_members_by_role(CommunityRole.BUILDER)
        assert len(builders) == 2

    def test_get_leader(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.set_role(1, CommunityRole.LEADER)
        leader = community.get_leader()
        assert leader == 1

    def test_elect_leader(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        fitness = {1: 0.9, 2: 0.5}
        leader = community.elect_leader(fitness)
        assert leader == 1

    def test_add_contribution(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_contribution(1, 5)
        assert community._members[1].contributions == 5

    def test_contribute_resource(self):
        community = BabyCommunity(0)
        community.contribute_resource("food", 100)
        assert community.get_resource("food") == 100

    def test_get_cohesion(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        cohesion = community.get_cohesion()
        assert 0.0 <= cohesion <= 1.0

    def test_get_top_contributors(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        community.add_contribution(1, 10)
        community.add_contribution(2, 5)
        top = community.get_top_contributors(1)
        assert top[0] == 1

    def test_summary(self):
        community = BabyCommunity(0)
        community.add_member(1)
        community.add_member(2)
        s = community.summary()
        assert s["size"] == 2
        assert s["id"] == 0


class TestBabyCommunitySystem:
    def test_create_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        assert community.id == 0

    def test_disband_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        success = system.disband_community(community.id)
        assert success
        assert community.id not in system._communities

    def test_add_baby_to_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        success = system.add_baby_to_community(1, community.id)
        assert success
        assert system._baby_community[1] == community.id

    def test_add_baby_to_community_not_found(self):
        system = BabyCommunitySystem()
        success = system.add_baby_to_community(1, 999)
        assert not success

    def test_add_baby_already_in_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        success = system.add_baby_to_community(1, community.id)
        assert not success

    def test_remove_baby_from_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        success = system.remove_baby_from_community(1)
        assert success
        assert 1 not in system._baby_community

    def test_remove_baby_not_in_community(self):
        system = BabyCommunitySystem()
        success = system.remove_baby_from_community(999)
        assert not success

    def test_get_baby_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        result = system.get_baby_community(1)
        assert result is not None
        assert result.id == community.id

    def test_get_baby_community_none(self):
        system = BabyCommunitySystem()
        result = system.get_baby_community(999)
        assert result is None

    def test_find_nearest_community(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        position = np.array([32, 0, 32], dtype=np.float64)
        result = system.find_nearest_community(position, max_distance=100.0)
        assert result is not None

    def test_tick(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        system.tick()

    def test_get_all_communities(self):
        system = BabyCommunitySystem()
        system.create_community()
        system.create_community()
        all_communities = system.get_all_communities()
        assert len(all_communities) == 2

    def test_summary(self):
        system = BabyCommunitySystem()
        community = system.create_community()
        system.add_baby_to_community(1, community.id)
        system.add_baby_to_community(2, community.id)
        s = system.summary()
        assert s["communities"] == 1
        assert s["total_members"] == 2
