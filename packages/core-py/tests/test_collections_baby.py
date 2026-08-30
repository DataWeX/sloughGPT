import numpy as np
import pytest

from domains.collections.baby_building import (
    StructureType, ToolType, BuildingConfig, Structure, Tool,
    BabyBuilding, BuildingRegistry,
)
from domains.collections.baby_community import (
    CommunityRole, RelationType, CommunityConfig, Relation, Member,
    BabyCommunity, BabyCommunitySystem,
)
from domains.collections.baby_economy import (
    ResourceType, TradeStatus, EconomyConfig, Resource, TradeOffer,
    BabyEconomy, MarketSystem,
)
from domains.collections.baby_evolution import (
    EvolutionConfig, Genome, FitnessTracker, SelectionOperator,
    EvolutionEngine,
)
from domains.collections.baby_social import (
    SocialConfig, Message, BabySocial, BabyCultural, BabySocialSystem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEntity:
    def __init__(self, eid: int = 1):
        self.id = eid
        self.entity_type = 1


class FakeBaby:
    def __init__(self, eid: int = 1, energy: float = 50.0,
                 position=None, alive: bool = True):
        self.entity = FakeEntity(eid)
        self.energy = energy
        self.position = np.array(position if position else [32, 0, 32],
                                 dtype=np.float64)
        self.alive = alive


class FakeWorldGrid:
    def __init__(self, size=(64, 4, 64)):
        self.size = size
        self.nx, self.ny, self.nz = size
        self.total = self.nx * self.ny * self.nz
        self.material = np.zeros(self.total, dtype=np.int32)
        self.energy = np.zeros(self.total, dtype=np.float32)

    def idx(self, x, y, z):
        return (x % self.nx) * self.ny * self.nz + (y % self.ny) * self.nz + (z % self.nz)


# ===========================================================================
# 1. baby_building.py
# ===========================================================================

class TestStructureType:
    def test_enum_values(self):
        assert StructureType.WALL == 1
        assert StructureType.LIBRARY == 8
        assert len(StructureType) == 8

    def test_name_lookup(self):
        assert StructureType(3).name == "STORAGE"


class TestToolType:
    def test_enum_values(self):
        assert ToolType.PICKAXE == 1
        assert ToolType.TOOLBOX == 8
        assert len(ToolType) == 8


class TestBuildingConfig:
    def test_defaults(self):
        cfg = BuildingConfig()
        assert cfg.build_range == 2
        assert cfg.build_cost_energy == 10.0
        assert cfg.max_structures == 50
        assert cfg.structure_health_decay == 0.01

    def test_custom(self):
        cfg = BuildingConfig(max_structures=5, build_cost_energy=20.0)
        assert cfg.max_structures == 5
        assert cfg.build_cost_energy == 20.0


class TestStructure:
    def test_alive(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 2, 3))
        assert s.is_alive()

    def test_dead(self):
        s = Structure(structure_type=StructureType.WALL, position=(1, 2, 3))
        s.health = 0.0
        assert not s.is_alive()

    def test_take_damage(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        s.take_damage(30.0)
        assert s.health == pytest.approx(70.0)

    def test_take_damage_floors_at_zero(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        s.take_damage(200.0)
        assert s.health == 0.0

    def test_repair(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        s.take_damage(50.0)
        s.repair(20.0)
        assert s.health == pytest.approx(70.0)

    def test_repair_caps_at_100(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        s.repair(100.0)
        assert s.health == 100.0

    def test_upgrade(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        s.take_damage(40.0)
        assert s.level == 1
        result = s.upgrade()
        assert result is True
        assert s.level == 2
        assert s.health == 100.0

    def test_upgrade_caps_at_5(self):
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0), level=5)
        result = s.upgrade()
        assert result is False
        assert s.level == 5

    def test_default_health(self):
        s = Structure(structure_type=StructureType.FLOOR, position=(0, 0, 0))
        assert s.health == 100.0
        assert s.owner_id == 0
        assert s.level == 1
        assert s.stored_items == 0
        assert s.metadata == {}


class TestTool:
    def test_usable(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        assert t.is_usable()

    def test_broken(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.durability = 0.0
        assert not t.is_usable()

    def test_use(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(25.0)
        assert t.durability == pytest.approx(75.0)

    def test_use_floors_at_zero(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(200.0)
        assert t.durability == 0.0

    def test_repair(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(50.0)
        t.repair(10.0)
        assert t.durability == pytest.approx(60.0)

    def test_repair_caps_at_100(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.repair(100.0)
        assert t.durability == 100.0

    def test_upgrade(self):
        t = Tool(tool_type=ToolType.PICKAXE)
        t.use(40.0)
        result = t.upgrade()
        assert result is True
        assert t.level == 2
        assert t.efficiency == pytest.approx(1.2)
        assert t.durability == 100.0

    def test_upgrade_caps_at_5(self):
        t = Tool(tool_type=ToolType.PICKAXE, level=5)
        assert t.upgrade() is False


class TestBabyBuilding:
    def _make(self, energy=100.0, max_struct=50, max_tools=10):
        baby = FakeBaby(energy=energy)
        cfg = BuildingConfig(max_structures=max_struct, max_tools=max_tools)
        return BabyBuilding(baby, cfg)

    def test_can_build_ok(self):
        bb = self._make()
        result = bb.can_build(StructureType.WALL)
        assert result["can_build"] is True

    def test_can_build_no_energy(self):
        bb = self._make(energy=0.0)
        result = bb.can_build(StructureType.WALL)
        assert result["can_build"] is False
        assert result["reason"] == "insufficient_energy"

    def test_can_build_max_structures(self):
        bb = self._make(max_struct=0)
        result = bb.can_build(StructureType.WALL)
        assert result["can_build"] is False
        assert result["reason"] == "max_structures"

    def test_build(self):
        baby = FakeBaby(eid=7, energy=100.0)
        bb = BabyBuilding(baby)
        s = bb.build(StructureType.WALL, (1, 2, 3))
        assert s is not None
        assert s.structure_type == StructureType.WALL
        assert s.position == (1, 2, 3)
        assert s.owner_id == 7
        assert baby.energy == pytest.approx(90.0)

    def test_build_fail(self):
        bb = self._make(energy=0.0)
        s = bb.build(StructureType.WALL, (0, 0, 0))
        assert s is None

    def test_can_craft_ok(self):
        bb = self._make()
        result = bb.can_craft(ToolType.PICKAXE)
        assert result["can_craft"] is True

    def test_can_craft_no_energy(self):
        bb = self._make(energy=0.0)
        result = bb.can_craft(ToolType.PICKAXE)
        assert result["can_craft"] is False

    def test_can_craft_max_tools(self):
        bb = self._make(max_tools=0)
        result = bb.can_craft(ToolType.PICKAXE)
        assert result["can_craft"] is False

    def test_craft(self):
        baby = FakeBaby(eid=3, energy=100.0)
        bb = BabyBuilding(baby)
        t = bb.craft(ToolType.HAMMER)
        assert t is not None
        assert t.tool_type == ToolType.HAMMER
        assert t.owner_id == 3
        assert baby.energy == pytest.approx(85.0)

    def test_craft_fail(self):
        bb = self._make(energy=0.0)
        t = bb.craft(ToolType.PICKAXE)
        assert t is None

    def test_get_structure(self):
        bb = self._make()
        bb.build(StructureType.WALL, (0, 0, 0))
        found = bb.get_structure(StructureType.WALL)
        assert found is not None
        assert bb.get_structure(StructureType.FLOOR) is None

    def test_get_structure_ignores_dead(self):
        bb = self._make()
        s = bb.build(StructureType.WALL, (0, 0, 0))
        s.health = 0.0
        assert bb.get_structure(StructureType.WALL) is None

    def test_get_tool(self):
        bb = self._make()
        bb.craft(ToolType.LANTERN)
        assert bb.get_tool(ToolType.LANTERN) is not None
        assert bb.get_tool(ToolType.SHOVEL) is None

    def test_get_tool_ignores_broken(self):
        bb = self._make()
        t = bb.craft(ToolType.LANTERN)
        t.durability = 0.0
        assert bb.get_tool(ToolType.LANTERN) is None

    def test_repair_structure(self):
        baby = FakeBaby(energy=100.0)
        bb = BabyBuilding(baby)
        s = bb.build(StructureType.WALL, (0, 0, 0))
        s.take_damage(40.0)
        result = bb.repair_structure(s)
        assert result is True
        assert s.health == pytest.approx(80.0)
        assert baby.energy == pytest.approx(85.0)

    def test_repair_structure_no_energy(self):
        baby = FakeBaby(energy=0.0)
        bb = BabyBuilding(baby)
        s = Structure(structure_type=StructureType.WALL, position=(0, 0, 0))
        result = bb.repair_structure(s)
        assert result is False

    def test_repair_structure_with_hammer(self):
        baby = FakeBaby(energy=100.0)
        bb = BabyBuilding(baby)
        hammer = bb.craft(ToolType.HAMMER)
        s = bb.build(StructureType.WALL, (0, 0, 0))
        s.take_damage(40.0)
        bb.repair_structure(s)
        assert s.health == pytest.approx(80.0)
        assert hammer.durability == pytest.approx(99.0)

    def test_use_tool_ok(self):
        bb = self._make()
        t = bb.craft(ToolType.PICKAXE)
        result = bb.use_tool(t, "mine")
        assert result["success"] is True
        assert result["action"] == "mine"
        assert t.durability == pytest.approx(99.0)

    def test_use_tool_broken(self):
        bb = self._make()
        t = bb.craft(ToolType.PICKAXE)
        t.durability = 0.0
        result = bb.use_tool(t)
        assert result["success"] is False

    def test_store_item(self):
        bb = self._make()
        s = bb.build(StructureType.STORAGE, (0, 0, 0))
        assert bb.store_item(s, 5) is True
        assert s.stored_items == 5

    def test_store_item_wrong_type(self):
        bb = self._make()
        s = bb.build(StructureType.WALL, (0, 0, 0))
        assert bb.store_item(s) is False

    def test_retrieve_item(self):
        bb = self._make()
        s = bb.build(StructureType.STORAGE, (0, 0, 0))
        s.stored_items = 10
        assert bb.retrieve_item(s, 3) is True
        assert s.stored_items == 7

    def test_retrieve_item_insufficient(self):
        bb = self._make()
        s = bb.build(StructureType.STORAGE, (0, 0, 0))
        s.stored_items = 2
        assert bb.retrieve_item(s, 5) is False
        assert s.stored_items == 2

    def test_retrieve_item_wrong_type(self):
        bb = self._make()
        s = bb.build(StructureType.WALL, (0, 0, 0))
        assert bb.retrieve_item(s) is False

    def test_tick_removes_dead_structures(self):
        bb = self._make()
        s = bb.build(StructureType.WALL, (0, 0, 0))
        s.health = 0.005
        bb.tick()
        assert len(bb._structures) == 0

    def test_tick_decays_health(self):
        bb = self._make()
        s = bb.build(StructureType.WALL, (0, 0, 0))
        bb.tick()
        assert s.health == pytest.approx(99.99)

    def test_summary(self):
        bb = self._make()
        bb.build(StructureType.WALL, (0, 0, 0))
        bb.build(StructureType.WALL, (1, 1, 1))
        bb.build(StructureType.STORAGE, (2, 2, 2))
        bb.craft(ToolType.PICKAXE)
        s = bb.summary()
        assert s["structures"] == 3
        assert s["tools"] == 1
        assert s["build_count"] == 3
        assert s["craft_count"] == 1
        assert s["structure_counts"]["WALL"] == 2
        assert s["structure_counts"]["STORAGE"] == 1
        assert s["tool_counts"]["PICKAXE"] == 1


class TestBuildingRegistry:
    def _make_registry_with_babies(self, count=3):
        reg = BuildingRegistry()
        for i in range(count):
            baby = FakeBaby(eid=i, energy=100.0)
            bb = BabyBuilding(baby)
            reg.register(i, bb)
        return reg

    def test_register_and_get(self):
        reg = self._make_registry_with_babies(2)
        assert reg.get(0) is not None
        assert reg.get(1) is not None
        assert reg.get(99) is None

    def test_unregister(self):
        reg = self._make_registry_with_babies(2)
        reg.unregister(0)
        assert reg.get(0) is None

    def test_unregister_nonexistent(self):
        reg = BuildingRegistry()
        reg.unregister(999)

    def test_tick(self):
        reg = self._make_registry_with_babies(2)
        b0 = reg.get(0)
        b0.build(StructureType.WALL, (0, 0, 0))
        reg.tick()
        for b in reg._buildings.values():
            for s in b._structures:
                assert s.health < 100.0

    def test_find_structures(self):
        reg = self._make_registry_with_babies(2)
        reg.get(0).build(StructureType.WALL, (10, 0, 10))
        reg.get(1).build(StructureType.WALL, (11, 0, 11))
        reg.tick()
        found = reg.find_structures(StructureType.WALL)
        assert len(found) == 2

    def test_find_structures_with_radius(self):
        reg = self._make_registry_with_babies(2)
        reg.get(0).build(StructureType.WALL, (10, 0, 10))
        reg.get(1).build(StructureType.WALL, (50, 0, 50))
        reg.tick()
        found = reg.find_structures(StructureType.WALL, radius=5, center=(10, 0, 10))
        assert len(found) == 1

    def test_find_structures_wrong_type(self):
        reg = self._make_registry_with_babies(1)
        reg.get(0).build(StructureType.WALL, (0, 0, 0))
        reg.tick()
        found = reg.find_structures(StructureType.LIBRARY)
        assert len(found) == 0

    def test_summary(self):
        reg = self._make_registry_with_babies(2)
        reg.get(0).build(StructureType.WALL, (0, 0, 0))
        reg.get(1).craft(ToolType.PICKAXE)
        s = reg.summary()
        assert s["registered_builders"] == 2
        assert s["total_structures"] == 1
        assert s["total_tools"] == 1


# ===========================================================================
# 2. baby_community.py
# ===========================================================================

class TestCommunityRole:
    def test_enum(self):
        assert CommunityRole.LEADER == 1
        assert CommunityRole.MEMBER == 7
        assert len(CommunityRole) == 7


class TestRelationType:
    def test_enum(self):
        assert RelationType.FRIEND == 1
        assert RelationType.STRANGER == 6
        assert len(RelationType) == 6


class TestCommunityConfig:
    def test_defaults(self):
        cfg = CommunityConfig()
        assert cfg.max_community_size == 20
        assert cfg.loyalty_decay == 0.98

    def test_custom(self):
        cfg = CommunityConfig(max_community_size=5)
        assert cfg.max_community_size == 5


class TestRelation:
    def test_strengthen(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND)
        r.strengthen(0.3)
        assert r.strength == pytest.approx(0.8)
        assert r.interactions == 1

    def test_strengthen_caps_at_1(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND,
                     strength=0.9)
        r.strengthen(0.5)
        assert r.strength == pytest.approx(1.0)

    def test_weaken(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND)
        r.weaken(0.2)
        assert r.strength == pytest.approx(0.3)
        assert r.interactions == 1

    def test_weaken_floors_at_0(self):
        r = Relation(baby_a=1, baby_b=2, relation_type=RelationType.FRIEND,
                     strength=0.1)
        r.weaken(0.5)
        assert r.strength == pytest.approx(0.0)


class TestMember:
    def test_defaults(self):
        m = Member(baby_id=1)
        assert m.role == CommunityRole.MEMBER
        assert m.loyalty == 0.5
        assert m.reputation == 0.5
        assert m.contributions == 0


class TestBabyCommunity:
    def _make(self, max_size=20):
        cfg = CommunityConfig(max_community_size=max_size)
        return BabyCommunity(community_id=0, config=cfg)

    def test_add_member(self):
        c = self._make()
        assert c.add_member(1) is True
        assert c.add_member(2) is True
        assert len(c._members) == 2

    def test_add_member_duplicate(self):
        c = self._make()
        c.add_member(1)
        assert c.add_member(1) is False

    def test_add_member_full(self):
        c = self._make(max_size=2)
        c.add_member(1)
        c.add_member(2)
        assert c.add_member(3) is False

    def test_remove_member(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        assert c.remove_member(1) is True
        assert len(c._members) == 1
        assert c.get_relation(1, 2) is None

    def test_remove_member_nonexistent(self):
        c = self._make()
        assert c.remove_member(99) is False

    def test_add_relation_on_join(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        r = c.get_relation(1, 2)
        assert r is not None
        assert r.relation_type == RelationType.STRANGER

    def test_get_relation(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        r = c.get_relation(2, 1)
        assert r is not None
        assert r.baby_a == 1
        assert r.baby_b == 2

    def test_update_relation(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        c.update_relation(1, 2, RelationType.FRIEND, 0.3)
        r = c.get_relation(1, 2)
        assert r.relation_type == RelationType.FRIEND
        assert r.strength == pytest.approx(0.8)
        assert r.interactions == 1

    def test_update_relation_negative(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        c.update_relation(1, 2, RelationType.RIVAL, -0.1)
        r = c.get_relation(1, 2)
        assert r.relation_type == RelationType.RIVAL
        assert r.strength == pytest.approx(0.4)

    def test_update_relation_new(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        c.update_relation(1, 2, RelationType.ALLY)
        r = c.get_relation(1, 2)
        assert r is not None
        assert r.relation_type == RelationType.ALLY

    def test_set_role(self):
        c = self._make()
        c.add_member(1)
        assert c.set_role(1, CommunityRole.LEADER) is True
        assert c._members[1].role == CommunityRole.LEADER

    def test_set_role_nonexistent(self):
        c = self._make()
        assert c.set_role(99, CommunityRole.LEADER) is False

    def test_get_members_by_role(self):
        c = self._make()
        c.add_member(1, CommunityRole.BUILDER)
        c.add_member(2, CommunityRole.BUILDER)
        c.add_member(3, CommunityRole.HUNTER)
        builders = c.get_members_by_role(CommunityRole.BUILDER)
        assert sorted(builders) == [1, 2]

    def test_get_leader(self):
        c = self._make()
        assert c.get_leader() is None
        c.add_member(1, CommunityRole.LEADER)
        assert c.get_leader() == 1

    def test_elect_leader(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        c.add_member(3)
        scores = {1: 0.5, 2: 1.0, 3: 0.3}
        winner = c.elect_leader(scores)
        assert winner == 2
        assert c.get_leader() == 2

    def test_elect_leader_no_candidates(self):
        c = self._make()
        c.add_member(1)
        scores = {1: 0.01}
        winner = c.elect_leader(scores)
        assert winner is None

    def test_add_contribution(self):
        c = self._make()
        c.add_member(1)
        c.add_contribution(1, 5)
        assert c._members[1].contributions == 5
        assert c._members[1].reputation == pytest.approx(0.6)

    def test_contribute_resource(self):
        c = self._make()
        c.contribute_resource("food", 100)
        assert c.get_resource("food") == 100

    def test_get_resource(self):
        c = self._make()
        assert c.get_resource("missing") is None

    def test_tick_decay_loyalty(self):
        c = self._make()
        c.add_member(1)
        initial = c._members[1].loyalty
        c.tick()
        assert c._members[1].loyalty == pytest.approx(initial * 0.98)

    def test_get_top_contributors(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        c.add_member(3)
        c.add_contribution(2, 10)
        c.add_contribution(3, 5)
        c.add_contribution(1, 1)
        top = c.get_top_contributors(2)
        assert top == [2, 3]

    def test_get_cohesion_empty(self):
        c = self._make()
        assert c.get_cohesion() == 0.0

    def test_get_cohesion(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        r = c.get_relation(1, 2)
        r.trust = 0.8
        assert c.get_cohesion() == pytest.approx(0.8)

    def test_summary(self):
        c = self._make()
        c.add_member(1)
        c.add_member(2)
        s = c.summary()
        assert s["id"] == 0
        assert s["size"] == 2
        assert s["relations"] == 1


class TestBabyCommunitySystem:
    def _make(self):
        return BabyCommunitySystem()

    def test_create_community(self):
        sys = self._make()
        c1 = sys.create_community()
        c2 = sys.create_community()
        assert c1.id == 0
        assert c2.id == 1

    def test_disband_community(self):
        sys = self._make()
        c = sys.create_community()
        sys.add_baby_to_community(1, 0)
        assert sys.disband_community(0) is True
        assert sys._communities.get(0) is None

    def test_disband_nonexistent(self):
        sys = self._make()
        assert sys.disband_community(99) is False

    def test_add_baby_to_community(self):
        sys = self._make()
        sys.create_community()
        assert sys.add_baby_to_community(1, 0) is True
        assert sys.get_baby_community(1) is not None

    def test_add_baby_nonexistent_community(self):
        sys = self._make()
        assert sys.add_baby_to_community(1, 99) is False

    def test_add_baby_already_in_community(self):
        sys = self._make()
        sys.create_community()
        sys.add_baby_to_community(1, 0)
        assert sys.add_baby_to_community(1, 0) is False

    def test_remove_baby_from_community(self):
        sys = self._make()
        sys.create_community()
        sys.add_baby_to_community(1, 0)
        assert sys.remove_baby_from_community(1) is True
        assert sys.get_baby_community(1) is None

    def test_remove_baby_not_in_any(self):
        sys = self._make()
        assert sys.remove_baby_from_community(99) is False

    def test_get_baby_community(self):
        sys = self._make()
        sys.create_community()
        sys.add_baby_to_community(1, 0)
        c = sys.get_baby_community(1)
        assert c is not None
        assert sys.get_baby_community(99) is None

    def test_tick(self):
        sys = self._make()
        sys.create_community()
        sys.add_baby_to_community(1, 0)
        sys.tick()
        c = sys._communities[0]
        assert c._members[1].loyalty < 0.5

    def test_get_all_communities(self):
        sys = self._make()
        sys.create_community()
        sys.create_community()
        assert len(sys.get_all_communities()) == 2

    def test_summary(self):
        sys = self._make()
        sys.create_community()
        sys.add_baby_to_community(1, 0)
        sys.add_baby_to_community(2, 0)
        s = sys.summary()
        assert s["communities"] == 1
        assert s["total_members"] == 2


# ===========================================================================
# 3. baby_economy.py
# ===========================================================================

class TestResourceType:
    def test_enum(self):
        assert ResourceType.FOOD == 1
        assert ResourceType.CURRENCY == 8
        assert len(ResourceType) == 8


class TestTradeStatus:
    def test_enum(self):
        assert TradeStatus.PENDING == 1
        assert TradeStatus.CANCELLED == 5
        assert len(TradeStatus) == 5


class TestEconomyConfig:
    def test_defaults(self):
        cfg = EconomyConfig()
        assert cfg.max_inventory == 100
        assert cfg.tax_rate == 0.05

    def test_custom(self):
        cfg = EconomyConfig(max_inventory=50)
        assert cfg.max_inventory == 50


class TestResource:
    def test_split(self):
        r = Resource(ResourceType.WOOD, amount=10.0, quality=0.8)
        split = r.split(3.0)
        assert split.amount == pytest.approx(3.0)
        assert split.quality == 0.8
        assert split.resource_type == ResourceType.WOOD
        assert r.amount == pytest.approx(7.0)

    def test_split_exceeds_amount(self):
        r = Resource(ResourceType.WOOD, amount=2.0)
        split = r.split(10.0)
        assert split.amount == pytest.approx(2.0)
        assert r.amount == pytest.approx(0.0)

    def test_merge_same_type(self):
        r1 = Resource(ResourceType.WOOD, amount=5.0, quality=0.5)
        r2 = Resource(ResourceType.WOOD, amount=5.0, quality=1.0)
        r1.merge(r2)
        assert r1.amount == pytest.approx(10.0)
        assert r1.quality == pytest.approx(0.75)

    def test_merge_different_type_noop(self):
        r1 = Resource(ResourceType.WOOD, amount=5.0)
        r2 = Resource(ResourceType.STONE, amount=5.0)
        r1.merge(r2)
        assert r1.amount == pytest.approx(5.0)
        assert r1.resource_type == ResourceType.WOOD

    def test_value(self):
        r = Resource(ResourceType.WOOD, amount=10.0, quality=0.5)
        assert r.value == pytest.approx(5.0)

    def test_split_preserves_metadata(self):
        r = Resource(ResourceType.WOOD, amount=5.0, metadata={"tag": "rare"})
        s = r.split(2.0)
        assert s.metadata["tag"] == "rare"


class TestTradeOffer:
    def _make_offer(self, status=TradeStatus.PENDING):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
            status=status,
        )
        return offer

    def test_accept(self):
        o = self._make_offer()
        assert o.accept() is True
        assert o.status == TradeStatus.ACCEPTED

    def test_accept_not_pending(self):
        o = self._make_offer(TradeStatus.ACCEPTED)
        assert o.accept() is False

    def test_reject(self):
        o = self._make_offer()
        assert o.reject() is True
        assert o.status == TradeStatus.REJECTED

    def test_reject_not_pending(self):
        o = self._make_offer(TradeStatus.COMPLETED)
        assert o.reject() is False

    def test_complete(self):
        o = self._make_offer(TradeStatus.ACCEPTED)
        assert o.complete() is True
        assert o.status == TradeStatus.COMPLETED

    def test_complete_not_accepted(self):
        o = self._make_offer()
        assert o.complete() is False

    def test_cancel(self):
        o = self._make_offer()
        assert o.cancel() is True
        assert o.status == TradeStatus.CANCELLED

    def test_cancel_completed(self):
        o = self._make_offer(TradeStatus.COMPLETED)
        assert o.cancel() is False

    def test_cancel_rejected(self):
        o = self._make_offer(TradeStatus.REJECTED)
        assert o.cancel() is False


class TestBabyEconomy:
    def _make(self, inventory_max=100):
        cfg = EconomyConfig(max_inventory=inventory_max)
        return BabyEconomy(baby_id=1, config=cfg)

    def test_add_resource(self):
        e = self._make()
        r = Resource(ResourceType.WOOD, amount=10.0)
        added = e.add_resource(r)
        assert added == pytest.approx(10.0)
        stored = e.get_resource(ResourceType.WOOD)
        assert stored is not None
        assert stored.amount == pytest.approx(10.0)

    def test_add_resource_merges(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=5.0))
        e.add_resource(Resource(ResourceType.WOOD, amount=3.0))
        stored = e.get_resource(ResourceType.WOOD)
        assert stored.amount == pytest.approx(8.0)

    def test_add_resource_capacity_limit(self):
        e = self._make(inventory_max=10)
        e.add_resource(Resource(ResourceType.WOOD, amount=8.0))
        e.add_resource(Resource(ResourceType.WOOD, amount=8.0))
        stored = e.get_resource(ResourceType.WOOD)
        assert stored.amount == pytest.approx(10.0)

    def test_remove_resource(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        removed = e.remove_resource(ResourceType.WOOD, 3.0)
        assert removed is not None
        assert removed.amount == pytest.approx(3.0)
        assert e.get_resource(ResourceType.WOOD).amount == pytest.approx(7.0)

    def test_remove_resource_not_enough(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=2.0))
        removed = e.remove_resource(ResourceType.WOOD, 5.0)
        assert removed is None

    def test_remove_resource_deletes_when_empty(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=5.0))
        e.remove_resource(ResourceType.WOOD, 5.0)
        assert e.get_resource(ResourceType.WOOD) is None

    def test_remove_resource_nonexistent(self):
        e = self._make()
        removed = e.remove_resource(ResourceType.WOOD, 1.0)
        assert removed is None

    def test_get_resource(self):
        e = self._make()
        assert e.get_resource(ResourceType.WOOD) is None
        e.add_resource(Resource(ResourceType.WOOD, amount=1.0))
        assert e.get_resource(ResourceType.WOOD) is not None

    def test_get_inventory_value(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0, quality=0.5))
        e.add_resource(Resource(ResourceType.STONE, amount=2.0, quality=1.0))
        val = e.get_inventory_value()
        assert val == pytest.approx(7.0)

    def test_get_inventory_summary(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0, quality=0.8))
        s = e.get_inventory_summary()
        assert "WOOD" in s
        assert s["WOOD"]["amount"] == pytest.approx(10.0)

    def test_create_offer(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        offer = e.create_offer(
            buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        assert offer is not None
        assert offer.seller_id == 1
        assert offer.status == TradeStatus.PENDING

    def test_create_offer_insufficient_resource(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=2.0))
        offer = e.create_offer(
            buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        assert offer is None

    def test_create_offer_no_resource(self):
        e = self._make()
        offer = e.create_offer(
            buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=1.0),
            want_resource=Resource(ResourceType.STONE, amount=1.0),
        )
        assert offer is None

    def test_create_offer_max_offers(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=100.0))
        cfg = EconomyConfig(max_offers=2)
        e2 = BabyEconomy(baby_id=1, config=cfg)
        e2.add_resource(Resource(ResourceType.WOOD, amount=100.0))
        e2.create_offer(2, Resource(ResourceType.WOOD, 1.0), Resource(ResourceType.STONE, 1.0))
        e2.create_offer(2, Resource(ResourceType.WOOD, 1.0), Resource(ResourceType.STONE, 1.0))
        o3 = e2.create_offer(2, Resource(ResourceType.WOOD, 1.0), Resource(ResourceType.STONE, 1.0))
        assert o3 is None

    def test_accept_offer(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        offer = e.create_offer(
            buyer_id=1,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = e.accept_offer(offer)
        assert result is True
        assert offer.status == TradeStatus.ACCEPTED

    def test_accept_offer_wrong_buyer(self):
        e1 = BabyEconomy(baby_id=1)
        e1.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        offer = e1.create_offer(
            buyer_id=3,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        e2 = BabyEconomy(baby_id=2)
        result = e2.accept_offer(offer)
        assert result is False

    def test_execute_trade_with(self):
        seller = BabyEconomy(baby_id=1)
        buyer = BabyEconomy(baby_id=2)
        seller.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        buyer.add_resource(Resource(ResourceType.STONE, amount=10.0))

        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = buyer.execute_trade_with(offer, seller)
        assert result is True
        assert offer.status == TradeStatus.COMPLETED
        assert buyer.get_resource(ResourceType.WOOD).amount == pytest.approx(5.0)
        assert seller.get_resource(ResourceType.STONE).amount == pytest.approx(3.0)
        assert seller.get_resource(ResourceType.WOOD).amount == pytest.approx(5.0)
        assert buyer.get_resource(ResourceType.STONE).amount == pytest.approx(7.0)

    def test_execute_trade_insufficient_seller(self):
        seller = BabyEconomy(baby_id=1)
        buyer = BabyEconomy(baby_id=2)
        seller.add_resource(Resource(ResourceType.WOOD, amount=2.0))
        buyer.add_resource(Resource(ResourceType.STONE, amount=10.0))

        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = buyer.execute_trade_with(offer, seller)
        assert result is False
        assert offer.status == TradeStatus.CANCELLED

    def test_execute_trade_insufficient_buyer(self):
        seller = BabyEconomy(baby_id=1)
        buyer = BabyEconomy(baby_id=2)
        seller.add_resource(Resource(ResourceType.WOOD, amount=10.0))

        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = buyer.execute_trade_with(offer, seller)
        assert result is False

    def test_price_tracking(self):
        e = self._make()
        e._update_price(ResourceType.WOOD, 5.0)
        e._update_price(ResourceType.WOOD, 10.0)
        avg = e.get_average_price(ResourceType.WOOD)
        assert avg == pytest.approx(7.5)

    def test_price_history_limit(self):
        cfg = EconomyConfig(price_history_size=3)
        e = BabyEconomy(baby_id=1, config=cfg)
        for i in range(10):
            e._update_price(ResourceType.WOOD, float(i))
        assert len(e._price_history[ResourceType.WOOD]) == 3

    def test_price_trend(self):
        e = self._make()
        for _ in range(15):
            e._update_price(ResourceType.WOOD, 1.0)
        for _ in range(15):
            e._update_price(ResourceType.WOOD, 10.0)
        trend = e.get_price_trend(ResourceType.WOOD)
        assert trend > 0

    def test_price_trend_no_history(self):
        e = self._make()
        assert e.get_price_trend(ResourceType.WOOD) == 0.0

    def test_get_average_price_no_history(self):
        e = self._make()
        assert e.get_average_price(ResourceType.WOOD) == 0.0

    def test_tick(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        offer = e.create_offer(
            buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        e.tick()
        assert offer.metadata.get("ticks_alive", 0) == 1

    def test_summary(self):
        e = self._make()
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        s = e.summary()
        assert s["baby_id"] == 1
        assert s["inventory_items"] == 1
        assert s["active_offers"] == 0


class TestMarketSystem:
    def _make(self):
        return MarketSystem()

    def test_register_and_get(self):
        m = self._make()
        e = BabyEconomy(baby_id=1)
        m.register(1, e)
        assert m.get_economy(1) is e
        assert m.get_economy(99) is None

    def test_unregister(self):
        m = self._make()
        m.register(1, BabyEconomy(baby_id=1))
        m.unregister(1)
        assert m.get_economy(1) is None

    def test_list_offers(self):
        m = self._make()
        e1 = BabyEconomy(baby_id=1)
        e1.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        e1.create_offer(2, Resource(ResourceType.WOOD, 5.0), Resource(ResourceType.STONE, 3.0))
        m.register(1, e1)
        offers = m.list_offers()
        assert len(offers) == 1

    def test_list_offers_filter_type(self):
        m = self._make()
        e1 = BabyEconomy(baby_id=1)
        e1.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        e1.add_resource(Resource(ResourceType.STONE, amount=10.0))
        e1.create_offer(2, Resource(ResourceType.WOOD, 5.0), Resource(ResourceType.STONE, 3.0))
        e1.create_offer(2, Resource(ResourceType.STONE, 3.0), Resource(ResourceType.WOOD, 5.0))
        m.register(1, e1)
        wood_offers = m.list_offers(ResourceType.WOOD)
        assert len(wood_offers) == 1

    def test_find_trade(self):
        m = self._make()
        e1 = BabyEconomy(baby_id=1)
        e1.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        e1.create_offer(2, Resource(ResourceType.WOOD, 5.0), Resource(ResourceType.STONE, 3.0))
        m.register(1, e1)
        found = m.find_trade(2, ResourceType.STONE, ResourceType.WOOD)
        assert found is not None

    def test_find_trade_not_found(self):
        m = self._make()
        assert m.find_trade(2, ResourceType.STONE, ResourceType.WOOD) is None

    def test_execute_trade(self):
        m = self._make()
        seller = BabyEconomy(baby_id=1)
        buyer = BabyEconomy(baby_id=2)
        seller.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        buyer.add_resource(Resource(ResourceType.STONE, amount=10.0))
        m.register(1, seller)
        m.register(2, buyer)

        offer = seller.create_offer(
            buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = m.execute_trade(offer, buyer)
        assert result is True

    def test_execute_trade_unknown_seller(self):
        m = self._make()
        buyer = BabyEconomy(baby_id=2)
        m.register(2, buyer)
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.WOOD, amount=5.0),
            want_resource=Resource(ResourceType.STONE, amount=3.0),
        )
        result = m.execute_trade(offer, buyer)
        assert result is False

    def test_tick(self):
        m = self._make()
        e = BabyEconomy(baby_id=1)
        m.register(1, e)
        e.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        e.create_offer(2, Resource(ResourceType.WOOD, 5.0), Resource(ResourceType.STONE, 3.0))
        m.tick()
        assert e._active_offers[0].metadata.get("ticks_alive", 0) == 1

    def test_get_market_stats(self):
        m = self._make()
        e1 = BabyEconomy(baby_id=1)
        e1.add_resource(Resource(ResourceType.WOOD, amount=10.0))
        m.register(1, e1)
        stats = m.get_market_stats()
        assert stats["participants"] == 1
        assert stats["active_offers"] == 0
        assert stats["total_value"] == pytest.approx(10.0)

    def test_summary(self):
        m = self._make()
        s = m.summary()
        assert "market_stats" in s
        assert s["economies"] == 0


# ===========================================================================
# 4. baby_evolution.py
# ===========================================================================

class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.population_size == 20
        assert cfg.mutation_rate == 0.1
        assert cfg.crossover_rate == 0.7

    def test_custom(self):
        cfg = EvolutionConfig(population_size=50)
        assert cfg.population_size == 50


class TestGenome:
    def test_init_random(self):
        g = Genome()
        assert len(g) == 16
        assert g.fitness == 0.0
        assert g.generation == 0

    def test_init_with_genes(self):
        genes = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        g = Genome(genes)
        assert len(g) == 3
        np.testing.assert_array_equal(g.genes, genes)

    def test_copy(self):
        g = Genome(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        g.fitness = 0.5
        g.generation = 3
        g2 = g.copy()
        assert g2.fitness == 0.5
        assert g2.generation == 3
        g2.genes[0] = 99.0
        assert g.genes[0] == 1.0

    def test_mutate(self):
        g = Genome(np.zeros(100, dtype=np.float32))
        g.mutate(rate=1.0, strength=0.1)
        changed = np.sum(g.genes != 0.0)
        assert changed > 0

    def test_crossover(self):
        g1 = Genome(np.ones(10, dtype=np.float32))
        g2 = Genome(np.zeros(10, dtype=np.float32))
        child = g1.crossover(g2)
        assert len(child) == 10
        assert id(g1) in child.parent_ids
        assert id(g2) in child.parent_ids

    def test_distance(self):
        g1 = Genome(np.zeros(5, dtype=np.float32))
        g2 = Genome(np.ones(5, dtype=np.float32))
        assert g1.distance(g2) == pytest.approx(np.sqrt(5.0))

    def test_to_dict(self):
        g = Genome(np.array([1.0, 2.0], dtype=np.float32))
        g.fitness = 0.5
        g.generation = 2
        d = g.to_dict()
        assert "genes" in d
        assert d["fitness"] == 0.5
        assert d["generation"] == 2

    def test_from_dict(self):
        d = {"genes": [1.0, 2.0, 3.0], "fitness": 0.8, "generation": 5}
        g = Genome.from_dict(d)
        assert len(g) == 3
        assert g.fitness == 0.8
        assert g.generation == 5

    def test_from_dict_defaults(self):
        d = {"genes": [1.0]}
        g = Genome.from_dict(d)
        assert g.fitness == 0.0
        assert g.generation == 0

    def test_gene_names(self):
        g = Genome(np.array([1.0, 2.0], dtype=np.float32), gene_names=["a", "b"])
        assert g.gene_names == ["a", "b"]

    def test_default_gene_names(self):
        g = Genome(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert g.gene_names == ["gene_0", "gene_1", "gene_2"]


class TestFitnessTracker:
    def test_record(self):
        ft = FitnessTracker()
        ft.record(1, 0.8)
        ft.record(1, 0.9)
        assert ft.get_fitness(1) == pytest.approx(0.85)

    def test_get_fitness_unknown(self):
        ft = FitnessTracker()
        assert ft.get_fitness(99) == 0.0

    def test_window_limit(self):
        ft = FitnessTracker(window=3)
        for i in range(10):
            ft.record(1, float(i))
        assert ft.get_fitness(1) == pytest.approx(np.mean([7.0, 8.0, 9.0]))

    def test_get_all_fitness(self):
        ft = FitnessTracker()
        ft.record(1, 0.5)
        ft.record(2, 0.8)
        all_f = ft.get_all_fitness()
        assert all_f[1] == pytest.approx(0.5)
        assert all_f[2] == pytest.approx(0.8)

    def test_get_trend(self):
        ft = FitnessTracker()
        for _ in range(15):
            ft.record(1, 1.0)
        for _ in range(15):
            ft.record(1, 5.0)
        trend = ft.get_trend(1)
        assert trend > 0

    def test_get_trend_short(self):
        ft = FitnessTracker()
        ft.record(1, 1.0)
        assert ft.get_trend(1) == 0.0

    def test_summary(self):
        ft = FitnessTracker()
        ft.record(1, 0.5)
        ft.record(2, 0.8)
        s = ft.summary()
        assert s["tracked_babies"] == 2
        assert s["avg_fitness"] == pytest.approx(0.65)
        assert s["max_fitness"] == pytest.approx(0.8)

    def test_summary_empty(self):
        ft = FitnessTracker()
        s = ft.summary()
        assert s["avg_fitness"] == 0.0
        assert s["max_fitness"] == 0.0


class TestSelectionOperator:
    def test_tournament_select(self):
        op = SelectionOperator(tournament_size=2)
        pop = [Genome() for _ in range(10)]
        for i, g in enumerate(pop):
            g.fitness = float(i)
        selected = op.tournament_select(pop)
        assert selected.fitness >= 0.0

    def test_rank_select(self):
        op = SelectionOperator()
        pop = [Genome() for _ in range(10)]
        for i, g in enumerate(pop):
            g.fitness = float(i)
        selected = op.rank_select(pop)
        assert selected is not None

    def test_roulette_select(self):
        op = SelectionOperator()
        pop = [Genome() for _ in range(10)]
        for i, g in enumerate(pop):
            g.fitness = float(i) + 0.1
        selected = op.roulette_select(pop)
        assert selected is not None

    def test_tournament_single_contestant(self):
        op = SelectionOperator(tournament_size=1)
        g = Genome()
        g.fitness = 1.0
        pop = [g]
        selected = op.tournament_select(pop)
        assert selected.fitness == 1.0


class TestEvolutionEngine:
    def _make(self, pop_size=5):
        cfg = EvolutionConfig(population_size=pop_size, elitism_count=1)
        return EvolutionEngine(config=cfg)

    def test_initialize_population(self):
        e = self._make()
        e.initialize_population(gene_size=8)
        assert e.population_size == 5
        assert len(e.get_population()[0]) == 8

    def test_add_genome(self):
        e = self._make(pop_size=0)
        g = Genome(np.array([1.0, 2.0], dtype=np.float32))
        e.add_genome(g)
        assert e.population_size == 1

    def test_record_fitness(self):
        e = self._make()
        e.record_fitness(1, 0.9)
        assert e._fitness_tracker.get_fitness(1) == pytest.approx(0.9)

    def test_step_small_population(self):
        e = self._make(pop_size=0)
        e.add_genome(Genome(np.ones(16, dtype=np.float32)))
        result = e.step()
        assert len(result) == 1

    def test_step_normal(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        pop = e.step()
        assert len(pop) == 5
        assert e.generation == 1

    def test_step_multiple(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        for _ in range(5):
            e.step()
        assert e.generation == 5
        assert len(e._history) == 5

    def test_get_best(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        e._population[2].fitness = 10.0
        best = e.get_best(1)
        assert best[0].fitness == 10.0

    def test_get_best_empty(self):
        e = self._make()
        assert e.get_best(1) == []

    def test_export_import(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        data = e.export_population()
        assert len(data) == 5
        e2 = self._make(pop_size=0)
        e2.import_population(data)
        assert e2.population_size == 5

    def test_diversity(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        div = e._diversity()
        assert div > 0.0

    def test_diversity_single(self):
        e = self._make(pop_size=1)
        e.add_genome(Genome())
        assert e._diversity() == 0.0

    def test_generation_stats(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        stats = e._generation_stats()
        assert "avg_fitness" in stats
        assert "diversity" in stats

    def test_summary(self):
        e = self._make(pop_size=5)
        e.initialize_population()
        s = e.summary()
        assert s["generation"] == 0
        assert s["population_size"] == 5
        assert "fitness_tracker" in s


# ===========================================================================
# 5. baby_social.py
# ===========================================================================

class TestSocialConfig:
    def test_defaults(self):
        cfg = SocialConfig()
        assert cfg.communication_range == 5.0
        assert cfg.teaching_range == 3.0
        assert cfg.imitation_chance == 0.3

    def test_custom(self):
        cfg = SocialConfig(communication_range=10.0)
        assert cfg.communication_range == 10.0


class TestMessage:
    def test_message(self):
        msg = Message(sender_id=1, receiver_id=2, content={"text": "hi"},
                      timestamp=1.0)
        assert msg.sender_id == 1
        assert msg.message_type == "info"


class TestBabySocial:
    def _make(self, eid=1, energy=50.0, position=None):
        baby = FakeBaby(eid=eid, energy=energy, position=position)
        return BabySocial(baby), baby

    def test_send_message(self):
        social, baby = self._make()
        msg = social.send_message(2, {"text": "hello"})
        assert msg.sender_id == 1
        assert msg.receiver_id == 2
        assert len(social._outbox) == 1

    def test_receive_message(self):
        social, _ = self._make()
        msg = Message(sender_id=2, receiver_id=1, content={}, timestamp=0.0)
        social.receive_message(msg)
        assert len(social._inbox) == 1

    def test_receive_message_buffer_limit(self):
        cfg = SocialConfig(max_message_buffer=3)
        baby = FakeBaby()
        social = BabySocial(baby, config=cfg)
        for i in range(5):
            social.receive_message(Message(sender_id=i, receiver_id=1,
                                           content={}, timestamp=0.0))
        assert len(social._inbox) == 3

    def test_get_messages_clears_inbox(self):
        social, _ = self._make()
        social.receive_message(Message(sender_id=2, receiver_id=1,
                                       content={}, timestamp=0.0))
        msgs = social.get_messages()
        assert len(msgs) == 1
        assert len(social._inbox) == 0

    def test_update_trust(self):
        social, _ = self._make()
        social.update_trust(2, 0.2)
        assert social.get_trust(2) == pytest.approx(0.7)

    def test_update_trust_clamp(self):
        social, _ = self._make()
        social.update_trust(2, 0.8)
        assert social.get_trust(2) == pytest.approx(1.0)
        social.update_trust(2, -2.0)
        assert social.get_trust(2) == pytest.approx(0.0)

    def test_get_trust_default(self):
        social, _ = self._make()
        assert social.get_trust(99) == 0.5

    def test_observe_action_success(self):
        social, _ = self._make()
        other = FakeBaby(eid=2, energy=40.0)
        social.observe_action(other, "move", {"success": True})
        assert len(social._social_memory) == 1

    def test_observe_action_failure(self):
        social, _ = self._make()
        other = FakeBaby(eid=2)
        social.observe_action(other, "move", {"success": False})
        assert len(social._social_memory) == 0

    def test_observe_action_buffer_limit(self):
        cfg = SocialConfig(max_message_buffer=3)
        baby = FakeBaby()
        social = BabySocial(baby, config=cfg)
        for i in range(5):
            other = FakeBaby(eid=i)
            social.observe_action(other, "move", {"success": True})
        assert len(social._social_memory) == 3

    def test_try_imitate_too_far(self):
        cfg = SocialConfig(imitation_chance=1.0)
        baby1 = FakeBaby(eid=1, position=[0, 0, 0])
        b1 = BabySocial(baby1, config=cfg)
        b2 = FakeBaby(eid=2, position=[100, 0, 0])
        result = b1.try_imitate(b2, FakeWorldGrid())
        assert result["success"] is False
        assert result["reason"] == "out_of_range"

    def test_try_imitate_low_trust(self):
        b1, _ = self._make(eid=1)
        b2 = FakeBaby(eid=2)
        b1.update_trust(2, -0.5)
        result = b1.try_imitate(b2, FakeWorldGrid())
        assert result["success"] is False

    def test_try_imitate_ok(self):
        b1, _ = self._make(eid=1, position=[32, 0, 32])
        b2 = FakeBaby(eid=2, position=[32, 0, 33])
        b1.update_trust(2, 0.5)
        np.random.seed(42)
        result = b1.try_imitate(b2, FakeWorldGrid())
        if result["success"]:
            assert result["action"] == "imitate"

    def test_try_imitate_energy_cost(self):
        cfg = SocialConfig(imitation_chance=1.0)
        baby = FakeBaby(eid=1, position=[32, 0, 32], energy=50.0)
        social = BabySocial(baby, config=cfg)
        b2 = FakeBaby(eid=2, position=[32, 0, 33])
        social.update_trust(2, 0.5)
        social.try_imitate(b2, FakeWorldGrid())
        assert baby.energy < 50.0

    def test_teach(self):
        teacher, _ = self._make(eid=1, position=[32, 0, 32], energy=50.0)
        learner_baby = FakeBaby(eid=2, position=[32, 0, 33])
        learner_social = BabySocial(learner_baby)
        learner_baby._social = learner_social

        result = teacher.teach(learner_baby, {"lesson": "fire"})
        assert result["success"] is True
        assert teacher._teach_count == 1
        assert len(learner_social._inbox) == 1

    def test_teach_out_of_range(self):
        teacher, _ = self._make(eid=1, position=[0, 0, 0], energy=50.0)
        learner = FakeBaby(eid=2, position=[100, 0, 0])
        result = teacher.teach(learner, {"lesson": "fire"})
        assert result["success"] is False
        assert result["reason"] == "out_of_range"

    def test_teach_insufficient_energy(self):
        teacher, _ = self._make(eid=1, energy=0.0)
        learner = FakeBaby(eid=2)
        result = teacher.teach(learner, {"lesson": "fire"})
        assert result["success"] is False
        assert result["reason"] == "insufficient_energy"

    def test_teach_energy_cost(self):
        teacher, _ = self._make(eid=1, position=[32, 0, 32], energy=50.0)
        learner = FakeBaby(eid=2, position=[32, 0, 33])
        learner._social = BabySocial(learner)
        teacher.teach(learner, {})
        assert teacher.baby.energy == pytest.approx(45.0)

    def test_share_knowledge(self):
        teacher, _ = self._make(eid=1, position=[32, 0, 32], energy=50.0)
        learner = FakeBaby(eid=2, position=[32, 0, 33])
        learner._social = BabySocial(learner)
        result = teacher.share_knowledge(learner)
        assert result["success"] is True

    def test_summary(self):
        social, _ = self._make()
        social.send_message(2, {"text": "hi"})
        social.update_trust(2, 0.1)
        s = social.summary()
        assert s["outbox_size"] == 1
        assert 2 in s["trust_scores"]


class TestBabyCultural:
    def _make(self):
        return BabyCultural()

    def test_record_culture(self):
        c = self._make()
        baby = FakeBaby(eid=1, energy=40.0)
        baby.position = np.array([32, 0, 32], dtype=np.float64)
        c.record_culture(baby, {"total": 5}, {"total_experiences": 10})
        assert len(c._cultural_memory) == 1
        assert len(c._current_generation) == 1

    def test_end_generation(self):
        c = self._make()
        baby = FakeBaby(eid=1, energy=40.0)
        baby.position = np.array([32, 0, 32], dtype=np.float64)
        c.record_culture(baby, {}, {})
        c.end_generation()
        assert len(c._generations) == 1
        assert len(c._current_generation) == 0

    def test_end_generation_empty(self):
        c = self._make()
        c.end_generation()
        assert len(c._generations) == 0

    def test_get_generation_stats(self):
        c = self._make()
        for i, energy in enumerate([20.0, 40.0, 60.0]):
            baby = FakeBaby(eid=i, energy=energy)
            baby.position = np.array([32, 0, 32], dtype=np.float64)
            c.record_culture(baby, {}, {})
        c.end_generation()
        stats = c.get_generation_stats(0)
        assert stats["size"] == 3
        assert stats["avg_energy"] == pytest.approx(40.0)
        assert stats["max_energy"] == pytest.approx(60.0)
        assert stats["min_energy"] == pytest.approx(20.0)

    def test_get_generation_stats_empty(self):
        c = self._make()
        assert c.get_generation_stats() == {}

    def test_get_cultural_trend(self):
        c = self._make()
        baby = FakeBaby(eid=1, energy=50.0)
        baby.position = np.array([32, 0, 32], dtype=np.float64)
        c.record_culture(baby, {}, {})
        c.end_generation()
        trend = c.get_cultural_trend()
        assert trend["generations"] == 1
        assert trend["total_records"] == 1

    def test_get_cultural_trend_empty(self):
        c = self._make()
        trend = c.get_cultural_trend()
        assert trend == {"generations": 0}

    def test_get_top_performers(self):
        c = self._make()
        for i, energy in enumerate([10.0, 80.0, 50.0]):
            baby = FakeBaby(eid=i, energy=energy)
            baby.position = np.array([32, 0, 32], dtype=np.float64)
            c.record_culture(baby, {}, {})
        top = c.get_top_performers(2)
        assert len(top) == 2
        assert top[0]["energy"] == pytest.approx(80.0)
        assert top[1]["energy"] == pytest.approx(50.0)

    def test_summary(self):
        c = self._make()
        baby = FakeBaby(eid=1, energy=30.0)
        baby.position = np.array([32, 0, 32], dtype=np.float64)
        c.record_culture(baby, {}, {})
        s = c.summary()
        assert s["total_records"] == 1
        assert s["current_generation_size"] == 1


class TestBabySocialSystem:
    def _make(self):
        return BabySocialSystem()

    def test_register_baby(self):
        sys = self._make()
        baby = FakeBaby(eid=1)
        sys.register_baby(baby)
        social = sys.get_social(baby)
        assert social is not None
        assert hasattr(baby, "_social")

    def test_unregister_baby(self):
        sys = self._make()
        baby = FakeBaby(eid=1)
        sys.register_baby(baby)
        sys.unregister_baby(baby)
        assert sys.get_social(baby) is None

    def test_get_social(self):
        sys = self._make()
        baby = FakeBaby(eid=1)
        assert sys.get_social(baby) is None
        sys.register_baby(baby)
        assert sys.get_social(baby) is not None

    def test_tick(self):
        sys = self._make()
        b1 = FakeBaby(eid=1, position=[32, 0, 32], energy=50.0)
        b2 = FakeBaby(eid=2, position=[32, 0, 33], energy=50.0)
        sys.register_baby(b1)
        sys.register_baby(b2)
        results = sys.tick([b1, b2], FakeWorldGrid())
        assert len(results) == 2
        assert len(sys._interaction_log) == 2

    def test_tick_dead_babies_excluded(self):
        sys = self._make()
        b1 = FakeBaby(eid=1, alive=True)
        b2 = FakeBaby(eid=2, alive=False)
        sys.register_baby(b1)
        sys.register_baby(b2)
        results = sys.tick([b1, b2], FakeWorldGrid())
        assert len(results) == 1

    def test_tick_teaches_nearby(self):
        sys = self._make()
        b1 = FakeBaby(eid=1, position=[32, 0, 32], energy=50.0)
        b2 = FakeBaby(eid=2, position=[32, 0, 33], energy=50.0)
        sys.register_baby(b1)
        sys.register_baby(b2)
        results = sys.tick([b1, b2], FakeWorldGrid())
        teaching_events = [r for r in results if "teach" in r]
        assert len(teaching_events) >= 1

    def test_end_generation(self):
        sys = self._make()
        sys.end_generation()
        assert len(sys._cultural._generations) == 0

    def test_record_culture(self):
        sys = self._make()
        b1 = FakeBaby(eid=1, energy=50.0)
        b1.position = np.array([32, 0, 32], dtype=np.float64)
        sys.register_baby(b1)
        sys.record_culture([b1])
        assert len(sys._cultural._cultural_memory) == 1

    def test_record_culture_dead_excluded(self):
        sys = self._make()
        b1 = FakeBaby(eid=1, alive=False)
        sys.register_baby(b1)
        sys.record_culture([b1])
        assert len(sys._cultural._cultural_memory) == 0

    def test_summary(self):
        sys = self._make()
        b1 = FakeBaby(eid=1)
        sys.register_baby(b1)
        s = sys.summary()
        assert s["social_modules"] == 1
        assert "cultural" in s
