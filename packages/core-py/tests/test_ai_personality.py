"""Tests for domains.ai_personality — PersonalityType, Personality, PersonalityManager."""

from domains.ai_personality import (
    PersonalityType, Personality, PersonalityManager,
    get_personality_manager, list_personalities,
)


class TestPersonalityType:
    def test_all_members(self):
        assert len(PersonalityType) >= 4

    def test_values(self):
        assert PersonalityType.HELPFUL.value == "helpful"


class TestPersonality:
    def test_apply(self):
        p = Personality(name="test", description="t", traits={}, examples=[])
        assert p.apply("Hello") == "Hello"

    def test_modify_temperature(self):
        p = Personality(name="test", description="t", traits={"creativity": 0.5}, examples=[])
        result = p.modify_temperature(1.0)
        assert result == 1.0 * (0.5 + 0.5)

    def test_low_creativity(self):
        p = Personality(name="test", description="t", traits={"creativity": 0.0}, examples=[])
        result = p.modify_temperature(1.0)
        assert result == 0.5


class TestPersonalityManager:
    def test_default(self):
        pm = PersonalityManager()
        p = pm.get_personality()
        assert p.name == "Helpful"

    def test_set_personality(self):
        pm = PersonalityManager()
        pm.set_personality(PersonalityType.CREATIVE)
        p = pm.get_personality()
        assert p.name == "Creative"

    def test_list_personalities(self):
        pm = PersonalityManager()
        items = pm.list_personalities()
        assert len(items) >= 4

    def test_apply_temperature(self):
        pm = PersonalityManager()
        t = pm.apply_temperature(0.5)
        assert isinstance(t, float)


class TestGlobalPersonalityManager:
    def test_singleton(self):
        pm1 = get_personality_manager()
        pm2 = get_personality_manager()
        assert pm1 is pm2


class TestListPersonalities:
    def test_returns_list(self):
        items = list_personalities()
        assert isinstance(items, list)
        assert len(items) >= 4
