"""Tests for domains.companion — CompanionTraits, ResponseStyle, CompanionSystem."""

from domains.companion import (
    CompanionTraits, ResponseStyle, CompanionSystem,
    get_companion, create_companion,
)


class TestResponseStyle:
    def test_all_members(self):
        assert len(ResponseStyle) >= 4

    def test_values(self):
        assert ResponseStyle.FORMAL.value == "formal"
        assert ResponseStyle.CASUAL.value == "casual"


class TestCompanionTraits:
    def test_basic(self):
        t = CompanionTraits(name="test", warmth=0.8)
        assert t.name == "test"
        assert t.warmth == 0.8

    def test_defaults(self):
        t = CompanionTraits(name="x")
        assert t.warmth >= 0.0


class TestCompanionSystem:
    def test_init(self):
        cs = CompanionSystem()
        prompt = cs.get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_set_personality_warm(self):
        cs = CompanionSystem()
        cs.set_personality("warm")
        prompt = cs.get_system_prompt()
        assert "warm" in prompt.lower()

    def test_set_personality_curious(self):
        cs = CompanionSystem()
        cs.set_personality("curious")
        prompt = cs.get_system_prompt()
        assert "curious" in prompt.lower()

    def test_set_personality_playful(self):
        cs = CompanionSystem()
        cs.set_personality("playful")
        prompt = cs.get_system_prompt()
        assert "playful" in prompt.lower()

    def test_adjust_for_mood(self):
        cs = CompanionSystem()
        cs.adjust_for_mood("happy")
        cs.adjust_for_mood("sad")

    def test_to_dict(self):
        cs = CompanionSystem()
        d = cs.to_dict()
        assert isinstance(d, dict)

    def test_clean_response(self):
        cs = CompanionSystem()
        result = cs.clean_response("  Hello  ")
        assert isinstance(result, str)

    def test_singleton(self):
        c1 = get_companion()
        c2 = get_companion()
        assert c1 is c2
