"""Tests for domains/companion.py — pure logic, no mocks."""

import pytest

from domains.companion import (
    CompanionSystem,
    CompanionTraits,
    ConversationContext,
    ResponseStyle,
    create_companion,
    get_companion,
)


# ---------------------------------------------------------------------------
# ResponseStyle enum
# ---------------------------------------------------------------------------

class TestResponseStyle:
    def test_values(self):
        assert ResponseStyle.CASUAL.value == "casual"
        assert ResponseStyle.FORMAL.value == "formal"
        assert ResponseStyle.PLAYFUL.value == "playful"
        assert ResponseStyle.EMPATHETIC.value == "empathetic"
        assert ResponseStyle.CURIOUS.value == "curious"

    def test_member_count(self):
        assert len(ResponseStyle) == 5


# ---------------------------------------------------------------------------
# CompanionTraits defaults
# ---------------------------------------------------------------------------

class TestCompanionTraits:
    def test_defaults(self):
        t = CompanionTraits()
        assert t.name == "Friend"
        assert t.warmth == 0.7
        assert t.curiosity == 0.6
        assert t.creativity == 0.5
        assert t.confidence == 0.5
        assert t.humor == 0.4
        assert t.response_length == "medium"
        assert t.use_questions is True
        assert t.share_personal is False
        assert t.avoid_topics == []
        assert t.no_robot_phrases is True

    def test_custom_traits(self):
        t = CompanionTraits(name="Echo", warmth=0.3, humor=0.9)
        assert t.name == "Echo"
        assert t.warmth == 0.3
        assert t.humor == 0.9


# ---------------------------------------------------------------------------
# ConversationContext defaults
# ---------------------------------------------------------------------------

class TestConversationContext:
    def test_defaults(self):
        ctx = ConversationContext()
        assert ctx.user_name is None
        assert ctx.topics == []
        assert ctx.user_mood is None
        assert ctx.shared_memories == []
        assert ctx.turn_count == 0


# ---------------------------------------------------------------------------
# CompanionSystem
# ---------------------------------------------------------------------------

class TestCompanionSystem:
    def test_init_defaults(self):
        c = CompanionSystem()
        assert c.traits.name == "Friend"
        assert c.traits.warmth == 0.7
        assert c.context.turn_count == 0

    def test_set_personality(self):
        c = CompanionSystem()
        c.set_personality(name="Echo", warmth=0.9, curiosity=0.8, humor=0.2)
        assert c.traits.name == "Echo"
        assert c.traits.warmth == 0.9
        assert c.traits.curiosity == 0.8
        assert c.traits.humor == 0.2

    def test_set_personality_rebuilds_prompt(self):
        c = CompanionSystem()
        old_prompt = c.get_system_prompt()
        c.set_personality(name="NewName")
        assert "NewName" in c.get_system_prompt()
        assert c.get_system_prompt() != old_prompt

    def test_system_prompt_contains_name(self):
        c = CompanionSystem()
        c.set_personality(name="Bob")
        prompt = c.get_system_prompt()
        assert "Bob" in prompt

    def test_system_prompt_contains_humor_guide(self):
        c = CompanionSystem()
        c.set_personality(humor=0.8)
        assert "humor" in c.get_system_prompt().lower()

    def test_system_prompt_contains_creativity_guide(self):
        c = CompanionSystem()
        c.set_personality(creativity=0.8)
        assert "creative" in c.get_system_prompt().lower()

    def test_clean_response_removes_robot_phrases(self):
        c = CompanionSystem()
        result = c.clean_response("I am an AI and I think so.")
        assert "I am an AI" not in result

    def test_clean_response_strips_double_space(self):
        c = CompanionSystem()
        result = c.clean_response("hello  world")
        assert "  " not in result

    def test_clean_response_adds_period(self):
        c = CompanionSystem()
        result = c.clean_response("hello there")
        assert result.endswith(".")

    def test_clean_response_preserves_punctuation(self):
        c = CompanionSystem()
        assert c.clean_response("hello!").endswith("!")
        assert c.clean_response("hello?").endswith("?")
        assert c.clean_response("hello.").endswith(".")

    def test_clean_response_empty(self):
        c = CompanionSystem()
        assert c.clean_response("") == ""

    def test_respond_increments_turn_count(self):
        c = CompanionSystem()
        c.respond("hi")
        assert c.context.turn_count == 1
        c.respond("hello")
        assert c.context.turn_count == 2

    def test_respond_includes_message(self):
        c = CompanionSystem()
        prompt = c.respond("How are you?")
        assert "How are you?" in prompt

    def test_respond_includes_name(self):
        c = CompanionSystem()
        c.set_personality(name="Sam")
        prompt = c.respond("hey")
        assert "Sam:" in prompt

    def test_respond_includes_user_name(self):
        c = CompanionSystem()
        c.context.user_name = "Alice"
        prompt = c.respond("hi")
        assert "Alice" in prompt

    def test_respond_uses_provided_context(self):
        c = CompanionSystem()
        ctx = ConversationContext(topics=["weather", "sports"])
        c.respond("hey", context=ctx)
        assert c.context.topics == ["weather", "sports"]

    def test_respond_without_context_keeps_topics(self):
        c = CompanionSystem()
        c.context.topics = ["existing"]
        c.respond("hey", context=None)
        assert c.context.topics == ["existing"]

    def test_adjust_for_mood_sad(self):
        c = CompanionSystem()
        original_warmth = c.traits.warmth
        original_humor = c.traits.humor
        c.adjust_for_mood("sad")
        assert c.traits.warmth == pytest.approx(min(1.0, original_warmth + 0.2))
        assert c.traits.humor == pytest.approx(max(0.0, original_humor - 0.2))
        assert c.context.user_mood == "sad"

    def test_adjust_for_mood_down(self):
        c = CompanionSystem()
        c.traits.warmth = 0.5
        c.adjust_for_mood("down")
        assert c.traits.warmth == pytest.approx(0.7)

    def test_adjust_for_mood_upset(self):
        c = CompanionSystem()
        c.traits.warmth = 0.9
        c.adjust_for_mood("upset")
        # min(1.0, 0.9 + 0.2) = 1.0
        assert c.traits.warmth == pytest.approx(1.0)

    def test_adjust_for_mood_happy(self):
        c = CompanionSystem()
        original_warmth = c.traits.warmth
        c.adjust_for_mood("happy")
        assert c.traits.warmth == pytest.approx(min(1.0, original_warmth + 0.1))

    def test_adjust_for_mood_excited(self):
        c = CompanionSystem()
        c.traits.warmth = 0.8
        c.adjust_for_mood("excited")
        assert c.traits.warmth == pytest.approx(0.9)

    def test_adjust_for_mood_neutral_no_change(self):
        c = CompanionSystem()
        original_warmth = c.traits.warmth
        c.adjust_for_mood("neutral")
        assert c.traits.warmth == original_warmth

    def test_to_dict_structure(self):
        c = CompanionSystem()
        d = c.to_dict()
        assert "traits" in d
        assert "system_prompt" in d
        assert d["traits"]["name"] == "Friend"
        assert isinstance(d["system_prompt"], str)

    def test_to_dict_reflects_changes(self):
        c = CompanionSystem()
        c.set_personality(name="Zara")
        d = c.to_dict()
        assert d["traits"]["name"] == "Zara"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestGetCompanion:
    def test_returns_singleton(self):
        import domains.companion as mod
        mod._companion = None
        c1 = get_companion()
        c2 = get_companion()
        assert c1 is c2

    def test_singleton_resets(self):
        import domains.companion as mod
        mod._companion = None
        c = get_companion()
        c.set_personality(name="ShouldPersist")
        assert get_companion().traits.name == "ShouldPersist"
        mod._companion = None  # cleanup


class TestCreateCompanion:
    def test_warm_preset(self):
        c = create_companion(name="W", personality="warm")
        assert c.traits.name == "W"
        assert c.traits.warmth == 0.9
        assert c.traits.humor == 0.3

    def test_curious_preset(self):
        c = create_companion(name="Q", personality="curious")
        assert c.traits.curiosity == 0.9

    def test_playful_preset(self):
        c = create_companion(name="P", personality="playful")
        assert c.traits.humor == 0.8

    def test_balanced_preset(self):
        c = create_companion(personality="balanced")
        assert c.traits.warmth == 0.7
        assert c.traits.curiosity == 0.6
        assert c.traits.humor == 0.5

    def test_unknown_personality_falls_back_to_balanced(self):
        c = create_companion(personality="unknown")
        assert c.traits.warmth == 0.7

    def test_default_name(self):
        c = create_companion()
        assert c.traits.name == "Friend"

    def test_returns_companion_system(self):
        c = create_companion()
        assert isinstance(c, CompanionSystem)
