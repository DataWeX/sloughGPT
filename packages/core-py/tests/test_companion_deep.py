"""Meaningful tests for CompanionSystem — clean_response, respond, adjust_for_mood, create_companion presets."""

import pytest
from domains.companion import (
    CompanionSystem, CompanionTraits, ConversationContext,
    ResponseStyle, create_companion, get_companion,
)


class TestResponseStyle:
    def test_values(self):
        assert ResponseStyle.CASUAL.value == "casual"
        assert ResponseStyle.FORMAL.value == "formal"
        assert len(ResponseStyle) == 5


class TestCompanionTraits:
    def test_defaults(self):
        t = CompanionTraits()
        assert t.name == "Friend"
        assert t.warmth == 0.7
        assert t.no_robot_phrases is True
        assert t.avoid_topics == []


class TestConversationContext:
    def test_defaults(self):
        ctx = ConversationContext()
        assert ctx.user_name is None
        assert ctx.turn_count == 0
        assert ctx.topics == []


class TestCompanionSystem:
    def test_init(self):
        c = CompanionSystem()
        assert c.traits.name == "Friend"
        assert c.context.turn_count == 0

    def test_set_personality(self):
        c = CompanionSystem()
        c.set_personality(name="Alex", warmth=0.9, humor=0.8)
        assert c.traits.name == "Alex"
        assert c.traits.warmth == 0.9
        assert c.traits.humor == 0.8

    def test_get_system_prompt(self):
        c = CompanionSystem()
        prompt = c.get_system_prompt()
        assert "Friend" in prompt
        assert "conversational" in prompt.lower()

    def test_clean_response_removes_robot_phrases(self):
        c = CompanionSystem()
        result = c.clean_response("As an AI, I think this is good.")
        assert "As an AI" not in result
        assert "I think this is good" in result

    def test_clean_response_removes_multiple_phrases(self):
        c = CompanionSystem()
        result = c.clean_response("I am an AI and I was trained on data.")
        assert "I am an AI" not in result
        assert "I was trained" not in result

    def test_clean_response_collapse_double_spaces(self):
        c = CompanionSystem()
        result = c.clean_response("hello  world  test")
        assert "  " not in result

    def test_clean_response_adds_punctuation(self):
        c = CompanionSystem()
        result = c.clean_response("hello world")
        assert result == "hello world."

    def test_clean_response_keeps_existing_punctuation(self):
        c = CompanionSystem()
        assert c.clean_response("hello!") == "hello!"
        assert c.clean_response("hello?") == "hello?"
        assert c.clean_response("hello.") == "hello."

    def test_clean_response_strips_whitespace(self):
        c = CompanionSystem()
        result = c.clean_response("  hello  ")
        assert result == "hello."

    def test_clean_response_empty(self):
        c = CompanionSystem()
        result = c.clean_response("")
        assert result == ""

    def test_respond_builds_prompt(self):
        c = CompanionSystem()
        c.set_personality(name="Alex")
        prompt = c.respond("Hey there!")
        assert "User: Hey there!" in prompt
        assert "Alex:" in prompt

    def test_respond_increments_turn_count(self):
        c = CompanionSystem()
        c.respond("hi")
        assert c.context.turn_count == 1
        c.respond("hello")
        assert c.context.turn_count == 2

    def test_respond_with_user_name(self):
        c = CompanionSystem()
        c.context.user_name = "Sam"
        prompt = c.respond("hi")
        assert "Sam" in prompt

    def test_respond_with_context_topics(self):
        c = CompanionSystem()
        ctx = ConversationContext(topics=["python", "coding"])
        c.respond("hi", context=ctx)
        assert c.context.topics == ["python", "coding"]

    def test_adjust_for_mood_sad(self):
        c = CompanionSystem()
        original_warmth = c.traits.warmth
        original_humor = c.traits.humor
        c.adjust_for_mood("sad")
        assert c.traits.warmth == min(1.0, original_warmth + 0.2)
        assert c.traits.humor == max(0, original_humor - 0.2)

    def test_adjust_for_mood_happy(self):
        c = CompanionSystem()
        original_warmth = c.traits.warmth
        c.adjust_for_mood("happy")
        assert c.traits.warmth == min(1.0, original_warmth + 0.1)

    def test_adjust_for_mood_neutral(self):
        c = CompanionSystem()
        c.adjust_for_mood("neutral")
        assert c.context.user_mood == "neutral"

    def test_to_dict(self):
        c = CompanionSystem()
        d = c.to_dict()
        assert "traits" in d
        assert "system_prompt" in d
        assert d["traits"]["name"] == "Friend"

    def test_build_prompt_high_warmth(self):
        c = CompanionSystem()
        c.set_personality(warmth=0.9)
        prompt = c.get_system_prompt()
        assert "Very warm" in prompt or "caring" in prompt.lower()

    def test_build_prompt_high_humor(self):
        c = CompanionSystem()
        c.set_personality(humor=0.8)
        prompt = c.get_system_prompt()
        assert "humor" in prompt.lower()

    def test_build_prompt_high_creativity(self):
        c = CompanionSystem()
        c.set_personality(creativity=0.8)
        prompt = c.get_system_prompt()
        assert "creative" in prompt.lower()

    def test_build_prompt_share_personal(self):
        c = CompanionSystem()
        c.traits.share_personal = True
        c._system_prompt = c._build_system_prompt()
        prompt = c.get_system_prompt()
        assert "personal" in prompt.lower()


class TestCreateCompanion:
    def test_warm_preset(self):
        c = create_companion(personality="warm")
        assert c.traits.warmth == 0.9
        assert c.traits.humor == 0.3

    def test_curious_preset(self):
        c = create_companion(personality="curious")
        assert c.traits.curiosity == 0.9

    def test_playful_preset(self):
        c = create_companion(personality="playful")
        assert c.traits.humor == 0.8

    def test_balanced_preset(self):
        c = create_companion(personality="balanced")
        assert c.traits.warmth == 0.7
        assert c.traits.humor == 0.5

    def test_unknown_preset_falls_back(self):
        c = create_companion(personality="nonexistent")
        assert c.traits.warmth == 0.7  # balanced default

    def test_custom_name(self):
        c = create_companion(name="Zara", personality="warm")
        assert c.traits.name == "Zara"


class TestGetCompanion:
    def test_singleton(self):
        c1 = get_companion()
        c2 = get_companion()
        assert c1 is c2
