"""Tests for the AI companion system — traits, prompts, cleaning, presets."""

import pytest
from domains.companion import (
    ResponseStyle,
    CompanionTraits,
    ConversationContext,
    CompanionSystem,
    get_companion,
    create_companion,
)


# ── ResponseStyle ──────────────────────────────────────────────────────────

class TestResponseStyle:

    def test_values(self):
        assert ResponseStyle.CASUAL.value == "casual"
        assert ResponseStyle.FORMAL.value == "formal"
        assert ResponseStyle.PLAYFUL.value == "playful"
        assert ResponseStyle.EMPATHETIC.value == "empathetic"
        assert ResponseStyle.CURIOUS.value == "curious"


# ── CompanionTraits / ConversationContext ─────────────────────────────────

class TestCompanionTraits:

    def test_defaults(self):
        t = CompanionTraits()
        assert t.name == "Friend"
        assert t.warmth == 0.7
        assert t.response_length == "medium"
        assert t.use_questions is True
        assert t.no_robot_phrases is True
        assert t.avoid_topics == []

    def test_custom(self):
        t = CompanionTraits(name="Alex", warmth=0.9, response_length="short")
        assert t.name == "Alex"
        assert t.warmth == 0.9
        assert t.response_length == "short"


class TestConversationContext:

    def test_defaults(self):
        c = ConversationContext()
        assert c.user_name is None
        assert c.topics == []
        assert c.user_mood is None
        assert c.shared_memories == []
        assert c.turn_count == 0


# ── CompanionSystem ────────────────────────────────────────────────────────

class TestCompanionSystem:

    def test_init_builds_prompt(self):
        c = CompanionSystem()
        assert c.traits.name == "Friend"
        assert "human friend" in c._system_prompt
        assert c.context.turn_count == 0

    def test_set_personality(self):
        c = CompanionSystem()
        c.set_personality(name="Alex", warmth=0.9, curiosity=0.8, humor=0.6)
        assert c.traits.name == "Alex"
        assert c.traits.warmth == 0.9
        assert c.traits.curiosity == 0.8
        assert c.traits.humor == 0.6
        assert "Alex" in c._system_prompt

    def test_system_prompt_contains_trait_guidance(self):
        c = CompanionSystem()
        prompt = c.get_system_prompt()
        assert "warmth" not in prompt  # guidance text, not raw values
        assert "Be conversational, not robotic." in prompt

    def test_high_humor_style_added(self):
        c = CompanionSystem()
        c.set_personality(humor=0.9)
        assert "Use light humor" in c.get_system_prompt()

    def test_low_humor_no_style(self):
        c = CompanionSystem()
        c.set_personality(humor=0.2)
        assert "Use light humor" not in c.get_system_prompt()

    def test_share_personal_style(self):
        c = CompanionSystem()
        c.traits.share_personal = True
        c._system_prompt = c._build_system_prompt()
        assert "Share relevant personal experiences" in c._system_prompt

    def test_clean_response_removes_robot_phrases(self):
        c = CompanionSystem()
        cleaned = c.clean_response("As an AI, I don't have feelings. But I can help!")
        assert "As an AI" not in cleaned
        assert "I don't have feelings" not in cleaned

    def test_clean_response_removal_is_case_sensitive(self):
        c = CompanionSystem()
        cleaned = c.clean_response("as an ai language model, here you go")
        assert cleaned == "as an ai language model, here you go."

    def test_clean_response_collapses_spaces(self):
        c = CompanionSystem()
        assert c.clean_response("hello   world  ") == "hello  world."

    def test_clean_response_adds_punctuation(self):
        c = CompanionSystem()
        assert c.clean_response("no ending punctuation") == "no ending punctuation."

    def test_clean_response_keeps_existing_punctuation(self):
        c = CompanionSystem()
        assert c.clean_response("already done!") == "already done!"
        assert c.clean_response("question?") == "question?"
        assert c.clean_response("ellipsis...") == "ellipsis..."

    def test_clean_response_empty(self):
        c = CompanionSystem()
        assert c.clean_response("") == ""

    def test_respond_increments_turn_count(self):
        c = CompanionSystem()
        c.respond("hello")
        assert c.context.turn_count == 1
        c.respond("again")
        assert c.context.turn_count == 2

    def test_respond_includes_user_and_name(self):
        c = CompanionSystem()
        c.context.user_name = "Sam"
        out = c.respond("how are you?")
        assert "Sam" in out
        assert "how are you?" in out
        assert "Friend:" in out

    def test_respond_merges_topics_from_context(self):
        c = CompanionSystem()
        c.respond("hi", context=ConversationContext(topics=["music"]))
        assert c.context.topics == ["music"]

    def test_respond_no_context_keeps_topics(self):
        c = CompanionSystem()
        c.respond("hi")
        assert c.context.topics == []

    def test_adjust_for_mood_sad(self):
        c = CompanionSystem()
        base_warmth, base_humor = c.traits.warmth, c.traits.humor
        c.adjust_for_mood("sad")
        assert c.context.user_mood == "sad"
        assert c.traits.warmth == pytest.approx(min(1.0, base_warmth + 0.2))
        assert c.traits.humor == pytest.approx(max(0, base_humor - 0.2))

    def test_adjust_for_mood_happy(self):
        c = CompanionSystem()
        base_warmth = c.traits.warmth
        c.adjust_for_mood("happy")
        assert c.traits.warmth == pytest.approx(min(1.0, base_warmth + 0.1))

    def test_adjust_for_mood_unknown_noop(self):
        c = CompanionSystem()
        before = (c.traits.warmth, c.traits.humor)
        c.adjust_for_mood("neutral")
        assert c.context.user_mood == "neutral"
        assert (c.traits.warmth, c.traits.humor) == before

    def test_warmth_capped_at_1(self):
        c = CompanionSystem()
        c.traits.warmth = 0.95
        c.adjust_for_mood("sad")
        assert c.traits.warmth == 1.0

    def test_to_dict(self):
        c = CompanionSystem()
        c.set_personality(name="Alex", warmth=0.8)
        d = c.to_dict()
        assert d["traits"]["name"] == "Alex"
        assert d["traits"]["warmth"] == 0.8
        assert d["system_prompt"] == c._system_prompt

    def test_robot_phrases_non_empty(self):
        assert len(CompanionSystem.ROBOT_PHRASES) >= 5


# ── Module-level helpers ───────────────────────────────────────────────────

class TestModuleHelpers:

    def test_get_companion_singleton(self):
        assert get_companion() is get_companion()
        assert isinstance(get_companion(), CompanionSystem)

    def test_create_companion_warm(self):
        c = create_companion(name="Alex", personality="warm")
        assert c.traits.name == "Alex"
        assert c.traits.warmth == 0.9
        assert c.traits.humor == 0.3

    def test_create_companion_playful(self):
        c = create_companion(name="Pip", personality="playful")
        assert c.traits.warmth == 0.7
        assert c.traits.humor == 0.8

    def test_create_companion_unknown_falls_back_to_balanced(self):
        c = create_companion(name="D", personality="nope")
        assert c.traits.warmth == 0.7
        assert c.traits.curiosity == 0.6
        assert c.traits.humor == 0.5

    def test_create_companion_default_name(self):
        assert create_companion().traits.name == "Friend"

    def test_respond_after_preset(self):
        c = create_companion(name="Alex", personality="warm")
        out = c.respond("hi there")
        assert "Alex:" in out
        assert c.context.turn_count == 1
