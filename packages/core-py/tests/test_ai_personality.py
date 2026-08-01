"""Tests for the personality system — types, temperature scaling, manager."""

import pytest
from domains.ai_personality import (
    Personality,
    PersonalityType,
    PersonalityManager,
    PERSONALITIES,
    get_personality_manager,
    list_personalities,
)


# ── PersonalityType ────────────────────────────────────────────────────────

class TestPersonalityType:

    def test_values(self):
        assert PersonalityType.HELPFUL.value == "helpful"
        assert PersonalityType.CREATIVE.value == "creative"
        assert PersonalityType.PROFESSIONAL.value == "professional"
        assert PersonalityType.CASUAL.value == "casual"
        assert PersonalityType.ACADEMIC.value == "academic"
        assert PersonalityType.SARCastic.value == "sarcastic"

    def test_all_defined_in_registry(self):
        for ptype in PersonalityType:
            assert ptype in PERSONALITIES


# ── Personality ────────────────────────────────────────────────────────────

class TestPersonality:

    def test_apply_returns_text_unchanged(self):
        p = PERSONALITIES[PersonalityType.HELPFUL]
        assert p.apply("hello") == "hello"
        assert p.apply("") == ""

    def test_modify_temperature_scales_by_creativity(self):
        p = Personality(name="x", description="d", traits={"creativity": 0.6}, examples=[])
        assert p.modify_temperature(1.0) == pytest.approx(1.1)
        assert p.modify_temperature(0.5) == pytest.approx(0.55)

    def test_modify_temperature_default_creativity(self):
        p = Personality(name="x", description="d", traits={}, examples=[])
        assert p.modify_temperature(1.0) == pytest.approx(1.0)
        assert p.modify_temperature(2.0) == pytest.approx(2.0)

    def test_creative_raises_temperature(self):
        helpful = PERSONALITIES[PersonalityType.HELPFUL]
        creative = PERSONALITIES[PersonalityType.CREATIVE]
        assert creative.modify_temperature(1.0) > helpful.modify_temperature(1.0)

    def test_professional_lowers_temperature(self):
        professional = PERSONALITIES[PersonalityType.PROFESSIONAL]
        creative = PERSONALITIES[PersonalityType.CREATIVE]
        assert professional.modify_temperature(1.0) < creative.modify_temperature(1.0)

    def test_registry_entries_complete(self):
        for ptype, p in PERSONALITIES.items():
            assert p.name
            assert p.description
            assert "creativity" in p.traits
            assert isinstance(p.examples, list)
            assert p.examples


# ── PersonalityManager ─────────────────────────────────────────────────────

class TestPersonalityManager:

    def test_default_personality(self):
        mgr = PersonalityManager()
        assert mgr.default == PersonalityType.HELPFUL
        assert mgr.current is PERSONALITIES[PersonalityType.HELPFUL]

    def test_custom_default(self):
        mgr = PersonalityManager(default_personality=PersonalityType.CREATIVE)
        assert mgr.current is PERSONALITIES[PersonalityType.CREATIVE]

    def test_set_personality(self):
        mgr = PersonalityManager()
        mgr.set_personality(PersonalityType.PROFESSIONAL)
        assert mgr.current.name == "Professional"
        assert mgr.get_personality() is mgr.current

    def test_list_personalities(self):
        items = PersonalityManager().list_personalities()
        assert len(items) == len(PERSONALITIES)
        assert {"type", "name", "description", "traits"} <= set(items[0].keys())
        types = {i["type"] for i in items}
        assert PersonalityType.CREATIVE.value in types

    def test_apply_temperature(self):
        mgr = PersonalityManager(default_personality=PersonalityType.CREATIVE)
        assert mgr.apply_temperature(1.0) == pytest.approx(1.4)

    def test_unknown_default_raises_keyerror(self):
        with pytest.raises(KeyError):
            PersonalityManager(default_personality="nope")  # type: ignore


# ── Module-level helpers ───────────────────────────────────────────────────

class TestModuleHelpers:

    def test_get_personality_manager_singleton(self):
        assert get_personality_manager() is get_personality_manager()
        assert isinstance(get_personality_manager(), PersonalityManager)

    def test_list_personalities_helper(self):
        items = list_personalities()
        assert len(items) == len(PERSONALITIES)
