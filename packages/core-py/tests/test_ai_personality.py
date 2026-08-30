"""Tests for domains.ai_personality — personality config and manager."""

import pytest

from domains.ai_personality import (
    PersonalityType,
    Personality,
    PersonalityManager,
    PERSONALITIES,
    get_personality_manager,
    list_personalities,
)


# ---------------------------------------------------------------------------
# PersonalityType enum
# ---------------------------------------------------------------------------

class TestPersonalityType:
    def test_all_values_are_strings(self):
        for pt in PersonalityType:
            assert isinstance(pt.value, str)

    def test_expected_count(self):
        assert len(PersonalityType) == 6

    def test_member_names(self):
        expected = {"HELPFUL", "CREATIVE", "PROFESSIONAL", "CASUAL", "ACADEMIC", "SARCastic"}
        assert {pt.name for pt in PersonalityType} == expected


# ---------------------------------------------------------------------------
# Personality dataclass
# ---------------------------------------------------------------------------

class TestPersonality:
    def test_apply_returns_text_unchanged(self):
        p = PERSONALITIES[PersonalityType.HELPFUL]
        assert p.apply("hello world") == "hello world"

    def test_apply_empty_string(self):
        p = PERSONALITIES[PersonalityType.CREATIVE]
        assert p.apply("") == ""

    def test_modify_temperature_default(self):
        p = PERSONALITIES[PersonalityType.HELPFUL]
        result = p.modify_temperature(1.0)
        assert result == pytest.approx(1.0 * (0.5 + 0.6))

    def test_modify_temperature_creative(self):
        p = PERSONALITIES[PersonalityType.CREATIVE]
        result = p.modify_temperature(2.0)
        assert result == pytest.approx(2.0 * (0.5 + 0.9))

    def test_modify_temperature_zero(self):
        p = PERSONALITIES[PersonalityType.ACADEMIC]
        assert p.modify_temperature(0.0) == pytest.approx(0.0)

    def test_modify_temperature_negative(self):
        p = PERSONALITIES[PersonalityType.CASUAL]
        result = p.modify_temperature(-1.0)
        assert result == pytest.approx(-1.0 * (0.5 + 0.5))

    def test_modify_temperature_missing_creativity(self):
        p = Personality(name="x", description="d", traits={}, examples=[])
        assert p.modify_temperature(1.0) == pytest.approx(1.0 * (0.5 + 0.5))

    def test_traits_are_floats(self):
        for p in PERSONALITIES.values():
            for v in p.traits.values():
                assert isinstance(v, float)

    def test_examples_are_lists_of_strings(self):
        for p in PERSONALITIES.values():
            assert isinstance(p.examples, list)
            for ex in p.examples:
                assert isinstance(ex, str)

    def test_all_personality_types_have_entry(self):
        for pt in PersonalityType:
            assert pt in PERSONALITIES


# ---------------------------------------------------------------------------
# PersonalityManager
# ---------------------------------------------------------------------------

class TestPersonalityManager:
    def test_default_personality(self):
        mgr = PersonalityManager()
        assert mgr.get_personality() is PERSONALITIES[PersonalityType.HELPFUL]

    def test_custom_default(self):
        mgr = PersonalityManager(PersonalityType.CREATIVE)
        assert mgr.get_personality().name == "Creative"

    def test_set_personality(self):
        mgr = PersonalityManager()
        mgr.set_personality(PersonalityType.ACADEMIC)
        assert mgr.get_personality().name == "Academic"

    def test_list_personalities(self):
        mgr = PersonalityManager()
        items = mgr.list_personalities()
        assert len(items) == 6
        names = {p["name"] for p in items}
        assert "Helpful" in names
        assert "Sarcastic" in names

    def test_list_personalities_structure(self):
        mgr = PersonalityManager()
        for item in mgr.list_personalities():
            assert "type" in item
            assert "name" in item
            assert "description" in item
            assert "traits" in item

    def test_apply_temperature(self):
        mgr = PersonalityManager(PersonalityType.CREATIVE)
        temp = mgr.apply_temperature(1.0)
        assert temp == pytest.approx(1.0 * (0.5 + 0.9))

    def test_switching_personalities(self):
        mgr = PersonalityManager()
        mgr.set_personality(PersonalityType.CREATIVE)
        mgr.set_personality(PersonalityType.PROFESSIONAL)
        p = mgr.get_personality()
        assert p.name == "Professional"
        assert p.traits["formality"] == 0.9


# ---------------------------------------------------------------------------
# Module-level singletons / convenience functions
# ---------------------------------------------------------------------------

class TestModuleSingletons:
    def test_get_personality_manager_returns_manager(self):
        mgr = get_personality_manager()
        assert isinstance(mgr, PersonalityManager)

    def test_get_personality_manager_returns_same_instance(self):
        assert get_personality_manager() is get_personality_manager()

    def test_list_personalities_matches_manager(self):
        from domains.ai_personality import _default_manager
        assert list_personalities() == _default_manager.list_personalities()
