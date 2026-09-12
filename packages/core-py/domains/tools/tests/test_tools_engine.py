"""Tests for the tools domain (profiles + engine)."""

import pytest
from domains.tools import (
    TOOL_PROFILES,
    ToolOption,
    ToolParam,
    ToolProfile,
    ToolsEngine,
    get_tool_profile,
    get_tools_engine,
)

PROFILE_IDS = [
    "writing",
    "translate",
    "rewrite",
    "brainstorm",
    "decide",
    "explain",
    "wellness",
]


def _engine():
    return get_tools_engine()


class TestToolProfiles:
    def test_all_profiles_registered(self):
        assert sorted(TOOL_PROFILES.keys()) == sorted(PROFILE_IDS)

    def test_get_tool_profile_returns_full_profile(self):
        profile = get_tool_profile("writing")
        assert profile is not None
        assert profile.id == "writing"
        assert profile.name
        assert profile.description
        assert profile.icon
        assert profile.system_prompt
        assert profile.max_tokens >= 100

    def test_every_profile_has_required_fields(self):
        for profile in TOOL_PROFILES.values():
            assert profile.id.isidentifier()
            assert profile.name
            assert profile.description
            assert profile.icon
            assert profile.max_tokens >= 100

    def test_profiles_are_frozen(self):
        profile = TOOL_PROFILES["decide"]
        with pytest.raises(FrozenInstanceError):
            profile.max_tokens = 1  # type: ignore[misc]

    def test_params_are_well_formed(self):
        for profile in TOOL_PROFILES.values():
            for param in profile.params:
                assert param.id
                assert param.label
                assert isinstance(param.optional, bool)

    def test_options_are_well_formed(self):
        for profile in TOOL_PROFILES.values():
            for group, options in profile.options.items():
                assert group
                for opt in options:
                    assert opt.id
                    assert opt.label
                    assert isinstance(opt.description, str)

    def test_default_options_reference_valid_option_ids(self):
        for profile in TOOL_PROFILES.values():
            for group, default in profile.default_options.items():
                assert group in profile.options, f"{profile.id} default group missing"
                ids = {o.id for o in profile.options[group]}
                assert default in ids, f"{profile.id} default {group}={default} not in options"


class TestToolsEngineList:
    def test_list_tools_returns_seven(self):
        tools = _engine().list_tools()
        assert len(tools) == 7

    def test_list_tools_returns_public_schema(self):
        tool = next(t for t in _engine().list_tools() if t["id"] == "writing")
        assert {"id", "name", "description", "icon", "params", "options"} <= set(tool)
        assert "system_prompt" in tool
        assert "default_options" in tool
        assert "max_tokens" in tool

    def test_list_tools_does_not_expose_render_fn(self):
        for tool in _engine().list_tools():
            assert "render_fn" not in tool
            assert "render_prompt" not in tool


class TestToolsEngineRender:
    def test_render_writing_defaults(self):
        prompt = _engine().render_prompt("writing", {"text": "hello"})
        assert "hello" in prompt
        assert "email" in prompt or "@" in prompt

    def test_render_writing_honors_options(self):
        prompt = _engine().render_prompt(
            "writing", {"text": "hello", "tone": "funny", "type": "poem"}
        )
        assert "poem" in prompt

    def test_render_translate(self):
        prompt = _engine().render_prompt(
            "translate", {"text": "bonjour", "target_lang": "French"}
        )
        assert "Translate" in prompt
        assert "French" in prompt
        assert "bonjour" in prompt

    def test_render_rewrite(self):
        prompt = _engine().render_prompt("rewrite", {"text": "raw", "action": "grammar"})
        assert "raw" in prompt

    def test_render_brainstorm_uses_history(self):
        prompt = _engine().render_prompt(
            "brainstorm",
            {"history": [{"role": "user", "content": "ideas?"}]},
        )
        assert "ideas?" in prompt
        assert "Assistant:" in prompt

    def test_render_decide(self):
        prompt = _engine().render_prompt(
            "decide",
            {"question": "Where?", "option_a": "Paris", "option_b": "Berlin"},
        )
        assert "Paris" in prompt
        assert "Berlin" in prompt
        assert "Where?" in prompt

    def test_render_explain_default_difficulty(self):
        prompt = _engine().render_prompt("explain", {"topic": "blobs"})
        assert "blobs" in prompt
        assert "everyday language" in prompt

    def test_render_explain_honors_difficulty(self):
        prompt = _engine().render_prompt(
            "explain", {"topic": "blobs", "difficulty": "simple"}
        )
        assert "5 years old" in prompt

    def test_render_wellness_default_kind(self):
        prompt = _engine().render_prompt("wellness", {})
        assert "sleep story" in prompt

    def test_render_wellness_honors_kind(self):
        prompt = _engine().render_prompt(
            "wellness", {"kind": "affirm", "preferences": "courage"}
        )
        assert "affirmations" in prompt
        assert "courage" in prompt

    def test_render_unknown_tool_raises(self):
        with pytest.raises(KeyError):
            _engine().render_prompt("nope", {})

    def test_get_profile_matches_engine(self):
        for tool_id in PROFILE_IDS:
            assert _engine().get_profile(tool_id) is not None


class TestToolsEngineDefaults:
    def test_default_options_applied_when_missing(self):
        engine = ToolsEngine()
        merged = engine._merge_defaults(TOOL_PROFILES["wellness"], {})
        assert merged["kind"] == "sleep"

    def test_default_options_not_overwritten(self):
        engine = ToolsEngine()
        merged = engine._merge_defaults(
            TOOL_PROFILES["wellness"], {"kind": "journal"}
        )
        assert merged["kind"] == "journal"

    def test_singleton(self):
        assert get_tools_engine() is get_tools_engine()


class TestToolParamOption:
    def test_tool_param_defaults(self):
        param = ToolParam("x", "Label", "placeholder")
        assert not param.multiline
        assert not param.optional
        assert param.multiline is False

    def test_tool_option_fields(self):
        opt = ToolOption("id", "Label", "Description")
        assert opt.id == "id"
        assert opt.label == "Label"
        assert opt.description == "Description"

    def test_tool_profile_render_factory(self):
        profile = ToolProfile(
            id="t",
            name="T",
            description="D",
            icon="i",
            render_fn=lambda p: f"rendered {p.get('x')}",
        )
        assert profile.render_prompt({"x": "hi"}) == "rendered hi"

    def test_tool_profile_without_render_fn_raises(self):
        profile = ToolProfile(id="t", name="T", description="D", icon="i")
        with pytest.raises(NotImplementedError):
            profile.render_prompt({})
