"""Extended tests for slo_format — save_soul, generate_sample_dialogue, create_soul_profile edge cases."""

import json
import os
import struct
import tempfile
import numpy as np
import pytest

from domains.inference.slo_format import (
    SOU_MAGIC,
    SOU_VERSION_V3,
    GenerationParams,
    ContextParams,
    PersonalityCore,
    BehavioralTraits,
    CognitiveSignature,
    EmotionalRange,
    SloProfile,
    SouParser,
    create_soul_profile,
    save_soul,
    load_soul,
    generate_sample_dialogue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal model with state_dict() returning numpy arrays."""

    def __init__(self, params=None):
        self._params = params or {
            "encoder.weight": np.random.randn(128, 64).astype(np.float32),
            "decoder.bias": np.random.randn(128).astype(np.float32),
        }

    def state_dict(self):
        return dict(self._params)


class FakeModelWithMetadata:
    """Model that also exposes a metadata dict and lineage attribute."""

    def __init__(self):
        self.metadata = {"framework": "slonet", "version": "0.1"}
        self.lineage = "custom-lineage"
        self._params = {"w": np.ones((2, 2), dtype=np.float32)}

    def state_dict(self):
        return dict(self._params)


class FakeModelDictValues:
    """Model whose state_dict values are plain Python lists (not numpy)."""

    def state_dict(self):
        return {"list_param": [[1.0, 2.0], [3.0, 4.0]], "scalar_param": 42.0}


class FakeModelEmpty:
    """Model whose state_dict returns empty dict."""

    def state_dict(self):
        return {}


class FakeModelBadValues:
    """Model whose state_dict has a non-convertible value (string)."""

    def state_dict(self):
        return {"good": np.ones(4, dtype=np.float32), "bad": "not_a_tensor"}


class FakeModelNoStateDict:
    """Model without state_dict method."""

    pass


class FakeModelDictSkip:
    """Model whose state_dict has a dict value (should be skipped)."""

    def state_dict(self):
        return {
            "valid": np.ones(3, dtype=np.float32),
            "nested_dict": {"a": 1, "b": 2},
        }


class FakeModelGenerate:
    """Model with generate() for dialogue tests."""

    def generate(self, idx, max_new_tokens=50, temperature=0.8):
        seq_len = idx.shape[1]
        out_len = min(seq_len + 10, max_new_tokens)
        return np.arange(out_len, dtype=np.int64).reshape(1, -1)


class FakeModelForward:
    """Model with only forward() (no generate)."""

    def forward(self, idx):
        return np.arange(8, dtype=np.int64).reshape(1, -1)


class FakeModelNoGen:
    """Model with neither generate nor forward."""

    pass


# ---------------------------------------------------------------------------
# save_soul tests
# ---------------------------------------------------------------------------

class TestSaveSoul:
    def test_writes_file_with_valid_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.soul")
            save_soul(FakeModel(), path)

            with open(path, "rb") as f:
                magic = f.read(4)
                version = struct.unpack("<I", f.read(4))[0]
                config_len = struct.unpack("<I", f.read(4))[0]

            assert magic == SOU_MAGIC
            assert version == SOU_VERSION_V3
            assert config_len > 0

    def test_meta_json_sidecar_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta.soul")
            save_soul(FakeModel(), path)

            meta_path = path + ".meta.json"
            assert os.path.exists(meta_path)

            with open(meta_path, "r") as f:
                meta = json.load(f)
            assert meta["name"] == "meta"
            assert "personality" in meta
            assert "generation" in meta

    def test_meta_json_matches_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "match.soul")
            soul = SloProfile(name="match-bot", base_model="gpt2")
            save_soul(FakeModel(), path, soul_profile=soul)

            soul2, _ = load_soul(path)
            assert soul2.name == "match-bot"
            assert soul2.base_model == "gpt2"

    def test_returns_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ret.soul")
            result = save_soul(FakeModel(), path)
            assert result == path

    def test_weights_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "roundtrip.soul")
            model = FakeModel()
            save_soul(model, path)

            soul, state = load_soul(path)
            original = model.state_dict()
            assert len(state) == len(original)
            for key in original:
                np.testing.assert_array_almost_equal(
                    state[key], original[key], decimal=5
                )

    def test_weights_only_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wo.soul")
            save_soul(FakeModel(), path, weights_only=True)

            soul, state = load_soul(path)
            assert len(state) == 0

    def test_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested", "deep", "test.soul")
            save_soul(FakeModel(), path)
            assert os.path.exists(path)

    def test_default_soul_profile_created_when_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "auto.soul")
            save_soul(FakeModel(), path)

            soul, _ = load_soul(path)
            assert soul.name == "auto"

    def test_metadata_from_model_copied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "meta.soul")
            model = FakeModelWithMetadata()
            save_soul(model, path)

            meta_path = path + ".meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["metadata"]["framework"] == "slonet"

    def test_lineage_from_model_copied_when_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lin.soul")
            model = FakeModelWithMetadata()
            soul = SloProfile(name="lin", lineage="")
            save_soul(model, path, soul_profile=soul)

            soul2, _ = load_soul(path)
            assert soul2.lineage == "custom-lineage"

    def test_lineage_not_overwritten_when_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lin2.soul")
            model = FakeModelWithMetadata()
            soul = SloProfile(name="lin2", lineage="original")
            save_soul(model, path, soul_profile=soul)

            soul2, _ = load_soul(path)
            assert soul2.lineage == "original"

    def test_integrity_hash_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hash.soul")
            soul = SloProfile(name="hash-bot")
            assert soul.integrity_hash == ""
            save_soul(FakeModel(), path, soul_profile=soul)

            soul2, _ = load_soul(path)
            assert soul2.integrity_hash != ""

    def test_list_values_converted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "list.soul")
            save_soul(FakeModelDictValues(), path)

            soul, state = load_soul(path)
            assert "list_param" in state
            assert state["list_param"].shape == (2, 2)

    def test_dict_values_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dictskip.soul")
            save_soul(FakeModelDictSkip(), path)

            soul, state = load_soul(path)
            assert "nested_dict" not in state
            assert "valid" in state

    def test_non_convertible_value_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.soul")
            save_soul(FakeModelBadValues(), path)

            soul, state = load_soul(path)
            assert "bad" not in state
            assert "good" in state

    def test_empty_state_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.soul")
            save_soul(FakeModelEmpty(), path)

            soul, state = load_soul(path)
            assert len(state) == 0

    def test_soul_profile_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "override.soul")
            soul = SloProfile(
                name="custom",
                version="2.0.0",
                base_model="custom-model",
                system_prompt="You are custom.",
            )
            save_soul(FakeModel(), path, soul_profile=soul)

            soul2, _ = load_soul(path)
            assert soul2.name == "custom"
            assert soul2.version == "2.0.0"
            assert soul2.base_model == "custom-model"
            assert soul2.system_prompt == "You are custom."

    def test_model_without_state_dict_no_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nosd.soul")
            save_soul(FakeModelNoStateDict(), path, weights_only=True)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# generate_sample_dialogue tests
# ---------------------------------------------------------------------------

class TestGenerateSampleDialogue:
    def _make_vocab(self):
        chars = list("abcdefghijklmnopqrstuvwxyz !?.")
        stoi = {c: i + 1 for i, c in enumerate(chars)}
        itos = {i + 1: c for i, c in enumerate(chars)}
        itos[0] = "?"
        return stoi, itos

    def test_basic_generation(self):
        stoi, itos = self._make_vocab()
        model = FakeModelGenerate()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=2, max_tokens=50)

        assert len(dialogue) == 4
        assert dialogue[0]["role"] == "user"
        assert dialogue[1]["role"] == "assistant"
        assert dialogue[2]["role"] == "user"
        assert dialogue[3]["role"] == "assistant"

    def test_num_turns_controls_length(self):
        stoi, itos = self._make_vocab()
        model = FakeModelGenerate()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=1)
        assert len(dialogue) == 2

    def test_max_num_turns(self):
        stoi, itos = self._make_vocab()
        model = FakeModelGenerate()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=10)
        assert len(dialogue) == 6  # max 3 prompts

    def test_zero_turns(self):
        stoi, itos = self._make_vocab()
        model = FakeModelGenerate()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=0)
        assert dialogue == []

    def test_model_with_forward_only(self):
        stoi, itos = self._make_vocab()
        model = FakeModelForward()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=1, max_tokens=50)
        assert len(dialogue) == 2
        assert dialogue[0]["role"] == "user"

    def test_model_no_generate_no_forward(self):
        stoi, itos = self._make_vocab()
        model = FakeModelNoGen()
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=1, max_tokens=50)
        assert len(dialogue) == 2
        # Falls back to passing idx through
        assert dialogue[1]["role"] == "assistant"

    def test_generation_failure_returns_placeholder(self):
        class BrokenModel:
            def generate(self, idx, **kw):
                raise RuntimeError("boom")

        stoi, itos = self._make_vocab()
        dialogue = generate_sample_dialogue(BrokenModel(), stoi, itos, num_turns=1)
        assert len(dialogue) == 2
        assert dialogue[1]["content"] == "[generation failed]"

    def test_response_truncated_to_100_chars(self):
        class LongModel:
            def generate(self, idx, **kw):
                return np.arange(200, dtype=np.int64).reshape(1, -1)

        stoi = {chr(i): i for i in range(300)}
        itos = {i: chr(i) for i in range(300)}
        dialogue = generate_sample_dialogue(LongModel(), stoi, itos, num_turns=1, max_tokens=200)
        assert len(dialogue[1]["content"]) <= 100

    def test_empty_stoi_uses_zero_fallback(self):
        model = FakeModelGenerate()
        dialogue = generate_sample_dialogue(model, {}, {}, num_turns=1, max_tokens=10)
        assert len(dialogue) == 2
        # Characters not in stoi map to index 0, which maps to "?" in itos
        assert dialogue[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# create_soul_profile edge cases
# ---------------------------------------------------------------------------

class TestCreateSoulProfileExtended:
    def test_default_system_prompt(self):
        sp = create_soul_profile(name="test")
        assert sp.system_prompt == "You are test, a thoughtful AI assistant."

    def test_custom_system_prompt(self):
        sp = create_soul_profile(name="test", system_prompt="Custom prompt.")
        assert sp.system_prompt == "Custom prompt."

    def test_empty_string_system_prompt_uses_default(self):
        sp = create_soul_profile(name="test", system_prompt="")
        assert "test" in sp.system_prompt

    def test_personality_passed_through(self):
        pc = PersonalityCore(warmth=0.9, creativity=0.1)
        sp = create_soul_profile(name="test", personality=pc)
        assert sp.personality.warmth == 0.9
        assert sp.personality.creativity == 0.1

    def test_generation_passed_through(self):
        gp = GenerationParams(temperature=0.3, top_p=0.8)
        sp = create_soul_profile(name="test", generation=gp)
        assert sp.generation.temperature == 0.3
        assert sp.generation.top_p == 0.8

    def test_tags_stored(self):
        sp = create_soul_profile(name="test", tags=["a", "b", "c"])
        assert sp.tags == ["a", "b", "c"]

    def test_kwargs_ignored(self):
        sp = create_soul_profile(name="test", unknown_param=42)
        assert sp.name == "test"

    def test_lineage_and_signature(self):
        sp = create_soul_profile(
            name="test",
            lineage="custom",
            dataset_signature="abc123",
        )
        assert sp.lineage == "custom"
        assert sp.dataset_signature == "abc123"

    def test_integrity_hash_deterministic(self):
        born = "2025-01-01T00:00:00Z"
        sp1 = SloProfile(name="deterministic", born_at=born)
        sp1.integrity_hash = sp1.compute_hash()
        sp2 = SloProfile(name="deterministic", born_at=born)
        sp2.integrity_hash = sp2.compute_hash()
        assert sp1.integrity_hash == sp2.integrity_hash

    def test_integrity_hash_changes_with_name(self):
        sp1 = create_soul_profile(name="alpha")
        sp2 = create_soul_profile(name="beta")
        assert sp1.integrity_hash != sp2.integrity_hash

    def test_base_model_default(self):
        sp = create_soul_profile(name="test")
        assert sp.base_model == "nanogpt"

    def test_created_by_fixed(self):
        sp = create_soul_profile(name="test")
        assert sp.created_by == "SloughGPT Training Pipeline"


# ---------------------------------------------------------------------------
# Edge cases for SloProfile / to_sou_string / round-trip
# ---------------------------------------------------------------------------

class TestSloProfileEdgeCases:
    def test_to_dict_all_fields(self):
        sp = SloProfile(
            name="full",
            tagline="A tagline",
            description="A description",
            system_prompt="System",
            sample_dialogue=[{"role": "user", "content": "Hi"}],
            lora_adapters=["adapter1"],
            quantization="4bit",
            acl_users=["alice"],
            watermark_enabled=True,
            watermark_strength=0.5,
            tags=["t1"],
            certifications=["cert1"],
            metadata={"key": "val"},
        )
        d = sp.to_dict()
        assert d["tagline"] == "A tagline"
        assert d["sample_dialogue"] == [{"role": "user", "content": "Hi"}]
        assert d["lora_adapters"] == ["adapter1"]
        assert d["quantization"] == "4bit"
        assert d["acl_users"] == ["alice"]
        assert d["watermark_enabled"] is True
        assert d["watermark_strength"] == 0.5
        assert d["tags"] == ["t1"]
        assert d["certifications"] == ["cert1"]
        assert d["metadata"] == {"key": "val"}

    def test_to_sou_string_roundtrip_preserves_sections(self):
        sp = SloProfile(name="sectest")
        sp.personality.warmth = 0.8
        sp.behavior.speaking_style = "formal"
        sp.behavior.follow_up_tendency = 0.9
        sp.cognition.pattern_recognition = 0.7
        sp.emotion.empathy_depth = 0.6
        sp.generation.stop = ["STOP"]
        sp.tags = ["x", "y"]
        sp.certifications = ["cert-a"]
        sp.lora_adapters = ["lora-1"]

        sou = sp.to_sou_string()
        assert "PERSONALITY" in sou
        assert "BEHAVIOR" in sou
        assert "COGNITION" in sou
        assert "EMOTION" in sou
        assert "stop STOP" in sou
        assert "TAG x,y" in sou
        assert "CERTIFICATION cert-a" in sou
        assert "ADAPTER" in sou

        parsed = SouParser.parse(sou)
        assert parsed.personality.warmth == 0.8
        assert parsed.behavior.speaking_style == "formal"
        assert abs(parsed.behavior.follow_up_tendency - 0.9) < 0.01
        assert abs(parsed.cognition.pattern_recognition - 0.7) < 0.01
        assert abs(parsed.emotion.empathy_depth - 0.6) < 0.01
        assert parsed.generation.stop == ["STOP"]
        assert parsed.tags == ["x", "y"]
        assert parsed.certifications == ["cert-a"]
        assert parsed.lora_adapters == ["lora-1"]

    def test_compute_hash_changes_with_modification(self):
        sp = SloProfile(name="mutate")
        h1 = sp.compute_hash()
        sp.personality.warmth = 0.99
        h2 = sp.compute_hash()
        assert h1 != h2

    def test_born_at_auto_populated(self):
        sp = SloProfile(name="auto")
        assert sp.born_at != ""
        assert "T" in sp.born_at

    def test_born_at_not_overwritten(self):
        sp = SloProfile(name="fixed", born_at="2025-01-01T00:00:00Z")
        assert sp.born_at == "2025-01-01T00:00:00Z"
