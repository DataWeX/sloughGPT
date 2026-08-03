"""Tests for the .soul format (SloProfile, SouParser, save/load soul)."""

import json
import math
import os
import struct
import string

import numpy as np
import pytest

from domains.inference.slo_format import (
    SOU_MAGIC,
    SOU_VERSION_V3,
    BehavioralTraits,
    CognitiveSignature,
    ContextParams,
    EmotionalRange,
    GenerationParams,
    PersonalityCore,
    SloProfile,
    SouParser,
    _soul_json_sanitize,
    create_soul_profile,
    generate_sample_dialogue,
    load_soul,
    save_soul,
    write_v3_sou,
)


def make_profile(**overrides):
    defaults = dict(
        name="soul-test",
        version="1.2.3",
        tagline="A test soul",
        description="Test description",
        base_model="nanogpt",
        training_dataset="shakespeare",
        epochs_trained=5,
        final_train_loss=1.5,
        final_val_loss=1.6,
        dataset_signature="abc123",
        system_prompt="You are a test soul.",
        sample_dialogue=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        lora_adapters=["adapter-a"],
        quantization="int4",
        tags=["test", "soul"],
        certifications=["cert-1"],
        metadata={"trained_by": "ci"},
    )
    defaults.update(overrides)
    return SloProfile(**defaults)


class TestParamDataclasses:
    def test_generation_params_to_dict(self):
        gp = GenerationParams(stop=["\n", "</s>"])
        d = gp.to_dict()
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.9
        assert d["top_k"] == 40
        assert d["max_tokens"] == 2048
        assert d["repeat_penalty"] == 1.1
        assert d["stop"] == ["\n", "</s>"]

    def test_context_params_to_dict(self):
        d = ContextParams().to_dict()
        assert d == {
            "context_window": 4096,
            "num_ctx": 4096,
            "num_gpu": 0,
            "num_thread": 0,
        }

    def test_personality_core_defaults(self):
        assert PersonalityCore().to_dict()["warmth"] == 0.5

    def test_behavioral_traits_defaults(self):
        assert BehavioralTraits().to_dict()["speaking_style"] == "conversational"

    def test_cognitive_signature_defaults(self):
        assert CognitiveSignature().to_dict()["abstract_reasoning"] == 0.5

    def test_emotional_range_defaults(self):
        assert EmotionalRange().to_dict()["empathy_depth"] == 0.5


class TestSloProfile:
    def test_born_at_autoset(self):
        sp = make_profile(born_at="")
        assert sp.born_at

    def test_born_at_preserved(self):
        sp = make_profile(born_at="2020-01-01T00:00:00Z")
        assert sp.born_at == "2020-01-01T00:00:00Z"

    def test_to_dict_contains_all_keys(self):
        d = make_profile().to_dict()
        for key in ["name", "version", "lineage", "born_at", "personality",
                    "behavior", "cognition", "emotion", "generation", "context",
                    "system_prompt", "sample_dialogue", "lora_adapters",
                    "quantization", "tags", "certifications", "integrity_hash"]:
            assert key in d

    def test_compute_hash_deterministic(self):
        sp = make_profile()
        assert sp.compute_hash() == sp.compute_hash()
        assert len(sp.compute_hash()) == 16

    def test_compute_hash_changes_with_fields(self):
        a = make_profile(tags=["x"])
        b = make_profile(tags=["y"])
        assert a.compute_hash() != b.compute_hash()


class TestSoulJsonSanitize:
    def test_nan_becomes_none(self):
        assert _soul_json_sanitize(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _soul_json_sanitize(float("inf")) is None

    def test_recurses_through_dict_and_list(self):
        result = _soul_json_sanitize({"a": [1.0, float("nan")], "b": float("-inf")})
        assert result == {"a": [1.0, None], "b": None}

    def test_passthrough_finite(self):
        assert _soul_json_sanitize({"x": 1.5}) == {"x": 1.5}


class TestToSouStringAndParse:
    def test_round_trip_full_profile(self):
        sp = make_profile()
        parsed = SouParser.parse(sp.to_sou_string())
        assert parsed.name == "soul-test"
        assert parsed.version == "1.2.3"
        assert parsed.lineage == "nanogpt"
        assert parsed.base_model == "nanogpt"
        assert parsed.training_dataset == "shakespeare"
        assert parsed.dataset_signature == "abc123"
        assert parsed.system_prompt == "You are a test soul."
        assert parsed.tags == ["test", "soul"]
        assert parsed.lora_adapters == ["adapter-a"]
        assert parsed.certifications == ["cert-1"]
        assert parsed.generation.temperature == 0.7
        assert parsed.personality.warmth == 0.5

    def test_round_trip_dialogue(self):
        sp = make_profile()
        parsed = SouParser.parse(sp.to_sou_string())
        assert parsed.sample_dialogue == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_round_trip_metadata(self):
        sp = make_profile()
        parsed = SouParser.parse(sp.to_sou_string())
        assert parsed.epochs_trained == 5
        assert parsed.final_train_loss == 1.5
        assert parsed.final_val_loss == 1.6

    def test_round_trip_stop_tokens(self):
        sp = make_profile()
        sp.generation.stop = ["END", "</s>"]
        parsed = SouParser.parse(sp.to_sou_string())
        assert parsed.generation.stop == ["END", "</s>"]

    def test_round_trip_custom_params(self):
        sp = make_profile()
        sp.personality.warmth = 0.9
        sp.generation.temperature = 1.5
        sp.behavior.speaking_style = "formal"
        parsed = SouParser.parse(sp.to_sou_string())
        assert parsed.personality.warmth == 0.9
        assert parsed.generation.temperature == 1.5
        assert parsed.behavior.speaking_style == "formal"

    def test_parse_ignores_comments_and_blank_lines(self):
        content = "# comment\n\nSOUL hello\n# another\nVERSION 2.0\n"
        sp = SouParser.parse(content)
        assert sp.name == "hello"
        assert sp.version == "2.0"

    def test_parse_defaults_to_unknown_name(self):
        sp = SouParser.parse("VERSION 1.0\n")
        assert sp.name == "unknown"

    def test_sou_string_has_marketing_line(self):
        assert "# SloughGPT Slo Unit" in make_profile().to_sou_string()


class TestCreateSoulProfile:
    def test_sets_integrity_hash(self):
        sp = create_soul_profile("test-soul")
        assert sp.integrity_hash
        assert len(sp.integrity_hash) == 16

    def test_default_system_prompt(self):
        sp = create_soul_profile("sara")
        assert sp.system_prompt == "You are sara, a thoughtful AI assistant."

    def test_custom_system_prompt(self):
        sp = create_soul_profile("sara", system_prompt="custom")
        assert sp.system_prompt == "custom"

    def test_passes_training_metadata(self):
        sp = create_soul_profile(
            "sara", base_model="gpt2", training_dataset="d",
            epochs_trained=3, final_train_loss=0.5,
        )
        assert sp.base_model == "gpt2"
        assert sp.epochs_trained == 3
        assert sp.final_train_loss == 0.5


class FakeStateDictModel:
    def __init__(self, params=None, metadata=None, lineage=None):
        self._params = params if params is not None else {
            "w1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "b1": np.array([0.5, -0.5], dtype=np.float64),
        }
        self.metadata = metadata or {"k": "v"}
        self.lineage = lineage or "fake-lineage"

    def state_dict(self):
        return dict(self._params)


class TestSaveLoadSoul:
    def test_save_load_round_trip(self, tmp_path):
        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel()
        sp = make_profile()
        out = save_soul(model, path, soul_profile=sp)
        assert out == path
        soul, state = load_soul(path)
        assert soul.name == "soul-test"
        assert soul.version == "1.2.3"
        assert soul.metadata["trained_by"] == "ci"
        assert state["w1"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert state["b1"].tolist() == [0.5, -0.5]
        assert state["w1"].dtype == np.float32

    def test_save_writes_meta_sidecar(self, tmp_path):
        path = str(tmp_path / "model.soul")
        save_soul(FakeStateDictModel(), path, soul_profile=make_profile())
        meta = json.loads((tmp_path / "model.soul.meta.json").read_text())
        assert meta["name"] == "soul-test"

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "model.soul")
        save_soul(FakeStateDictModel(), path, soul_profile=make_profile())
        assert (tmp_path / "deep" / "nested" / "model.soul").exists()

    def test_save_without_profile_uses_filename(self, tmp_path):
        path = str(tmp_path / "my-soul.soul")
        save_soul(FakeStateDictModel(), path)
        soul, _ = load_soul(path)
        assert soul.name == "my-soul"

    def test_save_weights_only_skips_weights(self, tmp_path):
        path = str(tmp_path / "model.soul")
        save_soul(FakeStateDictModel(), path, soul_profile=make_profile(), weights_only=True)
        soul, state = load_soul(path)
        assert state == {}
        assert soul.name == "soul-test"

    def test_save_metadata_merges_model_metadata(self, tmp_path):
        path = str(tmp_path / "model.soul")
        save_soul(FakeStateDictModel(metadata={"model_meta": 1}), path,
                  soul_profile=make_profile(metadata={}))
        soul, _ = load_soul(path)
        assert soul.metadata == {"model_meta": 1}

    def test_invalid_magic_raises(self, tmp_path):
        path = tmp_path / "bad.soul"
        path.write_bytes(b"BOGUS" + b"\x00" * 20)
        with pytest.raises(ValueError, match="magic"):
            load_soul(str(path))

    def test_load_v2_json_weights(self, tmp_path):
        path = tmp_path / "v2.soul"
        config = make_profile().to_dict()
        config_json = json.dumps(config, default=str).encode()
        state_json = json.dumps({"w": [[1.0, 2.0]], "b": [0.0]}).encode()
        data = (
            SOU_MAGIC
            + struct.pack("<I", 2)
            + struct.pack("<I", len(config_json))
            + config_json
            + struct.pack("<I", len(state_json))
            + state_json
        )
        path.write_bytes(data)
        soul, state = load_soul(str(path))
        assert soul.name == "soul-test"
        assert state["w"].tolist() == [[1.0, 2.0]]
        assert state["b"].tolist() == [0.0]

    def test_save_skips_dict_values(self, tmp_path):
        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel(params={
            "w": np.array([1.0, 2.0]),
            "nested": {"a": 1},
        })
        save_soul(model, path, soul_profile=make_profile())
        _, state = load_soul(path)
        assert "w" in state
        assert "nested" not in state


class TestWriteV3Sou:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "v3.soul")
        write_v3_sou(path, {"name": "v3-test", "traits": ["a"]},
                     {"p0": np.array([1.0, 2.0, 3.0]), "p1": np.array([[4.0, 5.0]])})
        soul, state = load_soul(str(path))
        assert soul.name == "v3-test"
        assert state["p0"].tolist() == [1.0, 2.0, 3.0]
        assert state["p1"].tolist() == [[4.0, 5.0]]
        assert state["p1"].shape == (1, 2)
        assert state["p0"].dtype == np.float32

    def test_accepts_lists(self, tmp_path):
        path = str(tmp_path / "v3.soul")
        write_v3_sou(path, {"x": 1}, {"p0": [1, 2, 3]})
        _, state = load_soul(str(path))
        assert state["p0"].tolist() == [1.0, 2.0, 3.0]


class FakeGenModel:
    def __init__(self, output, raise_error=False):
        self._output = output
        self._raise = raise_error

    def generate(self, idx, max_new_tokens=50, temperature=0.8):
        if self._raise:
            raise RuntimeError("gen failed")
        return self._output


def build_vocab(prompts):
    chars = sorted(set("".join(prompts) + " abc"))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return stoi, itos


class TestGenerateSampleDialogue:
    def test_generates_turns(self):
        prompts = [
            "Hello! How are you today?",
            "What's your favorite thing about helping people?",
        ]
        stoi, itos = build_vocab(prompts)
        response_chars = [c for c in "Hello! How are you today?"]
        idx = np.array([[stoi[c] for c in response_chars]], dtype=np.int64)
        model = FakeGenModel(idx)
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=2)
        assert len(dialogue) == 4
        assert dialogue[0]["role"] == "user"
        assert dialogue[1]["role"] == "assistant"

    def test_num_turns_caps_prompts(self):
        stoi, itos = build_vocab(["Hello! How are you today?"])
        idx = np.array([[stoi["H"]]], dtype=np.int64)
        model = FakeGenModel(idx)
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=5)
        assert len(dialogue) == 6

    def test_generation_failure_message(self):
        stoi, itos = build_vocab(["Hello! How are you today?"])
        model = FakeGenModel(None, raise_error=True)
        dialogue = generate_sample_dialogue(model, stoi, itos, num_turns=1)
        assert dialogue[1]["content"] == "[generation failed]"

    def test_model_without_generate_returns_empty_response(self):
        prompts = ["Hello! How are you today?"]
        stoi, itos = build_vocab(prompts)
        idx = np.array([[stoi[c] for c in prompts[0]]], dtype=np.int64)

        class NoGenModel:
            def forward(self, x):
                return x

        dialogue = generate_sample_dialogue(NoGenModel(), stoi, itos, num_turns=1)
        assert dialogue[1]["content"] == ""

    def test_model_without_generate_or_forward(self):
        stoi, itos = build_vocab(["Hello! How are you today?"])
        dialogue = generate_sample_dialogue(object(), stoi, itos, num_turns=1)
        # output falls back to the raw prompt tokens, so the response is empty
        assert dialogue[1]["content"] == ""


class TestSouParserEdgeBranches:
    def test_parse_non_numeric_parameter(self):
        sp = SouParser.parse("SOUL t\nPARAMETER\n    temperature fast\n")
        assert sp.generation.temperature == "fast"

    def test_parse_non_int_context(self):
        sp = SouParser.parse("SOUL t\nCONTEXT\n    context_window many\n")
        assert sp.context.context_window == "many"

    def test_parse_non_float_personality(self):
        sp = SouParser.parse("SOUL t\nPERSONALITY\n    warmth high\n    END\n")
        assert sp.personality.warmth == "high"

    def test_parse_quantization(self):
        sp = SouParser.parse("SOUL t\nQUANTIZATION int8\n")
        assert sp.quantization == "int8"

    def test_parse_metadata_generic_key(self):
        sp = SouParser.parse("SOUL t\nMETADATA custom_key custom_value\n")
        assert sp.metadata == {"custom_key": "custom_value"}

    def test_parse_behavior_float_conversion_overflow(self):
        big = "9" * 400
        sp = SouParser.parse(
            "SOUL t\nBEHAVIOR\n    speaking_style " + big + "\n    END\n"
        )
        assert math.isinf(sp.behavior.speaking_style)

    def test_parse_behavior_unicode_digit_conversion_failure(self):
        sp = SouParser.parse("SOUL t\nBEHAVIOR\n    speaking_style ²\n    END\n")
        assert sp.behavior.speaking_style == "²"

    def test_sou_parser_load_save_file(self, tmp_path):
        path = tmp_path / "p.soul"
        sp = SouParser.parse("SOUL rtr\nVERSION 3.0\nTAGLINE hello\n")
        SouParser.save(sp, str(path))
        sp2 = SouParser.load(str(path))
        assert sp2.name == "rtr"
        assert sp2.version == "3.0"
        assert sp2.tagline == "hello"


class TestSaveSoulEdgeBranches:
    def test_lineage_from_model_when_profile_empty(self, tmp_path):
        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel(lineage="model-lineage")
        save_soul(model, path, soul_profile=make_profile(lineage=""))
        soul, _ = load_soul(path)
        assert soul.lineage == "model-lineage"

    def test_state_value_type_branches(self, tmp_path):
        class TensorLike:
            def __init__(self, arr):
                self.data = arr

        class NumpyLike:
            def cpu(self):
                return self

            def numpy(self):
                return np.array([1.0, 2.0])

        class DetachLike:
            def detach(self):
                class Inner:
                    def cpu(self):
                        return self

                    def numpy(self):
                        return np.array([3.0, 4.0])

                return Inner()

        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel(params={
            "t": TensorLike(np.array([1.0], np.float64)),
            "n": NumpyLike(),
            "d": DetachLike(),
            "l": [5.0, 6.0],
            "s": 7.5,
        })
        save_soul(model, path, soul_profile=make_profile())
        _, state = load_soul(path)
        assert state["t"].tolist() == [1.0]
        assert state["n"].tolist() == [1.0, 2.0]
        assert state["d"].tolist() == [3.0, 4.0]
        assert state["l"].tolist() == [5.0, 6.0]
        assert float(state["s"]) == 7.5

    def test_skips_unconvertible_value(self, tmp_path):
        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel(params={"w": np.array([1.0]), "bad": {1, 2}})
        save_soul(model, path, soul_profile=make_profile())
        _, state = load_soul(path)
        assert "w" in state
        assert "bad" not in state

    def test_zero_params_logs_error(self, tmp_path):
        path = str(tmp_path / "model.soul")
        model = FakeStateDictModel(params={"only": {"nested": 1}})
        save_soul(model, path, soul_profile=make_profile())
        _, state = load_soul(path)
        assert state == {}

    def test_temp_cleanup_on_failure(self, tmp_path, monkeypatch):
        path = str(tmp_path / "model.soul")

        def boom(src, dst):
            raise OSError("rename failed")

        monkeypatch.setattr("os.rename", boom)
        with pytest.raises(OSError):
            save_soul(FakeStateDictModel(), path, soul_profile=make_profile())
        assert not list(tmp_path.glob("*.tmp"))

    def test_temp_cleanup_unlink_error_swallowed(self, tmp_path, monkeypatch):
        path = str(tmp_path / "model.soul")

        def boom_rename(src, dst):
            raise OSError("rename failed")

        def boom_unlink(p):
            raise OSError("unlink failed")

        monkeypatch.setattr("os.rename", boom_rename)
        monkeypatch.setattr("os.unlink", boom_unlink)
        with pytest.raises(OSError):
            save_soul(FakeStateDictModel(), path, soul_profile=make_profile())
        assert not os.path.exists(path)


class TestLoadSoulEdge:
    def test_v2_bad_state_json(self, tmp_path):
        path = tmp_path / "v2bad.soul"
        config = make_profile().to_dict()
        config_json = json.dumps(config, default=str).encode()
        state_json = b"{this is not valid json"
        data = (
            SOU_MAGIC
            + struct.pack("<I", 2)
            + struct.pack("<I", len(config_json))
            + config_json
            + struct.pack("<I", len(state_json))
            + state_json
        )
        path.write_bytes(data)
        soul, state = load_soul(str(path))
        assert state == {}
        assert soul.name == "soul-test"
