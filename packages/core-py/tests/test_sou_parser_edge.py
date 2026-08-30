"""Edge-case tests for SouParser.parse() not covered in test_slo_format.py."""

import math
from domains.inference.slo_format import (
    SouParser,
    SloProfile,
    PersonalityCore,
    BehavioralTraits,
    CognitiveSignature,
    EmotionalRange,
    GenerationParams,
    ContextParams,
)


class TestParseEdgeCases:

    def test_empty_string(self):
        sp = SouParser.parse("")
        assert sp.name == "unknown"

    def test_whitespace_only(self):
        sp = SouParser.parse("   \n  \n   ")
        assert sp.name == "unknown"

    def test_only_comments(self):
        sp = SouParser.parse("# comment1\n# comment2\n")
        assert sp.name == "unknown"

    def test_description_with_spaces(self):
        content = "SOUL bot\nDESCRIPTION This is a multi-word description.\n"
        sp = SouParser.parse(content)
        assert sp.description == "This is a multi-word description."

    def test_tagline_with_spaces(self):
        content = "SOUL bot\nTAGLINE Hello world tagline\n"
        sp = SouParser.parse(content)
        assert sp.tagline == "Hello world tagline"

    def test_quantization_line(self):
        content = "SOUL bot\nQUANTIZATION int8\n"
        sp = SouParser.parse(content)
        assert sp.quantization == "int8"

    def test_quantization_with_spaces(self):
        content = "SOUL bot\nQUANTIZATION  4bit  \n"
        sp = SouParser.parse(content)
        assert sp.quantization == "4bit"

    def test_single_certification(self):
        content = "SOUL bot\nCERTIFICATION iso-27001\n"
        sp = SouParser.parse(content)
        assert sp.certifications == ["iso-27001"]

    def test_multiple_certifications(self):
        content = (
            "SOUL bot\nCERTIFICATION iso-27001\n"
            "CERTIFICATION soc2\nCERTIFICATION gdpr\n"
        )
        sp = SouParser.parse(content)
        assert sp.certifications == ["iso-27001", "soc2", "gdpr"]

    def test_tag_with_spaces_around_commas(self):
        content = "SOUL bot\nTAG  python , ml , training\n"
        sp = SouParser.parse(content)
        assert sp.tags == ["python", "ml", "training"]

    def test_single_tag(self):
        content = "SOUL bot\nTAG python\n"
        sp = SouParser.parse(content)
        assert sp.tags == ["python"]

    def test_metadata_custom_key(self):
        content = (
            "SOUL bot\nMETADATA custom_key custom_value\n"
            "METADATA epochs_trained 5\n"
        )
        sp = SouParser.parse(content)
        assert sp.metadata["custom_key"] == "custom_value"
        assert sp.epochs_trained == 5

    def test_metadata_final_val_loss(self):
        content = "SOUL bot\nMETADATA final_val_loss 0.33\n"
        sp = SouParser.parse(content)
        assert sp.final_val_loss == 0.33

    def test_parameter_float_values(self):
        content = (
            "SOUL bot\nPARAMETER\ntemperature 0.9\ntop_p 0.95\n"
        )
        sp = SouParser.parse(content)
        assert sp.generation.temperature == 0.9
        assert sp.generation.top_p == 0.95

    def test_parameter_int_values(self):
        content = "SOUL bot\nPARAMETER\ntop_k 100\nmax_tokens 4096\n"
        sp = SouParser.parse(content)
        assert sp.generation.top_k == 100
        assert sp.generation.max_tokens == 4096

    def test_parameter_multiple_stop_tokens(self):
        content = (
            "SOUL bot\nPARAMETER\n"
            "stop END\nstop STOP\nstop DONE\n"
        )
        sp = SouParser.parse(content)
        assert sp.generation.stop == ["END", "STOP", "DONE"]

    def test_parameter_single_stop(self):
        content = "SOUL bot\nPARAMETER\nstop END\n"
        sp = SouParser.parse(content)
        assert sp.generation.stop == ["END"]

    def test_context_int_values(self):
        content = "SOUL bot\nCONTEXT\ncontext_window 8192\nnum_ctx 8192\n"
        sp = SouParser.parse(content)
        assert sp.context.context_window == 8192
        assert sp.context.num_ctx == 8192

    def test_context_non_int_value_stored_as_string(self):
        content = "SOUL bot\nCONTEXT\nnum_gpu auto\n"
        sp = SouParser.parse(content)
        assert sp.context.num_gpu == "auto"

    def test_personality_section(self):
        content = (
            "SOUL bot\nPERSONALITY\nwarmth 0.9\ncreativity 0.1\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.personality.warmth == 0.9
        assert sp.personality.creativity == 0.1

    def test_cognition_section(self):
        content = (
            "SOUL bot\nCOGNITION\npattern_recognition 0.8\n"
            "abstract_reasoning 0.2\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.cognition.pattern_recognition == 0.8
        assert sp.cognition.abstract_reasoning == 0.2

    def test_emotion_section(self):
        content = (
            "SOUL bot\nEMOTION\nempathy_depth 0.7\n"
            "mood_responsiveness 0.3\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.emotion.empathy_depth == 0.7
        assert sp.emotion.mood_responsiveness == 0.3

    def test_behavior_section_string_fields(self):
        content = (
            "SOUL bot\nBEHAVIOR\nspeaking_style formal\n"
            "reasoning_approach logical\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.behavior.speaking_style == "formal"
        assert sp.behavior.reasoning_approach == "logical"

    def test_behavior_section_float_fields(self):
        content = (
            "SOUL bot\nBEHAVIOR\nemotional_expressiveness 0.8\n"
            "formality_dynamic 0.2\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.behavior.emotional_expressiveness == 0.8
        assert sp.behavior.formality_dynamic == 0.2

    def test_behavior_mixed_string_and_float(self):
        content = (
            "SOUL bot\nBEHAVIOR\nspeaking_style casual\n"
            "emotional_expressiveness 0.6\nfollow_up_tendency 0.4\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.behavior.speaking_style == "casual"
        assert sp.behavior.emotional_expressiveness == 0.6
        assert sp.behavior.follow_up_tendency == 0.4

    def test_adapter_section(self):
        content = (
            "SOUL bot\nADAPTER\nlora-adapter-a\nlora-adapter-b\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.lora_adapters == ["lora-adapter-a", "lora-adapter-b"]

    def test_adapter_single(self):
        content = "SOUL bot\nADAPTER\nmy-adapter\nEND\n"
        sp = SouParser.parse(content)
        assert sp.lora_adapters == ["my-adapter"]

    def test_adapter_empty(self):
        content = "SOUL bot\nADAPTER\nEND\n"
        sp = SouParser.parse(content)
        assert sp.lora_adapters == []

    def test_message_single_word_content(self):
        content = "SOUL bot\nMESSAGE user Hello\n"
        sp = SouParser.parse(content)
        assert len(sp.sample_dialogue) == 1
        assert sp.sample_dialogue[0] == {"role": "user", "content": "Hello"}

    def test_message_multi_word_content(self):
        content = "SOUL bot\nMESSAGE assistant How are you doing today?\n"
        sp = SouParser.parse(content)
        assert sp.sample_dialogue[0] == {
            "role": "assistant",
            "content": "How are you doing today?",
        }

    def test_multiple_messages(self):
        content = (
            "SOUL bot\nMESSAGE user Hi\n"
            "MESSAGE assistant Hello!\nMESSAGE user Bye\n"
        )
        sp = SouParser.parse(content)
        assert len(sp.sample_dialogue) == 3

    def test_message_no_space_after_role(self):
        content = "SOUL bot\nMESSAGE user-only-word\n"
        sp = SouParser.parse(content)
        assert sp.sample_dialogue == []

    def test_system_prompt(self):
        content = "SOUL bot\nSYSTEM You are a helpful assistant.\n"
        sp = SouParser.parse(content)
        assert sp.system_prompt == "You are a helpful assistant."

    def test_all_header_fields(self):
        content = (
            "SOUL my-bot\nVERSION 2.0.0\nLINEAGE custom\n"
            "BORN 2025-01-01T00:00:00Z\nCREATED_BY Me\n"
            "BASEMODEL gpt2\nTRAINING_DATA dataset.csv\n"
            "DATA_SIGNATURE abc123\n"
        )
        sp = SouParser.parse(content)
        assert sp.name == "my-bot"
        assert sp.version == "2.0.0"
        assert sp.lineage == "custom"
        assert sp.born_at == "2025-01-01T00:00:00Z"
        assert sp.created_by == "Me"
        assert sp.base_model == "gpt2"
        assert sp.training_dataset == "dataset.csv"
        assert sp.dataset_signature == "abc123"

    def test_section_switching_without_end(self):
        content = (
            "SOUL bot\nPERSONALITY\nwarmth 0.9\n"
            "COGNITION\npattern_recognition 0.8\nEND\n"
        )
        sp = SouParser.parse(content)
        assert sp.personality.warmth == 0.5
        assert sp.cognition.pattern_recognition == 0.8

    def test_parameter_unknown_key_raises(self):
        import pytest
        content = "SOUL bot\nPARAMETER\nunknown_field 42\n"
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            SouParser.parse(content)

    def test_metadata_missing_value(self):
        content = "SOUL bot\nMETADATA only_key\n"
        sp = SouParser.parse(content)
        assert sp.metadata == {}

    def test_whitespace_lines_ignored(self):
        content = "SOUL bot\n   \n  \t  \nVERSION 3.0\n"
        sp = SouParser.parse(content)
        assert sp.name == "bot"
        assert sp.version == "3.0"

    def test_full_profile_round_trip(self):
        sp = SloProfile(name="full-bot")
        sp.tagline = "A full profile"
        sp.description = "Tests everything"
        sp.base_model = "gpt2"
        sp.training_dataset = "data.txt"
        sp.dataset_signature = "sig123"
        sp.system_prompt = "You are full-bot."
        sp.personality.warmth = 0.9
        sp.personality.creativity = 0.1
        sp.behavior.speaking_style = "formal"
        sp.behavior.emotional_expressiveness = 0.8
        sp.cognition.pattern_recognition = 0.7
        sp.emotion.empathy_depth = 0.6
        sp.generation.temperature = 0.5
        sp.generation.stop = ["END", "STOP"]
        sp.context.context_window = 8192
        sp.tags = ["a", "b"]
        sp.lora_adapters = ["adapter1"]
        sp.certifications = ["cert1"]
        sp.epochs_trained = 10
        sp.final_train_loss = 0.1
        sp.final_val_loss = 0.2

        sou = sp.to_sou_string()
        parsed = SouParser.parse(sou)

        assert parsed.name == "full-bot"
        assert parsed.tagline == "A full profile"
        assert parsed.description == "Tests everything"
        assert parsed.base_model == "gpt2"
        assert parsed.training_dataset == "data.txt"
        assert parsed.dataset_signature == "sig123"
        assert parsed.system_prompt == "You are full-bot."
        assert parsed.personality.warmth == 0.9
        assert parsed.personality.creativity == 0.1
        assert parsed.behavior.speaking_style == "formal"
        assert parsed.behavior.emotional_expressiveness == 0.8
        assert parsed.cognition.pattern_recognition == 0.7
        assert parsed.emotion.empathy_depth == 0.6
        assert parsed.generation.temperature == 0.5
        assert parsed.generation.stop == ["END", "STOP"]
        assert parsed.context.context_window == 8192
        assert parsed.tags == ["a", "b"]
        assert parsed.lora_adapters == ["adapter1"]
        assert parsed.certifications == ["cert1"]
        assert parsed.epochs_trained == 10
        assert parsed.final_train_loss == 0.1
        assert parsed.final_val_loss == 0.2
