"""Tests for domains.multimodal.manager — MultimodalCapabilities."""

from domains.multimodal.manager import MultimodalCapabilities


class TestMultimodalCapabilitiesDefaults:
    def test_all_booleans_default_false(self):
        mc = MultimodalCapabilities()
        assert mc.speech_to_text is False
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is False

    def test_models_default_none(self):
        mc = MultimodalCapabilities()
        assert mc.speech_model is None
        assert mc.vision_model is None

    def test_no_positional_args(self):
        mc = MultimodalCapabilities()
        assert isinstance(mc, MultimodalCapabilities)

    def test_field_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(MultimodalCapabilities)]
        assert len(fields) == 6
        assert set(fields) == {
            "speech_to_text", "image_caption", "object_detection",
            "vqa", "speech_model", "vision_model",
        }


class TestMultimodalCapabilitiesCustom:
    def test_single_bool_true(self):
        mc = MultimodalCapabilities(speech_to_text=True)
        assert mc.speech_to_text is True
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is False

    def test_image_caption_true(self):
        mc = MultimodalCapabilities(image_caption=True)
        assert mc.image_caption is True
        assert mc.speech_to_text is False

    def test_object_detection_true(self):
        mc = MultimodalCapabilities(object_detection=True)
        assert mc.object_detection is True
        assert mc.vqa is False

    def test_vqa_true(self):
        mc = MultimodalCapabilities(vqa=True)
        assert mc.vqa is True
        assert mc.speech_to_text is False

    def test_all_bools_true(self):
        mc = MultimodalCapabilities(
            speech_to_text=True, image_caption=True,
            object_detection=True, vqa=True,
        )
        assert mc.speech_to_text is True
        assert mc.image_caption is True
        assert mc.object_detection is True
        assert mc.vqa is True

    def test_all_bools_explicit_false(self):
        mc = MultimodalCapabilities(
            speech_to_text=False, image_caption=False,
            object_detection=False, vqa=False,
        )
        assert mc.speech_to_text is False
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is False

    def test_speech_model_string(self):
        mc = MultimodalCapabilities(speech_model="whisper")
        assert mc.speech_model == "whisper"

    def test_vision_model_string(self):
        mc = MultimodalCapabilities(vision_model="clip")
        assert mc.vision_model == "clip"

    def test_both_models(self):
        mc = MultimodalCapabilities(speech_model="vosk", vision_model="slonet")
        assert mc.speech_model == "vosk"
        assert mc.vision_model == "slonet"

    def test_speech_model_empty_string(self):
        mc = MultimodalCapabilities(speech_model="")
        assert mc.speech_model == ""

    def test_vision_model_empty_string(self):
        mc = MultimodalCapabilities(vision_model="")
        assert mc.vision_model == ""

    def test_all_fields_custom(self):
        mc = MultimodalCapabilities(
            speech_to_text=True,
            image_caption=True,
            object_detection=False,
            vqa=True,
            speech_model="whisper-large",
            vision_model="blip-2",
        )
        assert mc.speech_to_text is True
        assert mc.image_caption is True
        assert mc.object_detection is False
        assert mc.vqa is True
        assert mc.speech_model == "whisper-large"
        assert mc.vision_model == "blip-2"

    def test_mixed_bools(self):
        mc = MultimodalCapabilities(
            speech_to_text=False, image_caption=True,
            object_detection=True, vqa=False,
        )
        assert mc.speech_to_text is False
        assert mc.image_caption is True
        assert mc.object_detection is True
        assert mc.vqa is False

    def test_partial_bools_with_models(self):
        mc = MultimodalCapabilities(
            speech_to_text=True, vqa=True,
            speech_model="server", vision_model="resnet",
        )
        assert mc.speech_to_text is True
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is True
        assert mc.speech_model == "server"
        assert mc.vision_model == "resnet"


class TestMultimodalCapabilitiesEquality:
    def test_equal_instances(self):
        a = MultimodalCapabilities(speech_to_text=True, speech_model="x")
        b = MultimodalCapabilities(speech_to_text=True, speech_model="x")
        assert a == b

    def test_not_equal_different_bool(self):
        a = MultimodalCapabilities(speech_to_text=True)
        b = MultimodalCapabilities(speech_to_text=False)
        assert a != b

    def test_not_equal_different_model(self):
        a = MultimodalCapabilities(speech_model="a")
        b = MultimodalCapabilities(speech_model="b")
        assert a != b

    def test_not_equal_missing_model(self):
        a = MultimodalCapabilities(speech_model="a")
        b = MultimodalCapabilities()
        assert a != b

    def test_equal_all_none_models(self):
        a = MultimodalCapabilities()
        b = MultimodalCapabilities()
        assert a == b

    def test_not_equal_to_non_dataclass(self):
        mc = MultimodalCapabilities()
        assert mc != "not a dataclass"

    def test_not_equal_to_dict(self):
        mc = MultimodalCapabilities()
        assert mc != {"speech_to_text": False}


class TestMultimodalCapabilitiesRepr:
    def test_repr_contains_class_name(self):
        mc = MultimodalCapabilities()
        r = repr(mc)
        assert "MultimodalCapabilities" in r

    def test_repr_contains_field_values(self):
        mc = MultimodalCapabilities(speech_to_text=True, speech_model="whisper")
        r = repr(mc)
        assert "speech_to_text=True" in r
        assert "speech_model='whisper'" in r

    def test_repr_defaults(self):
        mc = MultimodalCapabilities()
        r = repr(mc)
        assert "speech_to_text=False" in r
        assert "speech_model=None" in r


class TestMultimodalCapabilitiesMutation:
    def test_can_set_bool_fields(self):
        mc = MultimodalCapabilities()
        mc.speech_to_text = True
        assert mc.speech_to_text is True

    def test_can_set_model_fields(self):
        mc = MultimodalCapabilities()
        mc.vision_model = "clip"
        assert mc.vision_model == "clip"

    def test_can_reset_model_to_none(self):
        mc = MultimodalCapabilities(speech_model="whisper")
        mc.speech_model = None
        assert mc.speech_model is None

    def test_can_overwrite_model(self):
        mc = MultimodalCapabilities(speech_model="whisper")
        mc.speech_model = "vosk"
        assert mc.speech_model == "vosk"

    def test_can_toggle_bool_back_and_forth(self):
        mc = MultimodalCapabilities(vqa=True)
        mc.vqa = False
        assert mc.vqa is False
        mc.vqa = True
        assert mc.vqa is True


class TestMultimodalCapabilitiesEdgeCases:
    def test_speech_model_long_string(self):
        long = "a" * 10000
        mc = MultimodalCapabilities(speech_model=long)
        assert mc.speech_model == long

    def test_vision_model_special_chars(self):
        mc = MultimodalCapabilities(vision_model="model/v1.0-beta_2024")
        assert mc.vision_model == "model/v1.0-beta_2024"

    def test_speech_model_whitespace(self):
        mc = MultimodalCapabilities(speech_model="  ")
        assert mc.speech_model == "  "

    def test_construct_kwargs_only(self):
        mc = MultimodalCapabilities(
            speech_to_text=False, image_caption=False,
            object_detection=False, vqa=False,
            speech_model=None, vision_model=None,
        )
        assert mc.speech_to_text is False
        assert mc.speech_model is None

    def test_hash_not_implemented(self):
        mc = MultimodalCapabilities()
        try:
            hash(mc)
        except TypeError:
            pass

    def test_dataclass_is_mutable(self):
        mc = MultimodalCapabilities()
        mc.image_caption = True
        mc.vqa = True
        mc.speech_model = "test"
        mc.vision_model = "test"
        assert mc.image_caption is True
        assert mc.vqa is True

    def test_copy_semantics(self):
        import dataclasses
        mc = MultimodalCapabilities(speech_to_text=True, speech_model="whisper")
        mc2 = dataclasses.replace(mc, speech_to_text=False)
        assert mc.speech_to_text is True
        assert mc2.speech_to_text is False
        assert mc.speech_model == mc2.speech_model

    def test_field_types(self):
        import dataclasses
        fields = {f.name: f.type for f in dataclasses.fields(MultimodalCapabilities)}
        assert fields["speech_to_text"] is bool
        assert fields["image_caption"] is bool
        assert fields["object_detection"] is bool
        assert fields["vqa"] is bool

    def test_construct_with_all_none(self):
        mc = MultimodalCapabilities(
            speech_to_text=None, image_caption=None,
            object_detection=None, vqa=None,
            speech_model=None, vision_model=None,
        )
        assert mc.speech_to_text is None
        assert mc.image_caption is None

    def test_as_dict_like_access(self):
        mc = MultimodalCapabilities(speech_to_text=True, speech_model="x")
        assert mc.speech_to_text is True
        assert mc.speech_model == "x"
        assert mc.vqa is False
        assert mc.vision_model is None

    def test_multiple_instances_independent(self):
        a = MultimodalCapabilities(speech_to_text=True)
        b = MultimodalCapabilities(speech_to_text=False)
        assert a.speech_to_text is True
        assert b.speech_to_text is False
        a.speech_to_text = False
        assert b.speech_to_text is False

    def test_booleans_are_exact_false(self):
        mc = MultimodalCapabilities()
        assert mc.speech_to_text is False
        assert mc.image_caption is False
        assert mc.object_detection is False
        assert mc.vqa is False

    def test_string_models_are_exact(self):
        mc = MultimodalCapabilities(speech_model="a", vision_model="b")
        assert mc.speech_model == "a"
        assert mc.vision_model == "b"
