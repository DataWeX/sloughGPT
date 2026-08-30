"""Tests for domains.multimodal — MultiModalConfig."""

from domains.multimodal import MultiModalConfig


class TestMultiModalConfigDefaults:
    def test_image_size_default(self):
        cfg = MultiModalConfig()
        assert cfg.image_size == 224

    def test_patch_size_default(self):
        cfg = MultiModalConfig()
        assert cfg.patch_size == 16

    def test_vision_hidden_size_default(self):
        cfg = MultiModalConfig()
        assert cfg.vision_hidden_size == 768

    def test_vision_num_layers_default(self):
        cfg = MultiModalConfig()
        assert cfg.vision_num_layers == 12

    def test_vision_num_heads_default(self):
        cfg = MultiModalConfig()
        assert cfg.vision_num_heads == 12

    def test_vocab_size_default(self):
        cfg = MultiModalConfig()
        assert cfg.vocab_size == 50257

    def test_text_hidden_size_default(self):
        cfg = MultiModalConfig()
        assert cfg.text_hidden_size == 768

    def test_text_num_layers_default(self):
        cfg = MultiModalConfig()
        assert cfg.text_num_layers == 12

    def test_text_num_heads_default(self):
        cfg = MultiModalConfig()
        assert cfg.text_num_heads == 12

    def test_max_seq_length_default(self):
        cfg = MultiModalConfig()
        assert cfg.max_seq_length == 512

    def test_fusion_type_default(self):
        cfg = MultiModalConfig()
        assert cfg.fusion_type == "cross_attention"

    def test_projection_dim_default(self):
        cfg = MultiModalConfig()
        assert cfg.projection_dim == 768

    def test_no_positional_args(self):
        cfg = MultiModalConfig()
        assert isinstance(cfg, MultiModalConfig)

    def test_field_count(self):
        import dataclasses
        fields = [f.name for f in dataclasses.fields(MultiModalConfig)]
        assert len(fields) == 12


class TestMultiModalConfigCustom:
    def test_image_size_custom(self):
        cfg = MultiModalConfig(image_size=512)
        assert cfg.image_size == 512

    def test_patch_size_custom(self):
        cfg = MultiModalConfig(patch_size=32)
        assert cfg.patch_size == 32

    def test_fusion_type_custom(self):
        cfg = MultiModalConfig(fusion_type="concat")
        assert cfg.fusion_type == "concat"

    def test_max_seq_length_custom(self):
        cfg = MultiModalConfig(max_seq_length=1024)
        assert cfg.max_seq_length == 1024

    def test_vision_hidden_size_custom(self):
        cfg = MultiModalConfig(vision_hidden_size=1024)
        assert cfg.vision_hidden_size == 1024

    def test_vocab_size_custom(self):
        cfg = MultiModalConfig(vocab_size=30000)
        assert cfg.vocab_size == 30000

    def test_projection_dim_custom(self):
        cfg = MultiModalConfig(projection_dim=512)
        assert cfg.projection_dim == 512

    def test_all_fields_custom(self):
        cfg = MultiModalConfig(
            image_size=384, patch_size=14,
            vision_hidden_size=1024, vision_num_layers=24,
            vision_num_heads=16, vocab_size=250000,
            text_hidden_size=1024, text_num_layers=24,
            text_num_heads=16, max_seq_length=2048,
            fusion_type="concat", projection_dim=1024,
        )
        assert cfg.image_size == 384
        assert cfg.patch_size == 14
        assert cfg.vision_hidden_size == 1024
        assert cfg.vision_num_layers == 24
        assert cfg.vision_num_heads == 16
        assert cfg.vocab_size == 250000
        assert cfg.text_hidden_size == 1024
        assert cfg.text_num_layers == 24
        assert cfg.text_num_heads == 16
        assert cfg.max_seq_length == 2048
        assert cfg.fusion_type == "concat"
        assert cfg.projection_dim == 1024

    def test_partial_override(self):
        cfg = MultiModalConfig(image_size=128, fusion_type="sum")
        assert cfg.image_size == 128
        assert cfg.fusion_type == "sum"
        assert cfg.patch_size == 16
        assert cfg.max_seq_length == 512

    def test_small_image_size(self):
        cfg = MultiModalConfig(image_size=32, patch_size=4)
        assert cfg.image_size == 32
        assert cfg.patch_size == 4

    def test_large_image_size(self):
        cfg = MultiModalConfig(image_size=1024, patch_size=64)
        assert cfg.image_size == 1024
        assert cfg.patch_size == 64

    def test_zero_seq_length(self):
        cfg = MultiModalConfig(max_seq_length=0)
        assert cfg.max_seq_length == 0

    def test_large_seq_length(self):
        cfg = MultiModalConfig(max_seq_length=100000)
        assert cfg.max_seq_length == 100000

    def test_empty_fusion_type(self):
        cfg = MultiModalConfig(fusion_type="")
        assert cfg.fusion_type == ""


class TestMultiModalConfigEquality:
    def test_equal_instances(self):
        a = MultiModalConfig()
        b = MultiModalConfig()
        assert a == b

    def test_not_equal_image_size(self):
        a = MultiModalConfig(image_size=224)
        b = MultiModalConfig(image_size=512)
        assert a != b

    def test_not_equal_fusion_type(self):
        a = MultiModalConfig(fusion_type="concat")
        b = MultiModalConfig(fusion_type="sum")
        assert a != b

    def test_equal_custom(self):
        a = MultiModalConfig(image_size=384, patch_size=14)
        b = MultiModalConfig(image_size=384, patch_size=14)
        assert a == b

    def test_not_equal_to_non_dataclass(self):
        cfg = MultiModalConfig()
        assert cfg != "not a config"

    def test_not_equal_to_dict(self):
        cfg = MultiModalConfig()
        assert cfg != {"image_size": 224}


class TestMultiModalConfigRepr:
    def test_repr_contains_class_name(self):
        cfg = MultiModalConfig()
        assert "MultiModalConfig" in repr(cfg)

    def test_repr_contains_field_values(self):
        cfg = MultiModalConfig(image_size=512, fusion_type="concat")
        r = repr(cfg)
        assert "image_size=512" in r
        assert "fusion_type='concat'" in r

    def test_repr_defaults(self):
        cfg = MultiModalConfig()
        r = repr(cfg)
        assert "image_size=224" in r
        assert "patch_size=16" in r
        assert "max_seq_length=512" in r


class TestMultiModalConfigMutation:
    def test_can_set_image_size(self):
        cfg = MultiModalConfig()
        cfg.image_size = 512
        assert cfg.image_size == 512

    def test_can_set_fusion_type(self):
        cfg = MultiModalConfig()
        cfg.fusion_type = "additive"
        assert cfg.fusion_type == "additive"

    def test_can_overwrite_fields(self):
        cfg = MultiModalConfig(image_size=224)
        cfg.image_size = 384
        cfg.image_size = 128
        assert cfg.image_size == 128

    def test_independent_instances(self):
        a = MultiModalConfig(image_size=128)
        b = MultiModalConfig(image_size=512)
        a.image_size = 256
        assert b.image_size == 512


class TestMultiModalConfigEdgeCases:
    def test_negative_image_size(self):
        cfg = MultiModalConfig(image_size=-1)
        assert cfg.image_size == -1

    def test_negative_patch_size(self):
        cfg = MultiModalConfig(patch_size=-1)
        assert cfg.patch_size == -1

    def test_negative_seq_length(self):
        cfg = MultiModalConfig(max_seq_length=-1)
        assert cfg.max_seq_length == -1

    def test_very_small_patch_size(self):
        cfg = MultiModalConfig(patch_size=1)
        assert cfg.patch_size == 1

    def test_large_num_layers(self):
        cfg = MultiModalConfig(vision_num_layers=100, text_num_layers=100)
        assert cfg.vision_num_layers == 100
        assert cfg.text_num_layers == 100

    def test_zero_num_heads(self):
        cfg = MultiModalConfig(vision_num_heads=0, text_num_heads=0)
        assert cfg.vision_num_heads == 0
        assert cfg.text_num_heads == 0

    def test_projection_dim_zero(self):
        cfg = MultiModalConfig(projection_dim=0)
        assert cfg.projection_dim == 0

    def test_vocab_size_zero(self):
        cfg = MultiModalConfig(vocab_size=0)
        assert cfg.vocab_size == 0

    def test_copy_semantics(self):
        import dataclasses
        a = MultiModalConfig(image_size=384, fusion_type="concat")
        b = dataclasses.replace(a, image_size=128)
        assert a.image_size == 384
        assert b.image_size == 128
        assert a.fusion_type == b.fusion_type

    def test_field_names(self):
        import dataclasses
        names = [f.name for f in dataclasses.fields(MultiModalConfig)]
        assert "image_size" in names
        assert "patch_size" in names
        assert "fusion_type" in names
        assert "max_seq_length" in names
        assert "vocab_size" in names
        assert "projection_dim" in names

    def test_fusion_types_string(self):
        for ft in ["cross_attention", "concat", "sum", "gated", "bilinear"]:
            cfg = MultiModalConfig(fusion_type=ft)
            assert cfg.fusion_type == ft

    def test_large_hidden_sizes(self):
        cfg = MultiModalConfig(vision_hidden_size=4096, text_hidden_size=4096)
        assert cfg.vision_hidden_size == 4096
        assert cfg.text_hidden_size == 4096
