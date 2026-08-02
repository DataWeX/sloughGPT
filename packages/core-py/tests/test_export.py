"""Tests for domains/training/export.py (model export utilities)."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import domains.training.onnx_export as onnx_mod
from domains.training import export as export_mod
from domains.training.export import (
    ExportConfig,
    GGUFExportOptions,
    ModelMetadata,
    ONNXExportOptions,
    _gguf_path,
    _replace_ext,
    create_model_metadata,
    export_all_formats,
    export_model,
    export_to_gguf,
    export_to_gguf_fp16,
    export_to_gguf_q4_k_m,
    export_to_onnx,
    export_to_safetensors,
    export_to_safetensors_bf16,
    export_to_sou,
    export_to_torch,
    export_to_torchscript,
    list_export_formats,
)


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
        self.cast_to = None

    def to(self, **kwargs):
        self.cast_to = kwargs.get("dtype")
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeScripted:
    def __init__(self, tag):
        self.tag = tag

    def save(self, path):
        Path(path).write_bytes(self.tag)


class _FakeJit:
    def __init__(self, torchref):
        self._torch = torchref

    def trace(self, model, example):
        self._torch.traced.append((model, example))
        return _FakeScripted(b"ts")

    def script(self, model):
        self._torch.scripted.append(model)
        return _FakeScripted(b"ts2")


class _FakeOnnx:
    def __init__(self, torchref):
        self._torch = torchref

    def export(self, *args, **kwargs):
        self._torch.onnx_exports.append((args, kwargs))


class _FakeTorch:
    __version__ = "9.9.9"
    long = "torch.long"

    def __init__(self):
        self.traced = []
        self.scripted = []
        self.onnx_exports = []
        self.saved = []
        self.zeros_calls = []
        self.from_numpy_calls = []
        self.jit = _FakeJit(self)
        self.onnx = _FakeOnnx(self)

    def from_numpy(self, arr):
        self.from_numpy_calls.append(arr)
        return _FakeTensor(arr)

    def save(self, obj, path):
        self.saved.append((obj, path))
        Path(path).write_bytes(b"pt")

    def zeros(self, *shape, dtype=None):
        self.zeros_calls.append((shape, dtype))
        return _FakeTensor(np.zeros(shape))


class FakeModel:
    def __init__(self, state=None, config=None, eval_log=None, **attrs):
        self._config = config
        self._state = state if state is not None else {}
        self.eval_calls = eval_log if eval_log is not None else []
        for k, v in attrs.items():
            setattr(self, k, v)

    def state_dict(self):
        return dict(self._state)

    def eval(self):
        self.eval_calls.append(1)


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = []

    def save_pretrained(self, path):
        self.saved_to.append(path)


class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if "output_path" in kwargs:
            return kwargs["output_path"]
        return args[1] if len(args) > 1 else "out"


@pytest.fixture
def fake_torch(monkeypatch):
    t = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", t)
    return t


@pytest.fixture
def fake_safetensors(monkeypatch):
    calls = []

    def save_file(state_dict, filename, metadata=None):
        Path(filename).write_bytes(b"st")
        calls.append((state_dict, str(filename), metadata))

    st = types.ModuleType("safetensors.torch")
    st.save_file = save_file
    top = types.ModuleType("safetensors")
    top.torch = st
    monkeypatch.setitem(sys.modules, "safetensors", top)
    monkeypatch.setitem(sys.modules, "safetensors.torch", st)
    return calls


@pytest.fixture
def stub_exports(monkeypatch):
    recs = {}
    for name in [
        "export_to_safetensors",
        "export_to_safetensors_bf16",
        "export_to_gguf",
        "export_to_gguf_fp16",
        "export_to_gguf_q4_k_m",
        "export_to_torch",
        "export_to_torchscript",
        "export_to_onnx",
        "export_to_sou",
        "export_all_formats",
    ]:
        rec = _Rec()
        recs[name] = rec
        monkeypatch.setattr(export_mod, name, rec)
    return recs


def _cfg(fmt, output="models/out", metadata=None, tokenizer=True, **kw):
    return ExportConfig(
        input_path="in.pt",
        output_path=output,
        format=fmt,
        metadata=metadata,
        include_tokenizer=tokenizer,
        **kw,
    )


class TestModelMetadata:
    def test_to_dict_roundtrip(self):
        md = ModelMetadata(name="n", vocab_size=512, n_embed=128)
        d = md.to_dict()
        assert d["name"] == "n"
        assert d["vocab_size"] == 512
        assert d["n_embed"] == 128
        assert isinstance(d["personality"], dict)
        assert isinstance(d["tags"], list)

    def test_from_dict_filters_unknown(self):
        md = ModelMetadata.from_dict(
            {"name": "x", "vocab_size": 64, "unknown_field": 123}
        )
        assert md.name == "x"
        assert md.vocab_size == 64
        assert not hasattr(md, "unknown_field")
        assert md.n_embed == 256

    def test_from_dict_minimal(self):
        md = ModelMetadata.from_dict({})
        assert md.name == "sloughgpt"
        assert md.vocab_size == 256

    def test_from_model_config(self):
        model = FakeModel(config={"vocab_size": 99, "n_embed": 77, "n_layer": 3})
        md = ModelMetadata.from_model(model, name="mine")
        assert md.name == "mine"
        assert md.vocab_size == 99
        assert md.n_embed == 77
        assert md.n_layer == 3
        assert md.created_at.endswith("Z")

    def test_from_model_partial_config(self):
        model = FakeModel(config={"vocab_size": 10})
        md = ModelMetadata.from_model(model)
        assert md.vocab_size == 10
        assert md.n_embed == 256

    def test_from_model_attributes(self):
        model = FakeModel(vocab_size=5, n_embed=33, n_head=2, block_size=8.5, unknown_int=7)
        md = ModelMetadata.from_model(model)
        assert md.vocab_size == 5
        assert md.n_embed == 33
        assert md.n_head == 2
        assert md.block_size == 128
        assert not hasattr(md, "unknown_int")

    def test_from_model_no_config(self):
        md = ModelMetadata.from_model(FakeModel())
        assert md.vocab_size == 256
        assert md.n_embed == 256

    def test_from_model_torch_version(self, fake_torch):
        md = ModelMetadata.from_model(FakeModel())
        assert md.torch_version == "9.9.9"

    def test_from_model_no_torch(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        md = ModelMetadata.from_model(FakeModel())
        assert md.torch_version == ""

    def test_add_training_info_sets_fields(self):
        md = ModelMetadata()
        md.add_training_info(dataset="d", epochs=5, train_loss=0.5, val_loss=0.3, steps=100)
        assert md.training_dataset == "d"
        assert md.epochs_trained == 5
        assert md.final_train_loss == 0.5
        assert md.final_val_loss == 0.3
        assert md.steps_trained == 100
        assert md.last_step == 100
        assert md.best_val_loss == 0.3
        assert md.trained_at.endswith("Z")

    def test_add_training_info_first_best(self):
        md = ModelMetadata()
        md.add_training_info(val_loss=0.5)
        assert md.best_val_loss == 0.5
        md.add_training_info(val_loss=0.2)
        assert md.best_val_loss == 0.2

    def test_add_training_info_worse_loss_keeps_best(self):
        md = ModelMetadata(best_val_loss=0.1)
        md.add_training_info(val_loss=0.5)
        assert md.best_val_loss == 0.1

    def test_add_training_info_zero_val_loss(self):
        md = ModelMetadata()
        md.add_training_info(val_loss=0.0)
        assert md.best_val_loss == 0.0

    def test_add_soul_info(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S", personality={"warmth": 0.9}, soul_hash="h")
        assert md.soul_name == "S"
        assert md.personality == {"warmth": 0.9}
        assert md.soul_hash == "h"

    def test_add_soul_info_no_personality_keeps_default(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S")
        assert md.personality == {}

    def test_add_soul_info_empty_dict_keeps_default(self):
        md = ModelMetadata()
        md.add_soul_info(soul_name="S", personality={})
        assert md.personality == {}

    def test_validate_ok_has_warnings(self):
        md = ModelMetadata(vocab_size=256, n_embed=128, n_layer=4, n_head=4)
        issues = md.validate()
        assert "warning: training_dataset not set" in issues
        assert "warning: epochs_trained is 0" in issues
        assert "warning: lineage not set" in issues

    def test_validate_positive_architecture(self):
        md = ModelMetadata()
        md.vocab_size = -1
        md.n_embed = 0
        md.n_layer = -2
        md.n_head = 0
        issues = md.validate()
        assert "vocab_size must be positive" in issues
        assert "n_embed must be positive" in issues
        assert "n_layer must be positive" in issues
        assert "n_head must be positive" in issues

    def test_validate_full_ok(self):
        md = ModelMetadata(
            vocab_size=10, n_embed=10, n_layer=2, n_head=2,
            training_dataset="d", epochs_trained=1, lineage="l",
        )
        assert md.validate() == []


class TestCreateModelMetadata:
    def test_minimal(self):
        md = create_model_metadata(FakeModel(), name="nm")
        assert md.name == "nm"
        assert md.exported_at.endswith("Z")

    def test_with_training_and_soul_info(self):
        md = create_model_metadata(
            FakeModel(),
            name="nm",
            training_info={"dataset": "d", "epochs": 3, "train_loss": 0.1, "val_loss": 0.2, "steps": 5},
            soul_info={"soul_name": "S", "personality": {"x": 1}, "soul_hash": "h"},
        )
        assert md.training_dataset == "d"
        assert md.epochs_trained == 3
        assert md.final_val_loss == 0.2
        assert md.best_val_loss == 0.2
        assert md.soul_name == "S"
        assert md.personality == {"x": 1}


class TestConfigs:
    def test_export_config_defaults(self):
        c = ExportConfig()
        assert c.input_path == ""
        assert c.output_path == ""
        assert c.format == "safetensors"
        assert c.quantization is None
        assert c.include_tokenizer is True
        assert c.metadata is None
        assert c.seq_len == 128
        assert c.opset_version == 17
        assert c.n_ctx == 2048

    def test_export_config_custom(self):
        c = ExportConfig(format="gguf_q4_k_m", quantization="Q8_0", seq_len=256, n_ctx=4096)
        assert c.format == "gguf_q4_k_m"
        assert c.quantization == "Q8_0"
        assert c.seq_len == 256
        assert c.n_ctx == 4096

    def test_onnx_options_defaults(self):
        o = ONNXExportOptions()
        assert o.input_names == ["input_ids"]
        assert o.output_names == ["logits"]
        assert o.dynamic_axes is None
        assert o.opset_version == 17
        assert o.optimize is True
        assert o.verbose is False
        assert o.external_data is True
        assert o.dynamo_export is True

    def test_onnx_options_custom(self):
        o = ONNXExportOptions(input_names=["ids"], output_names=["out"], dynamic_axes={}, opset_version=14, optimize=False, verbose=True, external_data=False, dynamo_export=False)
        assert o.input_names == ["ids"]
        assert o.output_names == ["out"]
        assert o.dynamic_axes == {}
        assert o.opset_version == 14
        assert o.optimize is False
        assert o.verbose is True
        assert o.external_data is False
        assert o.dynamo_export is False

    def test_gguf_options_defaults(self):
        g = GGUFExportOptions()
        assert g.model_name == "sloughgpt"
        assert g.model_version == "1.0"
        assert g.quantization == "Q4_K_M"
        assert g.n_ctx == 2048
        assert g.rope_freq_base == 10000.0
        assert g.rope_freq_scale == 1.0
        assert g.use_gpu is False

    def test_gguf_options_custom(self):
        g = GGUFExportOptions(model_name="m", quantization="Q5_K_M", n_ctx=4096, rope_freq_base=50000.0, rope_freq_scale=0.5, use_gpu=True)
        assert g.model_name == "m"
        assert g.quantization == "Q5_K_M"
        assert g.n_ctx == 4096
        assert g.rope_freq_base == 50000.0
        assert g.rope_freq_scale == 0.5
        assert g.use_gpu is True


class TestExportToTorchscript:
    def test_trace_with_example(self, tmp_path, fake_torch):
        model = FakeModel()
        out = str(tmp_path / "m.torchscript.pt")
        r = export_to_torchscript(model, out, example_input=object())
        assert r == out
        assert model.eval_calls == [1]
        assert len(fake_torch.traced) == 1
        assert (tmp_path / "m.torchscript.pt").read_bytes() == b"ts"

    def test_script_without_example(self, tmp_path, fake_torch):
        model = FakeModel()
        out = str(tmp_path / "m.torchscript.pt")
        r = export_to_torchscript(model, out)
        assert r == out
        assert len(fake_torch.scripted) == 1
        assert (tmp_path / "m.torchscript.pt").read_bytes() == b"ts2"


class TestExportToOnnx:
    def test_success(self, monkeypatch, fake_torch):
        rec = {}

        def stub(model, output_path, example_input=None, config=None, seq_len=128):
            rec["model"] = model
            rec["output_path"] = output_path
            rec["example_input"] = example_input
            rec["config"] = config
            rec["seq_len"] = seq_len
            return output_path

        monkeypatch.setattr(onnx_mod, "export_sloughgpt_to_onnx", stub)
        model = FakeModel()
        r = export_to_onnx(model, "out.onnx")
        assert r == "out.onnx"
        assert rec["model"] is model
        assert rec["seq_len"] == 128
        assert rec["config"].input_names == ["input_ids"]
        assert rec["config"].output_names == ["logits"]
        assert rec["config"].dynamic_axes == {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
        assert rec["config"].opset_version == 17
        assert fake_torch.onnx_exports == []

    def test_success_custom_names(self, monkeypatch, fake_torch):
        rec = {}

        def stub(model, output_path, example_input=None, config=None, seq_len=128):
            rec["config"] = config
            rec["seq_len"] = seq_len
            return "custom.onnx"

        monkeypatch.setattr(onnx_mod, "export_sloughgpt_to_onnx", stub)
        r = export_to_onnx(
            FakeModel(), "out.onnx",
            input_names=["ids"], output_names=["out"],
            dynamic_axes={"ids": {0: "b"}}, seq_len=256, opset_version=14,
        )
        assert r == "custom.onnx"
        assert rec["config"].input_names == ["ids"]
        assert rec["config"].output_names == ["out"]
        assert rec["config"].dynamic_axes == {"ids": {0: "b"}}
        assert rec["config"].opset_version == 14
        assert rec["seq_len"] == 256

    def test_fallback_no_example(self, monkeypatch, tmp_path, fake_torch):
        def boom(*a, **k):
            raise RuntimeError("advanced failed")

        monkeypatch.setattr(onnx_mod, "export_sloughgpt_to_onnx", boom)
        model = FakeModel()
        out = str(tmp_path / "m.onnx")
        r = export_to_onnx(model, out, seq_len=32)
        assert r == out
        assert model.eval_calls == [1]
        assert fake_torch.zeros_calls == [((1, 32), "torch.long")]
        assert len(fake_torch.onnx_exports) == 1
        args, kwargs = fake_torch.onnx_exports[0]
        assert args[0] is model
        assert isinstance(args[1], _FakeTensor)
        assert args[2] == out
        assert kwargs["input_names"] == ["input_ids"]
        assert kwargs["output_names"] == ["logits"]
        assert kwargs["dynamic_axes"] == {}
        assert kwargs["opset_version"] == 17
        assert kwargs["do_constant_folding"] is True

    def test_fallback_with_example(self, monkeypatch, tmp_path, fake_torch):
        def boom(*a, **k):
            raise RuntimeError("advanced failed")

        monkeypatch.setattr(onnx_mod, "export_sloughgpt_to_onnx", boom)
        example = object()
        out = str(tmp_path / "m.onnx")
        r = export_to_onnx(FakeModel(), out, example_input=example)
        assert r == out
        assert fake_torch.zeros_calls == []
        args, _ = fake_torch.onnx_exports[0]
        assert args[1] is example


class TestExportToSafetensors:
    def test_default_fp32(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(
            state={"w": np.ones((2, 2)), "config": {"a": 1}},
            vocab_size=10, n_embed=8, n_layer=2, n_head=2, block_size=4,
        )
        out = str(tmp_path / "m.safetensors")
        r = export_to_safetensors(model, out)
        assert r == out
        assert (tmp_path / "m.safetensors").exists()
        assert (tmp_path / "m.meta.json").exists()
        sd, fname, meta = fake_safetensors[0]
        assert "w" in sd
        assert "config" not in sd
        assert isinstance(sd["w"], _FakeTensor)
        assert meta["format"] == "safetensors"
        assert meta["format_version"] == "1.0"
        assert meta["precision"] == "fp32"
        assert "vocab_size" in meta
        assert "n_embed" in meta
        assert "n_layer" in meta
        assert "n_head" in meta
        assert "block_size" in meta
        assert "exported_at" in meta
        raw = json.loads((tmp_path / "m.meta.json").read_text())
        assert raw["precision"] == "fp32"

    def test_bf16(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((2, 2))})
        out = str(tmp_path / "m.safetensors")
        r = export_to_safetensors(model, out, dtype="bf16")
        assert r == out
        _, _, meta = fake_safetensors[0]
        assert meta["precision"] == "bf16"
        assert fake_safetensors[0][0]["w"].cast_to == "bfloat16"

    def test_fp16(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((2, 2))})
        r = export_to_safetensors(model, out := str(tmp_path / "m.safetensors"), dtype="fp16")
        assert r == out
        _, _, meta = fake_safetensors[0]
        assert meta["precision"] == "fp16"

    def test_metadata_modelmetadata(self, tmp_path, fake_torch, fake_safetensors):
        md = ModelMetadata(name="meta-name", vocab_size=777)
        model = FakeModel(state={"w": np.ones((1,))})
        export_to_safetensors(model, str(tmp_path / "m.safetensors"), metadata=md)
        _, _, meta = fake_safetensors[0]
        assert meta["name"] == "meta-name"
        assert meta["vocab_size"] == "777"
        assert meta["precision"] == "fp32"
        assert meta["export_format"] == "safetensors"
        assert "exported_at" in meta

    def test_metadata_dict_prefers_existing(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(
            config={"n_embed": 5},
            vocab_size=10, n_layer=3,
        )
        export_to_safetensors(
            model, str(tmp_path / "m.safetensors"),
            metadata={"vocab_size": 999, "custom": "x"},
        )
        _, _, meta = fake_safetensors[0]
        assert meta["vocab_size"] == "999"
        assert meta["custom"] == "x"
        assert meta["n_embed"] == "5"
        assert meta["n_layer"] == "3"

    def test_metadata_dict_extracts_config(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(config={"vocab_size": 42, "block_size": 8})
        export_to_safetensors(model, str(tmp_path / "m.safetensors"))
        _, _, meta = fake_safetensors[0]
        assert meta["vocab_size"] == "42"
        assert meta["block_size"] == "8"

    def test_to_str_all_types(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))})
        export_to_safetensors(
            model, str(tmp_path / "m.safetensors"),
            metadata={
                "s": "str",
                "i": 7,
                "f": 1.5,
                "b": True,
                "lst": [1, 2],
                "tup": (3, 4),
                "dct": {"k": "v"},
                "none": None,
            },
        )
        _, _, meta = fake_safetensors[0]
        assert meta["s"] == "str"
        assert meta["i"] == "7"
        assert meta["f"] == "1.5"
        assert meta["b"] == "True"
        assert meta["lst"] == "[1, 2]"
        assert meta["tup"] == "(3, 4)"
        assert meta["dct"] == '{"k": "v"}'
        assert meta["none"] == "None"

    def test_parent_dir_creation(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))})
        out = str(tmp_path / "nested" / "dir" / "m.safetensors")
        r = export_to_safetensors(model, out)
        assert r == out
        assert Path(out).exists()
        assert Path(out.replace(".safetensors", ".meta.json")).exists()


class TestExportToSafetensorsBf16:
    def test_wrapper(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))})
        out = str(tmp_path / "m.safetensors")
        r = export_to_safetensors_bf16(model, out, metadata={"k": "v"})
        assert r == out
        _, _, meta = fake_safetensors[0]
        assert meta["precision"] == "bf16"
        assert meta["k"] == "v"


class TestGGUFWrappers:
    def test_export_to_gguf(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None, config=None):
            rec["model"] = model
            rec["output_path"] = output_path
            rec["tokenizer"] = tokenizer
            rec["config"] = config
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf", stub)
        tok = object()
        r = export_to_gguf(object(), "out.gguf", "Q8_0", tok)
        assert r == "out.gguf"
        assert rec["tokenizer"] is tok
        assert rec["config"].quantization == "Q8_0"

    def test_export_to_gguf_default_quant(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None, config=None):
            rec["config"] = config
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf", stub)
        export_to_gguf(object(), "out.gguf")
        assert rec["config"].quantization == "Q4_K_M"

    def test_export_to_gguf_fp16(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None):
            rec["tokenizer"] = tokenizer
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf_fp16", stub)
        tok = object()
        r = export_to_gguf_fp16(object(), "out.gguf", tok)
        assert r == "out.gguf"
        assert rec["tokenizer"] is tok

    def test_export_to_gguf_q4_k_m(self, monkeypatch):
        rec = {}

        def stub(model, output_path, tokenizer=None):
            rec["tokenizer"] = tokenizer
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf_q4_k_m", stub)
        r = export_to_gguf_q4_k_m(object(), "out.gguf")
        assert r == "out.gguf"


class TestExportToTorch:
    def test_default_metadata(self, tmp_path, fake_torch):
        model = FakeModel(state={"w": np.ones((1,))})
        out = str(tmp_path / "m.pt")
        r = export_to_torch(model, out)
        assert r == out
        checkpoint, path = fake_torch.saved[0]
        assert path == out
        assert checkpoint["metadata"] == {"format": "torch"}
        assert "w" in checkpoint["model_state_dict"]

    def test_with_metadata(self, tmp_path, fake_torch):
        model = FakeModel(state={"w": np.ones((1,))})
        out = str(tmp_path / "m.pt")
        r = export_to_torch(model, out, metadata={"epochs": 5})
        assert r == out
        checkpoint, _ = fake_torch.saved[0]
        assert checkpoint["metadata"] == {"epochs": 5}


class TestExportToSou:
    def test_real_roundtrip(self, tmp_path):
        model = FakeModel(state={"w": np.array([1.0, 2.0, 3.0]), "b": np.array([0.5])})
        out = str(tmp_path / "m.soul")
        r = export_to_sou(model, out)
        assert r == out
        assert Path(out).exists()
        assert Path(out + ".meta.json").exists()
        meta = json.loads(Path(out + ".meta.json").read_text())
        assert meta["name"] == "m"

    def test_weights_only(self, tmp_path):
        model = FakeModel(state={"w": np.ones((2, 2))})
        out = str(tmp_path / "m.soul")
        r = export_to_sou(model, out, weights_only=True)
        assert r == out
        assert Path(out).exists()

    def test_with_soul_profile(self, tmp_path):
        from domains.inference import create_soul_profile

        model = FakeModel(state={"w": np.ones((1,))})
        profile = create_soul_profile(name="profile-name", lineage="base")
        out = str(tmp_path / "m.soul")
        export_to_sou(model, out, soul_profile=profile)
        meta = json.loads(Path(out + ".meta.json").read_text())
        assert meta["name"] == "profile-name"
        assert meta["lineage"] == "base"


class TestHelpers:
    def test_replace_ext(self):
        assert _replace_ext("model.pt", ".safetensors") == "model.safetensors"
        assert _replace_ext("path/to/model.tar.gz", ".onnx") == "path/to/model.tar.onnx"
        assert _replace_ext("model", ".soul") == "model.soul"
        assert _replace_ext("dir/model.bin", "-Q4_K_M.gguf") == "dir/model-Q4_K_M.gguf"

    def test_gguf_path(self):
        assert _gguf_path("model.pt", None) == "model-Q4_K_M.gguf"
        assert _gguf_path("model", "Q8_0") == "model-Q8_0.gguf"
        assert _gguf_path("dir/model.pt", "f16") == "dir/model-F16.gguf"


class TestExportAllFormats:
    def test_full(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))}, vocab_size=5)
        config = _cfg("all", output=str(tmp_path / "model"))
        results = {}
        export_all_formats(config, model, tokenizer=None, example_input=object(), results=results)
        assert "safetensors" in results
        assert "torch" in results
        assert "onnx" in results
        assert "sou" in results
        assert "gguf_q4_k_m" not in results
        assert Path(results["safetensors"]).exists()
        assert Path(results["torch"]).exists()
        assert Path(results["sou"]).exists()

    def test_gguf_success(self, tmp_path, fake_torch, fake_safetensors, monkeypatch):
        def stub_gguf(model, output_path, tokenizer=None, config=None):
            Path(output_path).write_bytes(b"gguf")
            return output_path

        monkeypatch.setattr("domains.training.gguf_export.export_to_gguf", stub_gguf)
        model = FakeModel(state={"w": np.ones((1,))})
        config = _cfg("all", output=str(tmp_path / "model"))
        results = {}
        export_all_formats(config, model, tokenizer=None, example_input=None, results=results)
        assert results["gguf_q4_k_m"].endswith("-Q4_K_M.gguf")
        assert Path(results["gguf_q4_k_m"]).exists()
        assert "safetensors" in results

    def test_no_example_skips_onnx(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))})
        config = _cfg("all", output=str(tmp_path / "model"))
        results = {}
        export_all_formats(config, model, tokenizer=None, example_input=None, results=results)
        assert "onnx" not in results
        assert "safetensors" in results

    def test_with_dict_metadata_soul(self, tmp_path, fake_torch, fake_safetensors):
        model = FakeModel(state={"w": np.ones((1,))})
        config = _cfg(
            "all",
            output=str(tmp_path / "model"),
            metadata={"name": "named-soul", "training_dataset": "d", "epochs_trained": 2, "lineage": "chain"},
        )
        results = {}
        export_all_formats(config, model, tokenizer=None, example_input=object(), results=results)
        meta = json.loads(Path(results["sou"] + ".meta.json").read_text())
        assert meta["name"] == "named-soul"
        assert meta["training_dataset"] == "d"
        assert meta["epochs_trained"] == 2
        assert meta["lineage"] == "chain"


class TestExportModel:
    def test_safetensors(self, stub_exports):
        r = export_model(_cfg("safetensors"), model=object())
        assert r == {"safetensors": "models/out.safetensors"}
        calls = stub_exports["export_to_safetensors"].calls
        assert len(calls) == 1
        assert calls[0][0][2] is None

    def test_safetensors_bf16(self, stub_exports):
        r = export_model(_cfg("safetensors_bf16"), model=object())
        assert r == {"safetensors_bf16": "models/out-bf16.safetensors"}

    def test_gguf_default(self, stub_exports):
        r = export_model(_cfg("gguf"), model=object())
        assert r == {"gguf": "models/out-Q4_K_M.gguf"}
        calls = stub_exports["export_to_gguf"].calls
        assert calls[0][0][2] == "Q4_K_M"

    def test_gguf_custom_quant(self, stub_exports):
        r = export_model(_cfg("gguf", quantization="Q5_K_M"), model=object())
        assert r == {"gguf": "models/out-Q5_K_M.gguf"}
        assert stub_exports["export_to_gguf"].calls[0][0][2] == "Q5_K_M"

    def test_gguf_q4_k_m(self, stub_exports):
        r = export_model(_cfg("gguf_q4_k_m"), model=object())
        assert r == {"gguf_q4_k_m": "models/out-Q4_K_M.gguf"}

    def test_gguf_fp16(self, stub_exports):
        r = export_model(_cfg("gguf_fp16"), model=object())
        assert r == {"gguf_fp16": "models/out-F16.gguf"}
        assert len(stub_exports["export_to_gguf_fp16"].calls) == 1

    def test_gguf_q8_0(self, stub_exports):
        r = export_model(_cfg("gguf_q8_0"), model=object())
        assert r == {"gguf_q8_0": "models/out-Q8_0.gguf"}

    def test_gguf_q5_k_m(self, stub_exports):
        r = export_model(_cfg("gguf_q5_k_m"), model=object())
        assert r == {"gguf_q5_k_m": "models/out-Q5_K_M.gguf"}

    def test_gguf_f16(self, stub_exports):
        r = export_model(_cfg("gguf_f16"), model=object())
        assert r == {"gguf_f16": "models/out-F16.gguf"}

    def test_gguf_f32(self, stub_exports):
        r = export_model(_cfg("gguf_f32"), model=object())
        assert r == {"gguf_f32": "models/out-F32.gguf"}

    def test_torch(self, stub_exports):
        r = export_model(_cfg("torch"), model=object())
        assert r == {"torch": "models/out.pt"}

    def test_pytorch_alias(self, stub_exports):
        r = export_model(_cfg("pytorch"), model=object())
        assert r == {"torch": "models/out.pt"}

    def test_torchscript_with_example(self, stub_exports):
        example = object()
        r = export_model(_cfg("torchscript"), model=object(), example_input=example)
        assert r == {"torchscript": "models/out.torchscript.pt"}
        calls = stub_exports["export_to_torchscript"].calls
        assert calls[0][0][2] is example

    def test_torchscript_without_example(self, stub_exports, caplog):
        r = export_model(_cfg("torchscript"), model=object())
        assert r == {}
        assert stub_exports["export_to_torchscript"].calls == []
        assert "example_input required for TorchScript export" in caplog.text

    def test_onnx(self, stub_exports):
        r = export_model(_cfg("onnx", seq_len=64, opset_version=14), model=object(), example_input=object())
        assert r == {"onnx": "models/out.onnx"}
        calls = stub_exports["export_to_onnx"].calls
        assert calls[0][1]["seq_len"] == 64
        assert calls[0][1]["opset_version"] == 14

    def test_sou_no_metadata(self, stub_exports):
        r = export_model(_cfg("sou"), model=object())
        assert r == {"sou": "models/out.soul"}
        calls = stub_exports["export_to_sou"].calls
        profile = calls[0][1]["soul_profile"]
        assert profile.name == "out"
        assert profile.lineage == "sloughgpt"

    def test_sou_with_metadata_lineage(self, stub_exports):
        metadata = {"name": "my-soul", "training_dataset": "d", "epochs_trained": 4, "final_train_loss": 0.1, "final_val_loss": 0.2, "lineage": "chain"}
        r = export_model(_cfg("sou", metadata=metadata), model=object())
        assert r == {"sou": "models/out.soul"}
        profile = stub_exports["export_to_sou"].calls[0][1]["soul_profile"]
        assert profile.name == "my-soul"
        assert profile.training_dataset == "d"
        assert profile.epochs_trained == 4
        assert profile.final_train_loss == 0.1
        assert profile.final_val_loss == 0.2
        assert profile.lineage == "chain"

    def test_all(self, stub_exports):
        r = export_model(_cfg("all"), model=object(), example_input=object(), tokenizer=None)
        assert r == {}
        assert len(stub_exports["export_all_formats"].calls) == 1

    def test_error_logged_and_continues(self, stub_exports, monkeypatch, caplog):
        def boom(*a, **k):
            raise ValueError("bad weights")

        monkeypatch.setattr(export_mod, "export_to_safetensors", boom)
        r = export_model(_cfg("safetensors,torch"), model=object())
        assert "safetensors" not in r
        assert r == {"torch": "models/out.pt"}
        assert "Export failed for format 'safetensors'" in caplog.text
        assert "bad weights" in caplog.text

    def test_comma_separated(self, stub_exports):
        r = export_model(_cfg("torch,gguf_f16"), model=object())
        assert r == {"torch": "models/out.pt", "gguf_f16": "models/out-F16.gguf"}

    def test_uppercase_stripped(self, stub_exports):
        r = export_model(_cfg("  TORCH  "), model=object())
        assert r == {"torch": "models/out.pt"}

    def test_tokenizer_saved(self, stub_exports):
        tok = _FakeTokenizer()
        r = export_model(_cfg("torch", output="models/named"), model=object(), tokenizer=tok)
        assert r["tokenizer"] == "models/tokenizer"
        assert tok.saved_to == ["models/tokenizer"]

    def test_tokenizer_not_saved_when_disabled(self, stub_exports):
        tok = _FakeTokenizer()
        r = export_model(_cfg("torch", tokenizer=False), model=object(), tokenizer=tok)
        assert "tokenizer" not in r
        assert tok.saved_to == []


class TestListExportFormats:
    def test_returns_formats(self):
        fmts = list_export_formats()
        assert isinstance(fmts, dict)
        for key in ["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m", "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0", "torch", "torchscript", "sou", "all"]:
            assert key in fmts
        assert "SafeTensors" in fmts["safetensors"]
