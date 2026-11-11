"""
Tests for SloNet ModelLoader.

Tests the torch-free model loading via .slnc format, including detection,
conversion, verification, and the singleton accessor.
"""

import json
import os
import struct
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from domain.infrastructure._internal.model_loader import (
    LoadResult,
    ModelLoader,
    get_model_loader,
    load_model,
)


def _write_safetensors(path: Path, tensors: dict, metadata: dict = None) -> None:
    """Write a minimal valid .safetensors file.

    Args:
        path: Destination path.
        tensors: Mapping of tensor name -> (dtype_str, np.ndarray).
        metadata: Optional dict stored under the __metadata__ header key.
    """
    header = {}
    offset = 0
    blob = b""
    for name, (dtype_str, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blob += raw
        offset += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


class TestLoadResult:
    """Test LoadResult dataclass."""

    def test_success_result(self):
        result = LoadResult(
            success=True,
            model_id="gpt2",
            model_type="slonet",
            provider=MagicMock(),
            model=MagicMock(),
        )
        assert result.success
        assert result.model_id == "gpt2"
        assert result.model_type == "slonet"
        assert result.error is None
        assert result.metrics == {}

    def test_failure_result(self):
        result = LoadResult(
            success=False,
            model_id="missing",
            model_type="slonet",
            error="Model not found",
        )
        assert not result.success
        assert result.error == "Model not found"
        assert result.provider is None

    def test_metrics_field(self):
        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            metrics={"quantized": True, "quant_bits": 8},
        )
        assert result.metrics["quantized"] is True
        assert result.metrics["quant_bits"] == 8


class TestLoadErrorClassification:
    """``error_kind`` tells retry loops what a failure means (card 20261104_062205).

    ``"permanent"`` = retrying the identical load cannot succeed (no
    artifacts, refused soul lookup) — startup breaks its backoff loop instead
    of burning 5 s + 10 s. ``"transient"`` = worth waiting for (load lock
    busy). ``None`` (the default) keeps every pre-existing caller's retry
    behaviour.
    """

    def test_missing_artifacts_is_permanent(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("nonexistent-model")
        assert not result.success
        assert "No .slnc or .soul file" in result.error
        assert result.error_kind == "permanent"

    def test_load_lock_busy_is_transient(self):
        loader = ModelLoader()
        assert ModelLoader._load_lock.acquire(blocking=False)
        try:
            result = loader.load("whatever")
        finally:
            ModelLoader._load_lock.release()
        assert not result.success
        assert "Another model is currently loading" in result.error
        assert result.error_kind == "transient"

    def test_soul_refusal_is_permanent(self):
        result = ModelLoader()._soul_refusal("some-model", "name", "some-key", [])
        assert not result.success
        assert result.error_kind == "permanent"

    def test_error_kind_defaults_to_none(self):
        result = LoadResult(success=False, model_id="x", model_type="slonet")
        assert result.error_kind is None


class TestModelLoader:
    """Test ModelLoader detection and routing."""

    def test_init_default(self):
        loader = ModelLoader()
        from domain.infrastructure._internal.model_loader import _REPO_ROOT

        assert loader.models_dir == _REPO_ROOT / "models"

    def test_init_custom_dir(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        assert loader.models_dir == tmp_path

    def test_try_load_slnc_returns_none_when_no_slnc(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        result = loader._try_load_slnc("nonexistent", "cpu", False, 8, "symmetric")
        assert result is None

    def test_load_returns_error_when_no_slnc(self, tmp_path):
        """load() returns error result when no .slnc file found."""
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("nonexistent-model")
        assert not result.success
        assert result.model_type == "slonet"
        assert "No .slnc or .soul file" in result.error

    def test_verify_model_passes_for_valid_model(self):
        """_verify_model returns True for a model that forward passes."""
        mock_model = MagicMock()
        mock_model.forward.return_value = np.zeros((1, 10))

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is True
        assert result.metrics["verified"] is True

    def test_verify_model_fails_for_none_output(self):
        """_verify_model returns False when model returns None."""
        mock_model = MagicMock()
        mock_model.forward.return_value = None

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is False
        assert result.success is False
        assert "empty output" in result.error

    def test_verify_model_fails_on_exception(self):
        """_verify_model returns False when model raises."""
        mock_model = MagicMock()
        mock_model.forward.side_effect = RuntimeError("boom")

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is False
        assert result.metrics["verified"] is False


class TestModelLoaderSlncLoad:
    """Test load()/_try_load_slnc() success and failure paths."""

    def test_load_detects_existing_slnc(self, tmp_path, monkeypatch):
        """has_slnc True when model.slnc exists; no tracker.fail on failure."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: tmp_path,
        )
        monkeypatch.setattr(
            "domain.inference._internal.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt")),
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2")
        assert not result.success
        assert "corrupt" in result.error

    def test_load_slnc_check_raises(self, tmp_path, monkeypatch):
        """_get_model_dir raising is tolerated (has_slnc=False)."""

        def _raise(model_id):
            raise OSError("boom")

        monkeypatch.setattr("domain.infrastructure._internal.model_resolver.get_model_dir", _raise)
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2")
        assert not result.success
        assert "No .slnc or .soul file" in result.error

    def test_load_success_path_with_verify(self, tmp_path, monkeypatch):
        """Full load() success: slnc exists, provider mocked, verify runs."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: tmp_path,
        )
        fake_model = MagicMock()
        fake_model.forward.return_value = np.zeros((1, 3))
        fake_provider = MagicMock()
        fake_provider._model = fake_model
        fake_provider._tokenizer = MagicMock()
        monkeypatch.setattr(
            "domain.inference._internal.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: fake_provider,
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2", quantize=True, quant_bits=4)
        assert result.success
        assert result.provider is fake_provider
        assert result.model is fake_model
        assert result.tokenizer is fake_provider._tokenizer
        assert result.metrics["quantized"] is True
        assert result.metrics["quant_bits"] == 4
        assert result.metrics["verified"] is True

    def test_try_load_slnc_returns_none_when_convert_fails(self, tmp_path, monkeypatch):
        """Missing slnc + failed conversion returns None."""
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: tmp_path,
        )
        loader = ModelLoader(models_dir=tmp_path)
        monkeypatch.setattr(loader, "_try_convert_to_slnc", lambda cache_dir, model_id: None)
        assert loader._try_load_slnc("gpt2", "cpu", False, 8, "symmetric") is None

    def test_try_load_slnc_converts_when_missing(self, tmp_path, monkeypatch):
        """Missing slnc + successful conversion loads the new file."""
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: tmp_path,
        )
        converted = tmp_path / "model.slnc"
        fake_provider = MagicMock()
        fake_provider._model = MagicMock()
        fake_provider._tokenizer = MagicMock()
        monkeypatch.setattr(
            "domain.inference._internal.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: fake_provider,
        )
        loader = ModelLoader(models_dir=tmp_path)
        monkeypatch.setattr(loader, "_try_convert_to_slnc", lambda cache_dir, model_id: converted)
        result = loader._try_load_slnc("gpt2", "cpu", True, 8, "symmetric")
        assert result is not None
        assert result.success
        assert result.metrics["slnc_path"] == str(converted)

    def test_try_load_slnc_returns_error_on_provider_exception(self, tmp_path, monkeypatch):
        """from_slnc raising produces an error LoadResult, not None."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: tmp_path,
        )
        monkeypatch.setattr(
            "domain.inference._internal.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt slnc")),
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader._try_load_slnc("gpt2", "cpu", False, 8, "symmetric")
        assert not result.success
        assert "corrupt slnc" in result.error


class TestModelLoaderConversion:
    """Test _try_convert_to_slnc (safetensors -> .slnc)."""

    def test_try_convert_returns_none_when_no_safetensors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_safetensors",
            lambda cache_dir: None,
        )
        loader = ModelLoader()
        assert loader._try_convert_to_slnc(tmp_path, "gpt2") is None

    def test_try_convert_success_all_dtypes(self, tmp_path, monkeypatch):
        """Full conversion walks a real safetensors file with mixed dtypes."""
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(
            st_path,
            {
                "w_f32": ("F32", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
                "w_f16": ("F16", np.array([1.0, 0.5], dtype=np.float16)),
                "w_bf16": ("BF16", np.array([0x3F80, 0x4000], dtype=np.uint16)),
                "w_i64": ("I64", np.array([1, 2], dtype=np.int64)),
                "w_i32": ("I32", np.array([1, 2], dtype=np.int32)),
                "w_other": ("U8", np.array([1, 2], dtype=np.uint32)),
            },
            metadata={"format": "pt"},
        )
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_safetensors",
            lambda cache_dir: st_path,
        )
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.load_model_config",
            lambda model_id: {"model_type": "gpt2"},
        )
        compiled = {}

        def _compile(compiler, config, weights, path):
            compiled["config"] = config
            compiled["weights"] = weights
            compiled["path"] = path

        monkeypatch.setattr(
            "domain.infrastructure._internal.slnc.compiler.SLNCCompiler.compile_from_dict", _compile
        )
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_protector.protect_model",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("protect failed")),
        )
        loader = ModelLoader()
        result = loader._try_convert_to_slnc(tmp_path, "gpt2")
        assert result == tmp_path / "model.slnc"
        assert compiled["config"] == {"model_type": "gpt2"}
        assert compiled["path"] == str(tmp_path / "model.slnc")
        assert compiled["weights"]["w_f32"].dtype == np.float32
        assert compiled["weights"]["w_f16"].dtype == np.float32
        assert compiled["weights"]["w_bf16"].dtype == np.float32
        np.testing.assert_allclose(compiled["weights"]["w_bf16"], [1.0, 2.0], rtol=1e-3)
        assert compiled["weights"]["w_i64"].dtype == np.int64
        assert compiled["weights"]["w_i32"].dtype == np.int32
        assert compiled["weights"]["w_other"].dtype == np.float32

    def test_try_convert_returns_none_on_corrupt_file(self, tmp_path, monkeypatch):
        bad = tmp_path / "model.safetensors"
        bad.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.find_safetensors",
            lambda cache_dir: bad,
        )
        loader = ModelLoader()
        assert loader._try_convert_to_slnc(tmp_path, "gpt2") is None


class TestModelLoaderSingleton:
    """Test the singleton accessor."""

    def test_get_model_loader_returns_same_instance(self):
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        assert loader1 is loader2

    def test_load_model_convenience_function(self):
        """Test the convenience load_model function."""
        with patch.object(ModelLoader, "load") as mock_load:
            mock_load.return_value = LoadResult(
                success=True,
                model_id="test",
                model_type="slonet",
            )
            result = load_model("test", device="cpu")
            assert result.success
            mock_load.assert_called_once_with(
                model_id="test",
                device="cpu",
                quantize=False,
                quant_bits=8,
                quant_mode="symmetric",
                verify=True,
            )


def _write_identity_soul(
    path: Path,
    *,
    base_model: str = "sloughgpt",
    dataset_signature: str = "",
) -> str:
    """Write a weights-less ``.soul`` with a given identity; return its hash."""
    from domain.inference._internal.slo_format import SloProfile, save_soul

    profile = SloProfile(
        name=path.stem,
        base_model=base_model,
        lineage=base_model or "nanogpt",
        dataset_signature=dataset_signature,
    )
    save_soul(None, str(path), soul_profile=profile, weights_only=True)
    return profile.integrity_hash


class TestModelLoaderSoulResolution:
    """A native ``.soul`` is loaded only when ``model_id`` actually names it.

    The previous resolver took the newest-mtime ``.soul`` whatever was asked
    for, so a request for one model could be answered by another and reported
    under the requested id (wrong-artifact class).
    """

    @staticmethod
    def _loader(tmp_path, monkeypatch):
        """Loader rooted in a temp tree + a recording ``from_soul`` fake."""
        cache = tmp_path / "hf-cache"
        cache.mkdir(exist_ok=True)
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_resolver.get_model_dir",
            lambda model_id: cache,
        )
        calls: list[str] = []

        def _from_soul(path, model_id="sloughgpt", **kwargs):
            calls.append(str(path))
            provider = MagicMock()
            provider._model = MagicMock()
            provider._tokenizer = MagicMock()
            return provider

        monkeypatch.setattr(
            "domain.inference._internal.slonet_provider.SloNetChatProvider.from_soul",
            _from_soul,
        )
        return ModelLoader(models_dir=tmp_path / "models"), calls

    @staticmethod
    def _native(tmp_path) -> Path:
        native = tmp_path / "models" / "slonet-native"
        native.mkdir(parents=True)
        return native

    def test_resolves_requested_checkpoint_not_newest_mtime(self, tmp_path, monkeypatch):
        """The named checkpoint loads even when another one is newer."""
        native = self._native(tmp_path)
        requested = native / "run-b.soul"
        _write_identity_soul(requested, dataset_signature="b" * 64)
        newer = native / "run-a.soul"
        _write_identity_soul(newer, dataset_signature="a" * 64)
        os.utime(requested, (time.time() - 3600, time.time() - 3600))
        os.utime(newer, (time.time(), time.time()))

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("run-b", verify=False)

        assert result.success, result.error
        assert result.metrics["soul_path"].endswith("run-b.soul")
        assert result.metrics["matched_by"] == "name"
        assert calls == [str(requested)]

    def test_unrelated_huggingface_id_never_loads_a_soul(self, tmp_path, monkeypatch):
        """A Qwen request must never be answered by a local native checkpoint."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "tinyshakespeare_1.soul")

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("Qwen/Qwen2.5-0.5B-Instruct", verify=False)

        assert not result.success
        assert calls == []
        assert "No .slnc or .soul file found" in result.error

    def test_unknown_plain_name_falls_through_to_slnc(self, tmp_path, monkeypatch):
        """Valid-looking ids that name no soul keep the old .slnc behaviour."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul")

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("gpt2", verify=False)

        assert not result.success
        assert calls == []
        assert "No .slnc or .soul file found for gpt2" in result.error

    def test_integrity_hash_prefix_resolves(self, tmp_path, monkeypatch):
        """``hash:<prefix>`` addresses exactly the matching checkpoint."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul", dataset_signature="a" * 64)
        wanted_hash = _write_identity_soul(native / "run-b.soul", dataset_signature="b" * 64)

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load(f"hash:{wanted_hash[:12]}", verify=False)

        assert result.success, result.error
        assert result.metrics["soul_path"].endswith("run-b.soul")
        assert result.metrics["matched_by"] == "hash"
        assert result.metrics["integrity_hash"] == wanted_hash
        assert len(calls) == 1

    def test_dataset_signature_resolves_equivalent_runs(self, tmp_path, monkeypatch):
        """Same dataset hash ⇒ interchangeable retrains; pick is deterministic."""
        native = self._native(tmp_path)
        signature = "c" * 64
        _write_identity_soul(native / "run-b.soul", dataset_signature=signature)
        _write_identity_soul(native / "run-a.soul", dataset_signature=signature)

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load(f"dataset:{signature}", verify=False)

        assert result.success, result.error
        assert result.metrics["candidate_count"] == 2
        assert result.metrics["soul_path"].endswith("run-a.soul")
        assert len(calls) == 1

    def test_family_alias_refuses_when_dataset_signatures_differ(self, tmp_path, monkeypatch):
        """Two models on different data are not interchangeable — refuse."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul", dataset_signature="a" * 64)
        _write_identity_soul(native / "run-b.soul", dataset_signature="b" * 64)

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("sloughgpt", verify=False)

        assert not result.success
        assert "not interchangeable" in result.error
        assert "run-a" in result.error and "run-b" in result.error
        assert calls == []

    def test_family_alias_refuses_when_dataset_signature_unrecorded(self, tmp_path, monkeypatch):
        """An unrecorded dataset hash cannot prove equivalence — refuse."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "tinyshakespeare_1.soul", dataset_signature="")

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("sloughgpt", verify=False)

        assert not result.success
        assert "dataset=unrecorded" in result.error
        assert calls == []

    def test_family_alias_resolves_when_candidates_are_equivalent(self, tmp_path, monkeypatch):
        """One model, one recorded dataset ⇒ the family id is unambiguous."""
        native = self._native(tmp_path)
        signature = "d" * 64
        _write_identity_soul(native / "run-a.soul", dataset_signature=signature)
        _write_identity_soul(native / "run-b.soul", dataset_signature=signature)

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("sloughgpt", verify=False)

        assert result.success, result.error
        assert result.metrics["matched_by"] == "alias"
        assert result.metrics["candidate_count"] == 2
        assert len(calls) == 1

    def test_soul_name_path_escape_is_not_a_soul_request(self, tmp_path, monkeypatch):
        """A path is not a checkpoint name and must not be resolved."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul")

        loader, calls = self._loader(tmp_path, monkeypatch)
        for request in ("../run-a", "..\\run-a", "dir/run-a"):
            result = loader.load(request, verify=False)
            assert not result.success, request
        assert calls == []

    def test_hash_prefix_matching_several_artifacts_is_refused(self, tmp_path, monkeypatch):
        """An ambiguous hash prefix is not a licence to pick one arbitrarily."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul")
        _write_identity_soul(native / "run-b.soul")

        shared = "abcdef012345"
        identities = {
            "run-a.soul": {"integrity_hash": shared + "aaaa", "base_model": "sloughgpt"},
            "run-b.soul": {"integrity_hash": shared + "bbbb", "base_model": "sloughgpt"},
        }
        monkeypatch.setattr(
            "domain.infrastructure._internal.model_loader._soul_identity",
            lambda path: identities[Path(path).name],
        )

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load(f"hash:{shared}", verify=False)

        assert not result.success
        assert "ambiguous" in result.error
        assert "run-a" in result.error and "run-b" in result.error
        assert calls == []

    def test_named_checkpoint_wins_over_the_family_alias_it_shares(self, tmp_path, monkeypatch):
        """``sloughgpt.soul`` is an explicit artifact, not a family id request."""
        native = self._native(tmp_path)
        _write_identity_soul(native / "run-a.soul", dataset_signature="a" * 64)
        _write_identity_soul(native / "sloughgpt.soul")

        loader, calls = self._loader(tmp_path, monkeypatch)
        result = loader.load("sloughgpt", verify=False)

        assert result.success, result.error
        assert result.metrics["soul_path"].endswith("sloughgpt.soul")
        assert result.metrics["matched_by"] == "name"
        assert len(calls) == 1
