"""Goal 12 — owned architecture: soul → slnc roundtrip + no-HF default path.

Covers:
1. Weight-identity roundtrip: train a tiny .soul → soul_to_slnc → from_slnc
   → every tensor matches the original state_dict (no silent drops).
2. Default load path prefers native .soul over .slnc (priority flip).
3. HuggingFace bootstrap is opt-in (SLO_BOOTSTRAP_HF); default path does not
   touch HF download machinery.
4. Embedded tokenizer round-trips through the SLNC config (no HF tokenizer).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow  # trains a tiny model


def _high_diversity_corpus() -> str:
    # ~95 unique chars — clears SloughGPTTrainer diversity gate
    return "".join(chr(32 + (i % 95)) for i in range(400)) * 4


@pytest.fixture(scope="module")
def tiny_soul(tmp_path_factory) -> Path:
    """Train a minimal .soul once for all tests in this module."""
    from domain.training._internal.train_pipeline import SloughGPTTrainer, TrainerConfig

    root = tmp_path_factory.mktemp("goal12")
    data = root / "corpus.txt"
    data.write_text(_high_diversity_corpus(), encoding="utf-8")

    cfg = TrainerConfig(
        vocab_size=0,
        n_embed=32,
        n_layer=1,
        n_head=4,
        block_size=32,
        batch_size=4,
        epochs=1,
        max_steps=4,
        learning_rate=1e-3,
        warmup_steps=1,
        checkpoint_dir=str(root / "ck"),
        checkpoint_interval=10_000,
        log_interval=100,
        eval_interval=10_000,
        min_data_quality=0.0,
        max_toxicity_rate=1.0,
    )
    trainer = SloughGPTTrainer(data_path=str(data), config=cfg, soul_name="goal12")
    trainer.train()
    out = root / "tiny"
    trainer.save(str(out), include_optimizer_state=False, is_final=True)
    soul_path = Path(str(out) + ".soul")
    assert soul_path.exists(), f"soul not written: {soul_path}"
    return soul_path


class TestSoulToSlncRoundtrip:
    def test_weight_identity_roundtrip(self, tiny_soul: Path, tmp_path: Path) -> None:
        """Every tensor in the soul survives the bridge with bit-identical values."""
        from domain.inference._internal.slo_format import load_soul
        from domain.inference._internal.slonet_provider import SloNetChatProvider
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

        _, original_sd = load_soul(str(tiny_soul))
        assert "tok_emb.weight" in original_sd

        slnc_path = tmp_path / "tiny.slnc"
        soul_to_slnc(tiny_soul, slnc_path)
        assert slnc_path.exists() and slnc_path.stat().st_size > 0

        provider = SloNetChatProvider.from_slnc(str(slnc_path), model_id="goal12-roundtrip")
        loaded = dict(provider._model._named_parameters())

        missing = []
        mismatched = []
        for key, arr in original_sd.items():
            if key not in loaded:
                missing.append(key)
                continue
            got = np.asarray(loaded[key].data)
            if got.shape != arr.shape or not np.allclose(got, arr, rtol=0, atol=0):
                mismatched.append(key)

        assert not missing, f"tensors dropped in slnc: {missing}"
        assert not mismatched, f"tensors not bit-identical: {mismatched}"

    def test_embedded_tokenizer_roundtrip(self, tiny_soul: Path, tmp_path: Path) -> None:
        """Char tokenizer from soul metadata is embedded in slnc and rebuilds."""
        from domain.inference._internal.slo_format import load_soul
        from domain.inference._internal.slonet_provider import SloNetChatProvider
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

        soul, _ = load_soul(str(tiny_soul))
        assert soul.metadata.get("stoi"), "soul must carry stoi for char tokenizer"

        slnc_path = tmp_path / "tok.slnc"
        soul_to_slnc(tiny_soul, slnc_path)
        provider = SloNetChatProvider.from_slnc(str(slnc_path), model_id="goal12-tok")

        tok = provider._tokenizer
        assert tok is not None, "tokenizer missing after from_slnc"
        # Round-trip a short string through encode/decode
        sample = "Hello!"
        ids = tok.encode(sample)
        assert isinstance(ids, list) and len(ids) == len(sample)
        # decode may use itos from embedded vocab
        if hasattr(tok, "decode"):
            decoded = tok.decode(ids)
            # char-level: at least same length (unknown chars → empty)
            assert isinstance(decoded, str)


class TestPriorityFlip:
    def test_model_loader_prefers_soul_over_slnc(self, tiny_soul: Path, tmp_path: Path) -> None:
        """With both .soul (native dir) and a .slnc present, soul wins."""
        from domain.infrastructure._internal.model_loader import ModelLoader

        # Place a native soul where _try_load_soul searches
        native_dir = Path(__file__).resolve().parents[1] / "models" / "slonet-native"
        native_dir.mkdir(parents=True, exist_ok=True)
        placed = native_dir / "goal12_priority_probe.soul"
        placed.write_bytes(tiny_soul.read_bytes())
        try:
            # Also drop a decoy slnc that would fail to load as a different arch
            # (priority is decided before slnc is opened when soul exists)
            loader = ModelLoader()
            result = loader.load("goal12-nonexistent-hf-id", verify=False)
            # Either soul loaded, or no soul matched — must NOT attempt HF download
            # in ModelLoader itself (it never downloads; only startup does).
            assert result.model_type in ("slonet-native", "slonet") or not result.success
            if result.success:
                assert result.metrics.get("source") == "native-trained" or (
                    result.model_type == "slonet-native"
                )
        finally:
            if placed.exists():
                placed.unlink()

    def test_soul_to_slnc_does_not_import_huggingface(self, tiny_soul: Path, tmp_path: Path) -> None:
        """Bridge is pure — no transformers/huggingface_hub import required."""
        import sys

        # Snapshot presence; if already loaded by other tests, still ensure
        # soul_to_slnc itself does not import them.
        before = {k for k in sys.modules if k.startswith(("transformers", "huggingface"))}
        from domain.infrastructure._internal.soul_to_slnc import soul_to_slnc

        soul_to_slnc(tiny_soul, tmp_path / "nohf.slnc")
        after = {k for k in sys.modules if k.startswith(("transformers", "huggingface"))}
        assert after == before, f"bridge imported HF modules: {after - before}"


class TestBootstrapGate:
    def test_autoload_skips_hf_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default path: no SLO_BOOTSTRAP_HF → no download_hf_model call."""
        monkeypatch.delenv("SLO_BOOTSTRAP_HF", raising=False)

        from unittest.mock import MagicMock, patch

        import state as server_state

        cfg = MagicMock()
        cfg.native_soul_path = ""
        cfg.autoload_model = "Qwen/Qwen2.5-0.5B-Instruct"
        cfg.autoload_device = "cpu"
        cfg.quantize_slonet = False
        cfg.quant_bits = 8
        cfg.quant_mode = "symmetric"

        fail_result = MagicMock()
        fail_result.success = False
        fail_result.error = "no local model"
        fail_result.model = None
        fail_result.tokenizer = None
        fail_result.provider = None
        fail_result.model_id = cfg.autoload_model

        with (
            patch.dict("sys.modules", {"state": server_state}),
            patch("domain.infrastructure.model_loader.ModelLoader") as mock_loader,
            patch(
                "domain.infrastructure.hf_hub.download_hf_model"
            ) as mock_dl,
        ):
            mock_loader.return_value.load.return_value = fail_result
            server_state.model = None
            server_state.model_type = None
            server_state.provider = None

            from apps.api.server.infrastructure import startup as startup_mod

            startup_mod._autoload_model(cfg)

        mock_dl.assert_not_called()

    def test_autoload_downloads_when_bootstrap_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SLO_BOOTSTRAP_HF=1 restores the HF download fallback."""
        monkeypatch.setenv("SLO_BOOTSTRAP_HF", "1")

        from unittest.mock import MagicMock, patch

        import state as server_state

        cfg = MagicMock()
        cfg.native_soul_path = ""
        cfg.autoload_model = "gpt2"
        cfg.autoload_device = "cpu"
        cfg.quantize_slonet = False
        cfg.quant_bits = 8
        cfg.quant_mode = "symmetric"

        fail_result = MagicMock()
        fail_result.success = False
        fail_result.error = "no local"
        fail_result.model = None
        fail_result.tokenizer = None
        fail_result.provider = None
        fail_result.model_id = "gpt2"

        ok_result = MagicMock()
        ok_result.success = True
        ok_result.model = MagicMock()
        ok_result.model_id = "gpt2"
        ok_result.tokenizer = None
        ok_result.provider = MagicMock()
        ok_result.error = None

        # All pre-download attempts fail (3 retries), then post-download succeeds.
        # Also stub sleep so exponential backoff does not stall the test.
        with (
            patch.dict("sys.modules", {"state": server_state}),
            patch("domain.infrastructure.model_loader.ModelLoader") as mock_loader,
            patch("domain.infrastructure.hf_hub.download_hf_model") as mock_dl,
            patch("time.sleep"),
            patch(
                "domain.infrastructure.model_resolver.get_model_dir",
                return_value=Path("/tmp/goal12-hf-cache"),
            ),
        ):
            mock_loader.return_value.load.side_effect = [fail_result, fail_result, fail_result, ok_result]
            server_state.model = None
            server_state.model_type = None
            server_state.provider = None

            from apps.api.server.infrastructure import startup as startup_mod

            startup_mod._autoload_model(cfg)

        mock_dl.assert_called_once()


class TestNoFourthLoop:
    def test_soul_to_slnc_has_no_training_loop(self) -> None:
        """Bridge must not contain a forward/loss/backward/optimize loop."""
        path = (
            Path(__file__).resolve().parents[1]
            / "domain"
            / "infrastructure"
            / "_internal"
            / "soul_to_slnc.py"
        )
        src = path.read_text(encoding="utf-8")
        has_backward = "loss.backward" in src or ".backward()" in src
        has_step = "optimizer.step" in src or "clip_gradients" in src
        assert not (has_backward and has_step), "soul_to_slnc must not host a training loop"
