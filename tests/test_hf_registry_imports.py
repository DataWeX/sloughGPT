"""Regression: HuggingFace registry must import and list models (no silent empty registry)."""

from __future__ import annotations


def test_domains_training_huggingface_package_imports():
    from domains.training import huggingface as hf_pkg  # noqa: F401

    assert hf_pkg.HF_MODELS
    assert hf_pkg.MODEL_REGISTRY is hf_pkg.HF_MODELS
    assert hf_pkg.LocalModelLoader is hf_pkg.HuggingFaceLocalLoader


def test_get_available_hf_models_includes_gpt2():
    from domains.training.model_registry import get_available_hf_models

    models = get_available_hf_models()
    ids = {m.id for m in models}
    assert "hf/gpt2" in ids, f"expected hf/gpt2 in {sorted(ids)[:10]}..."
