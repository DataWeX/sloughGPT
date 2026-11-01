"""MetaWeights drift contract — every aligned sampler default stays canonical.

Canonical source: ``domain/feedback/_internal/meta_weights.py::MetaWeights``
(temperature 0.7 / repetition_penalty 1.15 / top_p 0.85 / top_k 40). Phase 2
("these weights should be the weights set by default") aligned every inference
sampler default across the stack to that source BY VALUE. This test turns that
alignment into a gate: edit one literal on any registered site and CI fails
until either the site or MetaWeights is deliberately re-canonicalized.

Two layers:
  1. runtime pin — ``MetaWeights()`` itself must equal the hard literals, so
     the authority cannot drift silently;
  2. static pin — AST scan of every registered class field / function default
     in the source tree. AST (not import) keeps the gate free of engine/fastapi
     import side effects and reaches nested functions.

Methodology for each rule: a registered parameter must exist in AT LEAST one
matching definition (renames remove the site from the contract -> fail loudly),
and EVERY definition carrying it must match. ``"C"`` = compare to
``getattr(MetaWeights(), param)``; any other value = a pinned exception.

Deliberately excluded (by scope decision, phase-2 card eede63cc):
  - kernel identity: ``sample_token`` temperature 1.0 (sampling kernel), the
    SloNet ``generate``/``generate_with_logprobs``/``generate_with_stop``/
    ``generate_batch`` family (1.0 / top_p None / top_k None / rep 1.0 —
    callers always pass explicit values);
  - ``None`` sentinels pinned below (resolve at runtime from canonical
    ModelConfig/GenerationConfig);
  - training code (``train_step``/``train_batch``/``contrastive_loss``/
    lr schedulers), benchmark experiment configs, summary temps 0.3;
  - the persisted user file ``~/.config/sloughgpt/settings.json`` (runtime
    data — reset is a user decision, not code);
  - call-site literals (``engine.generate(..., temperature=0.7)`` args) —
    verified end-to-end by the live HTTP probe on the phase-2 card;
  - different-meaning knobs: ``search_hd_memory(top_k=5)`` (memory width),
    ``MultimodalEngine.generate`` (temperature 1.0, top_k 0 — raw multimodal
    sampling, untouched by phase 2).
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

import pytest

from domain.feedback import MetaWeights

ROOT = Path(__file__).resolve().parents[1]

# class field rules: file -> [(class_name, {attr: spec})]
CLASS_RULES: dict[str, list[tuple[str, dict[str, object]]]] = {
    "apps/api/server/config.py": [
        (
            "GenerationConfig",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
    "apps/api/server/routers/chat.py": [
        ("ChatRequest", {"temperature": "C"}),
    ],
    "apps/api/server/routers/meta_weights.py": [
        (
            "MetaWeightResponse",
            {"temperature": "C", "repetition_penalty": "C", "top_p": "C", "top_k": "C"},
        ),
    ],
    "apps/api/server/routers/multimodal.py": [
        ("VideoInferRequest", {"temperature": "C"}),
    ],
    "apps/api/server/routers/souls.py": [
        ("SloChatRequest", {"temperature": "C", "top_p": "C"}),
    ],
    "domain/chat/_internal/domain.py": [
        ("ChatRequest", {"temperature": "C"}),
    ],
    "domain/core/_internal/soul.py": [
        (
            "GenerationContext",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
    "domain/feedback/_internal/meta_weights.py": [
        (
            "MetaWeights",
            {"temperature": "C", "repetition_penalty": "C", "top_p": "C", "top_k": "C"},
        ),
    ],
    "domain/inference/_internal/slo_format.py": [
        ("GenerationParams", {"temperature": "C", "top_p": "C", "top_k": "C"}),
    ],
    "domain/infrastructure/_internal/config.py": [
        (
            "ModelConfig",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
    "domain/settings/_internal/persistent.py": [
        (
            "GenerationSettings",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
}

# function rules: file -> [(func_name, {param: spec})]; matched against EVERY
# definition of that name in the file (class methods, module fns, closures).
FUNC_RULES: dict[str, list[tuple[str, dict[str, object]]]] = {
    "domain/chat/_internal/domain.py": [
        ("respond", {"temperature": "C"}),
    ],
    "domain/chat/_internal/manager.py": [
        ("respond", {"temperature": "C"}),
        ("stream", {"temperature": "C"}),
    ],
    "domain/core/_internal/soul.py": [
        # None = resolve from GenerationContext (canonical fields) at runtime.
        (
            "generate",
            {"temperature": None, "top_p": None, "top_k": None, "repetition_penalty": "C"},
        ),
        ("generate_with_topk", {"temperature": "C", "top_p": "C", "top_k": "C"}),
    ],
    "domain/inference/_internal/api_provider.py": [
        ("chat", {"temperature": "C", "top_p": "C"}),
        ("chat_stream", {"temperature": "C", "top_p": "C"}),
    ],
    "domain/inference/_internal/native/engine.py": [
        # temperature 1.0 = sampling-kernel identity, deliberately not aligned.
        ("sample_token", {"temperature": 1.0, "top_p": "C", "top_k": "C"}),
        ("generate", {"temperature": "C", "top_p": "C", "top_k": "C"}),
        ("generate_stream", {"temperature": "C", "top_p": "C", "top_k": "C"}),
        ("chat", {"temperature": "C"}),
        ("chat_stream", {"temperature": "C", "top_p": "C", "top_k": "C"}),
    ],
    "domain/inference/_internal/slonet_provider.py": [
        ("chat", {"temperature": "C"}),
        ("chat_stream", {"temperature": "C"}),
        # top_p/top_k None = resolve from canonical ModelConfig at runtime.
        (
            "_generate_sync",
            {"temperature": "C", "repetition_penalty": "C", "top_p": None, "top_k": None},
        ),
        # kernel-identity family: callers always pass explicit values.
        ("generate", {"temperature": 1.0, "top_p": None, "top_k": None, "repetition_penalty": 1.0}),
        (
            "generate_with_logprobs",
            {"temperature": 1.0, "top_p": None, "top_k": None, "repetition_penalty": 1.0},
        ),
        (
            "generate_with_stop",
            {"temperature": 1.0, "top_p": None, "top_k": None, "repetition_penalty": 1.0},
        ),
        (
            "generate_batch",
            {"temperature": 1.0, "top_p": None, "top_k": None, "repetition_penalty": 1.0},
        ),
    ],
    "domain/infrastructure/_internal/inference_client.py": [
        ("chat", {"temperature": "C"}),
        ("chat_stream", {"temperature": "C"}),
    ],
    "domain/infrastructure/_internal/model_worker.py": [
        ("_generate", {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"}),
        ("_stream", {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"}),
        ("generate", {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"}),
        (
            "generate_stream",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
    "domain/infrastructure/_internal/numpy_engine.py": [
        ("generate", {"temperature": "C", "top_k": "C"}),
        ("generate_stream", {"temperature": "C", "top_k": "C"}),
    ],
    "domain/infrastructure/_internal/slonet_server.py": [
        (
            "_generate_sync",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
        (
            "_generate_stream_sync",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
        ("generate", {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"}),
        (
            "generate_stream",
            {"temperature": "C", "top_p": "C", "top_k": "C", "repetition_penalty": "C"},
        ),
    ],
    "domain/multimodal/_internal/engine.py": [
        ("chat", {"temperature": "C"}),
        ("chat_stream", {"temperature": "C"}),
        ("generate_vqa", {"temperature": "C"}),
    ],
    "domain/models/_internal/provider/slo_transformer.py": [
        ("chat", {"temperature": "C"}),
        ("chat_stream", {"temperature": "C"}),
    ],
}


def _expected(spec: object, param: str, mw: MetaWeights) -> object:
    return getattr(mw, param) if spec == "C" else spec


@cache
def _tree(rel: str) -> ast.Module:
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _field_literal(node: ast.AST, where: str) -> object:
    """Resolve ``0.7`` / ``Field(0.7)`` / ``Field(default=0.7, ...)`` nodes."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call):
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                return arg.value
        for kw in node.keywords:
            if kw.arg in ("default", "default_factory") and isinstance(kw.value, ast.Constant):
                return kw.value.value
        raise AssertionError(f"{where}: cannot resolve Field() literal — update the contract")
    raise AssertionError(
        f"{where}: unsupported default node {ast.dump(node)[:60]} — update the contract"
    )


def _sig_defaults(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr | None]:
    """param name -> default expr (None = no default)."""
    out: dict[str, ast.expr | None] = {}
    pos = list(fn.args.posonlyargs) + list(fn.args.args)
    ndef = len(fn.args.defaults)
    for i, a in enumerate(pos):
        out[a.arg] = fn.args.defaults[i - (len(pos) - ndef)] if i >= len(pos) - ndef else None
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        out[a.arg] = d
    return out


def test_meta_weights_is_canonical() -> None:
    """Layer 1: the authority itself equals the hard literals."""
    mw = MetaWeights()
    assert (mw.temperature, mw.top_p, mw.top_k, mw.repetition_penalty) == (0.7, 0.85, 40, 1.15)


@pytest.mark.parametrize(
    ("rel", "cls", "attrs"),
    [(f, c, a) for f, rules in CLASS_RULES.items() for c, a in rules],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_class_field_defaults(rel: str, cls: str, attrs: dict[str, object]) -> None:
    """Layer 2 (fields): registered class fields exist and match the contract."""
    mw = MetaWeights()
    classes = [n for n in ast.walk(_tree(rel)) if isinstance(n, ast.ClassDef) and n.name == cls]
    assert classes, (
        f"{rel}::{cls} missing — the contract site was renamed/removed, update {__file__}"
    )
    fields: dict[str, ast.AST] = {
        node.target.id: node.value
        for node in classes[0].body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    }
    for attr, spec in attrs.items():
        assert attr in fields, f"{rel}::{cls}.{attr} missing — update the contract"
        want = _expected(spec, attr, mw)
        got = _field_literal(fields[attr], f"{rel}::{cls}.{attr}")
        assert got == want, (
            f"{rel}::{cls}.{attr} = {got!r}, expected {want!r} "
            f"({'MetaWeights canonical' if spec == 'C' else 'pinned exception'})"
        )


@pytest.mark.parametrize(
    ("rel", "fn", "params"),
    [(f, n, p) for f, rules in FUNC_RULES.items() for n, p in rules],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_function_defaults(rel: str, fn: str, params: dict[str, object]) -> None:
    """Layer 2 (signatures): registered params exist and match the contract."""
    mw = MetaWeights()
    fns = [
        n
        for n in ast.walk(_tree(rel))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn
    ]
    assert fns, f"{rel}::{fn} missing — the contract site was renamed/removed, update {__file__}"
    sigs = [_sig_defaults(n) for n in fns]
    for param, spec in params.items():
        carriers = [s for s in sigs if param in s]
        assert carriers, f"{rel}::{fn}({param}) missing — update the contract"
        want = _expected(spec, param, mw)
        for i, sig in enumerate(carriers):
            default = sig[param]
            got = (
                _field_literal(default, f"{rel}::{fn}[{i}]({param})")
                if default is not None
                else "<no default>"
            )
            assert got == want, (
                f"{rel}::{fn}[{i}]({param}) = {got!r}, expected {want!r} "
                f"({'MetaWeights canonical' if spec == 'C' else 'pinned exception'})"
            )
