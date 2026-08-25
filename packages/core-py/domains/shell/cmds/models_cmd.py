"""Model management commands: models, unload, precision, quantize, dequantize.

Follows the cmds/ protocol:
    def run(argv, out, api, env) -> int
"""
from __future__ import annotations

help = "List, unload, or configure models"
names = ["models", "unload", "precision", "quantize", "dequantize"]

_VALID_PRECISION = {"auto", "fp32", "fp16"}
_VALID_BITS = {"4", "8"}
_VALID_SCHEME = {"symmetric", "asymmetric"}


def _format_error(e: Exception, cmd: str = "") -> str:
    """Format an exception into a user-friendly error message."""
    from domains.shell.error import format_error
    return format_error(e, cmd, color=False)


def run(argv: list[str], out, api, env: dict) -> int:
    cmd = argv[0] if argv else "models"
    args = argv[1:]

    handlers = {
        "models": _models,
        "unload": _unload,
        "precision": _precision,
        "quantize": _quantize,
        "dequantize": _dequantize,
    }

    handler = handlers.get(cmd, _models)
    return handler(args, out, api)


def _models(args, out, api):
    try:
        models = api.models()
    except Exception as e:
        out.write(_format_error(e, "models"))
        return 1
    if not models:
        out.write("No models available.")
        return 0

    for m in models:
        mid = m.get("model_id", m.get("name", "?"))
        mtype = m.get("type", "?")
        size = m.get("size_gb", "?")
        out.write(f"  {mid:20s}  type={mtype}  size={size} GB")

    try:
        health = api._api_get("/health")
        loaded = health.get("data", {}).get("model_type")
        if loaded:
            out.write(f"\n  Loaded: {loaded}")
    except Exception:
        pass  # non-critical; models list already shown

    return 0


def _unload(args, out, api):
    try:
        result = api.unload_model()
    except Exception as e:
        out.write(_format_error(e, "unload"))
        return 1
    status = result.get("status", "error")
    if status == "unloaded":
        out.write("Model unloaded.")
        return 0
    out.write(f"Unload result: {status}")
    return 0


def _precision(args, out, api):
    mode = args[0] if args else "auto"
    if mode not in _VALID_PRECISION:
        out.write(f"Invalid precision: {mode!r}. Must be one of: {', '.join(sorted(_VALID_PRECISION))}")
        return 1
    try:
        result = api.set_precision(mode)
    except Exception as e:
        out.write(_format_error(e, "precision"))
        return 1
    out.write(f"Precision set to: {result.get('mode', mode)}")
    return 0


def _quantize(args, out, api):
    bits = args[0] if args else "4"
    scheme = args[1] if len(args) > 1 else "symmetric"
    if bits not in _VALID_BITS:
        out.write(f"Invalid bits: {bits!r}. Must be one of: {', '.join(sorted(_VALID_BITS))}")
        return 1
    if scheme not in _VALID_SCHEME:
        out.write(f"Invalid scheme: {scheme!r}. Must be one of: {', '.join(sorted(_VALID_SCHEME))}")
        return 1
    try:
        result = api.quantize_model(bits=bits, scheme=scheme)
    except Exception as e:
        out.write(_format_error(e, "quantize"))
        return 1
    status = result.get("status", "error")
    if status == "ok":
        out.write(f"Quantized to {bits}-bit ({scheme}).")
        return 0
    out.write(f"Quantize result: {status}")
    return 0


def _dequantize(args, out, api):
    try:
        result = api.dequantize_model()
    except Exception as e:
        out.write(_format_error(e, "dequantize"))
        return 1
    status = result.get("status", "error")
    if status == "ok":
        out.write("Dequantized.")
        return 0
    out.write(f"Dequantize result: {status}")
    return 0
