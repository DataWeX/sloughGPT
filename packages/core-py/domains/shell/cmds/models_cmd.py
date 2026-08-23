"""models / unload — manage AI models in the inference server."""

from __future__ import annotations

from ..console import Console
from ..commands import ShellCommands, _api_get

help = "List models, unload/quantize/dequantize, or set compute precision"
names = ["models", "unload", "precision", "quantize", "dequantize"]


def run(argv: list[str], out: Console, api: ShellCommands,
        env: dict[str, str]) -> int:
    cmd = argv[0] if argv else "models"

    if cmd == "models":
        with out.spinner("Fetching models") as s:
            models = api.models()
            try:
                health = api._api_get("/health")
                if isinstance(health, dict):
                    hdata = health.get("data", health)
                    loaded_model = hdata.get("model_type") or hdata.get("model_id", "")
                    if loaded_model:
                        models = [m for m in models if m.get("model_id") != loaded_model]
                        models.insert(0, {"model_id": loaded_model, "status": "loaded", "loaded": True})
            except Exception:
                pass
        s.ok("Models loaded")
        if not models:
            out.print("  No models available")
            return 0
        rows = []
        for m in models:
            name = m.get("model_id", m.get("name", m.get("id", "?")))
            sz = m.get("size_gb", 0) or m.get("size_mb", 0) / 1024
            sz_str = f"{sz:.2f}G" if sz else ""
            loaded = m.get("status") == "loaded" or m.get("loaded")
            rows.append([name, m.get("type", ""), sz_str, "\u2713 loaded" if loaded else ""])
        out.table(rows, ["Model", "Type", "Size", "Status"])
        return 0

    if cmd == "unload":
        with out.spinner("Fetching models") as s:
            models = api.models()
        s.ok("Models loaded")
        if models:
            names = [m.get("model_id", m.get("name", m.get("id", "?"))) for m in models]
            selected = out.select("Select model to unload:", names)
            if not selected:
                out.print("  Cancelled")
                return 0
            if not out.confirm(f"Unload {selected}?"):
                out.print("  Cancelled")
                return 0
        with out.spinner("Unloading model") as s:
            result = api.unload_model()
        s.ok("Model unloaded")
        out.json(result)
        return 0

    if cmd in ("precision",):
        modes = {"auto": "auto", "fp32": "fp32", "fp16": "fp16"}
        mode = modes.get(argv[1]) if len(argv) > 1 else ""
        if not mode:
            mode = out.select("Select precision:", list(modes.values()))
        with out.spinner(f"Setting precision to {mode}") as s:
            result = api.set_precision(mode)
        s.ok(f"Precision set to {mode}")
        out.json(result)
        return 0

    if cmd == "quantize":
        bits = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 0
        if not bits:
            bits_str = out.select("Select quantization bits:", ["4", "8", "16"])
            bits = int(bits_str) if bits_str else 8
        qmode = argv[2] if len(argv) > 2 and argv[2] in ("symmetric", "asymmetric") else ""
        if not qmode:
            qmode = out.select("Select quantization mode:", ["symmetric", "asymmetric"])
        with out.spinner(f"Quantizing model ({bits}-bit, {qmode})") as s:
            result = api.quantize_model(bits, qmode)
        s.ok(f"Model quantized ({bits}-bit, {qmode})")
        out.json(result)
        return 0

    if cmd == "dequantize":
        if not out.confirm("Dequantize model?"):
            out.print("  Cancelled")
            return 0
        with out.spinner("Dequantizing model") as s:
            result = api.dequantize_model()
        s.ok("Model dequantized")
        out.json(result)
        return 0

    return 0
