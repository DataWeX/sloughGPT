"""
Model command group — list, inspect, download, export, and benchmark models.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register model commands with the CLI group."""

    @cli.group(help="List, inspect, download, export, and benchmark models")
    @click.pass_context
    def model(ctx):
        pass

    @model.command("list", help="List available models")
    @click.pass_context
    def model_list(ctx):
        from commands.models import cmd_models
        cmd_models(_ns(json_output=ctx.obj.get("json")))

    @model.command("status", help="Show cached/downloaded models with sizes")
    @click.pass_context
    def model_status(ctx):
        from commands.models import _cmd_models_status
        _cmd_models_status(_ns(json_output=ctx.obj.get("json")))

    @model.command("info", help="Show checkpoint info")
    @click.argument("checkpoint", default="models/sloughgpt.soul")
    @click.pass_context
    def model_info(ctx, checkpoint):
        from commands.models import _cmd_models_info
        _cmd_models_info(_ns(model=checkpoint, json_output=ctx.obj.get("json")))

    @model.command("download", help="Download model from HuggingFace")
    @click.argument("model_id", required=False, default=None)
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
    @click.pass_context
    def model_download(ctx, model_id, yes):
        from commands.models import _cmd_models_download
        _cmd_models_download(_ns(model_id=model_id, yes=yes))

    @model.command("export", help="Export model to different formats")
    @click.argument("checkpoint", default="models/sloughgpt.soul")
    @click.option("--output", "-o", help="Output path")
    @click.option("--format", "-f", "fmt",
        type=click.Choice(["safetensors", "safetensors_bf16", "onnx", "gguf_q4_k_m",
                           "gguf_fp16", "gguf_q5_k_m", "gguf_q8_0",
                           "sou", "all"]),
        default="safetensors", help="Export format")
    @click.option("--quantize", type=click.Choice(["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]))
    @click.option("--seq-len", default=128, type=int, help="Sequence length for ONNX")
    @click.option("--opset", default=17, type=int, help="ONNX opset")
    @click.option("--ctx", "n_ctx", default=2048, type=int, help="Context length for GGUF")
    @click.option("--soul-name", default=None, help="Slo name")
    @click.option("--metadata", multiple=True, help="Metadata KEY=VALUE")
    def model_export(checkpoint, output, fmt, quantize, seq_len, opset, n_ctx, soul_name, metadata):
        from commands.models import cmd_export_cli
        args = _ns(
            model=checkpoint, output=output, format=fmt, quantization=quantize,
            seq_len=seq_len, opset=opset, n_ctx=n_ctx, soul_name=soul_name,
            metadata=list(metadata) or None,
        )
        cmd_export_cli(args)

    @model.command("benchmark", help="Run performance benchmarks")
    @click.option("--checkpoint", "-m", default="gpt2", help="Model to benchmark")
    @click.option("--device", "-d", type=click.Choice(["auto", "cpu", "cuda", "mps"]), default="auto")
    @click.option("--test", "-t", type=click.Choice(["all", "latency", "throughput"]), default="all")
    @click.option("--runs", "-r", default=10, type=int, help="Number of runs")
    @click.option("--tokens", "-k", default=50, type=int, help="Max new tokens")
    @click.option("--prompt", "-p", default="The quick brown fox jumps over the lazy dog", help="Test prompt")
    @click.pass_context
    def model_benchmark(ctx, checkpoint, device, test, runs, tokens, prompt):
        from commands.models import cmd_benchmark
        args = _ns(model=checkpoint, device=device, test=test, runs=runs, tokens=tokens, prompt=prompt,
                  json_output=ctx.obj.get("json"))
        cmd_benchmark(args)

    @model.command("compare", help="Compare models or benchmarks")
    @click.pass_context
    def model_compare(ctx):
        from commands.models import _cmd_models_compare
        _cmd_models_compare(_ns(json_output=ctx.obj.get("json")))

    return model
