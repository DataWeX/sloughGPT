"""
Personality command group — soul personality file management.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register personality commands with the CLI group."""

    @cli.group(help="List, load, and manage .soul personality files")
    def personality():
        pass

    @personality.command("list", help="List built-in personalities")
    def personality_list():
        from commands.models import _cmd_models_personalities
        _cmd_models_personalities(_ns())

    @personality.command("load", help="Load soul via API")
    @click.argument("path")
    @click.pass_context
    def personality_load(ctx, path):
        from commands.models import cmd_soul
        cmd_soul(_ns(load=path, host=ctx.obj["host"], port=ctx.obj["port"]))

    @personality.command("info", help="Inspect soul file")
    @click.argument("path")
    def personality_info(path):
        from commands.models import cmd_soul
        cmd_soul(_ns(info=path))

    @personality.command("create", help="Create new soul from checkpoint")
    @click.option("--checkpoint", "-m", required=True, help="Weights path")
    @click.option("--name", "-n", required=True, help="Soul name")
    @click.option("--dataset", "-d", help="Dataset citation")
    @click.option("--epochs", "-e", default=0, type=int, help="Epoch count")
    @click.option("--lineage", default="nanogpt", help="Architecture label")
    @click.option("--tags", default="", help="Comma-separated tags")
    @click.option("--output", "-o", help="Output .soul path")
    def personality_create(checkpoint, name, dataset, epochs, lineage, tags, output):
        from commands.models import cmd_soul
        args = _ns(
            create=output or f"models/{name}.soul", model=checkpoint,
            name=name, dataset=dataset, epochs=epochs, lineage=lineage, tags=tags,
        )
        cmd_soul(args)

    return personality
