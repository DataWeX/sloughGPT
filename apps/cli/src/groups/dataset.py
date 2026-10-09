"""
Dataset command group — list, import, export, and validate datasets.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register dataset commands with the CLI group."""

    @cli.group(help="List, import, export, and validate datasets")
    @click.pass_context
    def dataset(ctx):
        pass

    @dataset.command("list", help="List available datasets")
    @click.pass_context
    def dataset_list(ctx):
        from commands.data import cmd_datasets
        cmd_datasets(_ns(json_output=ctx.obj.get("json")))

    @dataset.command("stats", help="Show dataset statistics")
    @click.argument("name")
    @click.pass_context
    def dataset_stats(ctx, name):
        from commands.data import cmd_dataset_stats
        args = _ns(name=name, json_output=ctx.obj.get("json"))
        cmd_dataset_stats(args)

    @dataset.command("search", help="Search online datasets")
    @click.argument("query")
    @click.option("--limit", "-n", default=10, type=int, help="Max results")
    @click.option("--source", type=click.Choice(["hf", "github"]), default="hf")
    @click.pass_context
    def dataset_search(ctx, query, limit, source):
        from commands.data import cmd_dataset_search
        args = _ns(query=query, limit=limit, source=source, json_output=ctx.obj.get("json"))
        cmd_dataset_search(args)

    @dataset.command("import", help="Import dataset from various sources")
    @click.argument("source", type=click.Choice(["github", "hf", "url", "local"]))
    @click.argument("identifier")
    @click.argument("name", required=False)
    def dataset_import(source, identifier, name):
        from commands.data import cmd_dataset_import
        args = _ns(**({"url": identifier} if source in ("github", "url") else {"dataset_id": identifier}), name=name)
        cmd_dataset_import(args, source)

    @dataset.command("export", help="Export dataset to zip")
    @click.argument("name")
    @click.option("--output", "-o", help="Output zip file")
    def dataset_export(name, output):
        from commands.data import cmd_dataset_export
        args = _ns(name=name, output=output)
        cmd_dataset_export(args)

    @dataset.command("validate", help="Validate dataset file")
    @click.argument("path")
    def dataset_validate(path):
        from commands.data import cmd_data_tool
        cmd_data_tool(_ns(path=path), "validate")

    @dataset.command("info", help="Show file or directory statistics")
    @click.argument("path")
    def dataset_info(path):
        from commands.data import cmd_data_tool
        cmd_data_tool(_ns(path=path), "stats")

    return dataset
