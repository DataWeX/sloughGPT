"""
Feedback command group — export and prepare feedback data.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register feedback commands with the CLI group."""

    @cli.group(help="Export and prepare feedback data")
    def feedback():
        pass

    @feedback.command("export", help="Export feedback data")
    @click.option("--format", type=click.Choice(["jsonl", "dpo"]), default="jsonl")
    @click.option("--output", default="data/training_feedback.jsonl")
    def feedback_export(fmt, output):
        from commands.train import _cmd_feedback_export
        args = _ns(format=fmt, output=output)
        _cmd_feedback_export(args)

    @feedback.command("prepare", help="Prepare training data from feedback")
    @click.option("--format", type=click.Choice(["all", "dpo", "sft", "reward"]), default="all")
    @click.option("--output")
    @click.option("--stats-only", is_flag=True)
    def feedback_prepare(fmt, output, stats_only):
        from commands.train import _cmd_feedback_train
        args = _ns(format=fmt, output=output, stats_only=stats_only)
        _cmd_feedback_train(args)

    return feedback
