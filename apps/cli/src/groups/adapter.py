"""
Adapter command group — per-user LoRA adapter management.
"""

from core.framework import click
from core.helpers import ns as _ns
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register adapter commands with the CLI group."""

    @cli.group(help="Manage per-user LoRA adapters")
    def adapter():
        pass

    @adapter.command("list", help="List LoRA adapters")
    def adapter_list():
        from commands.train import _cmd_user_adapters
        _cmd_user_adapters(_ns(action="list"))

    @adapter.command("info", help="Show adapter info")
    @click.argument("user")
    def adapter_info(user):
        from commands.train import _cmd_user_adapters
        _cmd_user_adapters(_ns(action="info", user=user))

    @adapter.command("merge", help="Merge adapters")
    @click.option("--users", required=True, help="Comma-separated user IDs")
    def adapter_merge(users):
        from commands.train import _cmd_user_adapters
        _cmd_user_adapters(_ns(action="merge", users=users))

    @adapter.command("delete", help="Delete adapter")
    @click.argument("user")
    @click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
    def adapter_delete(user, dry_run):
        from commands.train import _cmd_user_adapters
        if dry_run:
            log.info(f"Would delete adapter for user: {user}")
            return
        _cmd_user_adapters(_ns(action="delete", user=user))

    return adapter
