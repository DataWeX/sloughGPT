"""
Memory command group — inspect and manage the auto-memory layer.
"""

from core.framework import click
from core.helpers import ns as _ns
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register memory commands with the CLI group."""

    @cli.group(help="Inspect and manage the auto-memory layer (stats, search, store, consolidate, archive)")
    def memory():
        pass

    @memory.command("stats", help="Show memory statistics")
    def memory_stats():
        from commands.memory import cmd_memory_stats
        cmd_memory_stats(_ns())

    @memory.command("enable", help="Enable the memory layer at runtime")
    def memory_enable():
        from commands.memory import cmd_memory_enable
        cmd_memory_enable(_ns(enabled=True))

    @memory.command("disable", help="Disable the memory layer at runtime")
    def memory_disable():
        from commands.memory import cmd_memory_enable
        cmd_memory_enable(_ns(enabled=False))

    @memory.command("list", help="List stored memory items, most recent first")
    @click.option("--limit", "-n", default=50, type=int, help="Max items to show")
    def memory_list(limit):
        from commands.memory import cmd_memory_list
        cmd_memory_list(_ns(limit=limit))

    @memory.command("search", help="Semantic-search stored memory")
    @click.argument("query")
    @click.option("--limit", "-n", default=5, type=int, help="Max results")
    def memory_search(query, limit):
        from commands.memory import cmd_memory_search
        cmd_memory_search(_ns(query=query, limit=limit))

    @memory.command("store", help="Persist one explicit fact")
    @click.argument("content")
    @click.option("--topic", default="manual", help="Topic label")
    @click.option("--source", default="cli", help="Provenance label")
    def memory_store(content, topic, source):
        from commands.memory import cmd_memory_store
        cmd_memory_store(_ns(content=content, topic=topic, source=source))

    @memory.command("remember", help="Persist one completed turn (user + assistant)")
    @click.argument("user_message")
    @click.argument("assistant_response")
    def memory_remember(user_message, assistant_response):
        from commands.memory import cmd_memory_remember
        cmd_memory_remember(_ns(
            user_message=user_message, assistant_response=assistant_response,
        ))

    @memory.command("clear", help="Remove all stored memory")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--dry-run", is_flag=True, help="Show what would be cleared without clearing")
    def memory_clear(yes, dry_run):
        from commands.memory import cmd_memory_clear
        if dry_run:
            log.info("Would clear all stored memory")
            return
        cmd_memory_clear(_ns(yes=yes))

    @memory.command("consolidate", help="Merge near-duplicate facts, keeping the longest")
    @click.option("--threshold", type=float, default=None,
                  help="Min similarity for a merge (default from config)")
    def memory_consolidate(threshold):
        from commands.memory import cmd_memory_consolidate
        cmd_memory_consolidate(_ns(threshold=threshold))

    @memory.command("archive", help="Inspect or prune the task-backed provenance archive")
    @click.option("--limit", "-n", default=10, type=int, help="Recent records to show (0 = none)")
    @click.option("--prune-days", type=float, default=None,
                  help="Retention window in days; delete older records")
    def memory_archive(limit, prune_days):
        from commands.memory import cmd_memory_archive
        cmd_memory_archive(_ns(limit=limit, prune_days=prune_days))

    return memory
