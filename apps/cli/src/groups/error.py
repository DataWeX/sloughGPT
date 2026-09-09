"""
Error command group — error monitoring.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_delete, output_json, confirm


def register(cli):
    """Register error commands with the CLI group."""

    @cli.group(help="Error monitoring — recent, grouped, trends, clear")
    def error():
        pass

    @error.command("recent", help="Show recent errors")
    @click.option("--limit", "-n", default=20, type=int, help="Max errors to show")
    @click.pass_context
    def error_recent(ctx, limit):
        r = api_get(ctx, f"/errors/recent?limit={limit}")
        if r.status_code != 200:
            log.error(f"Failed to fetch errors: {r.text}")
            return
        data = r.json().get("data", {})
        errors = data.get("errors", [])
        if not errors:
            log.info("No recent errors")
            return
        if output_json(ctx, {"errors": errors}):
            return
        log.header(f"Recent Errors ({len(errors)})")
        for e in errors:
            ts = e.get("timestamp", "?")[:19]
            msg = e.get("message", "?")[:80]
            log.info(f"  [{ts}] {msg}")

    @error.command("grouped", help="Show errors grouped by message")
    @click.pass_context
    def error_grouped(ctx):
        r = api_get(ctx, "/errors/grouped")
        if r.status_code != 200:
            log.error(f"Failed to fetch grouped errors: {r.text}")
            return
        data = r.json().get("data", {})
        groups = data.get("groups", [])
        if not groups:
            log.info("No errors grouped")
            return
        if output_json(ctx, {"groups": groups}):
            return
        log.header(f"Error Groups ({len(groups)})")
        for g in groups:
            count = g.get("count", 0)
            msg = g.get("message", "?")[:70]
            log.info(f"  [{count}x] {msg}")

    @error.command("trends", help="Show error trends (last 24h)")
    @click.pass_context
    def error_trends(ctx):
        r = api_get(ctx, "/errors/trends")
        if r.status_code != 200:
            log.error(f"Failed to fetch trends: {r.text}")
            return
        data = r.json().get("data", {})
        trends = data.get("trends", [])
        if not trends:
            log.info("No error trends")
            return
        if output_json(ctx, {"trends": trends}):
            return
        log.header("Error Trends (24h)")
        for t in trends:
            hour = t.get("hour", "?")
            count = t.get("count", 0)
            bar = "#" * min(count, 40)
            log.info(f"  {hour}: {bar} ({count})")

    @error.command("clear", help="Clear all errors")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--dry-run", is_flag=True, help="Show what would be cleared")
    @click.pass_context
    def error_clear(ctx, yes, dry_run):
        if dry_run:
            log.info("Would clear all errors")
            return
        if not yes:
            confirm("Clear all errors?", abort=True)
        r = api_delete(ctx, "/errors/clear")
        if r.status_code == 200:
            log.success("Errors cleared")
        else:
            log.error(f"Failed to clear: {r.text}")

    @error.command("unread", help="Show unread error count")
    @click.pass_context
    def error_unread(ctx):
        r = api_get(ctx, "/errors/unread")
        if r.status_code != 200:
            log.error(f"Failed to fetch unread count: {r.text}")
            return
        data = r.json().get("data", {})
        count = data.get("count", 0)
        if output_json(ctx, {"unread": count}):
            return
        log.info(f"Unread errors: {count}")

    return error
