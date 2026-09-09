"""
Security command group — audit logs and API key management.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register security commands with the CLI group."""

    @cli.group(help="Security audit logs and API key management")
    def security():
        pass

    @security.command("audit", help="Show audit logs")
    @click.option("--limit", "-n", default=20, type=int, help="Max entries")
    @click.option("--type", "event_type", default="", help="Filter by event type")
    @click.option("--history", is_flag=True, help="Read from persisted audit.log")
    @click.pass_context
    def security_audit(ctx, limit, event_type, history):
        params = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        if history:
            params["history"] = True
        r = api_get(ctx, "/security/audit", params=params)
        if r.status_code != 200:
            log.error(f"Audit failed: {r.text}")
            return
        data = r.json()
        logs = data.get("data", data).get("logs", [])
        if output_json(ctx, {"logs": logs}):
            return
        log.header(f"Audit Logs ({len(logs)} entries)")
        for entry in logs:
            event = entry.get("event_type", "?")
            ts = entry.get("timestamp", "")[:19]
            detail = entry.get("detail", "")[:60]
            log.info(f"  [{ts}] {event} — {detail}")

    @security.command("keys", help="Show API key info")
    @click.pass_context
    def security_keys(ctx):
        r = api_get(ctx, "/security/keys")
        if r.status_code != 200:
            log.error(f"Keys failed: {r.text}")
            return
        data = r.json()
        info = data.get("data", data)
        if output_json(ctx, info):
            return
        count = info.get("count", 0)
        configured = info.get("configured", False)
        log.info(f"API keys configured: {configured} ({count} keys)")

    return security
