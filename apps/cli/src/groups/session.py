"""
Session command group — chat session management.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, output_json


def register(cli):
    """Register session commands with the CLI group."""

    @cli.group(help="Chat session management")
    def session():
        pass

    @session.command("list", help="List chat sessions")
    @click.pass_context
    def session_list(ctx):
        r = api_get(ctx, "/chat/sessions")
        if r.status_code != 200:
            log.error(f"Failed to list sessions: {r.text}")
            return
        data = r.json()
        sessions = data if isinstance(data, list) else data.get("sessions", [])
        if not sessions:
            log.info("No sessions found")
            return
        if output_json(ctx, {"sessions": sessions}):
            return
        log.header("Chat Sessions")
        for s in sessions:
            name = s.get("name", s.get("id", "?"))
            log.info(f"  {name}")

    @session.command("messages", help="Show messages in a session")
    @click.argument("session_id")
    @click.option("--limit", "-n", default=20, type=int, help="Max messages")
    @click.pass_context
    def session_messages(ctx, session_id, limit):
        r = api_get(ctx, f"/session/{session_id}/messages?limit={limit}")
        if r.status_code != 200:
            log.error(f"Failed to get messages: {r.text}")
            return
        data = r.json()
        messages = data.get("messages", [])
        if not messages:
            log.info("No messages in session")
            return
        if output_json(ctx, {"messages": messages}):
            return
        log.header(f"Session: {session_id}")
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")[:100]
            log.info(f"  [{role}] {content}")

    @session.command("search", help="Search chat sessions")
    @click.argument("query")
    @click.option("--limit", "-n", default=10, type=int, help="Max results")
    @click.pass_context
    def session_search(ctx, query, limit):
        r = api_get(ctx, f"/chat/sessions/search?q={query}&limit={limit}")
        if r.status_code != 200:
            log.error(f"Search failed: {r.text}")
            return
        data = r.json()
        results = data if isinstance(data, list) else data.get("results", [])
        if not results:
            log.info("No matching sessions")
            return
        if output_json(ctx, {"results": results}):
            return
        log.header(f"Search: {query}")
        for s in results:
            name = s.get("name", s.get("id", "?"))
            log.info(f"  {name}")

    return session
