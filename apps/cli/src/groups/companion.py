"""
Companion command group — AI companion management and chat.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json


def register(cli):
    """Register companion commands with the CLI group."""

    @cli.group(help="AI companion management and chat")
    def companion():
        pass

    @companion.command("status", help="Show companion status")
    @click.pass_context
    def companion_status(ctx):
        r = api_get(ctx, "/companion/status")
        if r.status_code != 200:
            log.error(f"Status failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Companion Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    @companion.command("chat", help="Chat with companion")
    @click.argument("message")
    @click.option("--user-name", default="", help="Your name")
    @click.option("--mood", default="", help="Your current mood")
    @click.pass_context
    def companion_chat(ctx, message, user_name, mood):
        payload = {"message": message}
        if user_name:
            payload["user_name"] = user_name
        if mood:
            payload["user_mood"] = mood
        r = api_post(ctx, "/companion/chat", json=payload)
        if r.status_code != 200:
            log.error(f"Chat failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        resp = data.get("data", data).get("response", str(data))
        log.info(resp)

    @companion.command("personality", help="Show companion personality")
    @click.pass_context
    def companion_personality(ctx):
        r = api_get(ctx, "/companion/personality")
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        p = data.get("data", data)
        log.header("Companion Personality")
        for k, v in p.items():
            log.info(f"  {k}: {v}")

    @companion.command("preset", help="Use a preset personality")
    @click.argument("name")
    @click.pass_context
    def companion_preset(ctx, name):
        r = api_post(ctx, "/companion/preset", json={"preset": name})
        if r.status_code != 200:
            log.error(f"Preset failed: {r.text}")
            return
        log.success(f"Applied preset: {name}")

    return companion
