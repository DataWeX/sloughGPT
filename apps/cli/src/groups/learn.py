"""
Learn command group — continual learning from web, feeds, and knowledge.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register learn commands with the CLI group."""

    @cli.group(help="Continual learning from web, feeds, and knowledge")
    def learn():
        pass

    @learn.command("search", help="Search web and learn from results")
    @click.argument("query")
    @click.option("--max-results", type=int, default=5, help="Max results")
    @click.pass_context
    def learn_search(ctx, query, max_results):
        r = api_post(ctx, "/learn/search",
                     json={"query": query, "max_results": max_results})
        if r.status_code != 200:
            log.error(f"Search failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        info = data.get("data", data)
        log.info(f"Tokens ingested: {info.get('tokens_ingested', 0)}")
        log.info(f"New facts: {info.get('new_facts', 0)}")
        log.info(f"Rejected: {info.get('rejected', 0)}")

    @learn.command("status", help="Show learner status")
    @click.pass_context
    def learn_status(ctx):
        r = api_get(ctx, "/learn/status")
        if r.status_code != 200:
            log.error(f"Status failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Learner Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    @learn.command("knowledge", help="Query learned knowledge")
    @click.argument("query", required=False)
    @click.option("--topic", default="", help="Filter by topic")
    @click.pass_context
    def learn_knowledge(ctx, query, topic):
        params = {}
        if query:
            params["q"] = query
        if topic:
            params["topic"] = topic
        r = api_get(ctx, "/learn/knowledge", params=params)
        if r.status_code != 200:
            log.error(f"Knowledge query failed: {r.text}")
            return
        data = r.json()
        facts = data.get("data", data).get("facts", [])
        if output_json(ctx, {"facts": facts}):
            return
        log.header("Learned Knowledge")
        for f in facts[:20]:
            topic = f.get("topic", "?")
            text = f.get("text", "")[:80]
            log.info(f"  [{topic}] {text}")

    @learn.command("train", help="Force a training step")
    @click.pass_context
    def learn_train(ctx):
        r = api_post(ctx, "/learn/train")
        if r.status_code != 200:
            log.error(f"Train failed: {r.text}")
            return
        log.success("Training step completed")

    @learn.command("ingest", help="Ingest raw text")
    @click.argument("text")
    @click.pass_context
    def learn_ingest(ctx, text):
        r = api_post(ctx, "/learn/ingest",
                     json={"text": text})
        if r.status_code != 200:
            log.error(f"Ingest failed: {r.text}")
            return
        log.success("Text ingested")

    return learn
