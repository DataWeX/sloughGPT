"""
Vector command group — vector store for semantic search.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register vector commands with the CLI group."""

    @cli.group(help="Vector store for semantic search")
    def vector():
        pass

    @vector.command("init", help="Initialize vector store")
    @click.option("--provider", default="in_memory", help="Provider: in_memory, chromadb")
    @click.option("--dimension", type=int, default=384, help="Embedding dimension")
    @click.pass_context
    def vector_init(ctx, provider, dimension):
        r = api_post(ctx, "/vector/init", json={"provider": provider, "dimension": dimension})
        if r.status_code != 200:
            log.error(f"Init failed: {r.text}")
            return
        log.success(f"Vector store initialized: {provider} (dim={dimension})")

    @vector.command("upsert", help="Insert or update vectors")
    @click.argument("texts")
    @click.option("--ids", default="", help="Comma-separated IDs")
    @click.pass_context
    def vector_upsert(ctx, texts, ids):
        text_list = [t.strip() for t in texts.split(",") if t.strip()]
        id_list = [i.strip() for i in ids.split(",") if i.strip()] if ids else None
        r = api_post(ctx, "/vector/upsert", json={"texts": text_list, "ids": id_list})
        if r.status_code != 200:
            log.error(f"Upsert failed: {r.text}")
            return
        log.success(f"Upserted {len(text_list)} vectors")

    @vector.command("search", help="Semantic search")
    @click.argument("query")
    @click.option("--top-k", type=int, default=5, help="Number of results")
    @click.pass_context
    def vector_search(ctx, query, top_k):
        r = api_post(ctx, "/vector/search", json={"query": query, "top_k": top_k})
        if r.status_code != 200:
            log.error(f"Search failed: {r.text}")
            return
        data = r.json()
        results = data.get("data", data).get("results", [])
        if output_json(ctx, {"results": results}):
            return
        log.header(f"Search: {query}")
        for i, res in enumerate(results):
            text = res.get("text", res.get("content", ""))[:80]
            score = res.get("score", 0)
            log.info(f"  {i+1}. [{score:.3f}] {text}")

    @vector.command("stats", help="Show vector store stats")
    @click.pass_context
    def vector_stats(ctx):
        r = api_get(ctx, "/vector/stats")
        if r.status_code != 200:
            log.error(f"Stats failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Vector Store Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    return vector
