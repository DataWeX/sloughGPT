"""
Vector command group — vector store for semantic search.
"""

from core.framework import click
from core.helpers import ns as _ns


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
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/init",
                          json={"provider": provider, "dimension": dimension}, timeout=timeout)
        if r.status_code != 200:
            log.error(f"Init failed: {r.text}")
            sys.exit(1)
        log.success(f"Vector store initialized: {provider} (dim={dimension})")

    @vector.command("upsert", help="Insert or update vectors")
    @click.argument("texts")
    @click.option("--ids", default="", help="Comma-separated IDs")
    @click.pass_context
    def vector_upsert(ctx, texts, ids):
        import requests
        text_list = [t.strip() for t in texts.split(",") if t.strip()]
        id_list = [i.strip() for i in ids.split(",") if i.strip()] if ids else None
        timeout = ctx.obj.get("timeout", 10)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/upsert",
                          json={"texts": text_list, "ids": id_list}, timeout=timeout)
        if r.status_code != 200:
            log.error(f"Upsert failed: {r.text}")
            sys.exit(1)
        log.success(f"Upserted {len(text_list)} vectors")

    @vector.command("search", help="Semantic search")
    @click.argument("query")
    @click.option("--top-k", type=int, default=5, help="Number of results")
    @click.pass_context
    def vector_search(ctx, query, top_k):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/search",
                          json={"query": query, "top_k": top_k}, timeout=timeout)
        if r.status_code != 200:
            log.error(f"Search failed: {r.text}")
            sys.exit(1)
        data = r.json()
        results = data.get("data", data).get("results", [])
        if ctx.obj.get("json"):
            _output(ctx, {"results": results})
        else:
            log.header(f"Search: {query}")
            for i, res in enumerate(results):
                text = res.get("text", res.get("content", ""))[:80]
                score = res.get("score", 0)
                log.info(f"  {i+1}. [{score:.3f}] {text}")

    @vector.command("stats", help="Show vector store stats")
    @click.pass_context
    def vector_stats(ctx):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/vector/stats", timeout=timeout)
        if r.status_code != 200:
            log.error(f"Stats failed: {r.text}")
            sys.exit(1)
        data = r.json()
        stats = data.get("data", data)
        if ctx.obj.get("json"):
            _output(ctx, stats)
        else:
            log.header("Vector Store Stats")
            for k, v in stats.items():
                log.info(f"  {k}: {v}")

    return vector
