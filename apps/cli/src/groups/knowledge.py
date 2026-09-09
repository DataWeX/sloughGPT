"""
Knowledge command group — semantic knowledge operations.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register knowledge commands with the CLI group."""

    @cli.group(help="Semantic knowledge operations — search, dedup, categorize, gaps")
    def knowledge():
        pass

    @knowledge.command("search", help="Search codebase with natural language")
    @click.argument("query")
    @click.option("--path", default=".", help="Directory to search")
    @click.option("--top-k", default=10, type=int, help="Max results")
    @click.option("--extensions", default=None, help="Comma-separated file extensions")
    @click.pass_context
    def knowledge_search(ctx, query, path, top_k, extensions):
        """Search your codebase using natural language."""
        exts = extensions.split(",") if extensions else None
        r = api_post(ctx, "/knowledge/search-files",
                     json={"query": query, "path": path, "top_k": top_k, "extensions": exts})
        if r.status_code != 200:
            log.error(f"Search failed: {r.text}")
            return
        data = r.json()
        log.header(f"Found {len(data['results'])} results (indexed {data['indexed_files']} files)")
        for i, res in enumerate(data["results"], 1):
            log.info(f"[{res['score']:.3f}] {res['path']}:{res['line']}")
            snippet = res['snippet'].replace('\n', ' ')[:100]
            log.info(f"  {snippet}")
            log.blank()

    @knowledge.command("dedup", help="Check for duplicate knowledge")
    @click.argument("content")
    @click.option("--threshold", default=0.85, type=float, help="Similarity threshold")
    @click.pass_context
    def knowledge_dedup(ctx, content, threshold):
        """Check if content already exists in the knowledge base."""
        r = api_post(ctx, "/knowledge/check-duplicate",
                     json={"content": content, "threshold": threshold})
        if r.status_code != 200:
            log.error(f"Check failed: {r.text}")
            return
        data = r.json()
        if data["is_duplicate"]:
            log.warning(f"DUPLICATE (score: {data['score']:.3f})")
            log.info(f"  Existing: {data['best_match'][:100]}")
        else:
            log.success(f"Unique (best match score: {data['score']:.3f})")

    @knowledge.command("categorize", help="Auto-categorize content")
    @click.argument("content")
    @click.pass_context
    def knowledge_categorize(ctx, content):
        """Auto-assign a topic to content based on existing categories."""
        r = api_post(ctx, "/knowledge/categorize", json={"content": content})
        if r.status_code != 200:
            log.error(f"Categorize failed: {r.text}")
            return
        data = r.json()
        log.success(f"Topic: {data['topic']}")
        if data["suggestions"]:
            log.info("Suggestions:")
            for s in data["suggestions"]:
                log.info(f"  {s['topic']} ({s['score']:.3f})")

    @knowledge.command("gaps", help="Find knowledge gaps")
    @click.pass_context
    def knowledge_gaps(ctx):
        """Show under-represented topics in your knowledge base."""
        r = api_get(ctx, "/knowledge/gaps")
        if r.status_code != 200:
            log.error(f"Gaps failed: {r.text}")
            return
        data = r.json()
        log.header(f"Knowledge gaps ({data['total_facts']} facts, {len(data['topics'])} topics)")
        if data["gaps"]:
            for g in data["gaps"]:
                log.info(f"  {g['topic']}: {g['suggestion']}")
        else:
            log.success("No significant gaps found")

    @knowledge.command("ingest", help="Bulk ingest texts with dedup")
    @click.argument("texts", nargs=-1)
    @click.option("--topic", default="imported", help="Topic tag")
    @click.option("--file", "file_path", default=None, help="Read texts from file (one per line)")
    @click.pass_context
    def knowledge_ingest(ctx, texts, topic, file_path):
        """Bulk ingest texts with automatic deduplication."""
        items = list(texts)
        if file_path:
            with open(file_path) as f:
                items.extend(line.strip() for line in f if line.strip())
        if not items:
            log.error("No texts to ingest")
            return
        r = api_post(ctx, "/knowledge/bulk-ingest",
                     json={"items": items, "topic": topic})
        if r.status_code != 200:
            log.error(f"Ingest failed: {r.text}")
            return
        data = r.json()
        log.success(f"Bulk ingest: {data['added']} added, {data['skipped']} skipped, {data['errors']} errors")

    return knowledge
