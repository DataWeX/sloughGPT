"""
Docstore command group — server-side document store.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_delete, output_json, confirm


def register(cli):
    """Register docstore commands with the CLI group."""

    @cli.group(help="Server-side document store (browser chat DB)")
    def docstore():
        pass

    @docstore.command("collections", help="List document collections")
    @click.pass_context
    def docstore_collections(ctx):
        r = api_get(ctx, "/docstore/collections")
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            return
        data = r.json()
        cols = data.get("data", data).get("collections", [])
        if output_json(ctx, {"collections": cols}):
            return
        log.header("Collections")
        for c in cols:
            log.info(f"  {c}")

    @docstore.command("list", help="List documents in a collection")
    @click.argument("collection")
    @click.option("--limit", "-n", default=20, type=int)
    @click.pass_context
    def docstore_list(ctx, collection, limit):
        r = api_get(ctx, f"/docstore/{collection}?limit={limit}")
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            return
        data = r.json()
        docs = data.get("data", data).get("documents", [])
        if output_json(ctx, {"documents": docs}):
            return
        log.header(f"{collection} ({len(docs)} docs)")
        for d in docs:
            doc_id = d.get("_id", d.get("id", "?"))
            log.info(f"  {doc_id}")

    @docstore.command("get", help="Get a document")
    @click.argument("collection")
    @click.argument("doc_id")
    @click.pass_context
    def docstore_get(ctx, collection, doc_id):
        r = api_get(ctx, f"/docstore/{collection}/{doc_id}")
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            return
        data = r.json()
        doc = data.get("data", data)
        if output_json(ctx, doc):
            return
        for k, v in doc.items():
            log.info(f"  {k}: {v}")

    @docstore.command("delete", help="Delete a document")
    @click.argument("collection")
    @click.argument("doc_id")
    @click.option("--yes", "-y", is_flag=True)
    @click.option("--dry-run", is_flag=True)
    @click.pass_context
    def docstore_delete(ctx, collection, doc_id, yes, dry_run):
        if dry_run:
            log.info(f"Would delete {collection}/{doc_id}")
            return
        if not yes:
            confirm(f"Delete {collection}/{doc_id}?", abort=True)
        r = api_delete(ctx, f"/docstore/{collection}/{doc_id}")
        if r.status_code == 200:
            log.success(f"Deleted {collection}/{doc_id}")
        else:
            log.error(f"Failed: {r.text}")

    return docstore
