"""
Docstore command group — server-side document store.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register docstore commands with the CLI group."""

    @cli.group(help="Server-side document store (browser chat DB)")
    def docstore():
        pass

    @docstore.command("collections", help="List document collections")
    @click.pass_context
    def docstore_collections(ctx):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/collections", timeout=timeout)
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            sys.exit(1)
        data = r.json()
        cols = data.get("data", data).get("collections", [])
        if ctx.obj.get("json"):
            _output(ctx, {"collections": cols})
        else:
            log.header("Collections")
            for c in cols:
                log.info(f"  {c}")

    @docstore.command("list", help="List documents in a collection")
    @click.argument("collection")
    @click.option("--limit", "-n", default=20, type=int)
    @click.pass_context
    def docstore_list(ctx, collection, limit):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}?limit={limit}",
                         timeout=timeout)
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            sys.exit(1)
        data = r.json()
        docs = data.get("data", data).get("documents", [])
        if ctx.obj.get("json"):
            _output(ctx, {"documents": docs})
        else:
            log.header(f"{collection} ({len(docs)} docs)")
            for d in docs:
                doc_id = d.get("_id", d.get("id", "?"))
                log.info(f"  {doc_id}")

    @docstore.command("get", help="Get a document")
    @click.argument("collection")
    @click.argument("doc_id")
    @click.pass_context
    def docstore_get(ctx, collection, doc_id):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}/{doc_id}",
                         timeout=timeout)
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            sys.exit(1)
        data = r.json()
        doc = data.get("data", data)
        if ctx.obj.get("json"):
            _output(ctx, doc)
        else:
            for k, v in doc.items():
                log.info(f"  {k}: {v}")

    @docstore.command("delete", help="Delete a document")
    @click.argument("collection")
    @click.argument("doc_id")
    @click.option("--yes", "-y", is_flag=True)
    @click.option("--dry-run", is_flag=True)
    @click.pass_context
    def docstore_delete(ctx, collection, doc_id, yes, dry_run):
        import requests
        if dry_run:
            log.info(f"Would delete {collection}/{doc_id}")
            return
        if not yes:
            confirm(f"Delete {collection}/{doc_id}?", abort=True)
        timeout = ctx.obj.get("timeout", 10)
        r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/docstore/{collection}/{doc_id}",
                            timeout=timeout)
        if r.status_code == 200:
            log.success(f"Deleted {collection}/{doc_id}")
        else:
            log.error(f"Failed: {r.text}")
            sys.exit(1)

    return docstore
