"""
Feeds command group — RSS and JSON feed generation from dev notes.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get


def register(cli):
    """Register feeds commands with the CLI group."""

    @cli.group(help="RSS and JSON feed generation from dev notes")
    def feeds():
        pass

    @feeds.command("rss", help="Generate RSS feed")
    @click.option("--tag", default="", help="Filter by tag")
    @click.option("--limit", "-n", default=20, type=int)
    @click.option("--output", "-o", help="Save to file")
    @click.pass_context
    def feeds_rss(ctx, tag, limit, output):
        params = {"limit": limit}
        if tag:
            params["tag"] = tag
        r = api_get(ctx, "/feeds/rss.xml", params=params)
        if r.status_code != 200:
            log.error(f"RSS failed: {r.status_code}")
            return
        content = r.text
        if output:
            with open(output, "w") as f:
                f.write(content)
            log.success(f"Saved to: {output}")
        else:
            print(content)

    @feeds.command("json", help="Generate JSON feed")
    @click.option("--tag", default="", help="Filter by tag")
    @click.option("--limit", "-n", default=20, type=int)
    @click.option("--output", "-o", help="Save to file")
    @click.pass_context
    def feeds_json(ctx, tag, limit, output):
        params = {"limit": limit}
        if tag:
            params["tag"] = tag
        r = api_get(ctx, "/feeds/feed.json", params=params)
        if r.status_code != 200:
            log.error(f"JSON feed failed: {r.status_code}")
            return
        content = r.text
        if output:
            with open(output, "w") as f:
                f.write(content)
            log.success(f"Saved to: {output}")
        else:
            print(content)

    return feeds
