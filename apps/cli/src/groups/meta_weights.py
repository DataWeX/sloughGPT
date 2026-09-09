"""
Meta-weights command group — feedback-driven meta-weight adaptation.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json


def register(cli):
    """Register meta-weights commands with the CLI group."""

    @cli.group(help="Feedback-driven meta-weight adaptation")
    def meta_weights():
        pass

    @meta_weights.command("get", help="Get meta-weight adjustments")
    @click.argument("message")
    @click.option("--k", type=int, default=5, help="Number of similar samples")
    @click.pass_context
    def meta_weights_get(ctx, message, k):
        r = api_post(ctx, "/meta-weights/get",
                     json={"user_message": message, "k": k})
        if r.status_code != 200:
            log.error(f"Failed: {r.text}")
            return
        data = r.json()
        if output_json(ctx, data):
            return
        w = data.get("data", data)
        log.header("Meta-Weights")
        for k, v in w.items():
            log.info(f"  {k}: {v}")

    @meta_weights.command("stats", help="Show meta-weight statistics")
    @click.pass_context
    def meta_weights_stats(ctx):
        r = api_get(ctx, "/meta-weights/stats")
        if r.status_code != 200:
            log.error(f"Stats failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Meta-Weight Stats")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    return meta_weights
