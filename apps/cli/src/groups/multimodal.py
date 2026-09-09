"""
Multimodal command group — vision, speech, and video capabilities.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, output_json
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register multimodal commands with the CLI group."""

    @cli.group(help="Multimodal capabilities (vision, speech, video)")
    def multimodal():
        pass

    @multimodal.command("status", help="Show multimodal engine status")
    @click.pass_context
    def multimodal_status(ctx):
        r = api_get(ctx, "/multimodal/status")
        if r.status_code != 200:
            log.error(f"Status failed: {r.text}")
            return
        data = r.json()
        stats = data.get("data", data)
        if output_json(ctx, stats):
            return
        log.header("Multimodal Status")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")

    @multimodal.command("dpo", help="Trigger DPO training")
    @click.option("--max-pairs", type=int, default=6, help="Max preference pairs")
    @click.option("--lr", type=float, default=5e-6, help="Learning rate")
    @click.pass_context
    def multimodal_dpo(ctx, max_pairs, lr):
        r = api_post(ctx, "/multimodal/dpo/trigger",
                     json={"max_pairs": max_pairs, "learning_rate": lr})
        if r.status_code != 200:
            log.error(f"DPO failed: {r.text}")
            return
        log.success("DPO training triggered")

    @multimodal.command("video-train", help="Train video model")
    @click.argument("data_path")
    @click.option("--epochs", type=int, default=5)
    @click.option("--batch-size", type=int, default=2)
    @click.pass_context
    def multimodal_video_train(ctx, data_path, epochs, batch_size):
        r = api_post(ctx, "/multimodal/video/train",
                     json={"data_path": data_path, "epochs": epochs, "batch_size": batch_size})
        if r.status_code != 200:
            log.error(f"Video train failed: {r.text}")
            return
        log.success("Video training started")

    return multimodal
