"""
Checkpoint command group — training checkpoint management.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, api_delete, output_json, confirm


def register(cli):
    """Register checkpoint commands with the CLI group."""

    @cli.group(help="List, load, and delete training checkpoints")
    def checkpoint():
        pass

    @checkpoint.command("list", help="List all training checkpoints")
    @click.option("--sort", type=click.Choice(["date", "size", "name"]), default="date", help="Sort order")
    @click.option("--json", "json_output", is_flag=True, help="JSON output")
    @click.pass_context
    def checkpoint_list(ctx, sort, json_output):
        """List all saved training checkpoints.

        \b
        Examples:
          sloughgpt checkpoint list
          sloughgpt checkpoint list --sort size
          sloughgpt checkpoint list --json
        """
        r = api_get(ctx, "/training/checkpoints")
        if r.status_code != 200:
            log.error(f"Failed to list checkpoints: {r.text}")
            return
        checkpoints = r.json()
        if not checkpoints:
            log.info("No checkpoints found")
            return

        if output_json(ctx, checkpoints):
            return

        log.header(f"Training Checkpoints ({len(checkpoints)})")
        rows = []
        for cp in checkpoints:
            name = cp.get("name", "unknown")
            size = cp.get("size_mb", 0)
            traits = cp.get("traits", {})
            trait_str = ", ".join(f"{k}={v:.2f}" for k, v in traits.items() if v != 0.5) if traits else ""
            rows.append([name, f"{size:.1f} MB", trait_str or "-"])
        log.table(["Name", "Size", "Traits"], rows)

    @checkpoint.command("load", help="Load a checkpoint into the model")
    @click.argument("name")
    @click.pass_context
    def checkpoint_load(ctx, name):
        """Load a training checkpoint into the active model.

        \b
        Example:
          sloughgpt checkpoint load my-checkpoint.soul
        """
        r = api_post(ctx, f"/training/checkpoints/{name}/load")
        if r.status_code == 200:
            data = r.json()
            log.success(f"Loaded checkpoint: {name}")
            for k, v in data.items():
                if k not in ("status",):
                    log.key_value(k, str(v))
        else:
            log.error(f"Failed to load: {r.text}")

    @checkpoint.command("delete", help="Delete a training checkpoint")
    @click.argument("name")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
    @click.pass_context
    def checkpoint_delete(ctx, name, yes, dry_run):
        """Delete a training checkpoint.

        \b
        Example:
          sloughgpt checkpoint delete my-checkpoint.soul
        """
        if not yes and not dry_run:
            confirm(f"Delete checkpoint '{name}'?", abort=True)
        if dry_run:
            log.info(f"Would delete: {name}")
            return
        r = api_delete(ctx, f"/training/checkpoints/{name}")
        if r.status_code == 200:
            log.success(f"Deleted: {name}")
        else:
            log.error(f"Failed to delete: {r.text}")

    return checkpoint
