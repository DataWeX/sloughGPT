"""
Experiment command group — ML experiment tracking.
"""

from core.framework import click
from core.helpers import ns as _ns, api_get, api_post, api_delete, output_json, confirm
from domains.logging import get_global
log = get_global()


def register(cli):
    """Register experiment commands with the CLI group."""

    @cli.group(help="ML experiment tracking — create, list, log metrics")
    def experiment():
        pass

    @experiment.command("list", help="List all experiments")
    @click.pass_context
    def experiment_list(ctx):
        r = api_get(ctx, "/experiments")
        if r.status_code != 200:
            log.error(f"Failed to list experiments: {r.text}")
            return
        data = r.json()
        exps = data.get("data", {}).get("experiments", [])
        if not exps:
            log.info("No experiments found")
            return
        if output_json(ctx, {"experiments": exps}):
            return
        log.header("Experiments")
        for exp in exps:
            log.info(f"  {exp}")

    @experiment.command("create", help="Create a new experiment")
    @click.argument("name")
    @click.pass_context
    def experiment_create(ctx, name):
        r = api_post(ctx, "/experiments", json={"name": name})
        if r.status_code != 200:
            log.error(f"Failed to create experiment: {r.text}")
            return
        data = r.json().get("data", {})
        log.success(f"Created experiment: {data.get('id', name)}")

    @experiment.command("info", help="Show experiment details")
    @click.argument("experiment_id")
    @click.pass_context
    def experiment_info(ctx, experiment_id):
        r = api_get(ctx, f"/experiments/{experiment_id}")
        if r.status_code != 200:
            log.error(f"Experiment not found: {r.text}")
            return
        data = r.json().get("data", {})
        if output_json(ctx, data):
            return
        log.header(f"Experiment: {experiment_id}")
        for k, v in data.items():
            log.key_value(k, str(v))

    @experiment.command("delete", help="Delete an experiment")
    @click.argument("experiment_id")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--dry-run", is_flag=True, help="Show what would be deleted")
    @click.pass_context
    def experiment_delete(ctx, experiment_id, yes, dry_run):
        if dry_run:
            log.info(f"Would delete experiment: {experiment_id}")
            return
        if not yes:
            confirm(f"Delete experiment '{experiment_id}'?", abort=True)
        r = api_delete(ctx, f"/experiments/{experiment_id}")
        if r.status_code == 200:
            log.success(f"Deleted experiment: {experiment_id}")
        else:
            log.error(f"Failed to delete: {r.text}")

    @experiment.command("metrics", help="Show experiment metrics")
    @click.argument("experiment_id")
    @click.pass_context
    def experiment_metrics(ctx, experiment_id):
        r = api_get(ctx, f"/experiments/{experiment_id}/data")
        if r.status_code != 200:
            log.error(f"Failed to get metrics: {r.text}")
            return
        data = r.json().get("data", {})
        if output_json(ctx, data):
            return
        log.header(f"Metrics: {experiment_id}")
        metrics = data.get("metrics", [])
        if not metrics:
            log.info("No metrics recorded yet")
            return
        for m in metrics[-20:]:
            log.info(f"  {m.get('step', '?')}: {m.get('key', '?')}={m.get('value', '?')}")

    return experiment
