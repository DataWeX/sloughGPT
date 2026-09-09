"""
Docker command group — container management workflows.
"""

from core.framework import click
from core.helpers import ns as _ns, docker_action


def register(cli):
    """Register docker commands with the CLI group."""

    @cli.group(help="Docker compose workflows")
    def docker():
        pass

    @docker.command("start", help="Start Docker services")
    @click.option("--gpu", is_flag=True, help="Use GPU profile")
    @click.option("--dev", is_flag=True, help="Use dev profile")
    def docker_start(gpu, dev):
        docker_action("start", _ns(gpu=gpu, dev=dev))

    @docker.command("stop", help="Stop Docker services")
    def docker_stop():
        docker_action("stop", _ns())

    @docker.command("status", help="Show Docker status")
    def docker_status():
        docker_action("status", _ns())

    @docker.command("logs", help="Show Docker logs")
    @click.argument("service", required=False)
    def docker_logs(service):
        docker_action("logs", _ns(service=service))

    @docker.command("build", help="Build Docker images")
    @click.option("--no-cache", is_flag=True, help="Build without cache")
    def docker_build(no_cache):
        docker_action("build", _ns(no_cache=no_cache))

    @docker.command("shell", help="Shell into container")
    @click.argument("service", default="api")
    def docker_shell(service):
        docker_action("shell", _ns(service=service))

    return docker
