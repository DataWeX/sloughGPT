"""
Build command group — Buildroot image building for v86 browser VM.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register build commands with the CLI group."""

    @cli.group(help="Buildroot image building for v86 browser VM")
    def build():
        pass

    @build.command("run", help="Build a Buildroot image")
    def build_run():
        from commands.build import cmd_build
        cmd_build(_ns())

    @build.command("init", help="Initialize Buildroot build environment")
    @click.option("--clean", is_flag=True, help="Clean first, then set up")
    def build_init(clean):
        from commands.build import cmd_build_init
        cmd_build_init(_ns(clean=clean))

    @build.command("clean", help="Clean build output")
    def build_clean():
        from commands.build import cmd_build_clean
        cmd_build_clean(_ns())

    @build.command("status", help="Show build status")
    def build_status():
        from commands.build import cmd_build_status
        cmd_build_status(_ns())

    @build.command("install", help="Install image to web public directory")
    @click.argument("image", required=False, default="sloughgpt-rootfs.img")
    def build_install(image):
        from commands.build import cmd_build_install
        cmd_build_install(_ns(image=image))

    return build
