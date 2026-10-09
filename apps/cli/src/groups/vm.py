"""
VM command group — x86 Virtual Machine console and management.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register vm commands with the CLI group."""

    @cli.group(help="x86 Virtual Machine console and management")
    def vm():
        pass

    @vm.command("status", help="Show VM status and information")
    def vm_status():
        from commands.vm import cmd_vm
        cmd_vm(_ns())

    @vm.command("run", help="Run assembly code in the VM")
    @click.argument("source", required=False)
    @click.option("--file", "-f", help="File containing assembly source")
    def vm_run(source, file):
        from commands.vm import cmd_vm_run
        cmd_vm_run(_ns(source=source, file=file))

    @vm.command("list", help="List available VM programs")
    def vm_list():
        from commands.vm import cmd_vm_list
        cmd_vm_list(_ns())

    @vm.command("info", help="Show detailed VM information")
    def vm_info_cmd():
        from commands.vm import cmd_vm_info
        cmd_vm_info(_ns())

    @vm.command("debug", help="Debug assembly code interactively or from script")
    @click.argument("source", required=False)
    @click.option("--file", "-f", help="File containing assembly source")
    @click.option("--script", "-s", help="Script file with debug commands (non-interactive)")
    def vm_debug(source, file, script):
        from commands.vm import cmd_vm_debug
        cmd_vm_debug(_ns(source=source, file=file, script=script))

    return vm
