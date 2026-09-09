"""
System command group — system information, health, and environment tools.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register system commands with the CLI group."""

    @cli.group(help="System information, health, and environment tools")
    def system():
        pass

    @system.command("status", help="Show live system status")
    @click.option("--watch", is_flag=True, help="Auto-refresh")
    @click.option("--interval", default=3, type=int, help="Refresh interval")
    @click.pass_context
    def system_status(ctx, watch, interval):
        from commands.system import cmd_status
        cmd_status(_ns(watch=watch, interval=interval, json_output=ctx.obj.get("json"), quiet=ctx.obj.get("quiet"),
                       timeout=ctx.obj.get("timeout", 10)))

    @system.command("info", help="Show system information")
    @click.pass_context
    def system_info(ctx):
        from commands.system import cmd_system
        cmd_system(_ns(json_output=ctx.obj.get("json")))

    @system.command("health", help="Quick API health check")
    @click.pass_context
    def system_health(ctx):
        from commands.dev import cmd_health
        args = _ns(host=ctx.obj["host"], port=ctx.obj["port"], json_output=ctx.obj.get("json"),
                   timeout=ctx.obj.get("timeout", 10))
        cmd_health(args)

    @system.command("stats", help="Show models/datasets statistics")
    @click.pass_context
    def system_stats(ctx):
        from commands.system import cmd_stats
        cmd_stats(_ns(json_output=ctx.obj.get("json")))

    @system.command("doctor", help="Run environment checks")
    @click.pass_context
    def system_doctor(ctx):
        from commands.system import cmd_config_check
        cmd_config_check(_ns(json_output=ctx.obj.get("json")))

    @system.command("config", help="Show or validate configuration")
    @click.option("--validate", "do_validate", is_flag=True, help="Validate .env file")
    @click.option("--env", default=".env", help="Dotenv file")
    @click.option("--generate", "do_generate", is_flag=True, help="Generate secrets")
    @click.option("--type", "secret_type", type=click.Choice(["api-key", "jwt-secret", "all"]), default="all")
    def system_config(do_validate, env, do_generate, secret_type):
        if do_generate:
            from commands.system import cmd_config_generate
            cmd_config_generate(_ns(type=secret_type))
        elif do_validate:
            from commands.system import cmd_config_validate
            cmd_config_validate(_ns(env=env))
        else:
            from commands.system import cmd_config_check
            cmd_config_check(_ns())

    @system.command("optimize", help="Show or apply optimization settings")
    @click.option("--apply", "do_apply", is_flag=True, help="Apply optimizations")
    def system_optimize(do_apply):
        from commands.system import cmd_optimize
        cmd_optimize(_ns(optimize=do_apply))

    @system.command("setup", help="Bootstrap environment")
    @click.option("--gpu", is_flag=True, help="GPU support")
    @click.option("--docker-only", is_flag=True, help="Docker only")
    @click.option("--local-only", is_flag=True, help="Local only")
    @click.option("--venv", default=".venv", help="Virtual env directory")
    def system_setup(gpu, docker_only, local_only, venv):
        from commands.system import cmd_setup
        args = _ns(gpu=gpu, docker_only=docker_only, local_only=local_only, venv=venv)
        cmd_setup(args)

    @system.command("api", help="Test API endpoints or authentication")
    @click.argument("action", type=click.Choice(["status", "test", "auth"]), default="status")
    @click.pass_context
    def system_api(ctx, action):
        from commands.dev import cmd_api_status, cmd_api_test, cmd_api_auth
        args = _ns(host=ctx.obj["host"], port=ctx.obj["port"])
        {
            "status": cmd_api_status,
            "test": cmd_api_test,
            "auth": cmd_api_auth,
        }[action](args)

    return system
