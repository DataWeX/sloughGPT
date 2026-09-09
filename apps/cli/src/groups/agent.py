"""
Agent command group — manage and execute AI agents.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register agent commands with the CLI group."""

    @cli.group(help="Manage and execute AI agents")
    def agent():
        pass

    @agent.command("list", help="List all agents")
    @click.pass_context
    def agent_list(ctx):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.get(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents", timeout=timeout)
        if r.status_code != 200:
            log.error(f"Failed to list agents: {r.text}")
            sys.exit(1)
        data = r.json()
        agents = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(agents, dict):
            agents = agents.get("agents", [])
        if not agents:
            log.info("No agents found")
            return
        if ctx.obj.get("json"):
            _output(ctx, {"agents": agents})
        else:
            log.header("Agents")
            for a in agents:
                name = a.get("name", a.get("id", "?"))
                desc = a.get("description", "")[:60]
                log.info(f"  {name} — {desc}")

    @agent.command("create", help="Create a new agent")
    @click.argument("name")
    @click.option("--description", "-d", default="", help="Agent description")
    @click.option("--instructions", "-i", default="", help="System instructions")
    @click.pass_context
    def agent_create(ctx, name, description, instructions):
        import requests
        timeout = ctx.obj.get("timeout", 10)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents",
                          json={"name": name, "description": description, "instructions": instructions},
                          timeout=timeout)
        if r.status_code != 200:
            log.error(f"Failed to create agent: {r.text}")
            sys.exit(1)
        log.success(f"Created agent: {name}")

    @agent.command("execute", help="Execute a task with an agent")
    @click.argument("agent_id")
    @click.argument("request")
    @click.pass_context
    def agent_execute(ctx, agent_id, request):
        import requests
        timeout = ctx.obj.get("timeout", 30)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/{agent_id}/execute",
                          json={"request": request}, timeout=timeout)
        if r.status_code != 200:
            log.error(f"Execution failed: {r.text}")
            sys.exit(1)
        data = r.json()
        if ctx.obj.get("json"):
            _output(ctx, data)
        else:
            result = data.get("data", data).get("result", str(data))
            log.info(result)

    @agent.command("orchestrate", help="Multi-agent orchestration")
    @click.argument("goal")
    @click.option("--context", "-c", default="", help="Additional context")
    @click.option("--agents", default="", help="Comma-separated agent IDs")
    @click.pass_context
    def agent_orchestrate(ctx, goal, context, agents):
        import requests
        agent_ids = [a.strip() for a in agents.split(",") if a.strip()] if agents else []
        timeout = ctx.obj.get("timeout", 60)
        r = requests.post(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/orchestrate",
                          json={"goal": goal, "context": context, "agent_ids": agent_ids},
                          timeout=timeout)
        if r.status_code != 200:
            log.error(f"Orchestration failed: {r.text}")
            sys.exit(1)
        data = r.json()
        if ctx.obj.get("json"):
            _output(ctx, data)
        else:
            result = data.get("data", data).get("result", str(data))
            log.info(result)

    @agent.command("delete", help="Delete an agent")
    @click.argument("agent_id")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--dry-run", is_flag=True, help="Show what would be deleted")
    @click.pass_context
    def agent_delete(ctx, agent_id, yes, dry_run):
        import requests
        if dry_run:
            log.info(f"Would delete agent: {agent_id}")
            return
        if not yes:
            confirm(f"Delete agent '{agent_id}'?", abort=True)
        timeout = ctx.obj.get("timeout", 10)
        r = requests.delete(f"http://{ctx.obj['host']}:{ctx.obj['port']}/agents/{agent_id}", timeout=timeout)
        if r.status_code == 200:
            log.success(f"Deleted agent: {agent_id}")
        else:
            log.error(f"Failed to delete: {r.text}")
            sys.exit(1)

    return agent
