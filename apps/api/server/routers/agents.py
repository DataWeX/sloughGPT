"""
Agents Router - Full CRUD for AI agent definitions with execution and orchestration.
"""

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, AsyncGenerator

from domains.api.sse_envelope import sse_event, sse_complete, sse_error

from schemas.common import raise_error, success_response, safe_audit_log, classify_and_raise

logger = logging.getLogger("slo.routers.agents")


class AgentOut(BaseModel):
    id: str
    name: str
    description: str
    instructions: str = ""
    tools: List[str] = []
    avatar: str = ""


class AgentCreate(BaseModel):
    id: str = ""
    name: str = Field(..., min_length=1)
    description: str = ""
    instructions: str = ""
    tools: List[str] = []
    avatar: str = ""


class AgentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    instructions: Optional[str] = Field(default=None, max_length=50000)
    tools: Optional[List[str]] = None
    avatar: Optional[str] = Field(default=None, max_length=500)


class ExecuteRequest(BaseModel):
    request: str = Field(..., min_length=1)
    session_id: str = ""
    user_id: str = "default"


class OrchestrateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="The goal for multi-agent orchestration")
    context: str = Field(default="", description="Additional context for the orchestrator")
    agent_ids: List[str] = Field(default_factory=list, description="Specific agent IDs to use (empty = all agents)")


class AgentsRouter:
    """Router for AI agent CRUD, execution and orchestration."""

    def __init__(self):
        self.router = APIRouter(prefix="/agents", tags=["agents"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_agents, methods=["GET"])
        self.router.add_api_route("", self.create_agent, methods=["POST"], status_code=201)
        self.router.add_api_route("/runs", self.list_runs, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}", self.get_run, methods=["GET"])
        self.router.add_api_route("/{agent_id}", self.get_agent, methods=["GET"])
        self.router.add_api_route("/{agent_id}", self.update_agent, methods=["PUT"])
        self.router.add_api_route("/{agent_id}", self.delete_agent, methods=["DELETE"])
        self.router.add_api_route("/{agent_id}/execute", self.execute_agent, methods=["POST"])
        self.router.add_api_route("/orchestrate", self.orchestrate_agents, methods=["POST"])

    def _get_system(self):
        """Get the agent system singleton."""
        from domains.agents.system import get_agent_system
        return get_agent_system()

    async def list_agents(self) -> dict:
        """List all agents stored in the agent system."""
        try:
            system = self._get_system()
            agents = await asyncio.to_thread(system.list)
            return success_response(data=[AgentOut(**a).model_dump() for a in agents])
        except Exception as e:
            classify_and_raise(e, source="agents.list")

    async def create_agent(self, req: AgentCreate) -> dict:
        try:
            """Create a new agent with the given name, description, tools, and instructions.

            Args:
                req: AgentCreate with name (required), description, instructions,
                    tools list, and avatar. If id is empty, a slug is derived from
                    the name.

            Returns:
                AgentOut with the created agent's data.

            Side effects:
                Persists the agent to the agent system store.
                Logs an audit entry for agent creation.
                Raises 409 if an agent with the same ID already exists.
            """
            agent_id = req.id or req.name.lower().replace(" ", "-").replace("_", "-")[:32]
            system = self._get_system()
            existing = await asyncio.to_thread(system.get, agent_id)
            if existing:
                raise_error("Agent ID already exists", "E_INFRA_BUSY", status_code=409)
            result = await asyncio.to_thread(
                system.create,
                agent_id=agent_id,
                name=req.name,
                description=req.description,
                instructions=req.instructions,
                tools=req.tools,
                avatar=req.avatar,
            )
            safe_audit_log("agent.create", resource=agent_id, detail=req.name, tools=list(req.tools or []))
            return success_response(data=AgentOut(**result).model_dump())

        except Exception as e:
            classify_and_raise(e, source="agents.create_agent")
    async def get_agent(self, agent_id: str) -> dict:
        """Get a specific agent by ID."""
        try:
            result = self._get_system().get(agent_id)
            if result is None:
                raise_error("Agent not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=AgentOut(**result).model_dump())
        except Exception as e:
            classify_and_raise(e, source="agents.get")

    async def update_agent(self, agent_id: str, req: AgentUpdate) -> dict:
        """Update an existing agent by ID with partial field changes."""
        try:
            system = self._get_system()
            result = await asyncio.to_thread(
                system.update,
                agent_id=agent_id,
                name=req.name,
                description=req.description,
                instructions=req.instructions,
                tools=req.tools,
                avatar=req.avatar,
            )
            if result is None:
                raise_error("Agent not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("agent.update", resource=agent_id)
            return success_response(data=AgentOut(**result).model_dump())
        except Exception as e:
            classify_and_raise(e, source="agents.update")

    async def delete_agent(self, agent_id: str) -> dict:
        """Delete an agent by its unique identifier."""
        try:
            if not await asyncio.to_thread(self._get_system().delete, agent_id):
                raise_error("Agent not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("agent.delete", resource=agent_id)
            return success_response(data={"status": "deleted"})
        except Exception as e:
            classify_and_raise(e, source="agents.delete")

    async def execute_agent(self, agent_id: str, req: ExecuteRequest) -> dict:
        """Execute an agent on a user request."""
        try:
            result = await self._get_system().execute(
                agent_id=agent_id,
                request=req.request,
                session_id=req.session_id,
                user_id=req.user_id,
            )
            if "error" in result:
                raise_error(result["error"], "E_NOT_FOUND", status_code=404)
            safe_audit_log("agent.execute", resource=agent_id, user_id=req.user_id or "", session_id=req.session_id or "")
            return result
        except Exception as e:
            classify_and_raise(e, source="agents.execute")

    # ── Orchestration ─────────────────────────────────────────────────────

    async def orchestrate_agents(self, req: OrchestrateRequest, request: Request) -> AsyncGenerator[str, None]:
        try:
            """Orchestrate multiple agents on a goal with SSE streaming.

            Streams plan → per-level task execution → composition → complete.
            Uses async HTTP for non-blocking inference calls.
            """
            from domains.agents.multi import MultiAgentOrchestrator
            from domains.agents.run_history import get_agent_run_store

            store = get_agent_run_store()

            async def event_stream() -> AsyncGenerator[str, None]:
                """event_stream."""
                run_id = None
                try:
                    orch = MultiAgentOrchestrator()
                    # Filter agents if specific IDs provided
                    if req.agent_ids:
                        filtered = {k: v for k, v in orch.agents.items() if k in req.agent_ids or v.name in req.agent_ids}
                        if filtered:
                            orch = MultiAgentOrchestrator(agents=filtered)
                    run_id = store.start(req.goal, req.context or "")
                    safe_audit_log("agent.orchestrate", resource=run_id, detail=req.goal[:200])
                    yield sse_event(
                        stream="agent-orchestrate",
                        phase="PLAN",
                        status="working",
                        data={"goal": req.goal, "run_id": run_id},
                        message="Planning orchestration...",
                    )

                    if await request.is_disconnected():
                        return

                    # Plan
                    tasks = await orch._async_plan(req.goal, req.context or "")
                    if not tasks:
                        store.fail(run_id, "Could not plan this goal")
                        yield sse_event(
                            stream="agent-orchestrate",
                            phase="PLAN",
                            status="error",
                            data={"error": "Could not plan this goal"},
                            message="Planning failed",
                        )
                        return

                    task_dicts = [t.to_dict() for t in tasks]
                    store.set_tasks(run_id, task_dicts)
                    yield sse_event(
                        stream="agent-orchestrate",
                        phase="PLAN",
                        status="success",
                        data={"tasks": task_dicts, "task_count": len(tasks), "run_id": run_id},
                        message=f"Planned {len(tasks)} subtasks",
                    )

                    if await request.is_disconnected():
                        return

                    # Execute level by level via async orchestrator
                    task_map = {t.id: t for t in tasks}
                    levels = orch._compute_levels(tasks)
                    results_ctx: dict = {}

                    yield sse_event(
                        stream="agent-orchestrate",
                        phase="EXECUTE",
                        status="working",
                        data={"levels": len(levels)},
                        message="Starting execution",
                    )

                    for level_idx, task_ids in enumerate(levels):
                        if await request.is_disconnected():
                            return

                        yield sse_event(
                            stream="agent-orchestrate",
                            phase="EXECUTE",
                            status="working",
                            data={"level": level_idx, "tasks": task_ids},
                            message=f"Executing level {level_idx + 1}/{len(levels)} ({len(task_ids)} tasks)",
                        )

                        async def run_and_yield(tid: str) -> dict:
                            """run_and_yield."""
                            task = task_map[tid]
                            task.status = "in_progress"
                            dep_context = orch._build_dep_context(task, task_map, results_ctx)
                            try:
                                result = await orch._async_run_agent(task, req.goal, dep_context)
                                task.result = result
                                task.status = "completed"
                                results_ctx[task.id] = result
                                store.set_tasks(run_id, [t.to_dict() for t in tasks])
                                return sse_event(
                                    stream="agent-orchestrate",
                                    phase="EXECUTE",
                                    status="success",
                                    data={
                                        "task_id": task.id,
                                        "agent": task.assigned_agent,
                                        "description": task.description,
                                        "result_preview": result[:200],
                                    },
                                    message=f"Completed: {task.description}",
                                )
                            except Exception as e:
                                error = str(e)
                                task.error = error
                                task.status = "failed"
                                results_ctx[task.id] = f"[error: {error}]"
                                store.set_tasks(run_id, [t.to_dict() for t in tasks])
                                return sse_event(
                                    stream="agent-orchestrate",
                                    phase="EXECUTE",
                                    status="error",
                                    data={
                                        "task_id": task.id,
                                        "agent": task.assigned_agent,
                                        "description": task.description,
                                        "error": error,
                                    },
                                    message=f"Failed: {task.description}",
                                )

                        events = await asyncio.gather(*[run_and_yield(tid) for tid in task_ids], return_exceptions=True)
                        for ev in events:
                            if isinstance(ev, str):
                                yield ev
                            elif isinstance(ev, Exception):
                                logger.warning("Task exception: %s", ev, extra={"tag": "MODEL"})

                        if await request.is_disconnected():
                            return

                    # Compose
                    yield sse_event(
                        stream="agent-orchestrate",
                        phase="COMPOSE",
                        status="working",
                        message="Composing final response...",
                    )

                    final = await orch._async_compose(req.goal, tasks)

                    store.complete(
                        run_id,
                        response=final,
                        tasks=[t.to_dict() for t in tasks],
                    )

                    yield sse_complete(
                        stream="agent-orchestrate",
                        phase="COMPLETE",
                        data={
                            "response": final,
                            "tasks": [t.to_dict() for t in tasks],
                            "completed": sum(1 for t in tasks if t.status == "completed"),
                            "failed": sum(1 for t in tasks if t.status == "failed"),
                            "run_id": run_id,
                        },
                        message="Orchestration complete",
                    )

                except Exception as e:
                    logger.exception("Orchestration error", extra={"tag": "MODEL"})
                    if run_id:
                        store.fail(run_id, str(e))
                    yield sse_error(
                        stream="agent-orchestrate",
                        phase="ERROR",
                        error=str(e),
                    )

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ── Run history ─────────────────────────────────────────────────────

        except Exception as e:
            classify_and_raise(e, source="agents.orchestrate_agents")
    async def list_runs(self, limit: int = 20) -> dict:
        try:
            """List orchestration run history, newest first."""
            from domains.agents.run_history import get_agent_run_store

            runs = await asyncio.to_thread(get_agent_run_store().list_runs, limit=max(1, min(int(limit), 200)))
            return success_response(data={"runs": runs, "count": len(runs)})

        except Exception as e:
            classify_and_raise(e, source="agents.list_runs")
    async def get_run(self, run_id: str) -> dict:
        try:
            """Return a single orchestration run record."""
            from domains.agents.run_history import get_agent_run_store

            record = await asyncio.to_thread(get_agent_run_store().get, run_id)
            if record is None:
                raise_error("Run not found", "E_NOT_FOUND", status_code=404)
            return success_response(data=record)


        except Exception as e:
            classify_and_raise(e, source="agents.get_run")
router = AgentsRouter().router
