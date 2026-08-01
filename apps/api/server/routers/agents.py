"""
Agents Router - Full CRUD for AI agent definitions with execution and orchestration.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from domains.api.sse_envelope import sse_event, sse_complete, sse_error

from schemas.common import success_response

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
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[str]] = None
    avatar: Optional[str] = None


class ExecuteRequest(BaseModel):
    request: str = Field(..., min_length=1)
    session_id: str = ""
    user_id: str = "default"


class OrchestrateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="The goal for multi-agent orchestration")
    context: str = Field(default="", description="Additional context for the orchestrator")


class AgentsRouter:
    """Router for AI agent CRUD, execution and orchestration."""

    def __init__(self):
        self.router = APIRouter(prefix="/agents", tags=["agents"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_agents, methods=["GET"], response_model=List[AgentOut])
        self.router.add_api_route("", self.create_agent, methods=["POST"], response_model=AgentOut, status_code=201)
        self.router.add_api_route("/runs", self.list_runs, methods=["GET"])
        self.router.add_api_route("/runs/{run_id}", self.get_run, methods=["GET"])
        self.router.add_api_route("/{agent_id}", self.get_agent, methods=["GET"], response_model=AgentOut)
        self.router.add_api_route("/{agent_id}", self.update_agent, methods=["PUT"], response_model=AgentOut)
        self.router.add_api_route("/{agent_id}", self.delete_agent, methods=["DELETE"])
        self.router.add_api_route("/{agent_id}/execute", self.execute_agent, methods=["POST"])
        self.router.add_api_route("/orchestrate", self.orchestrate_agents, methods=["POST"])

    def _get_system(self):
        """Get the agent system singleton."""
        from domains.agents.system import get_agent_system
        return get_agent_system()

    async def list_agents(self):
        """List all available agents."""
        return [AgentOut(**a) for a in self._get_system().list()]

    async def create_agent(self, req: AgentCreate):
        """Create a new agent."""
        agent_id = req.id or req.name.lower().replace(" ", "-").replace("_", "-")[:32]
        system = self._get_system()
        if system.get(agent_id):
            raise HTTPException(status_code=409, detail="Agent ID already exists")
        result = system.create(
            agent_id=agent_id,
            name=req.name,
            description=req.description,
            instructions=req.instructions,
            tools=req.tools,
            avatar=req.avatar,
        )
        return AgentOut(**result)

    async def get_agent(self, agent_id: str):
        """Get a specific agent by ID."""
        result = self._get_system().get(agent_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentOut(**result)

    async def update_agent(self, agent_id: str, req: AgentUpdate):
        """Update an existing agent."""
        system = self._get_system()
        result = system.update(
            agent_id=agent_id,
            name=req.name,
            description=req.description,
            instructions=req.instructions,
            tools=req.tools,
            avatar=req.avatar,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentOut(**result)

    async def delete_agent(self, agent_id: str):
        """Delete an agent."""
        if not self._get_system().delete(agent_id):
            raise HTTPException(status_code=404, detail="Agent not found")
        return success_response(data={"status": "deleted"})

    async def execute_agent(self, agent_id: str, req: ExecuteRequest):
        """Execute an agent on a user request."""
        result = await self._get_system().execute(
            agent_id=agent_id,
            request=req.request,
            session_id=req.session_id,
            user_id=req.user_id,
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    # ── Orchestration ─────────────────────────────────────────────────────

    async def orchestrate_agents(self, req: OrchestrateRequest, request: Request):
        """Orchestrate multiple agents on a goal with SSE streaming.

        Streams plan → per-level task execution → composition → complete.
        Uses async HTTP for non-blocking inference calls.
        """
        from domains.agents.multi import MultiAgentOrchestrator
        from domains.agents.run_history import get_agent_run_store

        store = get_agent_run_store()

        async def event_stream():
            run_id = None
            try:
                orch = MultiAgentOrchestrator()
                run_id = store.start(req.goal, req.context or "")
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

                    async def run_and_yield(tid: str):
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

    async def list_runs(self, limit: int = 20):
        """List orchestration run history, newest first."""
        from domains.agents.run_history import get_agent_run_store

        runs = get_agent_run_store().list_runs(limit=max(1, min(int(limit), 200)))
        return {"runs": runs, "count": len(runs)}

    async def get_run(self, run_id: str):
        """Return a single orchestration run record."""
        from domains.agents.run_history import get_agent_run_store

        record = get_agent_run_store().get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return record


router = AgentsRouter().router
