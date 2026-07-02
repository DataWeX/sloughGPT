"""
Agents Router - Full CRUD for AI agent definitions with execution and orchestration.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from domains.api.sse_envelope import sse_event, sse_complete, sse_error

logger = logging.getLogger("man.routers.agents")

router = APIRouter(prefix="/agents", tags=["agents"])


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


def _get_system():
    from domains.agents.system import get_agent_system
    return get_agent_system()


@router.get("", response_model=List[AgentOut])
async def list_agents():
    """List all available agents."""
    return [AgentOut(**a) for a in _get_system().list()]


@router.post("", response_model=AgentOut, status_code=201)
async def create_agent(req: AgentCreate):
    """Create a new agent."""
    agent_id = req.id or req.name.lower().replace(" ", "-").replace("_", "-")[:32]
    system = _get_system()
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


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(agent_id: str):
    """Get a specific agent by ID."""
    result = _get_system().get(agent_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return AgentOut(**result)


@router.put("/{agent_id}", response_model=AgentOut)
async def update_agent(agent_id: str, req: AgentUpdate):
    """Update an existing agent."""
    system = _get_system()
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


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if not _get_system().delete(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted"}


@router.post("/{agent_id}/execute")
async def execute_agent(agent_id: str, req: ExecuteRequest):
    """Execute an agent on a user request."""
    result = await _get_system().execute(
        agent_id=agent_id,
        request=req.request,
        session_id=req.session_id,
        user_id=req.user_id,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Orchestration ─────────────────────────────────────────────────────


class OrchestrateRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="The goal for multi-agent orchestration")
    context: str = Field(default="", description="Additional context for the orchestrator")


@router.post("/orchestrate")
async def orchestrate_agents(req: OrchestrateRequest, request: Request):
    """Orchestrate multiple agents on a goal with SSE streaming.

    Streams plan → per-level task execution → composition → complete.
    Uses async HTTP for non-blocking inference calls.
    """
    from domains.agents.multi import MultiAgentOrchestrator

    async def event_stream():
        try:
            orch = MultiAgentOrchestrator()
            yield sse_event(
                stream="agent-orchestrate",
                phase="PLAN",
                status="working",
                data={"goal": req.goal},
                message="Planning orchestration...",
            )

            if await request.is_disconnected():
                return

            # Plan
            tasks = await orch._async_plan(req.goal, req.context or "")
            if not tasks:
                yield sse_event(
                    stream="agent-orchestrate",
                    phase="PLAN",
                    status="error",
                    data={"error": "Could not plan this goal"},
                    message="Planning failed",
                )
                return

            task_dicts = [t.to_dict() for t in tasks]
            yield sse_event(
                stream="agent-orchestrate",
                phase="PLAN",
                status="success",
                data={"tasks": task_dicts, "task_count": len(tasks)},
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
                        logger.warning("Task exception: %s", ev)

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

            yield sse_complete(
                stream="agent-orchestrate",
                phase="COMPLETE",
                data={
                    "response": final,
                    "tasks": [t.to_dict() for t in tasks],
                    "completed": sum(1 for t in tasks if t.status == "completed"),
                    "failed": sum(1 for t in tasks if t.status == "failed"),
                },
                message="Orchestration complete",
            )

        except Exception as e:
            logger.exception("Orchestration error")
            yield sse_error(
                stream="agent-orchestrate",
                phase="ERROR",
                error=str(e),
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
