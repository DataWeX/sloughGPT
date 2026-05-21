"""
Agents Router - Full CRUD for AI agent definitions with execution.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

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
