"""
Personality Router - AI personality management
"""
from fastapi import APIRouter
from typing import Optional

router = APIRouter(tags=["personalities"])


@router.get("/personalities")
async def list_personalities():
    """List available AI personalities"""
    return {
        "personalities": [
            {"name": "Aria", "description": "Default SloughGPT personality"},
            {"name": "Sage", "description": "Research-focused personality"},
            {"name": "Companion", "description": "Friendly casual personality"},
        ]
    }