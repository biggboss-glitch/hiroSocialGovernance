"""FastAPI server for Hugging Face Spaces deployment."""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.environment import HiroSocialEnv
from src.models import Action, ActionType, Observation

# Global environment instance
_env: Optional[HiroSocialEnv] = None


class ResetRequest(BaseModel):
    """Request body for reset endpoint."""
    task: str = "easy"
    seed: Optional[int] = None


class ActionRequest(BaseModel):
    """Request body for action/step endpoint."""
    action_type: str
    target_id: Optional[str] = None
    reason: Optional[str] = None
    content: Optional[str] = None


class StepRequest(BaseModel):
    """Request body for step endpoint."""
    action: ActionRequest


def _serialize(obj: Any) -> Any:
    """Recursively serialize objects, handling datetime and Pydantic models."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return _serialize(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _action_from_request(req: ActionRequest) -> Action:
    """Convert ActionRequest to Action model."""
    action_type = req.action_type.lower()

    # Map to ActionType
    type_mapping = {
        "remove_post": ActionType.REMOVE_POST,
        "flag_post": ActionType.FLAG_POST,
        "downrank_post": ActionType.DOWNRANK_POST,
        "boost_post": ActionType.BOOST_POST,
        "warn_user": ActionType.WARN_USER,
        "suspend_user": ActionType.SUSPEND_USER,
        "inject_counter_info": ActionType.INJECT_COUNTER_INFO,
        "no_action": ActionType.NO_ACTION
    }

    return Action(
        action_type=type_mapping.get(action_type, ActionType.NO_ACTION),
        target_id=req.target_id,
        reason=req.reason,
        content=req.content
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _env
    _env = HiroSocialEnv()
    yield
    if _env:
        _env.close()


# Create FastAPI app
app = FastAPI(
    title="Hiro: Social Governance Environment",
    description="OpenEnv environment for multi-agent social media governance",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": "hiro-social-governance"
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with basic info."""
    return {
        "name": "Hiro: Social Governance Environment",
        "version": "1.0.0",
        "description": "OpenEnv environment for social media governance",
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/tasks",
            "/grade"
        ]
    }


@app.post("/reset")
async def reset(request: ResetRequest) -> Dict[str, Any]:
    """
    Reset environment and start new episode.

    Args:
        request: Reset request with task and optional seed

    Returns:
        Initial observation
    """
    global _env

    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        observation = _env.reset(task=request.task, seed=request.seed)
        return _serialize({
            "status": "success",
            "observation": observation.model_dump(),
            "task": request.task
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """
    Execute one environment step.

    Args:
        request: Step request with action

    Returns:
        Observation, reward, done flag, and info
    """
    global _env

    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        action = _action_from_request(request.action)
        obs, reward, done, info = _env.step(action)

        return _serialize({
            "status": "success",
            "observation": obs.model_dump() if obs else None,
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        })
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """
    Get current environment state.

    Returns:
        Current state dictionary
    """
    global _env

    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        state = _env.state()
        return _serialize({
            "status": "success",
            "state": state
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """
    List available tasks.

    Returns:
        Dictionary of available tasks
    """
    global _env

    if _env is None:
        _env = HiroSocialEnv()

    try:
        tasks = _env.get_available_tasks()
        return {
            "status": "success",
            "tasks": tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")


@app.get("/grade")
async def get_grade() -> Dict[str, Any]:
    """
    Get final task grade.

    Returns:
        Grade and detailed results
    """
    global _env

    if _env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        grade = _env.get_task_grade()
        details = _env.get_detailed_results()
        return _serialize({
            "status": "success",
            "grade": grade,
            "details": details
        })
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get grade: {str(e)}")


# For local testing
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting Hiro Social Governance Environment on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
