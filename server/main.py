from fastapi import FastAPI, Request, Body
from server.env import CodeReviewEnv, CodeReviewAction
from typing import Optional

app = FastAPI(title="OpenEnv Code Review Environment")

# Global environment instance
env = CodeReviewEnv()

@app.post("/reset")
async def reset(request: Request):
    """
    Resets the environment. Handles both empty body and JSON with task_type.
    """
    data = {}
    try:
        data = await request.json()
    except:
        pass
    
    task_type = data.get("task_type")
    observation = env.reset(task_type)
    return observation

@app.post("/step")
async def step(action: CodeReviewAction):
    """
    Takes a step in the environment.
    """
    result = env.step(action)
    return result

@app.get("/state")
async def state():
    """
    Returns the current state of the environment.
    """
    return env.get_state_dict()

@app.get("/health")
async def health():
    """
    Health check for HF Spaces.
    """
    return {"status": "ok"}
