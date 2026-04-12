import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from env import CodeReviewEnv, CodeReviewAction
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Using ORJSONResponse for ultra-fast serialization (Requirement 6)
app = FastAPI(title="OpenEnv Code Review Environment", default_response_class=ORJSONResponse)

@app.get("/")
async def root():
    return {
        "message": "OpenEnv Code Review Server is Running!",
        "status": "healthy",
        "documentation": "/docs"
    }

# Global environment instance
env = CodeReviewEnv()

@app.post("/reset")
async def reset(request: Request):
    """
    Resets the environment.
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

def main():
    """Entry point for the OpenEnv server."""
    # HF Spaces expects port 7860
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
