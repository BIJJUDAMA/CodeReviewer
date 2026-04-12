import os
import sys
import asyncio
import textwrap
import traceback
import httpx
import json
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI # Using Async client for concurrency

# Ensure local imports inside 'server' work from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "server")))
from env import CodeReviewEnv, CodeReviewAction

# ---------------------------------------------------------
# Environment Variables - PRE-FLIGHT FIXES
# ---------------------------------------------------------

if "API_KEY" not in os.environ and "HF_TOKEN" in os.environ:
    os.environ["API_KEY"] = os.environ["HF_TOKEN"]

if "API_BASE_URL" in os.environ:
    base_url = os.environ["API_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        os.environ["API_BASE_URL"] = base_url + "/v1"
else:
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"

if "MODEL_NAME" not in os.environ or not os.environ["MODEL_NAME"]:
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

# Clean proxies to avoid connection errors in certain environments
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ---------------------------------------------------------
# Script Constants
# ---------------------------------------------------------
BENCHMARK = "code-review-env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.4

def get_system_prompt() -> str:
    return textwrap.dedent("""
        You are an expert Python code reviewer and developer.
        You are in an interactive debugging terminal.
        You MUST interact with the environment using JSON commands:
        
        1. {"command": "RUN_TESTS", "payload": ""} -> Executes current code and returns logs.
        2. {"command": "EDIT_CODE", "payload": "NEW_CODE"} -> Replaces the current code.
        3. {"command": "SUBMIT", "payload": ""} -> Final submission for grading.
        
        Strategy:
        - First, RUN_TESTS to see the bug.
        - Then, EDIT_CODE to apply a fix.
        - RUN_TESTS again to verify.
        - When all tests pass, SUBMIT.
        
        ALWAYS respond with a valid JSON object.
    """).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Clean action for single-line logging
    action_str = json.dumps(action).replace("\n", " ")
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, observation: Any) -> str:
    # Use qualitative feedback instead of raw reward
    feedback = observation.feedback if observation.feedback else "No terminal output yet."
    return textwrap.dedent(f"""
        Step: {step}
        Current Code:
        {observation.code_snippet}
        
        Terminal Output:
        {feedback}
        
        Next Action (JSON):
    """).strip()

async def get_model_action(client: AsyncOpenAI, step: int, observation: Any, history: List[Dict[str, str]]) -> Dict[str, Any]:
    # Temperature Annealing: High (0.9) at start, Low (0.3) at end
    temp = max(0.3, 0.9 - (step - 1) * 0.1)
    
    user_msg = build_user_prompt(step, observation)
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    
    completion = await client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=messages,
        temperature=temp,
        response_format={"type": "json_object"} # Force JSON
    )
    
    content = completion.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except:
        # Fallback if model fails JSON
        return {"command": "RUN_TESTS", "payload": ""}

async def run_task(client: AsyncOpenAI, env_factory: Any, task_type: str) -> float:
    """Runs a single task interaction loop and returns the final score."""
    log_start(task=task_type, env=BENCHMARK, model=os.environ["MODEL_NAME"])
    
    env = env_factory()
    obs = env.reset(task_type)
    
    history: List[Dict[str, str]] = []
    rewards: List[float] = []
    steps_taken = 0
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done: break
        
        action_json = await get_model_action(client, step, obs, history)
        
        # Build multi-turn memory
        history.append({"role": "user", "content": build_user_prompt(step, obs)})
        history.append({"role": "assistant", "content": json.dumps(action_json)})
        
        action = CodeReviewAction(command=action_json.get("command", "RUN_TESTS"), payload=action_json.get("payload", ""))
        
        result = env.step(action)
        obs = result.observation
        reward = float(result.reward)
        done = bool(result.done)
        error = result.info.get("error")

        rewards.append(reward)
        steps_taken = step
        
        log_step(step=step, action=action_json, reward=reward, done=done, error=error)
        if done: break

    # Episode Score: Use max(rewards) for RL convergence goal (Requirement 9)
    task_score = max(rewards) if rewards else 0.0
    
    log_end(success=task_score >= SUCCESS_SCORE_THRESHOLD, steps=steps_taken, score=task_score, rewards=rewards)
    return task_score

async def main() -> None:
    try:
        client = AsyncOpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
            http_client=httpx.AsyncClient()
        )
        
        # Concurrent Evaluation (Requirement 11)
        tasks_to_run = ["identify_bug", "suggest_fix", "security_audit", "performance_refactor", "full_review"]
        
        # Using env factory to ensure isolated states during concurrency
        def env_factory(): return CodeReviewEnv()
        
        # Gather all tasks concurrently to finish under 20 mins
        task_scores = await asyncio.gather(*(run_task(client, env_factory, t) for t in tasks_to_run))

        avg_score = sum(task_scores) / len(task_scores)
        print(f"Final Evaluation Average Score: {avg_score:.2f}", file=sys.stderr)

    except Exception as e:
        print(f"[CRITICAL ERROR] Execution failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise e

if __name__ == "__main__":
    asyncio.run(main())
