import os
import sys
import asyncio
import textwrap
import traceback
import httpx
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI

# Ensure local imports inside 'server' work from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "server")))
from env import CodeReviewEnv, CodeReviewAction

# ---------------------------------------------------------
# Environment Variables - PRE-FLIGHT FIXES
# ---------------------------------------------------------

# 1. API_KEY Fix
if "API_KEY" not in os.environ and "HF_TOKEN" in os.environ:
    os.environ["API_KEY"] = os.environ["HF_TOKEN"]

# 2. API_BASE_URL Fix (Ensure /v1)
if "API_BASE_URL" in os.environ:
    base_url = os.environ["API_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        os.environ["API_BASE_URL"] = base_url + "/v1"
else:
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"

# 3. MODEL_NAME Fix
if "MODEL_NAME" not in os.environ or not os.environ["MODEL_NAME"]:
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

# 4. PROXY Fix
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ---------------------------------------------------------
# Script Constants
# ---------------------------------------------------------
BENCHMARK = "code-review-env"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.4

def get_system_prompt() -> str:
    return textwrap.dedent("""
        You are an expert Python developer in an interactive terminal.
        You MUST interact using JSON commands:
        1. {"command": "RUN_TESTS", "payload": ""} -> See test failures.
        2. {"command": "EDIT_CODE", "payload": "NEW_CODE"} -> Change the code.
        3. {"command": "SUBMIT", "payload": ""} -> Final grading.
        
        Strategy: RUN_TESTS first, then EDIT_CODE, then SUBMIT.
        ALWAYS respond with valid JSON.
    """).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", " ")
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, observation: Any) -> str:
    feedback = observation.feedback if observation.feedback else "No terminal output."
    return f"Step: {step}\nCode:\n{observation.code_snippet}\nFeedback:\n{feedback}"

def get_model_message(client: OpenAI, step: int, observation: Any, history: List[Dict[str, str]]) -> str:
    user_prompt = build_user_prompt(step, observation)
    messages = [{"role": "system", "content": get_system_prompt()}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    
    completion = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"}
    )
    return (completion.choices[0].message.content or "").strip()

async def run_task(client: OpenAI, env: CodeReviewEnv, task_type: str) -> float:
    """Runs a single task interaction loop."""
    log_start(task=task_type, env=BENCHMARK, model=os.environ["MODEL_NAME"])
    
    obs = env.reset(task_type)
    history: List[Dict[str, str]] = []
    rewards: List[float] = []
    steps_taken = 0
    done = False

    for step in range(1, MAX_STEPS + 1):
        if done: break
        
        action_text = get_model_message(client, step, obs, history)
        # Store in history
        history.append({"role": "user", "content": build_user_prompt(step, obs)})
        history.append({"role": "assistant", "content": action_text})
        
        # Wrap in 'response' for validator hybrid support
        action = CodeReviewAction(response=action_text)
        
        result = env.step(action)
        obs = result.observation
        reward = float(result.reward)
        done = bool(result.done)
        error = result.info.get("error")

        rewards.append(reward)
        steps_taken = step
        
        log_step(step=step, action=action_text, reward=reward, done=done, error=error)
        if done: break

    # Final Score: max(rewards) in [0.01, 0.99]
    task_score = max(0.01, min(0.99, max(rewards) if rewards else 0.01))
    
    log_end(success=task_score >= SUCCESS_SCORE_THRESHOLD, steps=steps_taken, score=task_score, rewards=rewards)
    return task_score

async def main() -> None:
    try:
        # Use synchronous client
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
            http_client=httpx.Client()
        )
        
        env = CodeReviewEnv()
        tasks_to_run = ["identify_bug", "suggest_fix", "security_audit"]
        
        task_scores = []
        for task_type in tasks_to_run:
            score = await run_task(client, env, task_type)
            task_scores.append(score)

        avg_score = sum(task_scores) / len(task_scores)
        print(f"Final Average Score: {avg_score:.2f}", file=sys.stderr)

    except Exception as e:
        print(f"[CRITICAL ERROR] Execution failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise e

if __name__ == "__main__":
    asyncio.run(main())
