import os
import sys
import asyncio
import textwrap
import traceback
from typing import List, Optional
from openai import OpenAI

# Ensure local imports inside 'server' work from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "server")))
from env import CodeReviewEnv, CodeReviewAction


# 1. API_KEY Fix
if "API_KEY" not in os.environ and "HF_TOKEN" in os.environ:
    os.environ["API_KEY"] = os.environ["HF_TOKEN"]

# 2. API_BASE_URL Fix (Ensure /v1)
if "API_BASE_URL" in os.environ:
    base_url = os.environ["API_BASE_URL"].rstrip("/")
    if not base_url.endswith("/v1"):
        os.environ["API_BASE_URL"] = base_url + "/v1"
else:
    # Safe fallback if missing entirely (should not happen based on logs)
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"

# 3. MODEL_NAME Fix (The cause of the recent crash)
if "MODEL_NAME" not in os.environ or not os.environ["MODEL_NAME"]:
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

# ---------------------------------------------------------
# Script Constants
# ---------------------------------------------------------
TASK_NAME = os.getenv("CODE_REVIEW_TASK") or os.getenv("TASK") or "identify_bug"
BENCHMARK = os.getenv("BENCHMARK") or "code-review-env"
MODEL_NAME = os.environ["MODEL_NAME"]

MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.5

def get_system_prompt(task_type: str) -> str:
    base_expert = "You are an expert Python code reviewer."
    if task_type == "security_audit":
        return textwrap.dedent(f"{base_expert} You are a Cyber Security Professional. Find vulnerabilities like Command Injection. Provide a SECURE fix.").strip()
    if task_type == "performance_refactor":
        return textwrap.dedent(f"{base_expert} You are a Senior Performance Engineer. Optimize algorithmic complexity.").strip()
    return textwrap.dedent(f"{base_expert} Provide a fix or analysis for the bug.").strip()

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

def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    if hasattr(observation, 'code_snippet'):
        code = observation.code_snippet
        desc = observation.task_description
    else:
        code = observation.get("code_snippet", "No code provided.")
        desc = observation.get("task_description", "No description.")
    return f"Step: {step}\nTask: {desc}\nCode:\n{code}\nLast Reward: {last_reward:.2f}"

def get_model_message(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    if hasattr(observation, 'task_type'):
        task_type = observation.task_type
    else:
        task_type = observation.get("task_type", "")
        
    system_prompt = get_system_prompt(task_type)
    user_prompt = build_user_prompt(step, observation, last_reward, history)
    
    # Using the strict MODEL_NAME from environment
    completion = client.chat.completions.create(
        model=os.environ["MODEL_NAME"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()

async def main() -> None:
    # Explicitly log environment to stderr so we can debug if it explodes
    print(f"DEBUG: Using API_BASE_URL={os.environ.get('API_BASE_URL')}", file=sys.stderr)
    print(f"DEBUG: Using API_KEY PRESENT: {'API_KEY' in os.environ}", file=sys.stderr)
    print(f"DEBUG: Using MODEL_NAME={os.environ.get('MODEL_NAME')}", file=sys.stderr)

    log_start(task=TASK_NAME, env=BENCHMARK, model=os.environ["MODEL_NAME"])
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # MANDATORY: THE EXACT LITERAL STRING THEY ASK FOR
        # base_url and api_key are read from os.environ which we fixed in Pre-flight.
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        env = CodeReviewEnv()
        obs = env.reset(TASK_NAME)
        if not obs:
            print("[ERROR] Environment reset failed.", file=sys.stderr, flush=True)
            return

        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break
            
            action_text = get_model_message(client, step, obs, last_reward, history)
            action = CodeReviewAction(response=action_text)
            
            result = env.step(action)
            obs = result.observation
            reward = float(result.reward)
            done = bool(result.done)
            error = result.info.get("error") if hasattr(result, 'info') else None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            
            log_step(step=step, action=action_text, reward=reward, done=done, error=error)
            if done: break

        total_reward = sum(rewards)
        score = min(max(total_reward, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[CRITICAL ERROR] Execution failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise e
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
