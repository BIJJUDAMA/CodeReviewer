import os
import asyncio
import textwrap
import requests
import json
import sys
import traceback
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# Environment Variables - PRIORITIZE validator variables strictly
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# Other environment variables
PING_URL = (os.getenv("HF_SPACE_URL") or "http://localhost:7860").rstrip("/")
TASK_NAME = os.getenv("CODE_REVIEW_TASK") or os.getenv("MY_ENV_V4_TASK") or os.getenv("TASK") or "identify_bug"
BENCHMARK = os.getenv("BENCHMARK") or "code-review-env"

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
    sys.stdout.flush()

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ")
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)
    sys.stdout.flush()

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
    sys.stdout.flush()

def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    code = observation.get("code_snippet", "No code provided.")
    desc = observation.get("task_description", "No description.")
    return f"Step: {step}\nTask: {desc}\nCode:\n{code}\nLast Reward: {last_reward:.2f}"

def get_model_message(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    system_prompt = get_system_prompt(observation.get("task_type", ""))
    try:
        user_prompt = build_user_prompt(step, observation, last_reward, history)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except:
        return "error"

class RemoteEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url
    async def reset(self, task_type: str):
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_type": task_type}, timeout=30)
            return resp.json() if resp.status_code == 200 else None
        except:
            return None
    async def step(self, action_str: str):
        try:
            resp = requests.post(f"{self.base_url}/step", json={"response": action_str}, timeout=30)
            return resp.json() if resp.status_code == 200 else None
        except:
            return None

async def main() -> None:
    # 1. Start Logging Immediately
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if not API_KEY:
            # We must log end even if we fail early
            return

        # Initialize client with strict environment variables
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        env = RemoteEnv(PING_URL)

        # Step 0: Reset
        obs = await env.reset(TASK_NAME)
        if not obs:
            return

        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break
            action_text = get_model_message(client, step, obs, last_reward, history)
            result = await env.step(action_text)
            if not result or "observation" not in result: break

            obs = result["observation"]
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error = result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward
            log_step(step=step, action=action_text, reward=reward, done=done, error=error)
            if done: break

        total_reward = sum(rewards)
        score = min(max(total_reward, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except:
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
