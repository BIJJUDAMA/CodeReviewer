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

# Do NOT override environment variables with .env - validator injects natively!
load_dotenv(override=False)

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

def get_model_message(client: OpenAI, model_name: str, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    system_prompt = get_system_prompt(observation.get("task_type", ""))
    try:
        user_prompt = build_user_prompt(step, observation, last_reward, history)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return "error"

class RemoteEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url
    async def reset(self, task_type: str):
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_type": task_type}, timeout=30)
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            return None
    async def step(self, action_str: str):
        try:
            resp = requests.post(f"{self.base_url}/step", json={"response": action_str}, timeout=30)
            return resp.json() if resp.status_code == 200 else None
        except Exception as e:
            return None

async def main() -> None:
    # Resolve target environment dynamically at runtime
    task_name = os.environ.get("CODE_REVIEW_TASK", os.environ.get("MY_ENV_V4_TASK", os.environ.get("TASK", "identify_bug")))
    benchmark = os.environ.get("BENCHMARK", "code-review-env")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    
    # 1. Start Logging Immediately 
    log_start(task=task_name, env=benchmark, model=model_name)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # EXACTLY match the validator's LLM Proxy injection
        base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        if not base_url:
            base_url = "https://router.huggingface.co/v1"
            
        api_key = os.environ.get("API_KEY", os.environ.get("HF_TOKEN"))
        if not api_key:
            return # Must silently exit and log_end

        # Initialize the OpenAI client precisely using the environment map
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        ping_url = (os.environ.get("HF_SPACE_URL") or "http://localhost:7860").rstrip("/")
        env = RemoteEnv(ping_url)

        # Step 0: Reset
        obs = await env.reset(task_name)
        if not obs:
            return

        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break
            action_text = get_model_message(client, model_name, step, obs, last_reward, history)
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

    except Exception as e:
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
