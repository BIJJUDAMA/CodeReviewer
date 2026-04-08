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

load_dotenv(override=True)

# ---------------------------------------------------------
# Exact matches to the sample script environment variables
# ---------------------------------------------------------
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "coder-reviewer-env")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CODE_REVIEW_TASK", os.getenv("TASK", "identify_bug"))
BENCHMARK = os.getenv("BENCHMARK", "code-review-env")

PING_URL = (os.getenv("HF_SPACE_URL") or "http://localhost:7860").rstrip("/")

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
    except Exception as e:
        # Print actual error to stderr so we can see why LLM failed
        print(f"[ERROR] LLM generation failed: {e}", file=sys.stderr, flush=True)
        sys.stderr.flush()
        return "error"

class RemoteEnv:
    def __init__(self, base_url: str):
        self.base_url = base_url
    async def reset(self, task_type: str):
        try:
            resp = requests.post(f"{self.base_url}/reset", json={"task_type": task_type}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] RemoteEnv reset failed for {self.base_url}: {e}", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return None
    async def step(self, action_str: str):
        try:
            resp = requests.post(f"{self.base_url}/step", json={"response": action_str}, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[ERROR] RemoteEnv step failed: {e}", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return None

async def main() -> None:
    # Always log start regardless of what happens next
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if not API_KEY:
            print("[ERROR] API_KEY not found", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return

        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        
        env = RemoteEnv(PING_URL)

        # Step 0: Reset
        obs = await env.reset(TASK_NAME)
        if not obs:
            print("[ERROR] Could not reset environment, exiting early.", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return

        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done: break
            action_text = get_model_message(client, step, obs, last_reward, history)
            result = await env.step(action_text)
            if not result or "observation" not in result: 
                print("[ERROR] Environment step returned invalid result.", file=sys.stderr, flush=True)
                sys.stderr.flush()
                break

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
        print(f"[ERROR] Unhandled exception in main: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
