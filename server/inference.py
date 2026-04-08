import os
import asyncio
import textwrap
import requests
import json
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# Environment Variables
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME (per checklist)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "coder-reviewer-env")

# Use HF_TOKEN as the API Key for the Hugging Face Router
API_KEY = HF_TOKEN

# Target Environment Configuration
PING_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860").rstrip("/")
TASK_NAME = os.getenv("CODE_REVIEW_TASK", "identify_bug")
BENCHMARK = "code-review-env"
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.5

def get_system_prompt(task_type: str) -> str:
    """Returns a tailored system prompt based on the task type."""
    base_expert = "You are an expert Python code reviewer."
    
    if task_type == "security_audit":
        return textwrap.dedent(f"""
            {base_expert} You are a Cyber Security Professional.
            Your mission is to find critical vulnerabilities like Command Injection or XSS.
            Provide a SECURE fix using best practices (e.g., subprocess with shell=False).
            """).strip()
    
    if task_type == "performance_refactor":
        return textwrap.dedent(f"""
            {base_expert} You are a Senior Performance Engineer.
            Analyze algorithmic complexity (O(n^2), etc.) and provide an optimized refactor.
            Prioritize efficient data structures like set() and dict() for linear time complexity.
            """).strip()

    return textwrap.dedent(f"""
        {base_expert}
        You will be given a code snippet that contains a bug or stylistic issue.
        Identify the issue and provide a fix or analysis.
        
        If identify_bug: respond with EXACTLY ONE keyword (e.g. IndexError). Do NOT explain. Do NOT add periods.
        If suggest_fix/full_review: provide corrected code in a ```python ... ``` block.
        """).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Normalize action for logging (remove newlines)
    action_clean = action.replace("\n", " ")[:50] + "..." if len(action) > 50 else action.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    code = observation.get("code_snippet", "No code provided.")
    desc = observation.get("task_description", "No description.")
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Task Description: {desc}
        Code Snippet:
        ```python
        {code}
        ```
        Last Reward: {last_reward:.2f}
        Previous turns:
        {" | ".join(history[-3:]) if history else "Start of episode"}
        
        Analyze the code and provide your response.
        """
    ).strip()

def get_model_message(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    system_prompt = get_system_prompt(observation.get("task_type", ""))
    try:
        user_prompt = build_user_prompt(step, observation, last_reward, history)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "error"

class RemoteEnv:
    """Helper to interact with the remote OpenEnv Space."""
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def reset(self, task_type: str):
        resp = requests.post(f"{self.base_url}/reset", json={"task_type": task_type}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    async def step(self, action_str: str):
        # The environment expects the field name 'response'
        resp = requests.post(f"{self.base_url}/step", json={"response": action_str}, timeout=30)
        if resp.status_code != 200:
            print(f"[DEBUG] Step failed: {resp.status_code} - {resp.text}", flush=True)
        resp.raise_for_status()
        return resp.json()

async def main() -> None:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is missing.", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = RemoteEnv(PING_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Step 0: Reset
        obs = await env.reset(TASK_NAME)
        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Step 1: Get action from model
            action_text = get_model_message(client, step, obs, last_reward, history)

            # Step 2: Push action to environment
            result = await env.step(action_text)
            
            # Step 3: Extract results
            obs = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            error = result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Step 4: Log step (clean action for log)
            log_step(step=step, action=action_text, reward=reward, done=done, error=error)

            history.append(f"Q: {TASK_NAME} R: {reward}")

            if done:
                break

        # Calculate final metrics
        total_reward = sum(rewards)
        # Assuming max reward is 1.0 for these tasks
        score = min(max(total_reward, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
