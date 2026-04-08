import os
import sys
import asyncio
import textwrap
import traceback
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Ensure local imports inside 'server' work from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "server")))
from env import CodeReviewEnv, CodeReviewAction

load_dotenv(override=True)

if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"
if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = os.getenv("HF_TOKEN", "")
if "MODEL_NAME" not in os.environ:
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

# Retrieve for other uses if needed
MODEL_NAME = os.environ["MODEL_NAME"]
TASK_NAME = os.getenv("CODE_REVIEW_TASK", os.getenv("TASK", "identify_bug"))
BENCHMARK = os.getenv("BENCHMARK", "code-review-env")

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
    action_clean = action.replace("\n", " ")
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    # Handle both old dict observation and pydantic observation
    if hasattr(observation, 'code_snippet'):
        code = observation.code_snippet
        desc = observation.task_description
        task_type = observation.task_type
    else:
        code = observation.get("code_snippet", "No code provided.")
        desc = observation.get("task_description", "No description.")
        task_type = observation.get("task_type", "")
        
    return f"Step: {step}\nTask: {desc}\nCode:\n{code}\nLast Reward: {last_reward:.2f}"

def get_model_message(client: OpenAI, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    if hasattr(observation, 'task_type'):
        task_type = observation.task_type
    else:
        task_type = observation.get("task_type", "")
        
    system_prompt = get_system_prompt(task_type)
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
        print(f"[ERROR] LLM generation failed: {e}", file=sys.stderr, flush=True)
        return "error"

async def main() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # STRICT ADHERENCE TO VALIDATOR INSTRUCTIONS
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        
        # Instantiate environment LOCALLY instead of using Remote HTTP calls
        # This prevents "runs completed successfully but no API calls" due to localhost not running
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
            
            if not result or not hasattr(result, 'observation'):
                print("[ERROR] Environment step returned invalid result.", file=sys.stderr, flush=True)
                break

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
        print(f"[ERROR] Unhandled exception in main: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
