import os
import requests
import json
from openai import OpenAI
from typing import List, Dict, Any

# Environment Variables
API_BASE_URL = os.getenv("OPEN_AI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("OPEN_AI_API_KEY", "your-api-key")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
PING_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("CODE_REVIEW_TASK", "identify_bug")

# OpenEnv Constants
BENCHMARK = "code-review-env"
SUCCESS_SCORE_THRESHOLD = 0.85
MAX_STEPS = 8 # Safety max

SYSTEM_PROMPTS = {
    "identify_bug": "You are a code reviewer. Identify the bug type in the snippet. Choose from: off_by_one, type_error, key_error, infinite_loop, name_error, index_error, logic_error, zero_division. Reply only with the bug name or a very short explanation.",
    "suggest_fix":  "You are a code reviewer. Provide corrected Python code that fixes the bug in the provided snippet. Wrap your code in ```python ... ``` blocks.",
    "full_review":  "You are a senior engineer doing a full code review. Identify the bug, provide a fix in a python code block, and list style improvements (variable naming, docstrings, etc.)."
}

def log_start(task: str, env: str, model: str):
    print(f"[START] Task: {task} | Env: {env} | Model: {model}")

def log_step(step: int, action: str, reward: float, done: bool, error: Any = None):
    status = "DONE" if done else "RUNNING"
    print(f"[STEP] {step} | Reward: {reward:.2f} | Status: {status}")
    if error:
        print(f"Error: {error}")

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    status = "SUCCESS" if success else "FAILURE"
    print(f"[END] Result: {status} | Steps: {steps} | Mean Reward: {score:.2f}")
    print(f"Reward Trace: {rewards}")

def get_model_message(client, step, observation, last_reward, history):
    prompt = f"""
Step: {step}
Task: {observation['task_description']}
Snippet:
{observation['code_snippet']}

Context: {observation.get('context', 'N/A')}
Last Reward: {last_reward}
History: {', '.join(history)}

Provide your response according to the task type.
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS.get(observation['task_type'], "You are a helpful assistant.")},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize Environment
    try:
        response = requests.post(f"{PING_URL}/reset", json={"task_type": TASK_NAME})
        response.raise_for_status()
        observation = response.json()
    except Exception as e:
        print(f"Failed to connect to environment at {PING_URL}: {e}")
        return

    last_reward = 0.0
    history = []
    rewards = []
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    # Episode Loop
    for step in range(1, observation['max_steps'] + 1):
        message = get_model_message(client, step, observation, last_reward, history)
        
        try:
            step_resp = requests.post(
                f"{PING_URL}/step",
                json={"response": message}
            )
            step_resp.raise_for_status()
            step_result = step_resp.json()
        except Exception as e:
            print(f"Error during step {step}: {e}")
            break
        
        reward = step_result["reward"]
        done = step_result["done"]
        observation = step_result["observation"]
        error = step_result["info"].get("error")
        
        rewards.append(reward)
        log_step(step=step, action=message, reward=reward, done=done, error=error)
        history.append(f"Step {step}: reward={reward:.2f}")
        last_reward = reward
        
        if done:
            break
    
    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

if __name__ == "__main__":
    main()
