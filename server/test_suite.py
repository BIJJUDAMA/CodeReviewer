import os
import requests
import asyncio
import textwrap
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# Configuration from Environment
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1").strip()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860").rstrip("/")
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

def get_system_prompt(task_type: str) -> str:
    base_expert = "You are an expert Python code reviewer."
    if task_type == "security_audit":
        return f"{base_expert} You are a Cyber Security Professional. Find vulnerabilities like Command Injection. Provide a SECURE fix using subprocess.run with shell=False. Never use os.system."
    if task_type == "performance_refactor":
        return f"{base_expert} You are a Senior Performance Engineer. Optimize O(n^2) bottlenecks to O(n) using set() or dict()."
    return f"{base_expert} Identify the issue and provide a fix. Use ```python ... ``` for code blocks."

def get_model_response(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error: {e}"

async def run_episode(client: OpenAI, task_type: str) -> float:
    print(f"\n[EPISODE START] Task: {task_type}")
    try:
        # 1. Reset Environment
        resp = requests.post(f"{HF_SPACE_URL}/reset", json={"task_type": task_type}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        
        history = [{"role": "system", "content": get_system_prompt(task_type)}]
        total_reward = 0.01 # Start at 0.01 to be in (0, 1)
        step = 1
        done = False
        
        while not done:
            code = obs.get("code_snippet", "")
            desc = obs.get("task_description", "")
            
            # Construct user message
            user_msg = f"Step {step}/{obs.get('max_steps')}\nCode:\n```python\n{code}\n```\nTask: {desc}"
            history.append({"role": "user", "content": user_msg})
            
            # Get Agent Response
            response = get_model_response(client, history)
            history.append({"role": "assistant", "content": response})
            
            if VERBOSE:
                print(f"\n--- STEP {step} AGENT RESPONSE ---")
                # Indent response for readability
                print(textwrap.indent(response, "  "))
            
            # Step in Environment
            resp = requests.post(f"{HF_SPACE_URL}/step", json={"response": response}, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            
            obs = result.get("observation", {})
            reward = float(result.get("reward", 0.01))
            done = result.get("done", False)
            total_reward = max(0.01, min(0.99, reward)) # Ensure strictly in (0, 1)
            
            print(f"  Step {step} Reward: {reward:.2f}")
            
            step += 1
            if step > obs.get("max_steps", 5):
                break
                
        print(f"[EPISODE END] Final Reward: {total_reward:.2f}")
        return total_reward
        
    except Exception as e:
        print(f"  [CRITICAL ERROR]: {e}")
        return 0.01 # Strictly in (0, 1)

async def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is missing in .env")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = ["identify_bug", "suggest_fix", "full_review", "security_audit", "performance_refactor"]
    
    print("="*60)
    print(f"Starting Multi-Turn Batch Test Suite")
    print(f"Target: {HF_SPACE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    results = {}
    for task in tasks:
        final_reward = await run_episode(client, task)
        results[task] = final_reward
    
    print("\n" + "="*40)
    print(f"{'TASK TYPE':<25} | {'REWARD':<10}")
    print("-" * 40)
    for task, reward in results.items():
        print(f"{task:<25} | {reward:<10.2f}")
    print("="*40)

if __name__ == "__main__":
    asyncio.run(main())
