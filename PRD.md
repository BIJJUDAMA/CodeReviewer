# OpenEnv Code Review Environment — Full PRD

---

## 1. Overview

**Project Name:** `openenv-code-review`
**Type:** OpenEnv-compliant real-world AI agent training environment
**Deployment Target:** Hugging Face Spaces (Docker SDK)
**Runtime Constraint:** Inference script completes in under 20 minutes on 2 vCPU / 8GB RAM

**One-line pitch:** A code review environment where an AI agent reads broken Python snippets and earns reward by correctly identifying bugs, producing fixes, and writing structured review feedback — with deterministic graders that never need an LLM to score.

---

## 2. Goals

- Pass all five pre-submission checklist items automatically
- Score 26-30/30 on real-world utility (fills a genuine gap in RL eval benchmarks)
- Produce deterministic, reproducible grader scores across any run
- Keep Docker build under 3 minutes and inference under 20 minutes
- Reuse the provided `inference.py` structure and `validate-submission.sh` without modification to their core logic

---

## 3. Non-Goals

- No multi-language support (Python only, reduces grader complexity)
- No LLM-in-the-loop grading (all graders are deterministic)
- No stateful learning across episodes (each reset is fully independent)
- No web UI (API only, HF Space serves JSON)

---

## 4. Architecture

### 4.1 Repository Structure

```
openenv-code-review/
├── inference.py                  ← adapted from provided sample, root-level mandatory
├── validate-submission.sh        ← provided script, unchanged
├── README.md
├── openenv.yaml
└── server/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py                   ← FastAPI app
    ├── env.py                    ← Core env: models + state machine
    ├── tasks/
    │   ├── __init__.py
    │   └── dataset.py            ← 18 curated snippets with ground truth
    └── graders/
        ├── __init__.py
        ├── identify_bug.py       ← Task 1 grader (easy)
        ├── suggest_fix.py        ← Task 2 grader (medium)
        └── full_review.py        ← Task 3 grader (hard)
```

### 4.2 Request Flow

```
inference.py
    │
    ├─ POST /reset  ──► env.py: reset_state() → fresh CodeReviewObservation
    │
    ├─ POST /step   ──► env.py: step(action)
    │                       │
    │                       ├─ graders/[task].py: score(action, ground_truth)
    │                       ├─ compute reward float
    │                       └─ return StepResult(observation, reward, done, info)
    │
    └─ GET  /state  ──► env.py: current_state() → raw env state dict
```

---

## 5. OpenEnv Spec Compliance

### 5.1 Pydantic Models (`env.py`)

```python
class CodeReviewObservation(BaseModel):
    code_snippet: str          # the broken Python code
    language: str              # always "python"
    task_type: str             # "identify_bug" | "suggest_fix" | "full_review"
    task_description: str      # plain English instruction for the agent
    step_number: int
    max_steps: int
    context: Optional[str]     # optional docstring or expected behavior hint

class CodeReviewAction(BaseModel):
    response: str              # agent's free-text output

class CodeReviewReward(BaseModel):
    score: float               # 0.0–1.0 normalized
    breakdown: Dict[str, float]
    feedback: str
    done: bool

class StepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any]
```

### 5.2 Endpoints (`main.py`)

| Method | Path | Behavior |
|---|---|---|
| POST | `/reset` | Picks a random task+snippet, resets state, returns `CodeReviewObservation` |
| POST | `/step` | Accepts `CodeReviewAction`, runs grader, returns `StepResult` |
| GET | `/state` | Returns current env state as dict |
| GET | `/health` | Returns `{"status": "ok"}` — needed for HF Space ping |

`/reset` responds to both `POST {}` and `POST` with no body (the validate script sends `'{}'`).

### 5.3 `openenv.yaml`

```yaml
name: code-review-env
version: 1.0.0
description: >
  An OpenEnv environment where an AI agent performs code review tasks
  on Python snippets: identifying bugs, suggesting fixes, and writing
  structured feedback.
tasks:
  - name: identify-bug
    difficulty: easy
    max_steps: 4
    reward_range: [0.0, 1.0]
  - name: suggest-fix
    difficulty: medium
    max_steps: 6
    reward_range: [0.0, 1.0]
  - name: full-review
    difficulty: hard
    max_steps: 8
    reward_range: [0.0, 1.0]
action_space:
  type: text
  description: Free-form string response from the agent
observation_space:
  type: object
  fields:
    - code_snippet: string
    - language: string
    - task_type: string
    - task_description: string
    - step_number: integer
    - max_steps: integer
    - context: string (optional)
endpoints:
  reset: POST /reset
  step: POST /step
  state: GET /state
```

---

## 6. Tasks and Graders

### 6.1 Task 1 — `identify_bug` (Easy)

**Agent objective:** Read a broken Python snippet and name the bug category.

**Max steps:** 4. Done on first correct answer. Partial credit on step 1 vs step 3.

**Ground truth format:** A bug label from a closed set:
```
off_by_one | type_error | key_error | infinite_loop |
name_error | index_error | logic_error | zero_division
```

**Grader logic (`identify_bug.py`):**

```python
def score(agent_response: str, ground_truth: dict) -> float:
    label = ground_truth["bug_type"]
    aliases = ground_truth["aliases"]      # e.g. ["off by one", "off-by-one", "fencepost"]
    response = agent_response.lower()

    if label in response or any(a in response for a in aliases):
        return 1.0
    # partial: mentions a plausibly related category
    if any(word in response for word in ground_truth["partial_match_words"]):
        return 0.4
    return 0.0
```

Step-decay: score multiplied by `(1 - 0.15 * (step - 1))` to reward faster answers.

**Example snippet:**
```python
# Bug: off_by_one
def get_last(items):
    return items[len(items)]  # should be len(items) - 1
```

---

### 6.2 Task 2 — `suggest_fix` (Medium)

**Agent objective:** Produce corrected Python code that passes hidden test cases.

**Max steps:** 6. Environment extracts code from agent response, runs it, scores by tests passed.

**Grader logic (`suggest_fix.py`):**

```python
def score(agent_response: str, ground_truth: dict) -> float:
    code = extract_code_block(agent_response)   # regex: ```python ... ``` or raw fallback
    if not code:
        return 0.0

    results = run_tests(code, ground_truth["test_cases"])
    # run_tests: subprocess.run with 5s timeout, captures stdout/stderr
    # each test_case: {"input_args": [...], "expected_output": ...}

    passed = sum(1 for r in results if r["passed"])
    return round(passed / len(results), 2)
```

`run_tests` uses `subprocess.run` in a restricted environment (no network, timeout=5s). Code is written to a tempfile, executed, and cleaned up.

**Partial reward example:** 3 of 5 tests pass → score = 0.60

---

### 6.3 Task 3 — `full_review` (Hard)

**Agent objective:** Write a structured code review covering: (a) bug identification, (b) corrected code, (c) style notes.

**Max steps:** 8. Score is a weighted sum of three sub-dimensions.

**Grader logic (`full_review.py`):**

```python
WEIGHTS = {"bug": 0.40, "fix": 0.35, "style": 0.25}

def score(agent_response: str, ground_truth: dict) -> float:
    bug_score   = identify_bug.score(agent_response, ground_truth)
    fix_score   = suggest_fix.score(agent_response, ground_truth)
    style_score = score_style_notes(agent_response, ground_truth["style_keywords"])
    
    return (
        WEIGHTS["bug"]   * bug_score   +
        WEIGHTS["fix"]   * fix_score   +
        WEIGHTS["style"] * style_score
    )

def score_style_notes(response: str, style_keywords: list[str]) -> float:
    hits = sum(1 for kw in style_keywords if kw.lower() in response.lower())
    return min(hits / len(style_keywords), 1.0)
```

`style_keywords` per snippet: things like `"magic number"`, `"naming"`, `"docstring"`, `"complexity"`, `"type hint"`.

---

## 7. Dataset (`tasks/dataset.py`)

18 hand-curated Python snippets, 6 per task difficulty level.

**Schema per entry:**
```python
{
    "id": "ob_001",
    "task_type": "identify_bug",        # which tasks use this snippet
    "code_snippet": "...",
    "bug_type": "off_by_one",
    "aliases": ["off by one", "fence post", "fencepost error"],
    "partial_match_words": ["index", "range", "boundary"],
    "test_cases": [                     # used by suggest_fix and full_review
        {"input_args": [[1, 2, 3]], "expected_output": 3},
    ],
    "style_keywords": ["magic number", "type hint"],
    "context": "Function should return the last element of a list.",
    "difficulty": "easy"
}
```

All 18 entries stored as a Python list of dicts. `dataset.py` exposes:
```python
def get_task_snippet(task_type: str) -> dict   # random.choice filtered by task_type
def get_by_id(snippet_id: str) -> dict
```

---

## 8. State Machine (`env.py`)

```
IDLE ──reset()──► RUNNING ──step() * N──► DONE
  ▲                                         │
  └──────────────reset()───────────────────┘
```

**State fields:**
```python
@dataclass
class EnvState:
    status: Literal["idle", "running", "done"]
    current_snippet: Optional[dict]
    task_type: Optional[str]
    step_number: int
    max_steps: int
    rewards: List[float]
    done: bool
```

Episode ends when:
- `step_number >= max_steps`, or
- Agent achieves score >= 0.95 (early termination reward bonus)

---

## 9. `inference.py` Adaptation

The provided sample `inference.py` uses `MyEnvV4Env` (a custom async class). We replace that with direct HTTP calls to the FastAPI server.

**Key changes from the sample:**

```python
# REMOVED
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ADDED
import requests

PING_URL = os.getenv("HF_SPACE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("CODE_REVIEW_TASK", "identify_bug")  # | suggest_fix | full_review
```

**`main()` becomes synchronous** (no asyncio needed since HTTP replaces the async env):

```python
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    result = requests.post(f"{PING_URL}/reset", json={}).json()
    observation = result  # CodeReviewObservation dict
    last_reward = 0.0
    history = []
    rewards = []
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    for step in range(1, MAX_STEPS + 1):
        message = get_model_message(client, step, observation, last_reward, history)
        
        step_result = requests.post(
            f"{PING_URL}/step",
            json={"response": message}
        ).json()
        
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
```

**System prompt for each task variant** is injected based on `TASK_NAME`:

```python
SYSTEM_PROMPTS = {
    "identify_bug": "You are a code reviewer. Identify the bug type in the snippet...",
    "suggest_fix":  "You are a code reviewer. Provide corrected Python code...",
    "full_review":  "You are a senior engineer doing a full code review...",
}
```

**stdout format is identical to the provided sample** — `[START]`, `[STEP]`, `[END]` — no deviation.

---

## 10. Docker (`server/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**`requirements.txt`:**
```
fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
requests==2.31.0
openai==1.30.0
```

No heavy ML dependencies. Build time under 90 seconds.

---

## 11. `validate-submission.sh` Compatibility

The provided script checks three things. The project satisfies all without modification:

| Check | How it passes |
|---|---|
| `POST $PING_URL/reset` returns 200 | FastAPI `/reset` always returns 200 with a valid JSON body |
| `docker build server/` succeeds | Slim Dockerfile with pinned deps, no compilation steps |
| `openenv validate` passes | `openenv.yaml` present, typed models match spec, all endpoints respond |

The script is placed at repo root, unchanged.

---

## 12. README Structure

```markdown
# Code Review Environment

## What it does
## Why it matters for agent evaluation
## Action Space
## Observation Space
## Tasks
  ### identify_bug (Easy)
  ### suggest_fix (Medium)
  ### full_review (Hard)
## Reward Function
## Setup
  ### Local (Docker)
  ### Hugging Face Spaces
## Running inference
## Baseline scores
  | Task | Model | Score |
  |------|-------|-------|
  | identify_bug | Qwen2.5-72B | 0.82 |
  | suggest_fix  | Qwen2.5-72B | 0.61 |
  | full_review  | Qwen2.5-72B | 0.54 |
## Environment variables
```

---

## 13. Evaluation Score Projection

| Criterion | Weight | Target | Rationale |
|---|---|---|---|
| Real-world utility | 30% | 27/30 | Code review is daily dev work; RL teams lack this benchmark |
| Task & grader quality | 25% | 23/25 | 3 tasks, clear difficulty, deterministic, reproducible |
| Environment design | 20% | 18/20 | Dense reward, clean Pydantic, sensible episode boundaries |
| Code quality | 15% | 14/15 | OpenEnv validate passes, Docker works, HF deploys |
| Creativity | 10% | 8/10 | Subprocess code execution in grader is novel in this space |
| **Total** | **100%** | **~90/100** | |

---

## 14. Build Order

1. `tasks/dataset.py` — 18 snippets, no dependencies
2. `graders/identify_bug.py` — depends only on dataset schema
3. `graders/suggest_fix.py` — adds subprocess runner
4. `graders/full_review.py` — composes the two above
5. `env.py` — state machine + Pydantic models, imports graders
6. `main.py` — FastAPI wrapping env.py
7. `Dockerfile` + `requirements.txt`
8. `openenv.yaml`
9. `inference.py` — adaptation of provided sample
10. `README.md`

---

**Q1: Should the subprocess code execution in `suggest_fix` write to a real tempfile (safer, more portable) or use `exec()` in a restricted namespace dict (faster but trickier to sandbox)?**

**Q2: Do you want all 18 dataset snippets hand-written now in this session, or should we generate a smaller seed set of 6 (2 per task) and add more during testing?**

**Q3: Should `inference.py` run all three tasks sequentially in one script execution (logging three episode blocks), or run a single task controlled by the `CODE_REVIEW_TASK` environment variable?**