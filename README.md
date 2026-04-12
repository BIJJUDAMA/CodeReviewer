---
title: CoderReviewer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OpenEnv Code Review

OpenEnv Code Review is an evaluation and training environment for AI agents to perform automated code review tasks. It provides a state-of-the-art, multi-turn Markov Decision Process (MDP) for identifying vulnerabilities, optimizing performance, and providing structured feedback through deterministic AST-based grading and interactive terminal simulation.

## Features
- **Deterministic Grading**: Evaluation using Abstract Syntax Tree (AST) analysis for 100% reliability.
- **Terminal Simulation**: Interactive environment supporting real-world tool use (RFC 003).
- **Security Auditing**: Specialized scenarios for SQL injection, command injection, and more.
- **Performance Refactoring**: Scenarios for optimizing algorithmic complexity (e.g. O(n^2) to O(n)).
- **Dense Rewards**: Wordle-style intermediate signals (+0.1 Investigation, +0.2 Syntax) for effective RL training.

## Environment Description
The environment simulates a professional developer's debugging workflow. Instead of a single guess, agents use an interactive terminal to:
- **Investigate**: Run hidden test suites to observe failure logs.
- **Iterate**: Edit code and check syntax in real-time.
- **Verify**: Re-run tests to confirm the fix before final submission.

## Action and Observation Spaces

### Observation Space
The observation is provided as a JSON object containing:
- `code_snippet`: The current state of the editor.
- `language`: The programming language (defaults to "python").
- `task_type`: The specific category (e.g. "security_audit", "performance_refactor").
- `task_description`: Instructions for the agent.
- `step_number`: Current step in the episode.
- `max_steps`: Maximum allowed steps.
- `context`: Optional metadata or expected behavior hints.
- `feedback`: **Terminal Output** from the previous action (e.g. AssertionError, SyntaxError).

### Action Space
The agent provides a response string. For advanced multi-turn interaction, it responds with structured JSON:
- `{"command": "RUN_TESTS", "payload": ""}`
- `{"command": "EDIT_CODE", "payload": "def new_fn()..."}`
- `{"command": "SUBMIT", "payload": ""}`

### Reward Signal
Rewards are calculated per step based on tool-use and functional correctness.
- **Range**: strictly [0.01, 0.99] per step.
- **Dense Signal**: +0.1 for first investigation, +0.2 for syntax-correct edits, +0.7 for resolution.
- **Termination**: The episode ends on `SUBMIT` or when `max_steps` is reached.

## Environment Configuration
Set these variables to connect to the model and environment:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token
HF_SPACE_URL=your_hf_space_url
```

## Local Execution

### 1. Running the Baseline Agent
Runs the multi-turn concurrent evaluation (< 2 mins total):
```bash
python inference.py
```

### 2. SOTA RL Training (Showcase)
Demonstrates how to hook the environment into `trl.GRPOTrainer`:
```bash
python train.py
```

### 3. Validation
```powershell
docker build -t openenv-validator -f Dockerfile.validator .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "${PWD}:/repo" openenv-validator https://bijjudama-coderreviewer.hf.space .
```

## Repository Structure
- `inference.py`: High-performance multi-turn baseline with tool-use logic.
- `train.py`: SOTA RL training pipeline (GRPO showcase).
- `server/`: Environment server root (HF Space source).
- `server/graders/`: Specialized AST-based evaluation engines.
- `openenv.yaml`: Task definitions and reward configurations.
