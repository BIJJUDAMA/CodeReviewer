---
title: CoderReviewer
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OpenEnv Code Review

OpenEnv Code Review is an evaluation environment for AI agents to perform automated code review tasks. It provides a standardized interface for identifying vulnerabilities, optimizing performance, and providing structured feedback through deterministic AST-based grading.

## Features
- **Deterministic Grading**: Evaluation using Abstract Syntax Tree (AST) analysis for 100% reliability.
- **Security Auditing**: Specialized scenarios for detecting command injection and other vulnerabilities.
- **Performance Refactoring**: Scenarios for optimizing algorithmic complexity using efficient data structures.
- **Tiered Rewards**: Dense reward signals (+0.2 Syntax, +0.8 Functional) for improved RL feedback.

## Environment Description
The environment simulates a code review process where an agent must analyze Python snippets across various categories:
- Bug Identification: Locating logical or syntax errors.
- Security Auditing: Finding vulnerabilities such as Command Injection or XSS.
- Performance Refactoring: Identifying algorithmic bottlenecks (e.g. O(n^2) complexity).
- Full Code Review: A comprehensive analysis combining all the above.

## Action and Observation Spaces

### Observation Space
The observation is provided as a JSON object containing:
- code_snippet: The Python source code to be reviewed.
- language: The programming language (defaults to "python").
- task_type: The specific review category (e.g. "identify_bug", "security_audit").
- task_description: A plain-text instruction for the agent.
- step_number: Current step in the episode.
- max_steps: Maximum steps allowed for the task.
- context: Optional metadata providing hints or expected behavior.

### Action Space
The agent provides a free-text response:
- response: A string containing the agent's review, fix, or analysis.

### Reward Signal
Rewards are calculated per step based on the grader's evaluation of the response.
- Range: [0.0, 1.0] per step.
- Termination: The episode ends when the reward matches/exceeds 0.95 or max_steps is reached.

## Environment Configuration
Set these variables to connect to the model and environment:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=deepseek-ai/DeepSeek-R1
HF_TOKEN=your_huggingface_token
HF_SPACE_URL=your_hf_space_url
```

## Local Execution

### 1. Running the Agent
```bash
python inference.py
```

### 2. Validation
Use the Docker validator to verify OpenEnv specification compliance:
```powershell
docker build -t openenv-validator -f Dockerfile.validator .
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v "${PWD}:/repo" openenv-validator https://bijjudama-coderreviewer.hf.space .
```
Use the Docker test to verify the multi-turn capability:
```powershell
docker build -t openenv-test -f Dockerfile.test .

docker run --rm --env-file .env openenv-test
```

## Repository Structure
- `inference.py`: Agent implementation with task-aware persona logic.
- `server/`: Environment server root (HF Space source).
- `server/graders/`: Specialized AST-based evaluation engines.
- `openenv.yaml`: Task definitions and reward configurations.
