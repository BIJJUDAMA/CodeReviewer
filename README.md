# OpenEnv Code Review Environment

## What it does
OpenEnv Code Review is an AI agent training environment designed to evaluate an agent's ability to identify bugs, suggest fixes, and provide structured code reviews for Python snippets.

## Why it matters
Code review is a fundamental skill for developers. This environment provides a deterministic, reproducible way to score agents on these tasks without requiring a "Human-in-the-loop" or "LLM-in-the-loop" grader.

## Action Space
- **Type:** Text
- **Format:** Free-form response from the agent containing bug identification or corrected code blocks.

## Observation Space
- `code_snippet`: The broken Python code.
- `language`: "python".
- `task_type`: "identify_bug" | "suggest_fix" | "full_review".
- `task_description`: Instructions for the task.
- `step_number`: Current step in the episode.
- `max_steps`: Total allowed steps.
- `context`: Optional usage hint.

## Tasks
1. **identify_bug (Easy):** Identify the bug category. (Max steps: 4)
2. **suggest_fix (Medium):** Provide corrected code that passes tests. (Max steps: 6)
3. **full_review (Hard):** Complete review (ID + Fix + Style). (Max steps: 8)

## Reward Function
- **Identify Bug:** Binary (1.0/0.0) with partial credit (0.4) for related terms. Rewards faster responses with step-decay.
- **Suggest Fix:** Based on the ratio of hidden test cases passed.
- **Full Review:** Weighted sum: 40% Bug ID, 35% Fix, 25% Style Notes.

## Setup

### Local (Docker)
1. Build the image: `docker build -t code-review-env server/`
2. Run the container: `docker run -p 7860:7860 code-review-env`

### Hugging Face Spaces
Deploy using the Docker SDK option and expose port 7860.

## Running Inference
Set your environment variables and run:
```bash
export OPEN_AI_API_KEY="your-key"
export CODE_REVIEW_TASK="full_review"
python inference.py
```

## Baseline scores
| Task | Model | Score |
|------|-------|-------|
| identify_bug | Qwen2.5-72B | 0.82 |
| suggest_fix  | Qwen2.5-72B | 0.61 |
| full_review  | Qwen2.5-72B | 0.54 |

## Environment Variables
- `HF_SPACE_URL`: URL of the environment server (Default: `http://localhost:7860`)
- `CODE_REVIEW_TASK`: Type of task to run (`identify_bug`, `suggest_fix`, `full_review`)
- `OPEN_AI_API_KEY`: API key for the model provider.
- `MODEL_NAME`: Model name to evaluate.
