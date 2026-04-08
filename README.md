# OpenEnv Code Review

OpenEnv Code Review is an evaluation environment designed for AI agents to perform automated code review tasks. It provides a standardized interface for agents to identify bugs, suggest fixes, and perform comprehensive code analysis across a variety of Python snippets.

## Features

- **Automated Grading**: Deterministic evaluation using test-case execution and pattern matching.
- **standardized Interface**: Implements the OpenEnv specification for Observation, Action, and Reward.
- **Multi-Mode Deployment**: Fully compliant with professional environment validation suites.
- **Production Ready**: Deployed on Hugging Face Spaces using the Docker SDK.

## Environment Configuration

Before running the agent or the environment, ensure the following environment variables are set in a `.env` file or exported to your shell:

```bash
# LLM Endpoint Configuration
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=deepseek-ai/DeepSeek-R1

# Authentication
HF_TOKEN=your_huggingface_token_here

# Local Server Connectivity (Optional)
HF_SPACE_URL=https://bijjudama-coderreviewer.hf.space
```

## Local Execution

### 1. Running the Agent
The main agent logic is contained in `inference.py`. It requires the `openai` and `python-dotenv` libraries.

```bash
python inference.py
```

### 2. Validation
To verify the environment against the OpenEnv specification, run the validation script:

```bash
bash validate-submission.sh https://bijjudama-coderreviewer.hf.space .
```

## Repository Structure

- `inference.py`: Primary agent implementation following mandatory logging formats.
- `server/`: Root directory for the environment server.
- `server/app.py`: FastAPI implementation with mandatory `main()` entry point.
- `server/Dockerfile`: Container specification for Hugging Face Spaces.
- `openenv.yaml`: Environment metadata and configuration.
- `pyproject.toml`: Project build and dependency specification.
- `uv.lock`: Dependency lockfile for reproducible builds.

## Deployment
The environment is designed to run as a Docker-based Hugging Face Space. All server logic, including graders and task datasets, is contained within the `server/` directory.
