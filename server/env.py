from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel
from dataclasses import dataclass, field
from tasks import dataset
from graders import identify_bug, suggest_fix, full_review, security_audit, performance_refactor
import random

# Pydantic Models for OpenEnv Compliance

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

class StepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any]

@dataclass
class EnvState:
    status: Literal["idle", "running", "done"] = "idle"
    current_snippet: Optional[dict] = None
    task_type: Optional[str] = None
    step_number: int = 0
    max_steps: int = 0
    rewards: List[float] = field(default_factory=list)
    done: bool = False

class CodeReviewEnv:
    def __init__(self):
        self.state = EnvState()

    def reset(self, task_type: Optional[str] = None) -> CodeReviewObservation:
        # If no task_type provided, pick a random one
        if not task_type:
            task_type = random.choice(["identify_bug", "suggest_fix", "full_review"])
        
        snippet = dataset.get_task_snippet(task_type)
        
        # Max steps logic from openenv.yaml
        max_steps_map = {
            "identify_bug": 4,
            "suggest_fix": 6,
            "full_review": 8,
            "security_audit": 6,
            "performance_refactor": 6
        }
        
        self.state = EnvState(
            status="running",
            current_snippet=snippet,
            task_type=task_type,
            step_number=1,
            max_steps=max_steps_map.get(task_type, 5),
            rewards=[],
            done=False
        )
        
        return self._get_observation()

    def step(self, action: CodeReviewAction) -> StepResult:
        if self.state.status != "running":
            raise ValueError("Environment not in running state. Call reset() first.")

        # Determine which grader to use
        grader_map = {
            "identify_bug": identify_bug,
            "suggest_fix": suggest_fix,
            "full_review": full_review,
            "security_audit": security_audit,
            "performance_refactor": performance_refactor
        }
        
        grader = grader_map.get(self.state.task_type)
        reward = grader.score(
            action.response, 
            self.state.current_snippet, 
            step=self.state.step_number
        )
        
        # ENSURE STRICT RANGE (0, 1) per validator requirement
        # This maps any [0, 1] input to [0.01, 0.99]
        reward = max(0.01, min(0.99, float(reward)))
        
        self.state.rewards.append(reward)
        self.state.step_number += 1
        
        # Check termination conditions
        if self.state.step_number > self.state.max_steps or reward >= 0.95:
            self.state.done = True
            self.state.status = "done"

        return StepResult(
            observation=self._get_observation(),
            reward=float(reward),
            done=self.state.done,
            info={"snippet_id": self.state.current_snippet["id"]}
        )

    def _get_observation(self) -> CodeReviewObservation:
        snippet = self.state.current_snippet
        task_desc_map = {
            "identify_bug": "Identify the type of bug in this code snippet.",
            "suggest_fix": "Provide a fixed version of this code snippet including the function definition.",
            "full_review": "Provide a full code review including bug identification, a fix, and style notes.",
            "security_audit": "Perform a security audit. Identify the vulnerability (e.g. Command Injection) and provide a secure fix.",
            "performance_refactor": "Analyze the performance bottleneck (e.g. O(n^2) complexity) and provide an optimized refactor."
        }
        
        return CodeReviewObservation(
            code_snippet=snippet["code_snippet"],
            language="python",
            task_type=self.state.task_type,
            task_description=task_desc_map.get(self.state.task_type, ""),
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
            context=snippet.get("context")
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "status": self.state.status,
            "task_type": self.state.task_type,
            "step_number": self.state.step_number,
            "max_steps": self.state.max_steps,
            "done": self.state.done,
            "rewards": self.state.rewards
        }

