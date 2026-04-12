from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel
from dataclasses import dataclass, field
from tasks import dataset
from graders import identify_bug, suggest_fix, full_review, security_audit, performance_refactor
import random

# Pydantic Models for OpenEnv Compliance

class CodeReviewObservation(BaseModel):
    code_snippet: str          # the current code state
    language: str              # always "python"
    task_type: str             # "identify_bug" | "suggest_fix" | "full_review" | "security_audit" | "performance_refactor"
    task_description: str      # instruction
    step_number: int
    max_steps: int
    context: Optional[str]     # optional context/hint
    feedback: Optional[str] = None # TERMINAL OUTPUT / FEEDBACK from last action

class CodeReviewAction(BaseModel):
    command: Literal["RUN_TESTS", "EDIT_CODE", "SUBMIT"]
    payload: str = ""          # The code if EDIT_CODE, else empty

class StepResult(BaseModel):
    observation: CodeReviewObservation
    reward: float
    done: bool
    info: Dict[str, Any]

@dataclass
class EnvState:
    status: Literal["idle", "running", "done"] = "idle"
    current_snippet: Optional[dict] = None
    working_code: str = ""     # Agent's current working code
    task_type: Optional[str] = None
    step_number: int = 0
    max_steps: int = 0
    rewards: List[float] = field(default_factory=list)
    done: bool = False
    last_feedback: Optional[str] = None
    reproduced_bug: bool = False # Track if agent ran initial broken tests

class CodeReviewEnv:
    def __init__(self):
        self.state = EnvState()

    def reset(self, task_type: Optional[str] = None) -> CodeReviewObservation:
        # All 5 tasks are now accessible
        tasks = ["identify_bug", "suggest_fix", "full_review", "security_audit", "performance_refactor"]
        if not task_type or task_type not in tasks:
            task_type = random.choice(tasks)
        
        snippet = dataset.get_task_snippet(task_type)
        
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
            working_code=snippet["code_snippet"],
            task_type=task_type,
            step_number=1,
            max_steps=max_steps_map.get(task_type, 5),
            rewards=[],
            done=False,
            last_feedback=None,
            reproduced_bug=False
        )
        
        return self._get_observation()

    def step(self, action: CodeReviewAction) -> StepResult:
        if self.state.status != "running":
            raise ValueError("Environment not in running state. Call reset() first.")

        reward = 0.0
        feedback = ""
        
        # 1. Handle Commands (Simulation logic)
        if action.command == "RUN_TESTS":
            # Wordle Reward (+0.1) for first investigation step
            if not self.state.reproduced_bug:
                reward += 0.1
                self.state.reproduced_bug = True
            
            # Execute current working_code against tests
            feedback = suggest_fix.run_execution_test(self.state.working_code, self.state.current_snippet)
            
        elif action.command == "EDIT_CODE":
            self.state.working_code = action.payload
            # Wordle Reward (+0.2) for syntax integrity
            syntax_ok, syntax_err = suggest_fix.check_syntax(self.state.working_code)
            if syntax_ok:
                reward += 0.2
                feedback = "Syntax Check: PASSED. Use RUN_TESTS to verify behavior."
            else:
                reward -= 0.1 # Penalty for carelessness
                feedback = f"Syntax Check: FAILED.\n{syntax_err}"
                
        elif action.command == "SUBMIT":
            grader_map = {
                "identify_bug": identify_bug,
                "suggest_fix": suggest_fix,
                "full_review": full_review,
                "security_audit": security_audit,
                "performance_refactor": performance_refactor
            }
            grader = grader_map.get(self.state.task_type)
            # Full logic reward (+0.7) handled inside the grader
            # Passing working_code instead of response for direct evaluation
            reward_val, feedback_val = grader.score(
                self.state.working_code, 
                self.state.current_snippet, 
                step=self.state.step_number,
                is_submission=True
            )
            reward += reward_val
            feedback = feedback_val
            
            # Termination logic (SUBMIT always ends or reaches max reward)
            if reward >= 0.95 or self.state.step_number >= self.state.max_steps:
                self.state.done = True
                self.state.status = "done"

        # Final Score Clamping strictly within (0, 1)
        # Using a slightly tighter margin to be safe
        reward = max(0.01, min(0.99, float(reward)))
        
        self.state.rewards.append(reward)
        self.state.last_feedback = feedback
        self.state.step_number += 1
        
        # Termination conditions (Max steps reached)
        if self.state.step_number > self.state.max_steps:
            self.state.done = True
            self.state.status = "done"

        return StepResult(
            observation=self._get_observation(),
            reward=float(reward),
            done=self.state.done,
            info={"snippet_id": self.state.current_snippet["id"], "command": action.command}
        )

    def _get_observation(self) -> CodeReviewObservation:
        snippet = self.state.current_snippet
        task_desc_map = {
            "identify_bug": "Analyze the terminal output and identify the bug type.",
            "suggest_fix": "Edit the code and run tests until all pass. Then SUBMIT.",
            "full_review": "Perform a comprehensive review: identify, fix, and style check.",
            "security_audit": "Identify and fix the vulnerability in the code.",
            "performance_refactor": "Refactor the code for better algorithmic complexity."
        }
        
        return CodeReviewObservation(
            code_snippet=self.state.working_code,
            language="python",
            task_type=self.state.task_type,
            task_description=task_desc_map.get(self.state.task_type, ""),
            step_number=self.state.step_number,
            max_steps=self.state.max_steps,
            context=snippet.get("context"),
            feedback=self.state.last_feedback
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "status": self.state.status,
            "task_type": self.state.task_type,
            "step_number": self.state.step_number,
            "max_steps": self.state.max_steps,
            "done": self.state.done,
            "rewards": self.state.rewards,
            "working_code": self.state.working_code,
            "last_feedback": self.state.last_feedback
        }
