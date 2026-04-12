from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel
from dataclasses import dataclass, field
from tasks import dataset
from graders import identify_bug, suggest_fix, full_review, security_audit, performance_refactor
import random
import json

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
    # Validator compatibility: accept a 'response' field as plain text
    response: str

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

        reward = 0.05 # Initial baseline in (0, 1)
        feedback = ""
        
        # 1. Attempt to parse command from response (JSON support for advanced agents)
        command = "SUBMIT"
        payload = action.response
        parsed = None
        
        try:
            parsed = json.loads(action.response)
            if isinstance(parsed, dict) and "command" in parsed:
                command = parsed.get("command", "SUBMIT")
                payload = parsed.get("payload", action.response)
        except:
            # Not JSON, default to SUBMIT with raw response as payload
            pass

        # 2. Handle Commands
        if command == "RUN_TESTS":
            if not self.state.reproduced_bug:
                reward += 0.1
                self.state.reproduced_bug = True
            feedback = suggest_fix.run_execution_test(self.state.working_code, self.state.current_snippet)
            
        elif command == "EDIT_CODE":
            self.state.working_code = payload
            syntax_ok, syntax_err = suggest_fix.check_syntax(self.state.working_code)
            if syntax_ok:
                reward += 0.2
                feedback = "Syntax Check: PASSED. Use RUN_TESTS to verify behavior."
            else:
                reward -= 0.1
                feedback = f"Syntax Check: FAILED.\n{syntax_err}"
                
        else: # SUBMIT (default)
            if command == "SUBMIT" and payload:
                # If it was structured JSON, use payload. If it was raw text, use response.
                if parsed and "payload" in parsed:
                    self.state.working_code = payload
                elif not parsed:
                    self.state.working_code = payload

            grader_map = {
                "identify_bug": identify_bug,
                "suggest_fix": suggest_fix,
                "full_review": full_review,
                "security_audit": security_audit,
                "performance_refactor": performance_refactor
            }
            grader = grader_map.get(self.state.task_type)
            
            # Graders now return Tuple[float, str]
            reward_val, feedback_val = grader.score(
                self.state.working_code, 
                self.state.current_snippet, 
                step=self.state.step_number,
                is_submission=True
            )
            reward += reward_val
            feedback = feedback_val
            
            # Auto-done on submission
            self.state.done = True
            self.state.status = "done"

        # FINAL STRICT CLAMPING within (0.05, 0.95) to satisfy validator
        reward = max(0.05, min(0.95, float(reward)))
        
        self.state.rewards.append(reward)
        self.state.last_feedback = feedback
        self.state.step_number += 1
        
        if self.state.step_number > self.state.max_steps:
            self.state.done = True
            self.state.status = "done"

        return StepResult(
            observation=self._get_observation(),
            reward=float(reward),
            done=self.state.done,
            info={"snippet_id": self.state.current_snippet["id"], "command": command}
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
