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
    feedback: Optional[str] = None # Terminal feedback

class CodeReviewAction(BaseModel):
    # accept a 'response' field as plain text for validator compatibility
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
    working_code: str = ""     # current code
    task_type: Optional[str] = None
    step_number: int = 0
    max_steps: int = 0
    rewards: List[float] = field(default_factory=list)
    done: bool = False
    last_feedback: Optional[str] = None
    reproduced_bug: bool = False

class CodeReviewEnv:
    def __init__(self):
        self.state = EnvState()

    def reset(self, task_type: Optional[str] = None) -> CodeReviewObservation:
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
            raise ValueError("Environment not in running state.")

        reward = 0.01
        feedback = ""
        
        # 1. Attempt to parse structured command (Smart RL support)
        command = "SUBMIT"
        payload = action.response
        is_json = False
        try:
            parsed = json.loads(action.response)
            if isinstance(parsed, dict) and "command" in parsed:
                command = parsed.get("command", "SUBMIT")
                payload = parsed.get("payload", action.response)
                is_json = True
        except:
            pass

        # 2. Handle Terminal Simulation
        if command == "RUN_TESTS":
            if not self.state.reproduced_bug:
                reward += 0.1
                self.state.reproduced_bug = True
            # Generate terminal output
            results = suggest_fix.run_tests(suggest_fix.extract_code_block(self.state.working_code), self.state.current_snippet.get("test_cases", []))
            passed = sum(1 for r in results if r.get("passed"))
            total = len(results)
            feedback = f"Ran {total} tests. {passed} PASSED."
            for r in results:
                if not r.get("passed"):
                    feedback += f"\n- Error: {r.get('error', 'AssertionError')}"
            
        elif command == "EDIT_CODE":
            self.state.working_code = payload
            if suggest_fix.check_syntax(suggest_fix.extract_code_block(payload)):
                reward += 0.2
                feedback = "Syntax Check: PASSED."
            else:
                reward -= 0.1
                feedback = "Syntax Check: FAILED."
                
        else: # SUBMIT (default or explicit)
            if not is_json: # from validator
                self.state.working_code = action.response
            
            grader_map = {
                "identify_bug": identify_bug,
                "suggest_fix": suggest_fix,
                "full_review": full_review,
                "security_audit": security_audit,
                "performance_refactor": performance_refactor
            }
            grader = grader_map.get(self.state.task_type)
            reward_val = grader.score(self.state.working_code, self.state.current_snippet, step=self.state.step_number)
            reward += reward_val
            feedback = f"Final Submission Reward: {reward_val:.2f}"
            self.state.done = True
            self.state.status = "done"

        # FINAL STRICT CLAMPING
        reward = max(0.01, min(0.99, float(reward)))
        
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
            "identify_bug": "Identify the bug type.",
            "suggest_fix": "Fix the code. RUN_TESTS then SUBMIT.",
            "full_review": "Full code review.",
            "security_audit": "Security audit and fix.",
            "performance_refactor": "Optimize performance."
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
            "last_feedback": self.state.last_feedback
        }
