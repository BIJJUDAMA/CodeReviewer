import re
import ast
from typing import Dict, Any

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Security Audit task.
    Rewards both identifying the vulnerability type and providing a secure fix.
    """
    response = agent_response.lower()
    
    # 1. Detection Score (40% weight)
    # Did it identify 'command injection' or 'shell injection'?
    detection_score = 0.01
    vulnerability_keywords = ["command injection", "shell injection", "os.system", "sanitize", "subprocess"]
    found_keywords = [kw for kw in vulnerability_keywords if kw in response]
    
    if "command injection" in response or "shell injection" in response:
        detection_score = 1.0
    elif found_keywords:
        detection_score = 0.5
        
    # 2. Fix Score (60% weight)
    # Did it provide a version using subprocess.run(..., shell=False)?
    fix_score = 0.01
    
    # Extract code blocks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    
    for code in code_blocks:
        try:
            tree = ast.parse(code)
            
            # Deterministic AST Checks:
            has_os_system = False
            has_subprocess_safe = False
            
            for node in ast.walk(tree):
                # Check for os.system
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
                            has_os_system = True
                            
                        # Check for subprocess.run or call
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess" and node.func.attr in ["run", "call", "check_output"]:
                            # Check if shell=True is present
                            shell_true = False
                            for kw in node.keywords:
                                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    shell_true = True
                            
                            if not shell_true:
                                has_subprocess_safe = True
            
            if has_subprocess_safe and not has_os_system:
                fix_score = 1.0
                break
            elif has_subprocess_safe:
                fix_score = 0.5 # Safe call present, but didn't remove os.system
                
        except SyntaxError:
            # If code is invalid syntax, maybe it's just a snippet. 
            # Fallback to simple string check
            if "subprocess.run" in code and "shell=false" in code.lower():
                fix_score = 0.8
            elif "subprocess" in code.lower():
                fix_score = 0.3

    final_score = (0.4 * detection_score) + (0.6 * fix_score)
    
    # Apply step-decay (optional for hard tasks, but good for consistency)
    decay_factor = max(0.5, 1 - 0.1 * (step - 1))
    
    # STRICT (0.01, 0.99) clamp
    return max(0.01, min(0.99, round(float(final_score * decay_factor), 2)))
