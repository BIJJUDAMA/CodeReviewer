import re
import ast
from typing import Dict, Any, Tuple

def score(agent_response: str, ground_truth: dict, step: int = 1) -> Tuple[float, str]:
    """
    Grades a Security Audit task.
    Rewards both identifying the vulnerability type and providing a secure fix.
    """
    response = agent_response.lower()
    feedback_lines = []
    
    # 1. Detection Score (40% weight)
    detection_score = 0.0
    vulnerability_keywords = ["command injection", "shell injection", "os.system", "sanitize", "subprocess"]
    found_keywords = [kw for kw in vulnerability_keywords if kw in response]
    
    if "command injection" in response or "shell injection" in response:
        detection_score = 1.0
        feedback_lines.append("Vulnerability Identification: Correct (Command Injection).")
    elif found_keywords:
        detection_score = 0.5
        feedback_lines.append(f"Vulnerability Identification: Partial. Found keywords: {', '.join(found_keywords)}.")
    else:
        feedback_lines.append("Vulnerability Identification: Failed.")
        
    # 2. Fix Score (60% weight)
    fix_score = 0.0
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    
    if not code_blocks:
        feedback_lines.append("Fix: No code block provided.")
    
    for code in code_blocks:
        try:
            tree = ast.parse(code)
            has_os_system = False
            has_subprocess_safe = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "system":
                            has_os_system = True
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess" and node.func.attr in ["run", "call", "check_output"]:
                            shell_true = False
                            for kw in node.keywords:
                                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    shell_true = True
                            if not shell_true:
                                has_subprocess_safe = True
            
            if has_subprocess_safe and not has_os_system:
                fix_score = 1.0
                feedback_lines.append("Fix: Secure (using subprocess without shell=True).")
                break
            elif has_subprocess_safe:
                fix_score = 0.5
                feedback_lines.append("Fix: Partial. Added subprocess but didn't remove os.system.")
            else:
                feedback_lines.append("Fix: Insecure or missing proper subprocess usage.")
                
        except SyntaxError:
            if "subprocess.run" in code and "shell=false" in code.lower():
                fix_score = 0.8
                feedback_lines.append("Fix: Likely secure (detected subprocess.run with shell=False).")
            else:
                feedback_lines.append("Fix: Syntax error in code block, unable to verify fully.")

    final_score = (0.4 * detection_score) + (0.6 * fix_score)
    decay_factor = max(0.5, 1 - 0.1 * (step - 1))
    
    return round(float(final_score * decay_factor), 2), "\n".join(feedback_lines)
