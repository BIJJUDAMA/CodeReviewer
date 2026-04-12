import re
import ast
from typing import Dict, Any, Tuple

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Security Audit task on SUBMIT.
    Returns score strictly within (0.01, 0.99).
    """
    if not is_submission:
        return 0.01, ""

    response = working_code.lower()
    feedback_lines = []
    
    # 1. Detection Score (0.3 weight)
    detection_score = 0.01
    vulnerability_keywords = ground_truth.get("vulnerability_keywords", [])
    found_keywords = [kw for kw in vulnerability_keywords if kw in response]
    
    if found_keywords:
        detection_score = 0.3 * (len(found_keywords) / len(vulnerability_keywords))
        feedback_lines.append(f"Vulnerability Identification: Found {len(found_keywords)} key indicators.")
    else:
        feedback_lines.append("Vulnerability Identification: Failed.")
        
    # 2. Fix Score (0.4 weight)
    fix_score = 0.01
    try:
        tree = ast.parse(working_code)
        forbidden = ground_truth.get("forbidden_patterns", [])
        has_forbidden = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ""
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        call_str = f"{node.func.value.id}.{node.func.attr}"
                elif isinstance(node.func, ast.Name):
                    call_str = node.func.id
                
                if call_str in forbidden:
                    has_forbidden = True
                    feedback_lines.append(f"Fix Review: Found insecure call: {call_str}")
        
        secure_patterns = ground_truth.get("secure_patterns", [])
        has_secure = all(pat in working_code for pat in secure_patterns)
        
        if has_secure and not has_forbidden:
            fix_score = 0.4
            feedback_lines.append("Fix Review: Code follows secure implementation.")
        elif has_secure:
            fix_score = 0.1
            feedback_lines.append("Fix Review: Incomplete fix.")
            
    except SyntaxError as e:
        feedback_lines.append(f"Fix Review: Syntax Error.")

    final_reward = detection_score + fix_score
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay
    
    return max(0.01, min(0.99, round(float(final_reward), 2))), "\n".join(feedback_lines)
