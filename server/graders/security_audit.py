import re
import ast
from typing import Dict, Any, Tuple

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Security Audit task on SUBMIT.
    Rewards both identifying the vulnerability type and providing a secure fix.
    """
    if not is_submission:
        return 0.0, ""

    response = working_code.lower()
    feedback_lines = []
    
    # 1. Detection Score (0.3 weight) - Data Driven
    detection_score = 0.0
    vulnerability_keywords = ground_truth.get("vulnerability_keywords", [])
    found_keywords = [kw for kw in vulnerability_keywords if kw in response]
    
    if found_keywords:
        detection_score = 0.3 * (len(found_keywords) / len(vulnerability_keywords))
        feedback_lines.append(f"Vulnerability Identification: Found {len(found_keywords)} key indicators.")
    else:
        feedback_lines.append("Vulnerability Identification: Failed to recognize the core vulnerability.")
        
    # 2. Fix Score (0.4 weight) - Data Driven / AST
    fix_score = 0.0
    try:
        tree = ast.parse(working_code)
        
        # Check for forbidden patterns (e.g., os.system)
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
                    feedback_lines.append(f"Fix Review: Found forbidden insecure call: {call_str}")
        
        # Check for mandatory secure patterns (e.g., subprocess.run with shell=False)
        secure_patterns = ground_truth.get("secure_patterns", [])
        has_secure = all(pat in working_code for pat in secure_patterns)
        
        if has_secure and not has_forbidden:
            fix_score = 0.4
            feedback_lines.append("Fix Review: Code follows secure implementation patterns.")
        elif has_secure:
            fix_score = 0.1
            feedback_lines.append("Fix Review: Secure pattern added but insecure code remains.")
            
    except SyntaxError as e:
        feedback_lines.append(f"Fix Review: Syntax Error in submission: {str(e)}")

    final_reward = detection_score + fix_score
    
    # Forgiving Step Decay: (max(0.5, 1 - 0.03 * max(0, step - 2)))
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay
    
    return round(float(final_reward), 2), "\n".join(feedback_lines)
