import re
import ast
from typing import Dict, Any

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Security Audit task.
    """
    response = agent_response.lower()
    
    # 1. Detection Score (30%)
    detection_score = 0.01
    vulnerability_keywords = ground_truth.get("vulnerability_keywords", [])
    found_keywords = [kw for kw in vulnerability_keywords if kw in response]
    if vulnerability_keywords:
        detection_score = len(found_keywords) / len(vulnerability_keywords)
    
    # 2. Fix Score (70%)
    fix_score = 0.01
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    for code in code_blocks:
        try:
            tree = ast.parse(code)
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
            
            secure_patterns = ground_truth.get("secure_patterns", [])
            has_secure = all(pat in code for pat in secure_patterns)
            
            if has_secure and not has_forbidden:
                fix_score = 1.0
                break
            elif has_secure:
                fix_score = 0.5
        except:
            pass

    final_score = (0.3 * detection_score) + (0.7 * fix_score)
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    
    return max(0.01, min(0.99, round(float(final_score * decay), 2)))
