import re
import ast
from typing import Dict, Any
from . import suggest_fix

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Performance Refactor task.
    """
    response = agent_response.lower()
    
    # 1. Performance Logic (20%)
    logic_score = 0.01
    performance_keywords = ground_truth.get("performance_keywords", ["complexity", "o(n)", "set", "dictionary"])
    found_keywords = [kw for kw in performance_keywords if kw in response]
    if performance_keywords:
        logic_score = len(found_keywords) / len(performance_keywords)
        
    # 2. Optimization Pattern (40%)
    opt_score = 0.01
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    for code in code_blocks:
        mandatory = ground_truth.get("optimized_patterns", ["set(", "dict("])
        if any(pat in code for pat in mandatory):
            opt_score = 1.0
            break

    # 3. Functional Correctness (40%)
    functional_score = 0.01
    test_cases = ground_truth.get("test_cases", [])
    for code in code_blocks:
        code_clean = suggest_fix.extract_code_block(code)
        results = suggest_fix.run_tests(code_clean, test_cases)
        if results:
            passed_ratio = sum(1 for r in results if r["passed"]) / len(results)
            functional_score = max(functional_score, passed_ratio)

    final_score = (0.2 * logic_score) + (0.4 * opt_score) + (0.4 * functional_score)
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    
    return max(0.01, min(0.99, round(float(final_score * decay), 2)))
