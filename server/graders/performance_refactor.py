import re
import ast
from typing import Dict, Any, Tuple
from . import suggest_fix

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Performance Refactor task on SUBMIT.
    Uses AST and execution to verify optimization.
    """
    if not is_submission:
        return 0.0, ""

    feedback_lines = []
    
    # 1. Performance Logic (0.2 weight) - Data Driven
    logic_score = 0.0
    performance_keywords = ground_truth.get("performance_keywords", ["complexity", "o(n)", "set", "dictionary"])
    found_keywords = [kw for kw in performance_keywords if kw in working_code.lower()]
    if found_keywords:
        logic_score = 0.2 * (len(found_keywords) / len(performance_keywords))
        feedback_lines.append(f"Analysis: Identified optimization concepts ({', '.join(found_keywords)}).")
    else:
        feedback_lines.append("Analysis: Fails to mention optimization concepts.")
        
    # 2. Optimization Pattern Score (0.3 weight) - AST
    opt_score = 0.0
    try:
        tree = ast.parse(working_code)
        
        # Check for mandatory optimized patterns (e.g., set() conversion)
        mandatory = ground_truth.get("optimized_patterns", ["set(", "dict("])
        has_optimized = any(pat in working_code for pat in mandatory)
        
        if has_optimized:
            opt_score = 0.3
            feedback_lines.append("Pattern Review: Found efficient data structure usage.")
        else:
            feedback_lines.append("Pattern Review: Missing algorithmic optimization.")
                
    except SyntaxError:
        feedback_lines.append("Pattern Review: Unable to verify optimization due to syntax error.")

    # 3. Functional Correctness (0.2 weight) - EXECUTION
    test_cases = ground_truth.get("test_cases", [])
    results = suggest_fix.run_tests(working_code, test_cases)
    passed_count = sum(1 for r in results if r["passed"])
    functional_score = 0.2 * (passed_count / len(results) if results else 1.0)
    feedback_lines.append(f"Verification: {passed_count}/{len(results)} functional tests passed.")

    final_reward = logic_score + opt_score + functional_score
    
    # Forgiving Step Decay: (max(0.5, 1 - 0.03 * max(0, step - 2)))
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay
    
    return round(float(final_reward), 2), "\n".join(feedback_lines)
