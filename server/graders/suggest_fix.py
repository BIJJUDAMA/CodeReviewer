import ast
import re
import sys
import io
from typing import List, Dict, Any

def extract_code_block(text: str) -> str:
    """Extracts the first python code block from a markdown string."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def check_syntax(code: str) -> bool:
    """Verifies if the code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except:
        return False

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Executes the provided code against test cases."""
    results = []
    for test in test_cases:
        input_args = test.get("input_args", [])
        expected = test.get("expected_output")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            def_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            if not def_match:
                raise ValueError("No function definition found.")
            fn_name = def_match.group(1)
            local_vars = {}
            exec(code, local_vars, local_vars)
            if fn_name not in local_vars:
                raise NameError(f"Function {fn_name} not found.")
            func = local_vars[fn_name]
            actual = func(*input_args)
            passed = (actual == expected)
            results.append({"passed": passed, "expected": expected, "actual": actual})
        except Exception as e:
            results.append({"passed": False, "error": str(e)})
        finally:
            sys.stdout = old_stdout
    return results

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Suggest Fix task.
    """
    code = extract_code_block(agent_response)
    if not code:
        return 0.01

    # 1. Syntax integrity (20%)
    syntax_reward = 1.0 if check_syntax(code) else 0.01

    # 2. Functional correctness (80%)
    test_cases = ground_truth.get("test_cases", [])
    if not test_cases:
        passed_ratio = 1.0
    else:
        results = run_tests(code, test_cases)
        passed_count = sum(1 for r in results if r["passed"])
        passed_ratio = passed_count / len(results)

    final_score = (0.2 * syntax_reward) + (0.8 * passed_ratio)
    
    # Scale based on step
    decay = max(0.5, 1 - 0.05 * (step - 1))
    
    return max(0.01, min(0.99, round(float(final_score * decay), 2)))
