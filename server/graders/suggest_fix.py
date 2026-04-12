import ast
import re
import sys
import io
from typing import List, Dict, Any, Tuple

def extract_code_block(text: str) -> str:
    """Extracts the first python code block from a markdown string."""
    pattern = r"```(?:python)?\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def check_syntax(code: str) -> Tuple[bool, str]:
    """Verifies if the code is valid Python syntax."""
    try:
        ast.parse(code)
        return True, ""
    except Exception as e:
        return False, str(e)

def run_execution_test(code: str, ground_truth: dict) -> str:
    """Simulates a test runner and returns stdout/stderr logs."""
    test_cases = ground_truth.get("test_cases", [])
    if not test_cases:
        return "No tests defined for this snippet."
    
    results = run_tests(code, test_cases)
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    
    output = [f"Ran {total} tests. {passed_count} PASSED, {total - passed_count} FAILED."]
    for i, res in enumerate(results):
        if not res["passed"]:
            if "error" in res:
                output.append(f"Test {i+1} ERROR: {res['error']}")
            else:
                output.append(f"Test {i+1} FAILED: Expected {res['expected']}, got {res['actual']}")
    
    return "\n".join(output)

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Executes the provided code against test cases."""
    results = []
    
    for test in test_cases:
        input_args = test.get("input_args", [])
        expected = test.get("expected_output")
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # Find the function name
            def_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            if not def_match:
                raise ValueError("No function definition found in code block.")
            
            fn_name = def_match.group(1)
            
            local_vars = {}
            exec(code, local_vars, local_vars)
            
            if fn_name not in local_vars:
                raise NameError(f"Function {fn_name} not found after execution.")
            
            func = local_vars[fn_name]
            actual = func(*input_args)
            
            passed = (actual == expected)
            results.append({
                "passed": passed,
                "input": input_args,
                "expected": expected,
                "actual": actual
            })
            
        except Exception as e:
            results.append({
                "passed": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "input": input_args,
                "expected": expected
            })
        finally:
            sys.stdout = old_stdout
            
    return results

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Suggest Fix task on SUBMIT.
    Returns score strictly within (0.01, 0.99).
    """
    if not is_submission:
        return 0.01, ""

    syntax_ok, syntax_err = check_syntax(working_code)
    if not syntax_ok:
        return 0.01, f"Submission Rejected: Syntax Error.\n{syntax_err}"

    test_cases = ground_truth.get("test_cases", [])
    if not test_cases:
        return 0.7, "Submission Accepted: No tests required."

    results = run_tests(working_code, test_cases)
    passed_count = sum(1 for r in results if r["passed"])
    passed_ratio = passed_count / len(results)
    
    # 0.7 base for full resolution
    final_reward = 0.7 * passed_ratio
    
    # Forgiving Step Decay
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay

    feedback = f"Final Submission Results: {passed_count}/{len(results)} tests passed."
    if passed_ratio == 1.0:
        feedback += " Excellent work!"
    
    # Strict (0.01, 0.99) clamping
    return max(0.01, min(0.99, round(float(final_reward), 2))), feedback
