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
    return ""

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Executes the provided code against test cases."""
    results = []
    
    for test in test_cases:
        input_args = test.get("input_args", [])
        expected = test.get("expected_output")
        
        # Capture stdout to prevent cluttering logs
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # We assume the code defines a function. 
            # We'll exec the code and then call the function by name from the snippet.
            # Heuristic: Find the first 'def function_name'
            def_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            if not def_match:
                raise ValueError("No function definition found in code block.")
            
            fn_name = def_match.group(1)
            
            # Local namespace for execution
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
                "error": str(e),
                "input": input_args,
                "expected": expected
            })
        finally:
            sys.stdout = old_stdout
            
    return results

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Suggest Fix task.
    Provides tiered rewards:
    - 20% for Syntax Integrity (is it valid Python?)
    - 80% for Functional Correctness (passing test cases)
    """
    code = extract_code_block(agent_response)
    if not code:
        return 0.0

    # 1. Syntax Integrity Reward (20% weight)
    # Check if code is valid Python using AST
    syntax_reward = 0.0
    try:
        ast.parse(code)
        syntax_reward = 1.0
    except:
        pass

    # 2. Functional Correctness (80% weight)
    test_cases = ground_truth.get("test_cases", [])
    if not test_cases:
        passed_ratio = 1.0 # Default if no tests
    else:
        results = run_tests(code, test_cases)
        results_count = len(results)
        if results_count == 0:
            passed_ratio = 1.0
        else:
            passed_ratio = sum(1 for r in results if r["passed"]) / results_count

    # Combine: 20% for syntax mastery, 80% for logical success
    final_score = (0.2 * syntax_reward) + (0.8 * passed_ratio)
    
    # Scale based on step (incentivize faster fixes)
    decay = max(0.5, 1 - 0.05 * (step - 1))
    return round(final_score * decay, 2)
