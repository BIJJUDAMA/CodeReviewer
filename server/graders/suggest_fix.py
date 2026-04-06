import re
import subprocess
import tempfile
import os
from typing import List, Dict, Any

def extract_code_block(response: str) -> str:
    """Extract code from within markdown code blocks or return a best guess."""
    # Try to find ```python ... ``` or just ``` ... ```
    match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: if no markdown blocks, try to see if there's any coherent-looking code
    # or just return the whole response as a last resort
    return response.strip()

def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run extracted code against test cases in a subprocess."""
    results = []
    
    for i, tc in enumerate(test_cases):
        passed = False
        error = None
        output = None
        
        # Prepare execution script
        # We wrap the code with an execution harness for the specific test case
        input_args = tc.get("input_args", [])
        expected = tc.get("expected_output")
        
        # Generate a wrapper script that imports our code and calls the last function defined
        # or we can just append a call to the code itself.
        # Assuming the code contains at least one function.
        
        wrapper_code = f"""
{code}

# Test Harness
try:
    # Find the last function defined in the code
    import types
    funcs = [v for k, v in locals().items() if isinstance(v, types.FunctionType)]
    if funcs:
        # Use the most recently defined function (usually the fix)
        target_func = funcs[-1]
        result = target_func(*{repr(input_args)})
        print(repr(result))
    else:
        print("Error: No function found in code.")
except Exception as e:
    print(f"Error: {{e}}")
"""

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(wrapper_code)
            tmp_path = tmp.name

        try:
            # Execute in a subprocess with timeout
            process = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if process.returncode == 0:
                stdout_val = process.stdout.strip()
                if stdout_val.startswith("Error:"):
                    error = stdout_val
                else:
                    try:
                        # Safely evaluate the output (which is repr-ed)
                        import ast
                        # If output is "None" (literal string), ast.literal_eval fails for some Python versions
                        # or handles it. In Python 3.x, it should be fine.
                        actual_result = ast.literal_eval(stdout_val)
                        if actual_result == expected:
                            passed = True
                        output = actual_result
                    except:
                        error = f"Malformed output: {stdout_val}"
            else:
                error = process.stderr.strip() or f"Process failed with exit code {process.returncode}"
                
        except subprocess.TimeoutExpired:
            error = "Execution timed out (5s limit)"
        except Exception as e:
            error = str(e)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        results.append({
            "test_case_id": i,
            "passed": passed,
            "error": error,
            "output": output,
            "expected": expected
        })
        
    return results

def score(agent_response: str, ground_truth: dict) -> float:
    code = extract_code_block(agent_response)
    if not code:
        return 0.0

    test_cases = ground_truth.get("test_cases", [])
    if not test_cases:
        return 1.0  # Or some predefined value if no tests exist

    results = run_tests(code, test_cases)
    passed_count = sum(1 for r in results if r["passed"])
    
    score_val = passed_count / len(results)
    return round(score_val, 2)
