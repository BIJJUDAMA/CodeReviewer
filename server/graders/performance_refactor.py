import re
import ast
from typing import Dict, Any, Tuple

def score(agent_response: str, ground_truth: dict, step: int = 1) -> Tuple[float, str]:
    """
    Grades a Performance Refactor task.
    Uses AST to verify if the agent optimized O(n^2) to O(n).
    Specifically looks for converting a search list to a set/dict.
    """
    response = agent_response.lower()
    feedback_lines = []
    
    # 1. Logic Identification (30% weight)
    logic_score = 0.0
    performance_keywords = ["complexity", "o(n)", "set", "dictionary", "linear time", "bottleneck"]
    found_keywords = [kw for kw in performance_keywords if kw in response]
    if found_keywords:
        logic_score = 1.0
        feedback_lines.append(f"Performance Analysis: Identified keywords ({', '.join(found_keywords)}).")
    else:
        feedback_lines.append("Performance Analysis: Fails to mention optimization concepts.")
        
    # 2. Optimization Score (70% weight)
    opt_score = 0.0
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    
    for code in code_blocks:
        try:
            tree = ast.parse(code)
            has_set_conversion = False
            has_efficient_lookup = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ["set", "dict"]:
                        has_set_conversion = True
                if isinstance(node, ast.Compare):
                    if isinstance(node.ops[0], ast.In):
                        has_efficient_lookup = True
            
            if has_set_conversion and has_efficient_lookup:
                opt_score = 1.0
                feedback_lines.append("Refactor: Efficient O(n) implementation detected.")
                break
            elif has_set_conversion:
                opt_score = 0.6
                feedback_lines.append("Refactor: Detected set conversion, but logical usage is unclear.")
            else:
                feedback_lines.append("Refactor: Missing set/dict optimization.")
                
        except SyntaxError:
            if "set(" in code.lower() and " in " in code.lower():
                opt_score = 0.8
                feedback_lines.append("Refactor: Optimized pattern detected via text match.")
            else:
                feedback_lines.append("Refactor: Unable to verify optimization due to syntax error or missing code.")

    final_score = (0.3 * logic_score) + (0.7 * opt_score)
    decay_factor = max(0.5, 1 - 0.1 * (step - 1))
    
    return round(float(final_score * decay_factor), 2), "\n".join(feedback_lines)
