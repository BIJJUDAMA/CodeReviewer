import re
import ast
from typing import Dict, Any

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades a Performance Refactor task.
    Uses AST to verify if the agent optimized O(n^2) to O(n).
    Specifically looks for converting a search list to a set/dict.
    """
    response = agent_response.lower()
    
    # 1. Logic Identification (30% weight)
    # Did it mention 'complexity', 'O(n)', 'set', or 'performance'?
    logic_score = 0.0
    performance_keywords = ["complexity", "o(n)", "set", "dictionary", "linear time", "bottleneck"]
    if any(kw in response for kw in performance_keywords):
        logic_score = 1.0
        
    # 2. Optimization Score (70% weight)
    # Did the code actually implement the set() optimization?
    opt_score = 0.0
    
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_response, re.DOTALL | re.IGNORECASE)
    
    for code in code_blocks:
        try:
            tree = ast.parse(code)
            
            has_set_conversion = False
            has_efficient_lookup = False
            
            for node in ast.walk(tree):
                # Look for set(some_list)
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ["set", "dict"]:
                        has_set_conversion = True
                
                # Look for 'x in some_set' where some_set is a variable (not a list literal)
                if isinstance(node, ast.Compare):
                    if isinstance(node.ops[0], ast.In):
                        # This is a bit complex for a simple grader, 
                        # so we use a heuristic: presence of set() + presence of 'in'
                        has_efficient_lookup = True
            
            if has_set_conversion and has_efficient_lookup:
                opt_score = 1.0
                break
            elif has_set_conversion:
                opt_score = 0.6 # Converted to set but maybe didn't use it right
                
        except SyntaxError:
            # Fallback to string matching for optimized patterns
            if "set(" in code.lower() and " in " in code.lower():
                opt_score = 0.8
            elif "set(" in code.lower():
                opt_score = 0.4

    final_score = (0.3 * logic_score) + (0.7 * opt_score)
    
    # Apply step-decay
    decay_factor = max(0.5, 1 - 0.1 * (step - 1))
    return round(final_score * decay_factor, 2)
