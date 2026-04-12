from typing import Tuple
from . import identify_bug, suggest_fix

WEIGHTS = {"bug": 0.30, "fix": 0.30, "style": 0.10} # Total 0.7 base for SUBMIT

def score_style_notes(response: str, style_keywords: list[str]) -> Tuple[float, str]:
    if not style_keywords:
        return 0.1, "Style: No specific requirements."
    hits = sum(1 for kw in style_keywords if kw.lower() in response.lower())
    score = min(hits / len(style_keywords), 1.0) * 0.1
    found = [kw for kw in style_keywords if kw.lower() in response.lower()]
    return score, f"Style: Found {len(found)}/{len(style_keywords)} keywords ({', '.join(found)})."

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Full Review task on SUBMIT.
    Composite score of Identify, Fix, and Style.
    """
    if not is_submission:
        return 0.0, ""

    # Call sub-graders with step=1 to AVOID DOUBLE PENALTY
    bug_score, bug_feedback = identify_bug.score(working_code, ground_truth, step=1, is_submission=True)
    fix_score, fix_feedback = suggest_fix.score(working_code, ground_truth, step=1, is_submission=True)
    
    # Map sub-grader internal 0.7 scale to full_review 0.3 weights
    # sub-grader returns 0.7 for max. So 0.7 * (0.3/0.7) = 0.3
    normalized_bug = bug_score * (0.3 / 0.7)
    normalized_fix = fix_score * (0.3 / 0.7)
    
    style_keywords = ground_truth.get("style_keywords", [])
    style_score, style_feedback = score_style_notes(working_code, style_keywords)
    
    final_reward = normalized_bug + normalized_fix + style_score
    
    # Single Forgiving Step Decay at top-level
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay
    
    combined_feedback = f"--- FULL REVIEW REPORT ---\n1. {bug_feedback}\n2. {fix_feedback}\n3. {style_feedback}"
    
    return round(float(final_reward), 2), combined_feedback
