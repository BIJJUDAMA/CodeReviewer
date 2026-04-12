from typing import Tuple
from . import identify_bug, suggest_fix

def score_style_notes(response: str, style_keywords: list[str]) -> Tuple[float, str]:
    if not style_keywords:
        return 0.1, "Style: OK."
    hits = sum(1 for kw in style_keywords if kw.lower() in response.lower())
    score = (hits / len(style_keywords)) * 0.1
    return max(0.01, min(0.1, score)), f"Style: {hits}/{len(style_keywords)}."

def score(working_code: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades a Full Review task on SUBMIT.
    Returns score strictly within (0.05, 0.95).
    """
    if not is_submission:
        return 0.05, ""

    # Sub-graders (already clamped internally to 0.05-0.95)
    bug_s, bug_f = identify_bug.score(working_code, ground_truth, step=1, is_submission=True)
    fix_s, fix_f = suggest_fix.score(working_code, ground_truth, step=1, is_submission=True)
    
    # Scale to composite weights (0.3 max for bug/fix)
    # sub-grader max is 0.7 normally, but we clamp to 0.95. 
    norm_bug = (bug_s / 0.7) * 0.3
    norm_fix = (fix_s / 0.7) * 0.3
    
    style_keywords = ground_truth.get("style_keywords", [])
    style_s, style_f = score_style_notes(working_code, style_keywords)
    
    final_reward = norm_bug + norm_fix + style_s
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_reward *= decay
    
    combined_feedback = f"FULL: {bug_f} | {fix_f} | {style_f}"
    
    return max(0.05, min(0.95, round(float(final_reward), 2))), combined_feedback
