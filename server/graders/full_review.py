from typing import Tuple
from . import identify_bug, suggest_fix

WEIGHTS = {"bug": 0.40, "fix": 0.35, "style": 0.25}

def score_style_notes(response: str, style_keywords: list[str]) -> Tuple[float, str]:
    if not style_keywords:
        return 1.0, "Style: No specific requirements."
    hits = sum(1 for kw in style_keywords if kw.lower() in response.lower())
    score = min(hits / len(style_keywords), 1.0)
    found = [kw for kw in style_keywords if kw.lower() in response.lower()]
    return score, f"Style: Found {len(found)}/{len(style_keywords)} keywords ({', '.join(found)})."

def score(agent_response: str, ground_truth: dict, step: int = 1) -> Tuple[float, str]:
    bug_score, bug_feedback = identify_bug.score(agent_response, ground_truth, step)
    fix_score, fix_feedback = suggest_fix.score(agent_response, ground_truth, step)
    style_keywords = ground_truth.get("style_keywords", [])
    style_score, style_feedback = score_style_notes(agent_response, style_keywords)
    
    final_score = (
        WEIGHTS["bug"] * bug_score +
        WEIGHTS["fix"] * fix_score +
        WEIGHTS["style"] * style_score
    )
    
    combined_feedback = f"--- FULL REVIEW REPORT ---\n1. {bug_feedback}\n2. {fix_feedback}\n3. {style_feedback}"
    
    return round(float(final_score), 2), combined_feedback
