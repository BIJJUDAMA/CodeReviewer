from . import identify_bug, suggest_fix

WEIGHTS = {"bug": 0.40, "fix": 0.35, "style": 0.25}

def score_style_notes(response: str, style_keywords: list[str]) -> float:
    if not style_keywords:
        return 1.0
    hits = sum(1 for kw in style_keywords if kw.lower() in response.lower())
    return min(hits / len(style_keywords), 1.0)

def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    # Use step=1 for sub-graders to avoid double penalty as discussed before, 
    # but the user snippet didn't specify. I will stick to user's logic mostly.
    bug_score = identify_bug.score(agent_response, ground_truth, step)
    fix_score = suggest_fix.score(agent_response, ground_truth, step) # Added step=step to match current suggest_fix signature
    style_keywords = ground_truth.get("style_keywords", [])
    style_score = score_style_notes(agent_response, style_keywords)
    
    final_score = (
        WEIGHTS["bug"] * bug_score +
        WEIGHTS["fix"] * fix_score +
        WEIGHTS["style"] * style_score
    )
    
    # STRICT (0.01, 0.99) clamp
    return max(0.01, min(0.99, round(final_score, 2)))
