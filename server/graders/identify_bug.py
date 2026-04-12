from typing import Tuple

def score(agent_response: str, ground_truth: dict, step: int = 1, is_submission: bool = False) -> Tuple[float, str]:
    """
    Grades an Identify Bug task on SUBMIT.
    Returns score strictly within (0.05, 0.95).
    """
    if not is_submission:
        return 0.05, ""

    label = ground_truth.get("bug_type", "").lower()
    aliases = [a.lower() for a in ground_truth.get("aliases", [])]
    
    # Clean response: lowercase, remove trailing periods/punctuation
    response = agent_response.lower().strip()

    base_score = 0.05
    feedback = ""
    
    # 1. Exact Match / Alias Match (0.7) - SUBMIT base reward
    if label in response or any(a in response for a in aliases):
        base_score = 0.7
        feedback = f"Correct! You identified the bug type: {label}."
        
        # Noise penalty
        if len(response.split()) > 100:
            base_score *= 0.8
            feedback += " (Penalty applied for overly verbose response)."
            
    # 2. Strong Keyword Match (0.3)
    elif any(word.lower() in response for word in ground_truth.get("partial_match_words", [])):
        base_score = 0.3
        feedback = "Close. You used relevant terminology but didn't pinpoint the specific bug type."
    else:
        feedback = "The response did not accurately identify the bug type."

    # Forgiving Step Decay: (max(0.5, 1 - 0.03 * max(0, step - 2)))
    decay = max(0.5, 1 - 0.03 * max(0, step - 2))
    final_score = base_score * decay

    # Strict (0.05, 0.95) clamping
    return max(0.05, min(0.95, round(float(final_score), 2))), feedback
