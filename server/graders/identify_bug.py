def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    """
    Grades an Identify Bug task.
    """
    label = ground_truth.get("bug_type", "").lower()
    aliases = [a.lower() for a in ground_truth.get("aliases", [])]
    
    # Clean response
    response = agent_response.lower().strip()

    base_score = 0.01
    
    # 1. Exact Match / Alias Match (1.0)
    if label in response or any(a in response for a in aliases):
        base_score = 1.0
    # 2. Strong Keyword Match (0.6)
    elif any(word.lower() in response for word in ground_truth.get("partial_match_words", [])):
        base_score = 0.6
    # 3. Categorical Match (0.2)
    elif any(cat in response for cat in ["error", "bug", "issue", "vulnerability", "optimization"]):
        base_score = 0.2

    # Apply step-decay
    decay_factor = max(0.5, 1 - 0.10 * (step - 1))
    final_score = base_score * decay_factor

    # STRICT (0.01, 0.99) clamp
    return max(0.01, min(0.99, round(float(final_score), 2)))
