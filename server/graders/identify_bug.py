def score(agent_response: str, ground_truth: dict, step: int = 1) -> float:
    label = ground_truth.get("bug_type", "").lower()
    aliases = [a.lower() for a in ground_truth.get("aliases", [])]
    response = agent_response.lower()

    # Base score calculation
    base_score = 0.0
    if label in response or any(a in response for a in aliases):
        base_score = 1.0
    elif any(word.lower() in response for word in ground_truth.get("partial_match_words", [])):
        base_score = 0.4

    # Apply step-decay: (1 - 0.15 * (step - 1))
    # step starts from 1
    decay_factor = max(0, 1 - 0.15 * (step - 1))
    final_score = base_score * decay_factor

    return round(final_score, 2)
