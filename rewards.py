"""Student extension C: training rewards. Evaluation always uses the raw game score."""
import numpy as np


def compute_reward(state, next_state, info, config):
    # STUDENT EDIT C: Add rewards here; keep info['score'] and evaluation metrics unchanged.
    if config.reward == "score":
        value = float(info["score_delta"])
    elif config.reward == "merge_bonus":
        before, after = int(np.max(state)), int(np.max(next_state))
        bonus = 0.1 * np.log2(after) if after > before else 0.0
        value = float(info["merges"]) + bonus
    else:
        raise ValueError(f"Unknown reward: {config.reward}")
    return float(value * config.reward_scale)
