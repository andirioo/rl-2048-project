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
    elif config.reward == "strategic":
        # 1. Base merge score
        score_gain = float(info["score_delta"])
        
        # 2. Empty cells bonus: encourage keeping breathing room on the board
        empty_count = float(np.count_nonzero(next_state == 0))
        empty_bonus = empty_count * 2.0
        
        # 3. Corner bonus: reward keeping the highest tile in one of the 4 corners
        max_tile = int(np.max(next_state))
        corners = [next_state[0, 0], next_state[0, 3], next_state[3, 0], next_state[3, 3]]
        if max_tile > 0:
            corner_scale = float(np.log2(max_tile))
            corner_bonus = (2.0 * corner_scale) if max_tile in corners else (-2.0 * corner_scale)
        else:
            corner_bonus = 0.0
            
        value = score_gain + empty_bonus + corner_bonus
    else:
        raise ValueError(f"Unknown reward: {config.reward}")
    return float(value * config.reward_scale)
