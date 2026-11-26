import unittest
import numpy as np

from settings import Config
from rewards import compute_reward


class Member3Tests(unittest.TestCase):
    def test_strategic_reward_corner_bonus(self):
        config = Config(reward="strategic")
        
        # Board with max tile (256) in corner (0, 0)
        state = np.zeros((4, 4), dtype=np.int64)
        next_corner = np.zeros((4, 4), dtype=np.int64)
        next_corner[0, 0] = 256
        info = {"score_delta": 0, "score": 256, "merges": 0}
        reward_corner = compute_reward(state, next_corner, info, config)
        
        # Board with max tile (256) in center (1, 1)
        next_center = np.zeros((4, 4), dtype=np.int64)
        next_center[1, 1] = 256
        reward_center = compute_reward(state, next_center, info, config)
        
        # Corner position should give strictly higher reward than center position
        self.assertGreater(reward_corner, reward_center)
        # Log2(256) is 8; corner bonus is +16, center penalty is -16, difference should be 32
        self.assertAlmostEqual(reward_corner - reward_center, 32.0)

    def test_strategic_reward_empty_tiles_bonus(self):
        config = Config(reward="strategic")
        state = np.zeros((4, 4), dtype=np.int64)
        
        # Board A has 15 empty cells (one tile in corner)
        board_a = np.zeros((4, 4), dtype=np.int64)
        board_a[0, 0] = 4
        
        # Board B has 10 empty cells (6 tiles on board, corner unchanged)
        board_b = board_a.copy()
        board_b[0, 1] = 2
        board_b[0, 2] = 2
        board_b[0, 3] = 2
        board_b[1, 0] = 2
        board_b[1, 1] = 2
        
        info = {"score_delta": 0, "score": 4, "merges": 0}
        reward_a = compute_reward(state, board_a, info, config)
        reward_b = compute_reward(state, board_b, info, config)
        
        # More empty cells should give higher reward
        self.assertGreater(reward_a, reward_b)
        # 5 more empty cells at 2.0 each = 10.0 difference
        self.assertAlmostEqual(reward_a - reward_b, 10.0)


if __name__ == "__main__":
    unittest.main()
