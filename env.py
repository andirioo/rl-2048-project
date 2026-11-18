"""Shared 2048 environment. Action IDs: 0 up, 1 left, 2 right, 3 down."""
from __future__ import annotations

import numpy as np

ACTION_NAMES = ("up", "left", "right", "down")


def validate_board(board):
    board = np.asarray(board)
    if board.shape != (4, 4) or not np.issubdtype(board.dtype, np.integer):
        raise ValueError("board must be a 4x4 integer array")
    if np.any(board < 0) or np.any((board > 0) & ((board & (board - 1)) != 0)) or np.any(board == 1):
        raise ValueError("Cells must be 0 or powers of two greater than or equal to 2")
    return board.astype(np.int64, copy=True)


def move_board(board, action):
    """Pure move: no tile spawning or input mutation. Return board, score gain, and merge count."""
    if action not in range(4):
        raise ValueError("action must be 0, 1, 2, or 3")
    rotations = (1, 0, 2, 3)
    rotated = np.rot90(np.asarray(board), rotations[action])
    result = np.zeros((4, 4), dtype=np.int64)
    score, merges = 0, 0
    for row in range(4):
        values = [int(x) for x in rotated[row] if x]
        merged = []
        i = 0
        while i < len(values):
            if i + 1 < len(values) and values[i] == values[i + 1]:
                value = 2 * values[i]
                merged.append(value)
                score += value
                merges += 1
                i += 2
            else:
                merged.append(values[i])
                i += 1
        result[row, :len(merged)] = merged
    return np.rot90(result, -rotations[action]).copy(), score, merges


def legal_action_mask(board):
    """True means the action changes the board. Terminal states have no legal actions."""
    return np.array([not np.array_equal(move_board(board, a)[0], board)
                     for a in range(4)], dtype=bool)


class Game2048:
    def __init__(self, seed=None, max_steps=None):
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be a positive integer or None")
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = self.steps = 0
        self.finished = True

    def _spawn(self):
        empty = np.argwhere(self.board == 0)
        if len(empty):
            row, col = empty[self.rng.integers(len(empty))]
            self.board[row, col] = 4 if self.rng.random() < 0.1 else 2

    def _info(self, score_delta=0, merges=0, moved=False):
        return {"score": self.score, "score_delta": score_delta,
                "merges": merges, "moved": moved, "steps": self.steps,
                "max_tile": int(self.board.max()),
                "action_mask": legal_action_mask(self.board)}

    def reset(self, seed=None, board=None):
        """Use board only for demonstrations or tests. Return observation and info."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = (np.zeros((4, 4), dtype=np.int64) if board is None
                      else validate_board(board))
        self.score = self.steps = 0
        if board is None:
            self._spawn()
            self._spawn()
        info = self._info()
        self.finished = not bool(info["action_mask"].any())
        return self.board.copy(), info

    def legal_actions(self):
        return np.flatnonzero(legal_action_mask(self.board)).tolist()

    def step(self, action):
        """Return obs, raw score gain, terminated, truncated, and info.

        Continue beyond 2048. Distinguish natural termination from a step limit.
        Invalid moves preserve the board and score without consuming tile-spawn randomness.
        """
        if self.finished:
            raise RuntimeError("The episode has ended; call reset() first")
        moved_board, delta, merges = move_board(self.board, action)
        moved = not np.array_equal(moved_board, self.board)
        self.steps += 1
        if moved:
            self.board = moved_board
            self.score += delta
            self._spawn()
        info = self._info(delta, merges, moved)
        terminated = not bool(info["action_mask"].any())
        truncated = bool(not terminated and self.max_steps is not None
                         and self.steps >= self.max_steps)
        self.finished = terminated or truncated
        return self.board.copy(), float(delta), terminated, truncated, info

    def render(self):
        lines = [" ".join(f"{int(v):7d}" if v else "      ." for v in row)
                 for row in self.board]
        return f"Score: {self.score}  Steps: {self.steps}\n" + "\n".join(lines)
