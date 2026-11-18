"""Student extension A: encode the raw board as neural network input."""
import numpy as np


def encode_state(board, encoding="one_hot", channels=32):
    """Input: (4, 4). Output: float32 (C, 4, 4), using PyTorch channel-first layout."""
    board = np.asarray(board)
    powers = np.zeros((4, 4), dtype=np.int64)
    positive = board > 0
    powers[positive] = np.log2(board[positive]).astype(np.int64)
    # STUDENT EDIT A: Add encodings here; also update input_channels and config validation.
    if encoding == "one_hot":
        if powers.max() >= channels:
            raise ValueError(f"Tile exceeds encoding range; increase channels (currently {channels})")
        return np.eye(channels, dtype=np.float32)[powers].transpose(2, 0, 1).copy()
    if encoding == "log2":
        return (powers.astype(np.float32) / 32.0)[None, :, :]
    raise ValueError(f"Unknown encoding: {encoding}")


def input_channels(config):
    return config.channels if config.encoding == "one_hot" else 1
