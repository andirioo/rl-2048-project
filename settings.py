"""Central configuration. Reject unknown fields to catch misspelled settings."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class Config:
    algorithm: str = "dqn"
    network: str = "cnn"
    encoding: str = "one_hot"
    channels: int = 32
    conv_depth: int = 32
    hidden_units: int = 128
    reward: str = "score"
    reward_scale: float = 1.0
    gamma: float = 0.99
    learning_rate: float = 0.0003
    batch_size: int = 64
    replay_capacity: int = 20000
    learning_starts: int = 1000
    train_every: int = 4
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    epsilon_schedule: str = "linear"
    double_dqn: bool = False
    grad_clip: float = 10.0
    total_steps: int = 20000
    checkpoint_every: int = 5000
    evaluation_episodes: int = 100
    seed: int = 42
    device: str = "cpu"
    torch_threads: int = 1
    output_dir: str = "outputs/dqn_cnn"

    def validate(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
        if self.epsilon_schedule not in {"linear", "cosine"}:
            raise ValueError("epsilon_schedule must be linear or cosine")
        if not isinstance(self.double_dqn, bool):
            raise ValueError("double_dqn must be true or false")
        positive = ("channels", "conv_depth", "hidden_units", "batch_size",
                    "replay_capacity", "train_every", "target_update_every",
                    "epsilon_decay_steps", "total_steps", "checkpoint_every",
                    "evaluation_episodes", "torch_threads")
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels < 2 or self.learning_starts < 0:
            raise ValueError("Require channels >= 2 and learning_starts >= 0")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.reward_scale <= 0:
            raise ValueError("learning_rate, grad_clip, and reward_scale must be > 0")
        if self.encoding not in {"one_hot", "log2"}:
            raise ValueError("encoding must be one_hot or log2")
        if self.reward not in {"score", "merge_bonus", "strategic"}:
            raise ValueError("reward must be score, merge_bonus, or strategic")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return Config(**json.loads(Path(path).read_text(encoding="utf-8-sig"))).validate()
