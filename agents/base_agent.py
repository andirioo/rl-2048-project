"""Student extension D: a general algorithm interface, independent of Q-learning or replay."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from encoders import encode_state
from models import build_model

ENVIRONMENT_ID = "2048-4x4-up-left-right-down-continue-v1"


@dataclass
class ActionResult:
    action: int
    # Algorithms such as PPO can store old_log_prob/value without changing the trainer.
    info: dict = field(default_factory=dict)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    action_info: dict = field(default_factory=dict)


class BaseAgent(ABC):
    @abstractmethod
    def act(self, state, action_mask, training=False) -> ActionResult:
        """Use training=False for evaluation; never update parameters or training counters here."""

    @abstractmethod
    def observe(self, transition: Transition):
        """Training only: store a transition or trajectory; action_info is available."""

    @abstractmethod
    def learn(self) -> dict:
        """Training only: choose when to update. Return numeric metrics, or {} if no update occurs."""

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class TorchAgent(BaseAgent):
    """Shared network and checkpoint helpers. Override methods for algorithms with multiple networks."""
    def __init__(self, config):
        self.config = config.validate()
        self.device = torch.device(config.device)
        self.model = build_model(config).to(self.device)
        self.rng = np.random.default_rng(config.seed + 1)
        self.env_steps = 0
        self.updates = 0

    def encode_batch(self, states):
        arrays = np.stack([encode_state(s, self.config.encoding, self.config.channels)
                           for s in states])
        return torch.as_tensor(arrays, dtype=torch.float32, device=self.device)

    def check_mask(self, action_mask):
        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape != (4,) or not mask.any():
            raise ValueError("act() requires at least one legal action; do not select actions in terminal states")
        return mask

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "environment": ENVIRONMENT_ID,
                   "config": self.config.to_dict(), "model": self.model.state_dict(),
                   "env_steps": self.env_steps, "updates": self.updates}
        if hasattr(self, "optimizer"):
            payload["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "target_model"):
            payload["target_model"] = self.target_model.state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path):
        payload = read_checkpoint(path)
        if payload["config"] != self.config.to_dict():
            # The device may differ; restore all other settings from the checkpoint.
            old, new = dict(payload["config"]), self.config.to_dict()
            old.pop("device", None)
            new.pop("device", None)
            if old != new:
                raise ValueError("Use registry.load_agent to restore the checkpoint configuration automatically")
        self.model.load_state_dict(payload["model"])
        if hasattr(self, "optimizer") and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        if hasattr(self, "target_model"):
            self.target_model.load_state_dict(payload.get("target_model", payload["model"]))
        self.env_steps = payload["env_steps"]
        self.updates = payload["updates"]
        self.model.eval()
        return self


def read_checkpoint(path):
    # Load only tensors and basic types, without custom pickled objects.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("environment") != ENVIRONMENT_ID:
        raise ValueError("Unsupported checkpoint format or mismatched environment/action order")
    return payload
