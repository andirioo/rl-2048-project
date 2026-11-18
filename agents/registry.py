from agents.base_agent import read_checkpoint
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import ReinforceAgent
from agents.student_agent import StudentAgent
from settings import Config

# STUDENT EDIT E4: Edit StudentAgent directly or register a new class here.
AGENTS = {"dqn": DQNAgent, "reinforce": ReinforceAgent, "student": StudentAgent}


def create_agent(config):
    if config.algorithm not in AGENTS:
        raise ValueError(f"Unknown algorithm {config.algorithm}; available options: {list(AGENTS)}")
    return AGENTS[config.algorithm](config)


def load_agent(path, device="cpu"):
    payload = read_checkpoint(path)
    config = Config(**{**payload["config"], "device": device}).validate()
    agent = create_agent(config)
    agent.load(path)
    return agent
