"""Student entry point for replacing the entire algorithm.

The default inherits REINFORCE and runs with algorithm='student'.
Keep a working copy before editing act/observe/learn. Inheriting DQN is not required.
See STUDENT_GUIDE.md. Update the description below when you change the algorithm.
"""
from agents.reinforce_agent import ReinforceAgent


class StudentAgent(ReinforceAgent):
    """Default behavior: REINFORCE. This template alone is not a new algorithm."""

    def act(self, state, action_mask, training=False):
        # STUDENT EDIT E1: Return ActionResult(action, info).
        # Example PPO metadata: info={'old_log_prob': ..., 'value': ...}.
        return super().act(state, action_mask, training)

    def observe(self, transition):
        # STUDENT EDIT E2: Use rollouts, complete episodes, or replay as your algorithm requires.
        # The transition includes terminated/truncated and both states' legal action masks.
        return super().observe(transition)

    def learn(self):
        # STUDENT EDIT E3: The algorithm controls update timing, targets, loss, and optimization.
        # The default uses neither Q-learning nor the DQN replay.py module.
        return super().learn()

    # If adding actor/critic networks, override save/load to include all parameters.
