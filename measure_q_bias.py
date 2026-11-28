"""Measure how optimistic a trained DQN is (Member 1).

Plays greedy games with a checkpoint, the same way evaluate.py does, and compares
  predicted:  Q(s, a), the network's estimate for the move it actually played
  actual:     G = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...  (the discounted
              training reward that really followed, until the game ended)
Overestimation = mean(predicted - actual). A positive value means the network
expects more future reward than it actually gets (the over-optimism Double DQN targets).

Example:
  python measure_q_bias.py --checkpoint outputs/kayzin_ddqn_softmax/final.pt --output outputs/kayzin_ddqn_softmax_qbias
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from agents.registry import load_agent
from env import Game2048
from rewards import compute_reward


def measure(agent, episodes=100, seed=2026, max_steps=20000):
    config = agent.config
    env = Game2048(max_steps=max_steps)
    predicted, actual, per_game = [], [], []
    agent.model.eval()
    with torch.no_grad():
        for episode in range(episodes):
            state, info = env.reset(seed=seed + episode)
            q_values, rewards = [], []
            while True:
                action = agent.act(state, info["action_mask"], training=False).action
                q = agent.model(agent.encode_batch([state]))[0, action].item()
                next_state, _, terminated, truncated, next_info = env.step(action)
                q_values.append(q)
                rewards.append(compute_reward(state, next_state, next_info, config))
                state, info = next_state, next_info
                if terminated or truncated:
                    break
            # Discounted return from every step to the end of the game (no bootstrap at the end).
            returns, running = np.zeros(len(rewards)), 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running = rewards[t] + config.gamma * running
                returns[t] = running
            predicted.extend(q_values)
            actual.extend(returns.tolist())
            per_game.append({"episode": episode + 1, "score": info["score"], "steps": len(rewards),
                             "mean_predicted_q": float(np.mean(q_values)),
                             "mean_actual_return": float(np.mean(returns)),
                             "mean_overestimation": float(np.mean(np.array(q_values) - returns))})
    predicted, actual = np.array(predicted), np.array(actual)
    summary = {"episodes": episodes, "evaluation_seed": seed, "steps": int(len(predicted)),
               "mean_predicted_q": float(predicted.mean()),
               "mean_actual_return": float(actual.mean()),
               "mean_overestimation": float((predicted - actual).mean()),
               "relative_overestimation": float((predicted - actual).mean() / actual.mean()),
               "share_of_steps_overestimated": float((predicted > actual).mean()),
               "double_dqn": bool(config.double_dqn), "epsilon_schedule": config.epsilon_schedule,
               "reward": config.reward}
    return summary, per_game


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    agent = load_agent(args.checkpoint)
    if agent.config.algorithm != "dqn":
        raise ValueError("measure_q_bias.py needs a DQN checkpoint (it reads Q-values)")
    summary, per_game = measure(agent, args.episodes, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "q_bias_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "q_bias_per_game.json").write_text(json.dumps(per_game, indent=2), encoding="utf-8")
    print(f"mean predicted Q {summary['mean_predicted_q']:.1f} | mean actual return "
          f"{summary['mean_actual_return']:.1f} | overestimation {summary['mean_overestimation']:+.1f} "
          f"({summary['relative_overestimation']:+.1%}) | steps overestimated "
          f"{summary['share_of_steps_overestimated']:.0%}")


if __name__ == "__main__":
    main()
