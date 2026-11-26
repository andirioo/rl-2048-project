"""Shared training loop, independent of any particular algorithm."""
import argparse
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from agents.base_agent import Transition
from agents.registry import create_agent
from env import Game2048
from reporting import prepare_output, write_json, plot_training
from rewards import compute_reward
from settings import load_config


def train(config):
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    agent = create_agent(config)
    output = prepare_output(config.output_dir)
    write_json(output / "config.json", config.to_dict())
    # Keep environment and exploration RNGs separate. Train without rendering on a step budget.
    env = Game2048(seed=config.seed)
    state, info = env.reset()
    episodes, updates = [], []
    episode_reward, episode_steps = 0.0, 0
    episode_index = 1
    started = time.perf_counter()
    episode_fields = ["episode", "env_steps", "score", "training_return", "max_tile",
                      "steps", "terminated", "truncated"]
    update_fields = ["env_steps", "updates", "loss", "epsilon", "metrics_json"]
    with (output / "train_episodes.csv").open("w", newline="", encoding="utf-8-sig") as ef, \
         (output / "train_updates.csv").open("w", newline="", encoding="utf-8-sig") as uf:
        ew, uw = csv.DictWriter(ef, fieldnames=episode_fields), csv.DictWriter(uf, fieldnames=update_fields)
        ew.writeheader()
        uw.writeheader()
        for step in range(1, config.total_steps + 1):
            decision = agent.act(state, info["action_mask"], training=True)
            if decision.action not in range(4) or not info["action_mask"][decision.action]:
                raise ValueError("Agent returned an invalid action; check the action selection implementation")
            next_state, _, terminated, truncated, next_info = env.step(decision.action)
            # Flag budget truncation explicitly. DQN still bootstraps from this nonterminal next state.
            truncated = bool(truncated or (step == config.total_steps and not terminated))
            reward = compute_reward(state, next_state, next_info, config)
            if not np.isfinite(reward):
                raise FloatingPointError("Training rewards must be finite")
            agent.observe(Transition(state.copy(), decision.action, reward, next_state.copy(),
                                     terminated, truncated, info["action_mask"].copy(),
                                     next_info["action_mask"].copy(), decision.info))
            metrics = agent.learn()
            if metrics:
                row = {"env_steps": step, "updates": metrics.get("updates"),
                       "loss": metrics.get("loss"), "epsilon": metrics.get("epsilon"),
                       "metrics_json": json.dumps(metrics, allow_nan=False)}
                uw.writerow(row)
                updates.append(row)
            episode_reward += reward
            episode_steps += 1
            state, info = next_state, next_info
            if terminated or truncated:
                row = {"episode": episode_index, "env_steps": step, "score": info["score"],
                       "training_return": episode_reward, "max_tile": info["max_tile"],
                       "steps": episode_steps, "terminated": terminated, "truncated": truncated}
                ew.writerow(row)
                ef.flush()
                uf.flush()
                episodes.append(row)
                if episode_index % 10 == 0 or step == config.total_steps:
                    print(f"Episode {episode_index} | step {step}/{config.total_steps} | "
                          f"score {info['score']} | max tile {info['max_tile']}", flush=True)
                if step < config.total_steps:
                    state, info = env.reset()
                episode_index += 1
                episode_reward, episode_steps = 0.0, 0
            if step % config.checkpoint_every == 0 and step != config.total_steps:
                agent.save(output / f"step_{step}.pt")
                print(f"Saved checkpoint at step {step}", flush=True)
    agent.save(output / "final.pt")
    elapsed = time.perf_counter() - started
    write_json(output / "run_info.json", {
        "algorithm": config.algorithm, "network": config.network,
        "environment_steps": config.total_steps,
        "completed_episodes": sum(r["terminated"] for r in episodes),
        "truncated_episodes": sum(r["truncated"] for r in episodes),
        "updates": getattr(agent, "updates", None), "seconds": elapsed,
        "steps_per_second": config.total_steps / elapsed,
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "device": config.device,
        "note": "Training run only. Run evaluate.py for independent game-score statistics."})
    plot_training(episodes, updates, output)
    print(f"Finished in {elapsed:.1f}s. Model: {output / 'final.pt'}", flush=True)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train an agent for the 2048 reinforcement learning project")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--steps", type=int, help="Override total_steps")
    parser.add_argument("--output", help="Override the output directory; it must be empty")
    parser.add_argument("--device", help="cpu or cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config.total_steps = args.steps
    if args.output is not None:
        config.output_dir = args.output
    if args.device is not None:
        config.device = args.device
    train(config)


if __name__ == "__main__":
    main()
