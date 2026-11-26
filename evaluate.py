"""Evaluate each model over 100 episodes by default, with shared metrics for all algorithms."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import torch
from agents.registry import load_agent
from baselines import BaselineAgent
from env import Game2048
from reporting import prepare_output, write_csv, write_json, summarize, plot_evaluation


def evaluate_agent(agent, episodes=100, seed=2026, max_steps=20000, progress=True,
                   render=False, delay=0.0):
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    env = Game2048(max_steps=max_steps)
    records = []
    # Use no_grad and training=False. Never call observe or learn during evaluation.
    with torch.no_grad():
        for episode in range(1, episodes + 1):
            state, info = env.reset(seed=seed + episode - 1)
            if render:
                print(env.render(), flush=True)
            while True:
                decision = agent.act(state, info["action_mask"], training=False)
                if decision.action not in range(4) or not info["action_mask"][decision.action]:
                    raise ValueError(f"In episode {episode}, the agent returned invalid action {decision.action}")
                state, _, terminated, truncated, info = env.step(decision.action)
                if render:
                    print(env.render(), flush=True)
                    if delay:
                        time.sleep(delay)
                if terminated or truncated:
                    break
            records.append({"episode": episode, "environment_seed": seed + episode - 1,
                            "score": info["score"], "max_tile": info["max_tile"],
                            "steps": info["steps"], "terminated": terminated,
                            "truncated": truncated})
            if progress and (episode % 10 == 0 or episode == episodes):
                print(f"Evaluation {episode}/{episodes} | score {info['score']} | "
                      f"max tile {info['max_tile']}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2048 agents and export episode statistics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Model checkpoint (.pt) saved during training")
    group.add_argument("--baseline", choices=["random", "greedy"])
    parser.add_argument("--episodes", type=int, default=None, help="Default: 100, or the value stored in the model configuration")
    parser.add_argument("--seed", type=int, default=2026, help="Starting environment seed for evaluation, separate from training")
    parser.add_argument("--max-steps", type=int, default=20000, help="Per-episode step limit; truncated episodes make the evaluation incomplete")
    parser.add_argument("--output", help="New evaluation output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--render", action="store_true", help="Render each board in the terminal; use one episode for a demonstration")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.checkpoint:
        agent = load_agent(args.checkpoint, args.device)
        model_name = f"{agent.config.algorithm}_{agent.config.network}"
        episodes = args.episodes if args.episodes is not None else agent.config.evaluation_episodes
    else:
        agent = BaselineAgent(args.baseline, seed=args.seed + 1000000)
        model_name = args.baseline
        episodes = args.episodes if args.episodes is not None else 100
    output = prepare_output(args.output or f"outputs/eval_{model_name}_{datetime.now():%Y%m%d_%H%M%S_%f}")
    records = evaluate_agent(agent, episodes, args.seed, args.max_steps, render=args.render, delay=args.delay)
    summary = summarize(records, model_name)
    write_csv(output / "episodes.csv", records)
    write_csv(output / "summary.csv", [summary])
    write_json(output / "summary.json", summary)
    write_json(output / "evaluation_config.json", {
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "baseline": args.baseline, "episodes": episodes, "environment_seed_start": args.seed,
        "max_steps": args.max_steps, "device": args.device,
        "policy": "greedy masked actions for learned agents; random tie-break for baselines",
        "metric_denominator": "completed natural-terminal episodes only",
        "training_config": agent.config.to_dict() if args.checkpoint else None})
    plot_evaluation(records, summary, output)
    print(f"\nModel: {model_name}; completed {summary['episodes_completed']}/{episodes}")
    print(f"Mean score: {summary['mean_score']}; sample std: {summary['score_std']}")
    for tile in (512, 1024, 2048):
        rate = summary[f"rate_{tile}"]
        print(f">={tile}: {rate:.1%}" if rate is not None else f">={tile}: unavailable")
    print(f"Results: {output}")
    if not summary["evaluation_complete"]:
        print("INCOMPLETE: truncated games were excluded. Increase --max-steps and evaluate again.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
