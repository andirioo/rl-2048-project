"""Algorithm-independent evaluation statistics and plots."""
import csv
import json
from pathlib import Path
import numpy as np


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new directory to preserve existing results.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No records to write")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(episodes, model_name):
    """Summarize complete natural-terminal episodes; explicitly flag incomplete evaluations."""
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    completed = [r for r in episodes if bool(r["terminated"]) and not bool(r["truncated"])]
    scores = np.array([r["score"] for r in completed], dtype=float)
    maxima = np.array([r["max_tile"] for r in completed], dtype=int)
    n = len(completed)
    result = {"model": model_name, "episodes_requested": len(episodes),
              "episodes_completed": n, "episodes_truncated": len(episodes) - n,
              "evaluation_complete": n == len(episodes),
              "mean_score": float(scores.mean()) if n else None,
              # Sample standard deviation uses ddof=1; return null when only one episode is available.
              "score_std": float(scores.std(ddof=1)) if n > 1 else None,
              "std_ddof": 1}
    for threshold in (512, 1024, 2048):
        count = int((maxima >= threshold).sum())
        result[f"reached_{threshold}"] = count
        result[f"rate_{threshold}"] = count / n if n else None
    return result


def _pyplot():
    import os
    cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_evaluation(episodes, summary, output_dir):
    plt = _pyplot()
    completed = [r for r in episodes if r["terminated"] and not r["truncated"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    if completed:
        scores = [r["score"] for r in completed]
        axes[0].hist(scores, bins=min(15, max(1, len(scores))), color="#376c9d", edgecolor="white")
        axes[0].axvline(summary["mean_score"], color="#c55a35", label="Mean")
        axes[0].legend()
        rates = [summary[f"rate_{t}"] * 100 for t in (512, 1024, 2048)]
        bars = axes[1].bar(["512", "1024", "2048"], rates, color="#528573")
        axes[1].bar_label(bars, labels=[f"{r:.1f}%" for r in rates], padding=3)
    else:
        axes[0].text(.5, .5, "No completed episodes", ha="center", transform=axes[0].transAxes)
    axes[0].set(xlabel="Final game score", ylabel="Episodes", title="Score distribution")
    axes[1].set(xlabel="Maximum tile >= threshold", ylabel="Episodes (%)",
                ylim=(0, 110), title="Tile achievement rates")
    status = "complete" if summary["evaluation_complete"] else "INCOMPLETE: truncated episodes excluded"
    fig.suptitle(f"{summary['model']} | {len(completed)} completed episodes | {status}")
    fig.savefig(Path(output_dir) / "evaluation.png", dpi=150)
    plt.close(fig)


def plot_training(episodes, updates, output_dir):
    plt = _pyplot()
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    complete = [r for r in episodes if r["terminated"]]
    for ax, key, label in [(axes[0], "score", "Game score"),
                            (axes[1], "training_return", "Training reward sum")]:
        if complete:
            x = [r["env_steps"] for r in complete]
            y = np.array([r[key] for r in complete], dtype=float)
            ax.plot(x, y, alpha=.3, color="#376c9d", label="Episode")
            window = min(20, len(y))
            smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            ax.plot(x[window - 1:], smooth, color="#376c9d", label=f"Mean of {window} episodes")
            ax.legend()
        ax.set(xlabel="Environment steps", ylabel=label)
    valid = [r for r in updates if r.get("loss") is not None]
    if valid:
        axes[2].plot([r["env_steps"] for r in valid], [r["loss"] for r in valid], color="#a36140")
    axes[2].set(xlabel="Environment steps", ylabel="Training loss")
    fig.suptitle("Training diagnostics (not independent evaluation)")
    fig.savefig(Path(output_dir) / "training.png", dpi=150)
    plt.close(fig)
