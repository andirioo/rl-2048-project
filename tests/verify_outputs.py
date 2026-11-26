"""Independently check saved evaluation statistics and the truncation CLI exit code.

Run the validation commands in README or use the included validation_* outputs first.
This script does not retrain models.
"""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    checked = []
    for name in ("dqn", "reinforce", "student", "random", "greedy"):
        directory = ROOT / "outputs" / f"validation_{name}_eval"
        with (directory / "episodes.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert len(rows) == 100
        assert all(r["terminated"] == "True" and r["truncated"] == "False" for r in rows)
        scores = [int(r["score"]) for r in rows]
        assert abs(statistics.mean(scores) - summary["mean_score"]) < 1e-9
        assert abs(statistics.stdev(scores) - summary["score_std"]) < 1e-9
        assert summary["episodes_completed"] == 100 and summary["evaluation_complete"]
        for threshold in (512, 1024, 2048):
            count = sum(int(r["max_tile"]) >= threshold for r in rows)
            assert count == summary[f"reached_{threshold}"]
            assert count / 100 == summary[f"rate_{threshold}"]
        checked.append(name)
    with tempfile.TemporaryDirectory() as directory:
        result = subprocess.run([sys.executable, str(ROOT / "evaluate.py"),
                                 "--baseline", "random", "--episodes", "2",
                                 "--max-steps", "1", "--output", directory],
                                cwd=ROOT, capture_output=True)
        assert result.returncode == 2, result.stderr.decode(errors="replace")
        data = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
        assert data["episodes_truncated"] == 2 and data["episodes_completed"] == 0
        assert data["mean_score"] is None
    print("Verified 100 complete episodes and all CSV statistics for:", ", ".join(checked))
    print("Verified truncation CLI exit code: 2")


if __name__ == "__main__":
    main()
