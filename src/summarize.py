"""Collect all runs' metrics.json into one table: python -m src.summarize [--split test]"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .config import get_paths


# Loads every metrics.json under outputs/metrics into a flat DataFrame, one row per run and split.
def load_runs(split):
    rows = []
    for f in sorted(Path(get_paths()["metrics"]).glob("*/metrics.json")):
        r = json.loads(f.read_text())
        rows.append({"run": f.parent.name, "model": r["model"], "fold": r["fold"], "seed": r["seed"],
                     "lam": r["lam"], "steps": r["train_steps"], **r[split]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    df = load_runs(ap.parse_args().split)
    cols = ["run", "cumulative_return", "annualized_return", "sharpe", "sortino", "max_drawdown",
            "avg_daily_turnover", "total_cost_pct_of_v0"]
    print(df[cols].round(4).to_string(index=False) if len(df) else "no runs found")