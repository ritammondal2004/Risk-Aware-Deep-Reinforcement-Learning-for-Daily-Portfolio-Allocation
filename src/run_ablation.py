"""Single entry point for all four ablation models.

Usage:
    python src/run_ablation.py --model M1 --seed 42 --fold 1
"""

from __future__ import annotations

import argparse
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["M1", "M2", "M3", "M4"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Model={args.model} | seed={args.seed} | fold={args.fold}")
    print("Training pipeline scaffold is ready. Implement the common PPO rollout/update loop here.")
    print("Keep environment, feature processing, folds, budgets and evaluation identical across M1-M4.")


if __name__ == "__main__":
    main()
