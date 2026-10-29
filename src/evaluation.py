"""Greedy out-of-sample evaluation, the common metrics, and the equal-weight benchmark."""
import json
from pathlib import Path

import numpy as np
import torch

ANN = 252  # trading days per year


# Runs the policy's mean action (Dirichlet mean) through one segment and records daily accounting.
def run_policy(model, env, device):
    obs, _ = env.reset() 
    rec = {k: [] for k in ("index", "r_net", "turnover", "cost", "v", "target")}
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs, _, _, done, info = env.step(model.mean_action(x))   # env applies threshold itself
        for k in rec:
            rec[k].append(info[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}


# Equal-weight benchmark: 1/N in every stock, zero cash, rebalanced daily, same env/costs/metrics.
def run_equal_weight(env):
    env.reset()
    w = np.zeros(env.action_dim)
    w[:env.N] = 1.0 / env.N
    rec = {k: [] for k in ("index", "r_net", "turnover", "cost", "v")}
    done = False
    while not done:
        _, _, _, done, info = env.step(w)
        for k in rec:
            rec[k].append(info[k])
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}


# Computes the agreed metrics from daily net returns, turnover and costs.
def compute_metrics(r_net, turnover, cost, v0):
    r = np.asarray(r_net, dtype=np.float64)
    wealth = np.concatenate([[v0], v0 * np.cumprod(1.0 + r)])
    cum = wealth[-1] / v0 - 1.0
    ann_ret = (1.0 + cum) ** (ANN / len(r)) - 1.0
    sd = r.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))         # semi-deviation vs 0
    peak = np.maximum.accumulate(wealth)
    return {
        "cumulative_return": float(cum),
        "annualized_return": float(ann_ret),
        "volatility": float(sd * np.sqrt(ANN)),
        "sharpe": float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
        "sortino": float(r.mean() / downside * np.sqrt(ANN)) if downside > 0 else float("nan"),
        "max_drawdown": float((wealth / peak - 1.0).min()),
        "avg_daily_turnover": float(np.mean(turnover)),
        "total_cost_pct_of_v0": float(np.sum(cost) / v0),
        "n_days": int(len(r)),
    }


# Writes EW metrics for every fold into outputs/metrics/EW_fold{k}/metrics.json (same schema as model runs).
def main():
    from .config import CFG, config_hash, get_paths
    from .data import build_folds
    from .environment import EnvParams, PortfolioEnv
    from .features import build_dataset

    ds, _ = build_dataset()
    params = EnvParams.from_cfg("M1")                           # accounting is identical for every model
    for k, fold in enumerate(build_folds(ds.dates), start=1):
        res = {"model": "EW", "seed": 0, "fold": k, "train_steps": 0, "reward": "none", "lam": 0.0,
               "config_hash": config_hash(CFG)}
        for split in ("val", "test"):
            rec = run_equal_weight(PortfolioEnv(ds.X, ds.ret_next, *fold[split], params))
            res[split] = compute_metrics(rec["r_net"], rec["turnover"], rec["cost"], params.v0)
        out = Path(get_paths()["metrics"]) / f"EW_fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
        print(f"fold {k}: val sharpe {res['val']['sharpe']:.3f} | test sharpe {res['test']['sharpe']:.3f} "
              f"| test cum {res['test']['cumulative_return']:.3f}")


if __name__ == "__main__":
    main()