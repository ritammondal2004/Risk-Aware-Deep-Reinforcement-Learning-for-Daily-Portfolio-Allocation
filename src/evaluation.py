
"""Greedy out-of-sample evaluation and the common metrics, identical for M1-M4."""
import numpy as np
import torch

ANN = 252  # trading days per year
          

# Runs the policy's mean action (Dirichlet mean) through one segment and records daily accounting.
def run_policy(model, env, device):
    obs, _ = env.reset()
    rec = {k: [] for k in ("index", "r_net", "turnover", "cost", "v")}
    done = False
    while not done:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs, _, _, done, info = env.step(model.mean_action(x))   # env applies threshold itself
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
