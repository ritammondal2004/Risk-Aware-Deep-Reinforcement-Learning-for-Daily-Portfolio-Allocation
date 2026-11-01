
"""Results analysis for the M1-M4 ablation: tables, paired comparisons and plots from saved run folders."""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["M1", "M2", "M3", "M4"]
METRICS = ["cumulative_return", "annualized_return", "volatility", "sharpe", "sortino",
           "max_drawdown", "avg_daily_turnover", "total_cost_pct_of_v0"]
# +1: higher is better, -1: lower is better (used to count how often a step improves a metric).
DIRECTION = {"cumulative_return": 1, "annualized_return": 1, "sharpe": 1, "sortino": 1, "max_drawdown": 1,
             "volatility": -1, "avg_daily_turnover": -1, "total_cost_pct_of_v0": -1}
STEPS = [("M1", "M2", "Temporal memory (M2 - M1)"), ("M2", "M3", "DSR reward (M3 - M2)"),
         ("M3", "M4", "Turnover penalty (M4 - M3)"), ("M1", "M4", "All combined (M4 - M1)")]
# Colour-blind-safe categorical palette (validated: lightness band, CVD separation, contrast); EW is neutral grey.
COLORS = {"M1": "#0072B2", "M2": "#D55E00", "M3": "#009E73", "M4": "#7B5EA7", "EW": "#555555"}
LABELS = {"M1": "M1 PPO (MLP)", "M2": "M2 LSTM-PPO", "M3": "M3 + DSR", "M4": "M4 + turnover penalty",
          "EW": "Equal-weight"}


# Folder name of one final-grid run (M4 carries the chosen lambda tag, like run_ablation names it).
def run_name(model, fold, seed, lam="0.1"):
    return f"{model}_fold{fold}_seed{seed}" + (f"_lam{lam}" if model == "M4" else "")


# Reads metrics.json of every final run into one long table (one row per model/fold/seed/split).
def load_runs(metrics_dir, lam="0.1", folds=(1, 2, 3, 4), seeds=(0, 1, 2)):
    rows, missing = [], []
    for m in MODELS:
        for f in folds:
            for s in seeds:
                p = Path(metrics_dir) / run_name(m, f, s, lam) / "metrics.json"
                if not p.exists():
                    missing.append(p.parent.name)
                    continue
                r = json.loads(p.read_text())
                for split in ("val", "test"):
                    rows.append({"model": m, "fold": f, "seed": s, "split": split,
                                 "selected_step": r["selected_step"], **r[split]})
    if missing:
        print("MISSING RUNS:", missing)
    return pd.DataFrame(rows)


# Formats mean and std as text;
def _fmt(mean, std, digits):
    return f"{mean:.{digits}f}" if (std is None or np.isnan(std)) else f"{mean:.{digits}f} ± {std:.{digits}f}"


# Mean ± std over seeds for each model and fold, for one metric (test by default). 
def table_by_fold(df, metric, split="test", digits=3):
    g = df[df.split == split].groupby(["model", "fold"])[metric].agg(["mean", "std"])
    s = pd.Series([_fmt(r["mean"], r["std"], digits) for _, r in g.iterrows()], index=g.index)
    order = [m for m in MODELS + ["EW"] if m in s.index.get_level_values(0)]
    return s.unstack("fold").reindex(order)


# Averages the folds within each seed, then reports mean ± std over seeds; profit = cum. return x V0 per window.
def table_pooled(df, split="test", v0=None, digits=3):
    d = df[df.split == split].copy()
    cols = list(METRICS)
    if v0 is not None:
        d["profit_per_window"] = d["cumulative_return"] * v0
        cols.append("profit_per_window")
    per_seed = d.groupby(["model", "seed"])[cols].mean()
    out = {}
    for m in [m for m in MODELS + ["EW"] if m in per_seed.index.get_level_values(0)]:
        x = per_seed.loc[m]
        out[m] = {c: _fmt(x[c].mean(), x[c].std(ddof=1) if len(x) > 1 else np.nan, digits if c != "profit_per_window" else 0)
                  for c in cols}
    return pd.DataFrame(out).T


# Paired difference b - a over (fold, seed); CI from a bootstrap that resamples whole folds (seeds averaged inside).
def paired_delta(df, a, b, metric, split="test", n_boot=10000, seed=0):
    p = df[df.split == split].pivot_table(index=["fold", "seed"], columns="model", values=metric)
    diff = (p[b] - p[a]).dropna()
    by_fold = diff.groupby(level="fold").mean()
    rng = np.random.default_rng(seed)
    v = by_fold.values
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sign = DIRECTION[metric]
    return {"metric": metric, "mean_diff": diff.mean(), "ci95_lo": lo, "ci95_hi": hi,
            "folds_improved": f"{int((sign * by_fold > 0).sum())}/{len(by_fold)}",
            "pairs_improved": f"{int((sign * diff > 0).sum())}/{len(diff)}"}


# Table of the ablation steps (memory, reward, penalty, combined) for the chosen metrics.
def delta_table(df, metrics=("sharpe", "cumulative_return", "max_drawdown", "avg_daily_turnover",
                            "total_cost_pct_of_v0"), split="test"):
    rows = [{"step": label, **paired_delta(df, a, b, m, split)} for a, b, label in STEPS for m in metrics]
    return pd.DataFrame(rows)


# Loads one run's saved daily arrays (dates, r_net, turnover, cost, v, target) for a split.
def load_daily(metrics_dir, name, split="test"):
    z = np.load(Path(metrics_dir) / name / f"daily_{split}.npz")
    return {k: z[k] for k in z.files}


# Chains per-fold daily parts into one continuous date-sorted series (duplicate dates dropped).
def chain(parts, keys=("r_net", "turnover", "cost", "target")):
    dates = np.concatenate([p["dates"] for p in parts]).astype("datetime64[D]")
    order = np.argsort(dates, kind="stable")
    dates = dates[order]
    keep = np.r_[True, dates[1:] != dates[:-1]]
    out = {"dates": dates[keep]}
    for k in keys:
        if k in parts[0]:
            out[k] = np.concatenate([p[k] for p in parts])[order][keep]
    return out


# One chained 2018-2025 series for a model and seed (the four test windows back to back).
def stitched(metrics_dir, model, seed, lam="0.1", folds=(1, 2, 3, 4)):
    return chain([load_daily(metrics_dir, run_name(model, f, seed, lam)) for f in folds])


# Equal-weight daily series per fold, from the same environment, costs and test windows as the models.
def ew_daily(ds, folds, params, split="test"):
    from .environment import PortfolioEnv
    from .evaluation import run_equal_weight
    parts = []
    for fold in folds:
        rec = run_equal_weight(PortfolioEnv(ds.X, ds.ret_next, *fold[split], params))
        parts.append({"dates": np.asarray(ds.dates)[rec["index"].astype(int)].astype("datetime64[D]"), **rec})
    return parts


# Equal-weight metrics per fold as rows shaped like the model rows, so the same tables include them.
def ew_rows(ew_parts, v0):
    from .evaluation import compute_metrics
    rows = [{"model": "EW", "fold": k, "seed": 0, "split": "test", "selected_step": 0,
             **compute_metrics(p["r_net"], p["turnover"], p["cost"], v0)} for k, p in enumerate(ew_parts, 1)]
    return pd.DataFrame(rows)


# Light axes: no top/right frame, faint grid.
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


# Saves a figure to out_dir/name.png when out_dir is given.
def _save(fig, out_dir, name):
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{name}.png", dpi=140, bbox_inches="tight")


# Cumulative profit of the stitched 2018-2025 test period: seed-mean line with min-max band per model, plus EW.
def plot_equity(metrics_dir, ew_parts, v0, out_dir=None, lam="0.1", seeds=(0, 1, 2)):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in MODELS:
        W = []
        for s in seeds:
            ch = stitched(metrics_dir, m, s, lam)
            W.append(v0 * np.cumprod(1 + ch["r_net"]) - v0)
        W, d = np.array(W), ch["dates"]
        ax.fill_between(d, W.min(0), W.max(0), color=COLORS[m], alpha=0.12, linewidth=0)
        ax.plot(d, W.mean(0), color=COLORS[m], lw=1.8, label=LABELS[m])
    ch = chain(ew_parts)
    ax.plot(ch["dates"], v0 * np.cumprod(1 + ch["r_net"]) - v0, color=COLORS["EW"], lw=1.8, ls="--", label=LABELS["EW"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Cumulative profit on the out-of-sample test windows (4 folds chained, seed mean and range)")
    ax.set_ylabel("profit (currency units, start capital reinvested)")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, out_dir, "equity_stitched")
    return fig


# One panel per fold: wealth minus start capital over that fold's test window, seed mean, with EW.
def plot_folds(metrics_dir, ew_parts, v0, out_dir=None, lam="0.1", folds=(1, 2, 3, 4), seeds=(0, 1, 2)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    for ax, f in zip(axes.ravel(), folds):
        for m in MODELS:
            runs = [load_daily(metrics_dir, run_name(m, f, s, lam)) for s in seeds]
            ax.plot(runs[0]["dates"], np.mean([r["v"] for r in runs], axis=0) - v0, color=COLORS[m], lw=1.6, label=LABELS[m])
        e = ew_parts[f - 1]
        ax.plot(e["dates"], e["v"] - v0, color=COLORS["EW"], lw=1.6, ls="--", label=LABELS["EW"])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"Fold {f} test window")
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Profit per fold test window (seed mean)")
    fig.tight_layout()
    _save(fig, out_dir, "equity_per_fold")
    return fig


# Drawdown of the stitched test period for each model (seed mean) and EW.
def plot_drawdown(metrics_dir, ew_parts, v0, out_dir=None, lam="0.1", seeds=(0, 1, 2)):
    fig, ax = plt.subplots(figsize=(10, 4))

    def dd(r):
        w = np.concatenate([[1.0], np.cumprod(1 + r)])
        return (w / np.maximum.accumulate(w) - 1)[1:]

    for m in MODELS:
        D = [dd(stitched(metrics_dir, m, s, lam)["r_net"]) for s in seeds]
        ax.plot(stitched(metrics_dir, m, seeds[0], lam)["dates"], np.mean(D, axis=0) * 100, color=COLORS[m], lw=1.4, label=LABELS[m])
    ch = chain(ew_parts)
    ax.plot(ch["dates"], dd(ch["r_net"]) * 100, color=COLORS["EW"], lw=1.4, ls="--", label=LABELS["EW"])
    ax.set_title("Drawdown from the running peak (stitched test windows, seed mean)")
    ax.set_ylabel("drawdown (%)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _style(ax)
    _save(fig, out_dir, "drawdown")
    return fig


# Grouped bars: test profit per fold window (mean over seeds, error bar = std over seeds) for each model and EW.
def plot_profit_bars(df, v0, out_dir=None):
    d = df[df.split == "test"].copy()
    d["profit"] = d["cumulative_return"] * v0
    folds = sorted(d.fold.unique())
    names = MODELS + ["EW"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(names)
    for i, m in enumerate(names):
        g = d[d.model == m].groupby("fold")["profit"].agg(["mean", "std"]).reindex(folds)
        ax.bar(np.arange(len(folds)) + i * w, g["mean"], w * 0.92, yerr=g["std"].fillna(0), color=COLORS[m],
               label=LABELS[m], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(folds)) + 0.4 - w / 2)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("profit over the 2-year test window")
    ax.set_title("Test-window profit by model (mean ± std over seeds)")
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _style(ax)
    _save(fig, out_dir, "profit_bars")
    return fig


# Two panels: average daily turnover and total transaction cost (% of start capital) per model, test windows.
def plot_turnover_cost(df, out_dir=None):
    d = df[df.split == "test"]
    names = MODELS + ["EW"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], "avg_daily_turnover", "Average daily turnover"),
                           (axes[1], "total_cost_pct_of_v0", "Transaction cost over the window (fraction of start capital)")):
        g = d.groupby(["model", "seed"])[col].mean().groupby("model").agg(["mean", "std"]).reindex(names)
        ax.bar(range(len(names)), g["mean"], 0.6, yerr=g["std"].fillna(0), color=[COLORS[m] for m in names], capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10)
        _style(ax)
    fig.tight_layout()
    _save(fig, out_dir, "turnover_cost")
    return fig


# Validation Sharpe and turnover against training step (mean over folds and seeds) - shows the overtraining effect.
def plot_val_curves(metrics_dir, out_dir=None, lam="0.1", folds=(1, 2, 3, 4), seeds=(0, 1, 2)):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in MODELS:
        curves = [json.loads((Path(metrics_dir) / run_name(m, f, s, lam) / "metrics.json").read_text())["val_curve"]
                  for f in folds for s in seeds]
        steps = [c["step"] for c in curves[0]]
        for ax, key in ((axes[0], "sharpe"), (axes[1], "turnover")):
            ax.plot(steps, np.mean([[c[key] for c in cv] for cv in curves], axis=0), color=COLORS[m], lw=1.8, label=LABELS[m])
    axes[0].set_title("Validation Sharpe vs training step")
    axes[1].set_title("Validation turnover vs training step")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("environment steps")
        _style(ax)
    fig.tight_layout()
    _save(fig, out_dir, "validation_curves")
    return fig


# Writes the tables to CSV files in out_dir.
def save_tables(out_dir, **tables):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(Path(out_dir) / f"{name}.csv")  
                   