
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from src import report


# Builds a fake metrics folder with the exact schema run_ablation writes (M2 = M1 + 0.5 Sharpe by construction).
@pytest.fixture
def mock(tmp_path):
    rng = np.random.default_rng(0)
    start = np.datetime64("2018-01-01", "D")
    for f in (1, 2, 3, 4):
        n = 100
        dates = start + np.arange(n) + (f - 1) * n
        for mi, m in enumerate(report.MODELS):
            for s in (0, 1, 2):
                name = report.run_name(m, f, s)
                d = tmp_path / name
                d.mkdir()
                base = {k: float(rng.normal()) for k in report.METRICS}
                base["sharpe"] = 1.0 + 0.5 * (m != "M1") + 0.01 * f
                base["avg_daily_turnover"], base["total_cost_pct_of_v0"] = 0.1 / (mi + 1), 0.05 / (mi + 1)
                res = {"model": m, "fold": f, "seed": s, "selected_step": 1000,
                       "val_curve": [{"step": 100, "sharpe": 1.0, "turnover": 0.1}, {"step": 200, "sharpe": 2.0, "turnover": 0.2}],
                       "val": base, "test": base}
                (d / "metrics.json").write_text(json.dumps(res))
                r = rng.normal(0.001, 0.01, n)
                tgt = rng.dirichlet(np.ones(11), n)
                for split in ("val", "test"):
                    np.savez(d / f"daily_{split}.npz", dates=dates, r_net=r, turnover=np.full(n, 0.1),
                             cost=np.full(n, 1.0), v=1e6 * np.cumprod(1 + r), target=tgt)
    return tmp_path


# The loader finds all 48 final runs and both splits.
def test_load_runs(mock):
    df = report.load_runs(mock)
    assert len(df) == 4 * 4 * 3 * 2 and set(df.model) == set(report.MODELS)


# Paired delta recovers the constructed +0.5 Sharpe step, improving in every fold and pair.
def test_paired_delta_known_effect(mock):
    r = report.paired_delta(report.load_runs(mock), "M1", "M2", "sharpe")
    assert np.isclose(r["mean_diff"], 0.5) and r["folds_improved"] == "4/4" and r["pairs_improved"] == "12/12"
    assert r["ci95_lo"] <= 0.5 <= r["ci95_hi"]


# Lower-is-better metrics count a decrease as an improvement (turnover falls from M1 to M2 by construction).
def test_direction_for_turnover(mock):
    r = report.paired_delta(report.load_runs(mock), "M1", "M2", "avg_daily_turnover")
    assert r["mean_diff"] < 0 and r["folds_improved"] == "4/4"


# Chaining sorts by date and drops a duplicated boundary date.
def test_chain_drops_duplicates():
    a = {"dates": np.array(["2018-01-01", "2018-01-02"], dtype="datetime64[D]"), "r_net": np.array([0.1, 0.2])}
    b = {"dates": np.array(["2018-01-02", "2018-01-03"], dtype="datetime64[D]"), "r_net": np.array([0.2, 0.3])}
    c = report.chain([a, b])
    assert len(c["dates"]) == 3 and np.allclose(c["r_net"], [0.1, 0.2, 0.3])


# Tables have one row per model and render mean ± std; every plot function runs and writes its PNG.
def test_tables_and_plots(mock, tmp_path):
    df = report.load_runs(mock)
    assert list(report.table_by_fold(df, "sharpe").index) == report.MODELS
    pooled = report.table_pooled(df, "test", v0=1e6)
    assert "±" in pooled.loc["M2", "sharpe"] and "profit_per_window" in pooled.columns
    out = tmp_path / "figs"
    start = np.datetime64("2018-01-01", "D")
    ew = [{"dates": start + np.arange(100) + 100 * k, "r_net": np.full(100, 0.0005),
           "v": 1e6 * (1.0005) ** np.arange(1, 101)} for k in range(4)]
    report.plot_equity(mock, ew, 1e6, out); report.plot_folds(mock, ew, 1e6, out)
    report.plot_drawdown(mock, ew, 1e6, out); report.plot_profit_bars(df, 1e6, out)
    report.plot_turnover_cost(df, out); report.plot_val_curves(mock, out)
    assert len(list(out.glob("*.png"))) == 6  
          