import numpy as np
from src.evaluation import compute_metrics


# Hand-computed: +10% then -10% gives cum -1%, max drawdown -10%.
def test_metrics_hand_computed():
    m = compute_metrics([0.1, -0.1], [0.5, 0.5], [1.0, 2.0], v0=100.0)
    assert np.isclose(m["cumulative_return"], -0.01)
    assert np.isclose(m["max_drawdown"], 0.99 / 1.1 - 1.0)
    assert np.isclose(m["avg_daily_turnover"], 0.5) and np.isclose(m["total_cost_pct_of_v0"], 0.03)


# A constant positive return has no drawdown and zero volatility.
def test_metrics_constant_return():
    m = compute_metrics([0.01] * 10, [0.0] * 10, [0.0] * 10, v0=1.0)
    assert abs(m["max_drawdown"]) < 1e-12 and m["volatility"] < 1e-12  

     