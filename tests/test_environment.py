import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.environment import PortfolioEnv


def test_threshold_and_normalization():
    returns = np.zeros((2, 2))
    features = np.zeros((2, 1, 1))
    env = PortfolioEnv(returns, features, action_threshold=0.01)
    env.reset()
    state, reward, done, info = env.step(np.array([0.005, 0.995, 0.0]))
    assert np.isclose(env.current_weights.sum(), 1.0)
    assert env.current_weights[0] == 0.0


def test_transaction_cost_is_applied_once():
    returns = np.zeros((1, 1))
    features = np.zeros((1, 1, 1))
    env = PortfolioEnv(returns, features, transaction_cost=0.001, initial_value=1_000_000)
    env.reset()
    _, _, _, info = env.step(np.array([0.5, 0.5]))
    assert info["transaction_cost"] > 0
    assert np.isclose(info["portfolio_value"], 999_000.0)
