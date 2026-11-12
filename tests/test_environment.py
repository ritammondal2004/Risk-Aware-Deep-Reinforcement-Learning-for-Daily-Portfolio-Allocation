
import numpy as np
import pytest

from src.environment import DSR, EnvParams, PortfolioEnv, threshold_and_renormalize

N, F, W, T = 3, 13, 5, 80


def make(reward="net_return", lam=0.0, tau=0.01, c=0.001, start=10, end=60, seed=0):
    rng = np.random.default_rng(seed)
    X, R = rng.normal(size=(T, N * F)), rng.normal(0, 0.01, size=(T, N))
    p = EnvParams(window=W, c_trans=c, v0=1e6, tau=tau, reward=reward, lam=lam, eta=0.01, eps=1e-8)
    return X, R, PortfolioEnv(X, R, start, end, p)
           

def run(env, acts):
    env.reset()  
    out = []
    for a in acts:
        _, r, _, trunc, info = env.step(a)
        out.append((r, info))
        if trunc:
            break
    return out


ACTS = np.random.default_rng(1).dirichlet(np.ones(N + 1), size=50)


def test_hand_computed_accounting():
    X, R, env = make()
    env.reset()
    _, r, _, _, info = env.step(np.array([0.5, 0.5, 0.0, 0.0]))
    cost = 0.001 * 1e6 * 1.0
    v = (1e6 - cost) * (0.5 * (1 + R[10, 0]) + 0.5 * (1 + R[10, 1]))
    assert info["cost"] == pytest.approx(cost)
    assert info["v"] == pytest.approx(v)
    assert r == pytest.approx(v / 1e6 - 1)               # reward is R_net: cost not subtracted twice

          
def test_all_cash_costs_nothing_and_earns_nothing():
    _, _, env = make()
    env.reset()
    _, r, _, _, info = env.step(np.array([0, 0, 0, 1.0]))
    assert info["cost"] == 0 and info["v"] == 1e6 and r == 0


def test_holding_the_drifted_portfolio_costs_nothing():
    _, _, env = make()
    env.reset()
    _, _, _, _, i1 = env.step(np.array([0.5, 0.5, 0, 0]))
    _, _, _, _, i2 = env.step(i1["w_next"])
    assert i2["turnover"] == pytest.approx(0, abs=1e-9)


def test_threshold_rule():
    w = threshold_and_renormalize([0.005, 0.2, 0.3, 0.495], 0.01)
    assert w[0] == 0 and w.sum() == pytest.approx(1) and (w >= 0).all()
    assert w[1] == pytest.approx(0.2 / 0.995)


def test_weights_always_valid():
    _, _, env = make()
    for _, info in run(env, ACTS):
        for k in ("target", "w_next", "w_before"):
            assert (info[k] >= 0).all() and info[k].sum() == pytest.approx(1)


def test_ablation_isolation_same_actions_same_wealth():
    runs = {k: run(make(*a)[2], ACTS) for k, a in
            {"net": ("net_return", 0.0), "dsr": ("dsr", 0.0), "dsr_pen": ("dsr", 0.05)}.items()}
    v = {k: np.array([i["v"] for _, i in o]) for k, o in runs.items()}
    assert np.array_equal(v["net"], v["dsr"]) and np.array_equal(v["dsr"], v["dsr_pen"])
    for (r3, i3), (r4, i4) in zip(runs["dsr"], runs["dsr_pen"]):
        assert r4 == pytest.approx(r3 - i4["penalty"])
        dw = i4["target"][:N] - i4["w_before"][:N]
        assert i4["penalty"] == pytest.approx(0.05 * np.sum(dw ** 2))


def test_dsr_matches_formula_and_resets():
    rets = np.random.default_rng(3).normal(0.0005, 0.01, 300)
    d, A, B = DSR(0.01, 1e-8), 0.0, 0.0
    for k, r in enumerate(rets):
        A1, B1 = A + 0.01 * (r - A), B + 0.01 * (r * r - B)
        expected = (B * (A1 - A) - 0.5 * A * (B1 - B)) / ((B - A * A + 1e-8) ** 1.5)
        got = d.step(r)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)
        if k == 0:
            assert got == 0.0
        A, B = A1, B1
    d.reset()
    assert d.step(0.01) == 0.0


def test_env_dsr_reward_equals_independent_dsr_on_r_net():
    out = run(make("dsr")[2], ACTS)
    ref = DSR(0.01, 1e-8)
    for r, info in out:
        assert r == pytest.approx(ref.step(info["r_net"]))


def test_observation_layout():
    X, _, env = make()
    obs, _ = env.reset()
    assert obs.shape == (W, N * F + N + 1) and obs.dtype == np.float32
    cash = np.array([0, 0, 0, 1.0])
    assert np.allclose(obs[-1, :N * F], X[10]) and np.allclose(obs[-1, N * F:], cash)
    assert np.allclose(obs[:-1, N * F:], cash)           # pre-episode rows: all-cash weights
    obs2, *_ = env.step(np.array([0.4, 0.3, 0.2, 0.1]))
    assert np.allclose(obs2[-1, :N * F], X[11]) and np.allclose(obs2[-2, :N * F], X[10])


def test_start_is_clamped_so_window_is_full():
    _, _, env = make(start=2)
    _, info = env.reset()
    assert info["index"] == W - 1


def test_observation_does_not_see_the_future():
    X, R, env1 = make()
    X2 = X.copy()
    X2[30:] = np.random.default_rng(9).normal(size=X2[30:].shape)
    _, _, env2 = make()
    env2.X = X2
    o1, _ = env1.reset(); o2, _ = env2.reset()
    for a in ACTS:
        assert np.array_equal(o1, o2)
        o1, _, _, t1, _ = env1.step(a); o2, _, _, t2, _ = env2.step(a)
        if env1.i >= 30:
            break


def test_episode_length_and_truncation():
    _, _, env = make()
    out = run(env, ACTS)
    assert len(out) == 50 and env.done
    with pytest.raises(RuntimeError):
        env.step(ACTS[0])