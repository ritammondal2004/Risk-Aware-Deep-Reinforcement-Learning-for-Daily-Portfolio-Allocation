from dataclasses import dataclass

import numpy as np

from .config import CFG, MODEL_SPECS


@dataclass(frozen=True)
class EnvParams:
    window: int
    c_trans: float
    v0: float
    tau: float
    reward: str       # "net_return" | "dsr"
    lam: float        # turnover penalty weight; 0.0 unless the model uses it
    eta: float
    eps: float 

    @classmethod
    def from_cfg(cls, model, cfg=CFG):
        spec = MODEL_SPECS[model]
        return cls(cfg.state.window, cfg.cost.c_trans, cfg.cost.initial_value, cfg.model.action_threshold,
                   spec["reward"], cfg.reward.lambda_turnover if spec["turnover_penalty"] else 0.0,
                   cfg.reward.dsr_eta, cfg.reward.dsr_epsilon)


def threshold_and_renormalize(w, tau):
    """Action transformation: zero weights below tau, renormalise survivors to sum 1."""
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"invalid raw action {w}")
    w = w / w.sum()
    w = np.where(w >= tau, w, 0.0)
    return w / w.sum()


class DSR:
    """Differential Sharpe Ratio, Millea (2021) Eqs 6-7. A, B start at 0 each episode."""

    def __init__(self, eta, eps):
        self.eta, self.eps = eta, eps
        self.reset()

    def reset(self):
        self.A, self.B = 0.0, 0.0

    def step(self, r):
        A0, B0 = self.A, self.B
        self.A = A0 + self.eta * (r - A0)
        self.B = B0 + self.eta * (r * r - B0)
        dA, dB = self.A - A0, self.B - B0
        d = (B0 * dA - 0.5 * A0 * dB) / ((B0 - A0 * A0 + self.eps) ** 1.5)
        if not np.isfinite(d):
            raise FloatingPointError(f"non-finite DSR (A={A0}, B={B0}, r={r})")
        return d


class PortfolioEnv:
    """Daily long-only allocation over N assets + cash.

    Decision index i: observe s_i = [x_i ; w_cur], pick target weights, earn ret_next[i] (close_i -> close_{i+1}).
    Observation is always the window (W, 14N+1); the encoder decides whether to use all rows (LSTM) or the last (MLP).
    """

    def __init__(self, X, ret_next, start, end, params):
        X, ret_next = np.asarray(X, np.float64), np.asarray(ret_next, np.float64)
        assert X.shape[0] == ret_next.shape[0]
        self.N = ret_next.shape[1]
        assert X.shape[1] % self.N == 0
        self.X, self.R, self.p = X, ret_next, params
        self.first, self.end = max(int(start), params.window - 1), int(end) 
                
        assert self.end - self.first >= 2 and self.end <= X.shape[0]
        assert np.isfinite(self.R[self.first:self.end]).all()
        self.action_dim = self.N + 1
        self.obs_shape = (params.window, X.shape[1] + self.N + 1)
        self.dsr = DSR(params.eta, params.eps)
        self.done = True

    def _cash(self):
        w = np.zeros(self.N + 1)
        w[-1] = 1.0
        return w

    def _obs(self):
        W, i = self.p.window, self.i
        self._last_obs = np.concatenate([self.X[i - W + 1:i + 1], self.wbuf], axis=1).astype(np.float32)
        return self._last_obs

    def reset(self):
        self.i, self.V, self.w, self.done = self.first, self.p.v0, self._cash(), False
        self.wbuf = np.tile(self.w, (self.p.window, 1))
        self.dsr.reset()
        return self._obs(), {"index": self.i}

    def step(self, raw_action):
        if self.done:
            raise RuntimeError("step() after episode end; call reset()")
        p, N, i = self.p, self.N, self.i
        raw = np.asarray(raw_action, np.float64)
        if raw.shape != (N + 1,):
            raise ValueError(f"expected action of shape {(N + 1,)}, got {raw.shape}")
        target = threshold_and_renormalize(raw, p.tau)

        w_before = self.w.copy()
        dw = target[:N] - w_before[:N]                       # cash leg is free
        turnover = float(np.abs(dw).sum())
        cost = p.c_trans * self.V * turnover
        growth = float((target[:N] * (1.0 + self.R[i])).sum() + target[N])
        v_new = (self.V - cost) * growth
        r_net = v_new / self.V - 1.0                         # cost already inside v_new

        w_new = np.empty(N + 1)
        w_new[:N] = target[:N] * (1.0 + self.R[i]) / growth
        w_new[N] = target[N] / growth

        dsr = float("nan")
        if p.reward == "net_return":
            reward = r_net
        elif p.reward == "dsr":
            dsr = self.dsr.step(r_net)
            reward = dsr
        else:
            raise ValueError(p.reward)
        penalty = p.lam * float((dw ** 2).sum())
        reward -= penalty                                    # learning regulariser only, never touches V

        info = {"index": i, "v": v_new, "r_net": r_net, "cost": cost, "turnover": turnover,
                "penalty": penalty, "dsr": dsr, "w_before": w_before, "target": target, "w_next": w_new}

        self.V, self.w, self.i = v_new, w_new, i + 1
        self.wbuf = np.roll(self.wbuf, -1, axis=0)      
        self.wbuf[-1] = w_new
        self.done = self.i >= self.end
        info["next_obs_valid"] = self.i < self.X.shape[0]          # true next state exists?
        obs = self._obs() if info["next_obs_valid"] else self._last_obs
        return obs, float(reward), False, self.done, info          # terminated=False, truncated=done


if __name__ == "__main__":
    from .data import build_folds
    from .features import build_dataset

    ds, _ = build_dataset()
    a, b = build_folds(ds.dates)[0]["train"]
    print(f"fold-1 train steps: {b - a} | obs shape (W, 14N+1) = {(CFG.state.window, CFG.state_dim)}")
    finals = {}
    for m in MODEL_SPECS:
        env = PortfolioEnv(ds.X, ds.ret_next, a, b, EnvParams.from_cfg(m))
        env.reset()
        rng = np.random.default_rng(0)                       # same random policy for every model
        rs, tos, costs, done = [], [], 0.0, False
        while not done:
            _, r, _, done, info = env.step(rng.dirichlet(np.ones(env.action_dim)))
            rs.append(r); tos.append(info["turnover"]); costs += info["cost"]
        finals[m] = env.V
        print(f"{m}: final V {env.V:,.2f} | mean turnover {np.mean(tos):.3f} | total cost {costs:,.0f} | "
              f"mean reward {np.mean(rs):+.5f} | max reward| {np.max(np.abs(rs)):.5f}")
    v = np.array(list(finals.values()))
    assert np.allclose(v, v[0], rtol=1e-12), "wealth differs across models for identical actions"
    print("OK: identical actions give identical wealth for M1-M4; only the reward differs")