import numpy as np, torch
from torch.distributions import Dirichlet
from src.models import ActorCritic
from src.ppo import compute_gae


# Stored raw-action log-prob matches an independent Dirichlet computation.
def test_logp_matches_scipy():
    from scipy.stats import dirichlet
    torch.manual_seed(0)
    m = ActorCritic("mlp", 6, 4)
    obs = torch.randn(1, 3, 6)
    raw, logp, _ = m.act(obs, np.random.default_rng(0))
    alpha = m(obs)[0].exp().detach().double().numpy()[0]
    assert abs(logp - dirichlet.logpdf(raw / raw.sum(), alpha)) < 1e-6


# Before any update the ratio is exactly 1.
def test_ratio_one_before_update():
    m = ActorCritic("lstm", 6, 4)
    obs = torch.randn(1, 3, 6)
    raw, logp, _ = m.act(obs, np.random.default_rng(1))
    lp, _, _ = m.evaluate(obs, torch.as_tensor(raw).unsqueeze(0))
    assert abs(float(lp) - logp) < 1e-5


# Truncation bootstraps with V(final); termination uses 0 (hand-computed, gamma=1, lam=1).
def test_gae_truncation_vs_termination():
    r, v = [1.0, 1.0], [0.5, 0.5]  
    adv_tr, ret_tr = compute_gae(r, v, [False, True], [0, 2.0], 9.9, 1.0, 1.0)
    assert np.isclose(ret_tr[1], 3.0)                    # r + V(final)
    adv_te, ret_te = compute_gae(r, v, [False, True], [0, 0.0], 9.9, 1.0, 1.0)
    assert np.isclose(ret_te[1], 1.0)                    # r only
    assert np.isclose(adv_tr[1], 2.5) and np.isclose(adv_tr[0], 3.5)


# M1 ignores earlier window rows; the LSTM does not.  
def test_mlp_ignores_history_lstm_uses_it():
    x = torch.randn(1, 5, 6); y = x.clone(); y[:, 0] += 5
    mlp, lstm = ActorCritic("mlp", 6, 4), ActorCritic("lstm", 6, 4)
    assert torch.allclose(mlp(x)[0], mlp(y)[0])
    assert not torch.allclose(lstm(x)[0], lstm(y)[0])


# Same seed gives identical actor/critic heads in M1 and M2.
def test_head_init_parity():
    torch.manual_seed(7); a = ActorCritic("mlp", 6, 4)
    torch.manual_seed(7); b = ActorCritic("lstm", 6, 4)
    assert torch.equal(a.actor.weight, b.actor.weight) and torch.equal(a.critic.weight, b.critic.weight)