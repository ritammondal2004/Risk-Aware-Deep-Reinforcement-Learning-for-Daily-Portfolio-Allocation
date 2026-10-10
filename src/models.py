"""Actor-critic with Dirichlet policy; encoder = MLP (M1) or LSTM (M2-M4)."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet


# Builds the shared 512-d state representation; only this part differs between M1 and M2-M4.
class Encoder(nn.Module):
    def __init__(self, kind, d_in, hidden):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        elif kind == "lstm":
            self.net = nn.LSTM(d_in, hidden, batch_first=True)
        else:
            raise ValueError(kind)

    # obs: (B, W, D). MLP sees only the last row; LSTM sees the whole window.
    def forward(self, obs):
        if self.kind == "mlp":
            return self.net(obs[:, -1, :])
        out, _ = self.net(obs)
        return out[:, -1]


class ActorCritic(nn.Module):
    # Heads are built BEFORE the encoder so the same seed gives identical heads in every model.
    def __init__(self, kind, d_in, n_out, hidden=512, alpha_init=1.0,
                 log_alpha_min=-3.0, log_alpha_max=6.0):
        super().__init__()
        self.actor = nn.Linear(hidden, n_out)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, float(np.log(alpha_init)))
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        self.enc = Encoder(kind, d_in, hidden)
        self.lo, self.hi = log_alpha_min, log_alpha_max

    # Returns clamped log-alpha (so alpha = exp(m)) and the state value.
    def forward(self, obs):
        h = self.enc(obs)
        return self.actor(h).clamp(self.lo, self.hi), self.critic(h).squeeze(-1)

    # Samples the RAW Dirichlet action with numpy gamma draws and returns (raw, logp, value).
    @torch.no_grad()
    def act(self, obs, rng):
        la, v = self(obs)
        alpha = la.exp().double().cpu().numpy()[0]
        g = rng.gamma(alpha)
        raw = np.clip(g / g.sum(), 1e-12, None)
        raw = raw / raw.sum()
        logp = Dirichlet(torch.as_tensor(alpha), validate_args=False).log_prob(torch.as_tensor(raw))
        return raw, float(logp), float(v)

    # Log-prob, entropy and value of stored raw actions under the CURRENT policy.
    def evaluate(self, obs, raw):
        la, v = self(obs)
        dist = Dirichlet(la.exp().double(), validate_args=False)
        raw = raw.double()
        return dist.log_prob(raw).float(), dist.entropy().float(), v

    # State value only (used for bootstrapping at segment end).
    @torch.no_grad()
    def value(self, obs):
        return float(self(obs)[1])

    # Deterministic action = Dirichlet mean, for evaluation.
    @torch.no_grad()
    def mean_action(self, obs):
        a = self(obs)[0].exp()
        return (a / a.sum(-1, keepdim=True)).cpu().numpy()[0]