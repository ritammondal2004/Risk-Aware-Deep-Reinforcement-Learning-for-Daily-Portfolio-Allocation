"""MLP/LSTM encoders and Dirichlet policy components.

This file provides the core network/distribution building blocks. The full
PPO rollout/update loop should be kept in one implementation so M1-M4 share
exactly the same optimizer and training protocol.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Dirichlet


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return h[-1]


class DirichletActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, n_actions: int):
        super().__init__()
        self.encoder = encoder
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        h = self.encoder(obs)
        concentration = torch.exp(self.actor(h)).clamp(min=1e-4, max=1e4)
        dist = Dirichlet(concentration)
        value = self.critic(h).squeeze(-1)
        return dist, value
