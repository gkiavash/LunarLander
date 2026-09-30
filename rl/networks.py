"""Networks: Q-networks for DQN (state in, one value per action out), actor and critics for SAC."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPQNetwork(nn.Module):
    """For small vector states (e.g. LunarLander's 8 numbers)."""

    def __init__(self, state_dim, n_actions, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


class AtariQNetwork(nn.Module):
    """The Atari DQN convnet: a stack of grayscale uint8 frames in, one value per action out."""

    def __init__(self, stack, n_actions, hidden, frame_shape=(96, 96)):
        super().__init__()
        self.conv1 = nn.Conv2d(stack, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        with torch.no_grad():
            n_flat = self._features(torch.zeros(1, stack, *frame_shape)).shape[1]
        self.fc = nn.Linear(n_flat, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def _features(self, x):
        return F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))).flatten(1)

    def forward(self, x):
        return self.out(F.relu(self.fc(self._features(x.float() / 255.0))))


def mlp(sizes):
    """Linear layers with ReLU between them (none after the last)."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """SAC policy for continuous actions in [-1, 1]: a Gaussian whose samples are squashed through tanh."""

    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.body = mlp([state_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state, deterministic=False):
        """Returns (action, log-probability of that action)."""
        h = F.relu(self.body(state))
        mu = self.mu(h)
        std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        dist = torch.distributions.Normal(mu, std)
        u = mu if deterministic else dist.rsample()        # rsample keeps the gradient path (reparameterisation)
        # Log-prob of tanh(u): Gaussian log-prob minus the log-derivative of tanh, written in a stable form.
        log_prob = dist.log_prob(u).sum(-1) - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return torch.tanh(u), log_prob


class TwinQCritic(nn.Module):
    """Two independent Q(s, a) estimates; SAC uses the smaller one to avoid overestimating."""

    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.q1 = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.q2 = mlp([state_dim + action_dim, hidden, hidden, 1])

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)
