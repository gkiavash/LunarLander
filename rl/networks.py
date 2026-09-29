"""Q-networks: a state goes in, one value per action comes out."""
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
