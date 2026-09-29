"""DQN agent, independent of the game: the caller supplies the network and the replay buffer.

Config keys used: lr, gamma, batch_size, tau, and optionally double (Double DQN, default off).
"""
import random

import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, make_network, buffer, n_actions, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.n_actions = n_actions
        self.device = device
        self.q = make_network().to(device)
        self.target = make_network().to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = buffer

    def act(self, state, eps):
        """Epsilon-greedy action for one state (an unbatched numpy array)."""
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.q(torch.as_tensor(state).unsqueeze(0).to(self.device)).argmax())

    def learn(self):
        s, a, r, s2, done = (t.to(self.device) for t in self.buffer.sample(self.cfg["batch_size"]))
        with torch.no_grad():
            if self.cfg.get("double", False):
                # Double DQN: the online net picks the next action, the target net scores it.
                next_q = self.target(s2).gather(1, self.q(s2).argmax(1, keepdim=True)).squeeze(1)
            else:
                next_q = self.target(s2).max(1).values
            target = r + self.cfg["gamma"] * next_q * (1 - done)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        tau = self.cfg["tau"]
        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.q.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
