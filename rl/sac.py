"""Soft Actor-Critic for continuous actions in [-1, 1], independent of the game.

Off-policy like DQN (learns from a replay buffer), but instead of picking the best of a few discrete actions it
trains a stochastic policy (the actor) against two Q-estimates (the critics), and is rewarded for keeping the policy
random where it can afford to (entropy), with that trade-off tuned automatically.

Config keys used: hidden, lr, gamma, batch_size, tau.
"""
import torch
import torch.nn.functional as F

from rl.networks import SquashedGaussianActor, TwinQCritic


class SACAgent:
    def __init__(self, state_dim, action_dim, buffer, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.device = device
        self.actor = SquashedGaussianActor(state_dim, action_dim, cfg["hidden"]).to(device)
        self.critic = TwinQCritic(state_dim, action_dim, cfg["hidden"]).to(device)
        self.critic_target = TwinQCritic(state_dim, action_dim, cfg["hidden"]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        # Entropy temperature alpha, learned so the policy's entropy stays near -action_dim.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])
        self.target_entropy = -float(action_dim)
        self.buffer = buffer
        self.last_losses = {}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.critic_target.load_state_dict(d["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"])
        self.actor_optimizer.load_state_dict(d["actor_optimizer"])
        self.critic_optimizer.load_state_dict(d["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(d["alpha_optimizer"])

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def act(self, state, deterministic=False):
        """Action for one state (an unbatched numpy array): sampled while training, the mean when evaluating."""
        with torch.no_grad():
            action, _ = self.actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device), deterministic)
        return action[0].cpu().numpy()

    def learn(self):
        s, a, r, s2, done = (t.to(self.device) for t in self.buffer.sample(self.cfg["batch_size"]))
        alpha = self.log_alpha.exp().detach()

        # Critics: regress both Q(s, a) onto r + gamma * (min target Q(s', a') - alpha * log pi(a'|s')), a' ~ pi.
        with torch.no_grad():
            a2, log_prob2 = self.actor(s2)
            q1_t, q2_t = self.critic_target(s2, a2)
            target = r + self.cfg["gamma"] * (1 - done) * (torch.min(q1_t, q2_t) - alpha * log_prob2)
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor: pick actions the critics like, while staying random (entropy bonus weighted by alpha).
        a_new, log_prob = self.actor(s)
        q_new = torch.min(*self.critic(s, a_new))
        actor_loss = (alpha * log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Temperature: raise alpha when the policy is less random than the target entropy, lower it when more.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        tau = self.cfg["tau"]
        with torch.no_grad():
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
        self.last_losses = {"critic": critic_loss.item(), "actor": actor_loss.item(), "q": q_new.mean().item()}
