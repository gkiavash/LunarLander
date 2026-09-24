"""DQN for LunarLander-v3 with checkpointing.

    python train.py                  # resume from checkpoints/latest.pt if it exists, else start fresh
    python train.py --episodes 3000  # raise the target and keep going
    python train.py --fresh          # ignore the checkpoint and start over

Ctrl+C saves a checkpoint before exiting.
"""
import argparse
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_DIR = "checkpoints"
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")

CONFIG = {
    "hidden": 128,
    "lr": 5e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 100_000,
    "learn_every": 4,
    "tau": 1e-3,              # soft update rate for the target network
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995,       # per episode
    "save_every": 25,         # episodes
    "solved_score": 200.0,
}


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.pos
        self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i] = s, a, r, s2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def state_dict(self):
        n = self.size
        return {
            "states": self.states[:n].copy(), "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(), "next_states": self.next_states[:n].copy(),
            "dones": self.dones[:n].copy(), "pos": self.pos, "size": self.size,
        }

    def load_state_dict(self, d):
        n = d["size"]
        self.states[:n], self.actions[:n], self.rewards[:n] = d["states"], d["actions"], d["rewards"]
        self.next_states[:n], self.dones[:n] = d["next_states"], d["dones"]
        self.pos, self.size = d["pos"], n


class Agent:
    def __init__(self, state_dim, n_actions, cfg):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = QNetwork(state_dim, n_actions, cfg["hidden"])
        self.target = QNetwork(state_dim, n_actions, cfg["hidden"])
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"], state_dim)

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return int(self.q(torch.as_tensor(state, dtype=torch.float32)).argmax())

    def learn(self):
        s, a, r, s2, done = self.buffer.sample(self.cfg["batch_size"])
        with torch.no_grad():
            target = r + self.cfg["gamma"] * self.target(s2).max(1).values * (1 - done)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        tau = self.cfg["tau"]
        with torch.no_grad():
            for tp, p in zip(self.target.parameters(), self.q.parameters()):
                tp.mul_(1 - tau).add_(tau * p)


def save_checkpoint(agent, progress):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        "config": agent.cfg,
        "q": agent.q.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "buffer": agent.buffer.state_dict(),
        "progress": progress,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    # Write to a temp file first so a crash mid-save can't corrupt the existing checkpoint.
    tmp = LATEST_PATH + ".tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, LATEST_PATH)


def load_checkpoint(agent):
    checkpoint = torch.load(LATEST_PATH, weights_only=False)
    agent.q.load_state_dict(checkpoint["q"])
    agent.target.load_state_dict(checkpoint["target"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.buffer.load_state_dict(checkpoint["buffer"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    return checkpoint["progress"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000, help="total episodes to train up to")
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring any checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    agent = Agent(state_dim, env.action_space.n, CONFIG)

    if os.path.exists(LATEST_PATH) and not args.fresh:
        progress = load_checkpoint(agent)
        print(f"Resumed from {LATEST_PATH}: episode {progress['episode']}, "
              f"step {progress['step']}, epsilon {progress['eps']:.3f}")
        state, _ = env.reset()
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        progress = {"episode": 0, "step": 0, "eps": CONFIG["eps_start"], "returns": [], "best_avg": -np.inf}
        state, _ = env.reset(seed=args.seed)
        print("Starting fresh")

    recent = deque(progress["returns"][-100:], maxlen=100)

    try:
        while progress["episode"] < args.episodes:
            episode_return, done = 0.0, False
            while not done:
                action = agent.act(state, progress["eps"])
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Only a real ending (crash/land) cuts off the future value; a time-limit cut does not.
                agent.buffer.add(state, action, reward, next_state, float(terminated))
                state = next_state
                episode_return += reward
                progress["step"] += 1
                done = terminated or truncated

                if progress["step"] % CONFIG["learn_every"] == 0 and agent.buffer.size >= CONFIG["batch_size"]:
                    agent.learn()

            state, _ = env.reset()
            progress["episode"] += 1
            progress["returns"].append(episode_return)
            progress["eps"] = max(CONFIG["eps_end"], progress["eps"] * CONFIG["eps_decay"])
            recent.append(episode_return)
            avg = float(np.mean(recent))

            if len(recent) == 100 and avg > progress["best_avg"]:
                progress["best_avg"] = avg
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save(agent.q.state_dict(), BEST_PATH)

            if progress["episode"] % 10 == 0:
                print(f"episode {progress['episode']:5d} | step {progress['step']:7d} | "
                      f"return {episode_return:7.1f} | avg100 {avg:7.1f} | eps {progress['eps']:.3f}")

            if progress["episode"] % CONFIG["save_every"] == 0:
                save_checkpoint(agent, progress)

            if len(recent) == 100 and avg >= CONFIG["solved_score"] and "solved_at" not in progress:
                progress["solved_at"] = progress["episode"]
                print(f"Solved at episode {progress['episode']} (avg100 {avg:.1f})")
    except KeyboardInterrupt:
        print("\nInterrupted")

    # The unfinished episode is dropped; resuming starts a new one.
    save_checkpoint(agent, progress)
    print(f"Saved {LATEST_PATH} at episode {progress['episode']}, step {progress['step']}")
    env.close()


if __name__ == "__main__":
    main()
