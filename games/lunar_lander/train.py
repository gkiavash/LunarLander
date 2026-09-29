"""DQN for LunarLander-v3 with checkpointing.

    python -m games.lunar_lander.train                  # resume from checkpoints/lunar_lander/latest.pt if it exists
    python -m games.lunar_lander.train --episodes 3000  # raise the target and keep going
    python -m games.lunar_lander.train --fresh          # ignore the checkpoint and start over

Ctrl+C saves a checkpoint before exiting.
"""
import argparse
import os
import random
from collections import deque

import numpy as np
import torch

from games.lunar_lander.env import GAME, make_env
from rl.buffers import ReplayBuffer
from rl.checkpoint import checkpoint_paths, load_checkpoint, save_checkpoint, save_weights
from rl.dqn import DQNAgent
from rl.networks import MLPQNetwork

CHECKPOINT_DIR, LATEST_PATH, BEST_PATH = checkpoint_paths(GAME)

CONFIG = {
    "hidden": 128,
    "lr": 5e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 100_000,
    "learn_every": 4,
    "tau": 1e-3,              # soft update rate for the target network
    "double": False,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995,       # per episode
    "save_every": 25,         # episodes
    "solved_score": 200.0,
}


def make_agent(env, cfg=CONFIG, device=torch.device("cpu")):
    state_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    return DQNAgent(
        lambda: MLPQNetwork(state_dim, n_actions, cfg["hidden"]),
        ReplayBuffer(cfg["buffer_size"], state_dim),
        n_actions, cfg, device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000, help="total episodes to train up to")
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring any checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env()
    agent = make_agent(env)

    if os.path.exists(LATEST_PATH) and not args.fresh:
        progress = load_checkpoint(agent, LATEST_PATH)
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
                save_weights(agent.q, BEST_PATH)

            if progress["episode"] % 10 == 0:
                print(f"episode {progress['episode']:5d} | step {progress['step']:7d} | "
                      f"return {episode_return:7.1f} | avg100 {avg:7.1f} | eps {progress['eps']:.3f}")

            if progress["episode"] % CONFIG["save_every"] == 0:
                save_checkpoint(agent, progress, LATEST_PATH)

            if len(recent) == 100 and avg >= CONFIG["solved_score"] and "solved_at" not in progress:
                progress["solved_at"] = progress["episode"]
                print(f"Solved at episode {progress['episode']} (avg100 {avg:.1f})")
    except KeyboardInterrupt:
        print("\nInterrupted")

    # The unfinished episode is dropped; resuming starts a new one.
    save_checkpoint(agent, progress, LATEST_PATH)
    print(f"Saved {LATEST_PATH} at episode {progress['episode']}, step {progress['step']}")
    env.close()


if __name__ == "__main__":
    main()
