"""DQN for CarRacing-v3 (discrete actions, pixel input) with checkpointing.

Every episode is a new random track, so the agent has to learn to drive, not memorise a map.

    python -m games.car_racing.train                  # resume from checkpoints/car_racing/latest.pt if it exists
    python -m games.car_racing.train --episodes 2000  # raise the target and keep going
    python -m games.car_racing.train --fresh          # ignore the checkpoint and start over

Ctrl+C saves a checkpoint before exiting.
"""
import argparse
import os
import random
from collections import deque

import numpy as np
import torch

from games.car_racing.env import GAME, make_env
from rl.buffers import FrameReplayBuffer
from rl.checkpoint import checkpoint_paths, load_checkpoint, pick_device, save_checkpoint, save_weights
from rl.dqn import DQNAgent
from rl.networks import AtariQNetwork

CHECKPOINT_DIR, LATEST_PATH, BEST_PATH = checkpoint_paths(GAME)

CONFIG = {
    "hidden": 512,
    "lr": 1e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "buffer_size": 100_000,   # frames; 96x96 uint8 each, ~0.9 GB
    "frame_skip": 4,          # repeat each chosen action for this many frames
    "stack": 4,               # frames stacked into one state so the network can see motion
    "learn_start": 5_000,     # decisions collected before the first gradient step
    "learn_every": 1,         # decisions
    "tau": 5e-3,              # soft update rate for the target network
    "double": True,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 0.99,        # per episode
    "max_neg_steps": 25,      # end the episode after this many decisions in a row without progress
    "save_every": 25,         # episodes
    "solved_score": 900.0,
}


def make_agent(env, cfg=CONFIG, device=torch.device("cpu")):
    frame_shape, n_actions = env.observation_space.shape, env.action_space.n
    return DQNAgent(
        lambda: AtariQNetwork(cfg["stack"], n_actions, cfg["hidden"], frame_shape),
        FrameReplayBuffer(cfg["buffer_size"], frame_shape, cfg["stack"]),
        n_actions, cfg, device,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="total episodes to train up to")
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring any checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    env = make_env(frame_skip=CONFIG["frame_skip"], max_neg_steps=CONFIG["max_neg_steps"])
    agent = make_agent(env, device=device)

    if os.path.exists(LATEST_PATH) and not args.fresh:
        progress = load_checkpoint(agent, LATEST_PATH)
        print(f"Resumed from {LATEST_PATH}: episode {progress['episode']}, "
              f"step {progress['step']}, epsilon {progress['eps']:.3f}")
        frame, _ = env.reset()
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        progress = {"episode": 0, "step": 0, "eps": CONFIG["eps_start"], "returns": [], "lap_times": [],
                    "best_avg": -np.inf}
        frame, _ = env.reset(seed=args.seed)
        print(f"Starting fresh on {device}")

    recent = deque(progress["returns"][-100:], maxlen=100)

    try:
        while progress["episode"] < args.episodes:
            # A new episode starts with the first frame repeated to fill the stack.
            state_ids = [agent.buffer.add_frame(frame)] * CONFIG["stack"]
            episode_return, done, info = 0.0, False, {}
            while not done:
                action = agent.act(agent.buffer.stacked(state_ids), progress["eps"])
                frame, reward, terminated, truncated, info = env.step(action)
                next_ids = state_ids[1:] + [agent.buffer.add_frame(frame)]
                # Only a real ending (lap done / left the map) cuts off the future value; a time-limit cut does not.
                agent.buffer.add(state_ids, action, reward, next_ids, float(terminated))
                state_ids = next_ids
                episode_return += reward
                progress["step"] += 1
                done = terminated or truncated

                if progress["step"] % CONFIG["learn_every"] == 0 and agent.buffer.size >= CONFIG["learn_start"]:
                    agent.learn()

            frame, _ = env.reset()
            progress["episode"] += 1
            progress["returns"].append(episode_return)
            progress["lap_times"].append(info.get("lap_time"))
            progress["eps"] = max(CONFIG["eps_end"], progress["eps"] * CONFIG["eps_decay"])
            recent.append(episode_return)
            avg = float(np.mean(recent))

            if len(recent) == 100 and avg > progress["best_avg"]:
                progress["best_avg"] = avg
                save_weights(agent.q, BEST_PATH)

            if info.get("lap_time"):
                print(f"episode {progress['episode']:5d} | lap finished in {info['lap_time']:.2f}s")

            if progress["episode"] % 10 == 0:
                laps = [t for t in progress["lap_times"][-100:] if t]
                best_lap = f"{min(laps):6.2f}s" if laps else "     -"
                print(f"episode {progress['episode']:5d} | step {progress['step']:7d} | "
                      f"return {episode_return:7.1f} | avg100 {avg:7.1f} | "
                      f"laps100 {len(laps):3d} | best lap {best_lap} | eps {progress['eps']:.3f}")

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
