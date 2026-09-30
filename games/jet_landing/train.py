"""SAC for JetLanding (F-16 on JSBSim, continuous stick/rudder/throttle) with a curriculum and checkpointing.

Training starts on short, easy approaches and moves them further out and further off the ideal path each time the
agent lands reliably. Every few episodes the deterministic policy flies a fixed set of full-difficulty approaches;
best.pt keeps the actor with the best score there.

    python -m games.jet_landing.train                    # resume from checkpoints/jet_landing/latest.pt if it exists
    python -m games.jet_landing.train --episodes 5000    # raise the target and keep going
    python -m games.jet_landing.train --fresh            # ignore the checkpoint and start over
    python -m games.jet_landing.train --fresh --demos 200 --name demos   # seed the replay buffer with autopilot
                                                         # flights; saves to checkpoints/jet_landing_demos/

Ctrl+C saves a checkpoint before exiting.
"""
import argparse
import os
import random
from collections import Counter, deque

import numpy as np
import torch

from games.jet_landing.env import GAME, make_env
from rl.buffers import ReplayBuffer
from rl.checkpoint import checkpoint_paths, load_checkpoint, save_checkpoint, save_weights
from rl.sac import SACAgent

CHECKPOINT_DIR, LATEST_PATH, BEST_PATH = checkpoint_paths(GAME)

CONFIG = {
    "hidden": 256,
    "lr": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 5e-3,               # soft update rate for the target critics
    "learn_start": 10_000,     # steps of random actions before learning starts
    "learn_every": 1,          # steps
    "max_wind_kt": 10.0,
    "difficulty_start": 0.0,   # curriculum: 0 = 400 m out on the glide path, 1 = up to 8 km out and well off it
    "difficulty_step": 0.1,
    "promote_window": 50,      # episodes at the current difficulty used to judge it
    "promote_rate": 0.7,       # landing rate over that window needed to move up
    "eval_every": 25,          # episodes
    "eval_episodes": 20,       # fixed full-difficulty approaches
    "save_every": 25,          # episodes
    "torch_threads": 1,        # small networks train fastest single-threaded on CPU
}


def make_agent(env, cfg=CONFIG, device=torch.device("cpu")):
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    return SACAgent(state_dim, action_dim, ReplayBuffer(cfg["buffer_size"], state_dim, action_dim), cfg, device)


def add_demonstrations(agent, episodes, seed=20_000):
    """Fill the replay buffer with autopilot flights spread over all difficulties, so the critics see real landings
    (and what they're worth) from the first update. The agent still learns its own policy."""
    from games.jet_landing.autopilot import Autopilot

    env = make_env(max_wind_kt=CONFIG["max_wind_kt"])
    outcomes = Counter()
    for i in range(episodes):
        env.difficulty = i / max(episodes - 1, 1)
        state, _ = env.reset(seed=seed + i)
        autopilot, done, info = Autopilot(), False, {}
        while not done:
            action = autopilot(env.state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.buffer.add(state, action, reward, next_state, float(terminated))
            state = next_state
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
    return outcomes


def evaluate(agent, env, episodes, seed=10_000):
    """Fly fixed full-difficulty approaches with the deterministic policy. Returns (mean return, landing rate, outcomes)."""
    returns, outcomes = [], Counter()
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done, total, info = False, 0.0, {}
        while not done:
            state, reward, terminated, truncated, info = env.step(agent.act(state, deterministic=True))
            total += reward
            done = terminated or truncated
        returns.append(total)
        outcomes[info["outcome"]] += 1
    return float(np.mean(returns)), outcomes["landed"] / episodes, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000, help="total episodes to train up to")
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring any checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demos", type=int, default=0, help="autopilot episodes to seed the replay buffer with (fresh runs)")
    parser.add_argument("--name", default="", help="keep this run's checkpoints separate: checkpoints/jet_landing_<name>/")
    args = parser.parse_args()

    _, latest_path, best_path = checkpoint_paths(f"{GAME}_{args.name}" if args.name else GAME)

    torch.set_num_threads(CONFIG["torch_threads"])
    env = make_env(max_wind_kt=CONFIG["max_wind_kt"])
    eval_env = make_env(max_wind_kt=CONFIG["max_wind_kt"], difficulty=1.0)
    agent = make_agent(env)

    if os.path.exists(latest_path) and not args.fresh:
        progress = load_checkpoint(agent, latest_path)
        print(f"Resumed from {latest_path}: episode {progress['episode']}, step {progress['step']}, "
              f"difficulty {progress['difficulty']:.2f}")
        env.difficulty = progress["difficulty"]
        state, _ = env.reset()
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        progress = {"episode": 0, "step": 0, "returns": [], "outcomes": [], "difficulty": CONFIG["difficulty_start"],
                    "at_difficulty": [], "evals": [], "best_eval": -np.inf}
        env.difficulty = progress["difficulty"]
        state, _ = env.reset(seed=args.seed)
        print("Starting fresh")
        if args.demos:
            outcomes = add_demonstrations(agent, args.demos)
            print(f"Replay buffer seeded with {args.demos} autopilot episodes ({agent.buffer.size} transitions): "
                  f"{dict(outcomes)}")

    recent = deque(progress["returns"][-100:], maxlen=100)
    recent_outcomes = deque(progress["outcomes"][-100:], maxlen=100)

    try:
        while progress["episode"] < args.episodes:
            episode_return, done, info = 0.0, False, {}
            while not done:
                if progress["step"] < CONFIG["learn_start"]:
                    action = env.action_space.sample()
                else:
                    action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                # Only a real ending (touchdown / crash / lost) cuts off the future value; a time-limit cut does not.
                agent.buffer.add(state, action, reward, next_state, float(terminated))
                state = next_state
                episode_return += reward
                progress["step"] += 1
                done = terminated or truncated

                if progress["step"] >= CONFIG["learn_start"] and progress["step"] % CONFIG["learn_every"] == 0:
                    agent.learn()

            state, _ = env.reset()
            progress["episode"] += 1
            outcome = info["outcome"]
            progress["returns"].append(episode_return)
            progress["outcomes"].append(outcome)
            recent.append(episode_return)
            recent_outcomes.append(outcome)

            # Curriculum: move the start positions further out once the agent lands reliably at this difficulty.
            progress["at_difficulty"].append(outcome == "landed")
            window = progress["at_difficulty"][-CONFIG["promote_window"]:]
            if (progress["difficulty"] < 1.0 and len(window) == CONFIG["promote_window"]
                    and np.mean(window) >= CONFIG["promote_rate"]):
                progress["difficulty"] = min(1.0, progress["difficulty"] + CONFIG["difficulty_step"])
                progress["at_difficulty"] = []
                env.difficulty = progress["difficulty"]
                print(f"episode {progress['episode']:5d} | curriculum -> difficulty {progress['difficulty']:.2f}")

            if progress["episode"] % 10 == 0:
                counts = Counter(recent_outcomes)
                losses = agent.last_losses
                print(f"episode {progress['episode']:5d} | step {progress['step']:8d} | return {episode_return:7.1f} | "
                      f"avg100 {np.mean(recent):7.1f} | landed100 {counts['landed']:3d} | "
                      f"difficulty {progress['difficulty']:.2f} | alpha {agent.alpha:.3f} | q {losses.get('q', 0):6.1f}")

            if progress["episode"] % CONFIG["eval_every"] == 0 and progress["step"] >= CONFIG["learn_start"]:
                score, land_rate, outcomes = evaluate(agent, eval_env, CONFIG["eval_episodes"])
                progress["evals"].append((progress["episode"], score, land_rate))
                marker = ""
                if score > progress["best_eval"]:
                    progress["best_eval"] = score
                    save_weights(agent.actor, best_path)
                    marker = "  <- new best.pt"
                print(f"episode {progress['episode']:5d} | EVAL full difficulty: return {score:7.1f} | "
                      f"landed {land_rate:4.0%} | {dict(outcomes.most_common())}{marker}")

            if progress["episode"] % CONFIG["save_every"] == 0:
                save_checkpoint(agent, progress, latest_path)
    except KeyboardInterrupt:
        print("\nInterrupted")

    # The unfinished episode is dropped; resuming starts a new one.
    save_checkpoint(agent, progress, latest_path)
    print(f"Saved {latest_path} at episode {progress['episode']}, step {progress['step']}")


if __name__ == "__main__":
    main()
