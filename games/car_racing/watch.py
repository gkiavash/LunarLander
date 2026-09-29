"""Watch a trained CarRacing agent drive. Every episode is a new random track.

    python -m games.car_racing.watch                            # load checkpoints/car_racing/best.pt
    python -m games.car_racing.watch --checkpoint checkpoints/car_racing/latest.pt
    python -m games.car_racing.watch --episodes 5 --seed 42     # same seed = same tracks, handy for comparing checkpoints
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import torch

from games.car_racing.env import ACTIONS, FPS, make_env
from games.car_racing.train import BEST_PATH, CONFIG
from rl.checkpoint import load_weights
from rl.networks import AtariQNetwork
from rl.viewer import WHITE, YELLOW, Viewer, q_value_lines, step_lines


def load_q_network(path, n_actions):
    weights = load_weights(path)
    q = AtariQNetwork(weights["conv1.weight"].shape[1], n_actions, hidden=weights["fc.weight"].shape[0])
    q.load_state_dict(weights)
    q.eval()
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_PATH)
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = until the window is closed)")
    parser.add_argument("--seed", type=int, default=None, help="track seed for the first episode")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No checkpoint at {args.checkpoint}. Train first: python -m games.car_racing.train "
                 f"(best.pt appears once 100 episodes are done)")

    # Watching should show whole laps, not stop when the car hesitates.
    env = make_env(render_mode="rgb_array", frame_skip=CONFIG["frame_skip"], max_neg_steps=float("inf"))
    q = load_q_network(args.checkpoint, env.action_space.n)
    frame, _ = env.reset(seed=args.seed)
    stack = deque([frame] * CONFIG["stack"], maxlen=CONFIG["stack"])
    viewer = Viewer(env, f"CarRacing - {os.path.basename(args.checkpoint)}", fps=round(FPS / CONFIG["frame_skip"]))

    episode, step, episode_return = 1, 0, 0.0
    returns, lap_times = [], []

    while viewer.is_open():
        with torch.no_grad():
            q_values = q(torch.from_numpy(np.stack(stack)).unsqueeze(0))[0].tolist()
        action = max(range(len(q_values)), key=q_values.__getitem__)
        frame, reward, terminated, truncated, info = env.step(action)
        stack.append(frame)
        step += 1
        episode_return += reward

        lines = step_lines(episode, step, action, reward, episode_return, ACTIONS) + [("", WHITE)]
        lines += q_value_lines(q_values, action, ACTIONS) + [("", WHITE)]
        lines += [(f"race time  {env.frames / FPS:.2f}s", WHITE)]
        if lap_times:
            lines += [(f"best lap   {min(lap_times):.2f}s", YELLOW)]
        viewer.draw(env.render(), lines)

        if terminated or truncated:
            returns.append(episode_return)
            if info.get("lap_time"):
                lap_times.append(info["lap_time"])
                result = f"lap {info['lap_time']:.2f}s"
            else:
                result = "did not finish"
            print(f"episode {episode}: {result}, return {episode_return:.1f} "
                  f"(average so far {sum(returns) / len(returns):.1f}, finished {len(lap_times)}/{len(returns)})")
            if args.episodes and episode >= args.episodes:
                break
            frame, _ = env.reset()
            stack = deque([frame] * CONFIG["stack"], maxlen=CONFIG["stack"])
            episode, step, episode_return = episode + 1, 0, 0.0

    env.close()
    viewer.close()


if __name__ == "__main__":
    main()
