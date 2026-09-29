"""Watch a trained LunarLander agent fly.

    python -m games.lunar_lander.watch                  # load checkpoints/lunar_lander/best.pt
    python -m games.lunar_lander.watch --checkpoint checkpoints/lunar_lander/latest.pt
    python -m games.lunar_lander.watch --episodes 5
"""
import argparse
import os
import sys

import torch

from games.lunar_lander.env import ACTIONS, STATE_LABELS, make_env
from games.lunar_lander.train import BEST_PATH
from rl.checkpoint import load_weights
from rl.networks import MLPQNetwork
from rl.viewer import WHITE, Viewer, q_value_lines, state_lines, step_lines


def load_q_network(path, state_dim, n_actions):
    weights = load_weights(path)
    q = MLPQNetwork(state_dim, n_actions, hidden=weights["fc1.weight"].shape[0])
    q.load_state_dict(weights)
    q.eval()
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_PATH)
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = until the window is closed)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No checkpoint at {args.checkpoint}. Train first: python -m games.lunar_lander.train "
                 f"(best.pt appears once 100 episodes are done)")

    env = make_env(render_mode="rgb_array")
    q = load_q_network(args.checkpoint, env.observation_space.shape[0], env.action_space.n)
    state, _ = env.reset()
    viewer = Viewer(env, f"LunarLander - {os.path.basename(args.checkpoint)}")

    episode, step, episode_return = 1, 0, 0.0
    returns = []

    while viewer.is_open():
        with torch.no_grad():
            q_values = q(torch.as_tensor(state, dtype=torch.float32)).tolist()
        action = max(range(len(q_values)), key=q_values.__getitem__)
        state, reward, terminated, truncated, _ = env.step(action)
        step += 1
        episode_return += reward

        lines = step_lines(episode, step, action, reward, episode_return, ACTIONS) + [("", WHITE)]
        lines += q_value_lines(q_values, action, ACTIONS) + [("", WHITE)] + state_lines(state, STATE_LABELS)
        viewer.draw(env.render(), lines)

        if terminated or truncated:
            returns.append(episode_return)
            print(f"episode {episode}: return {episode_return:.1f} in {step} steps "
                  f"(average so far {sum(returns) / len(returns):.1f})")
            if args.episodes and episode >= args.episodes:
                break
            state, _ = env.reset()
            episode, step, episode_return = episode + 1, 0, 0.0

    env.close()
    viewer.close()


if __name__ == "__main__":
    main()
