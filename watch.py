"""Watch a trained agent fly.

    python watch.py                                  # load checkpoints/best.pt
    python watch.py --checkpoint checkpoints/latest.pt
    python watch.py --episodes 5
"""
import argparse
import os
import sys

import gymnasium as gym
import torch

from train import BEST_PATH, QNetwork
from viewer import ACTIONS, GREY, WHITE, YELLOW, Viewer, state_lines, step_lines


def load_q_network(path, state_dim, n_actions):
    checkpoint = torch.load(path, weights_only=False)
    # best.pt holds bare weights; latest.pt holds a full training checkpoint.
    weights = checkpoint["q"] if "q" in checkpoint else checkpoint
    q = QNetwork(state_dim, n_actions, hidden=weights["fc1.weight"].shape[0])
    q.load_state_dict(weights)
    q.eval()
    return q


def q_value_lines(q_values, action):
    lines = [("action values", GREY)]
    for i, (name, value) in enumerate(zip(ACTIONS, q_values)):
        lines.append((f"{name:<10} {value:+7.2f}", YELLOW if i == action else WHITE))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_PATH)
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = until the window is closed)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No checkpoint at {args.checkpoint}. Train first: python train.py "
                 f"(best.pt appears once 100 episodes are done)")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
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

        lines = step_lines(episode, step, action, reward, episode_return) + [("", WHITE)]
        lines += q_value_lines(q_values, action) + [("", WHITE)] + state_lines(state)
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
