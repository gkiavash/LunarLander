"""Watch a trained SAC pilot fly random full-difficulty approaches.

    python -m games.jet_landing.watch                       # load checkpoints/jet_landing/best.pt
    python -m games.jet_landing.watch --checkpoint checkpoints/jet_landing/latest.pt
    python -m games.jet_landing.watch --seed 42 --episodes 5
"""
import argparse
import math
import os
import sys

import torch

from games.jet_landing.env import ACTIONS, AGENT_HZ, GEAR_HEIGHT, make_env
from games.jet_landing.train import BEST_PATH
from rl.checkpoint import load_weights
from rl.networks import SquashedGaussianActor
from rl.viewer import GREY, WHITE, Viewer

def flight_lines(episode, step, episode_return, action, s):
    lines = [(f"episode    {episode}", WHITE), (f"step       {step}", WHITE), (f"return     {episode_return:+.1f}", WHITE),
             ("", WHITE), ("controls", GREY)]
    lines += [(f"{name:<10} {value:+.2f}", WHITE) for name, value in zip(ACTIONS, action)]
    lines += [("", WHITE), ("flight", GREY),
              (f"distance   {-s['x']:7.0f} m", WHITE), (f"height     {s['h'] - GEAR_HEIGHT:7.1f} m", WHITE),
              (f"glide err  {s['gs_error']:+7.1f} m", WHITE), (f"offset     {s['y']:+7.1f} m", WHITE),
              (f"airspeed   {s['airspeed_kt']:7.1f} kt", WHITE), (f"sink       {-s['vz'] / 0.3048:7.1f} ft/s", WHITE),
              (f"bank       {math.degrees(s['roll']):+7.1f} deg", WHITE)]
    return lines


def run_viewer(make_policy, title, seed=None, episodes=0):
    """Fly episodes in a window. make_policy() is called per episode and returns policy(obs, state_dict) -> action."""
    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    policy = make_policy()
    viewer = Viewer(env, title, fps=AGENT_HZ * 2)
    episode, step, episode_return, results = 1, 0, 0.0, []

    while viewer.is_open():
        action = policy(obs, env.state)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        episode_return += reward
        lines = flight_lines(episode, step, episode_return, action, env.state)
        viewer.draw(env.render(), lines)

        if terminated or truncated:
            results.append(info["outcome"])
            detail = f", sink {info['sink_fps']:.1f} ft/s, {info['touchdown_x']:.0f} m past threshold, " \
                     f"{info['touchdown_y']:+.1f} m off centre" if "sink_fps" in info else ""
            print(f"episode {episode}: {info['outcome']}{detail}, return {episode_return:.1f} "
                  f"(landed {results.count('landed')}/{len(results)})")
            for _ in range(AGENT_HZ * 2):           # hold the final frame for a moment
                if viewer.is_open():
                    viewer.draw(env.render(), lines)
            if episodes and episode >= episodes:
                break
            obs, _ = env.reset()
            policy = make_policy()
            episode, step, episode_return = episode + 1, 0, 0.0

    env.close()
    viewer.close()


def load_actor(path):
    weights = load_weights(path, key="actor")
    hidden, state_dim = weights["body.0.weight"].shape
    actor = SquashedGaussianActor(state_dim, weights["mu.weight"].shape[0], hidden)
    actor.load_state_dict(weights)
    actor.eval()
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=BEST_PATH)
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = until the window is closed)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"No checkpoint at {args.checkpoint}. Train first: python -m games.jet_landing.train "
                 f"(best.pt appears at the first evaluation)")

    actor = load_actor(args.checkpoint)

    def make_policy():
        def policy(obs, _state):
            with torch.no_grad():
                action, _ = actor(torch.as_tensor(obs).unsqueeze(0), deterministic=True)
            return action[0].numpy()
        return policy

    run_viewer(make_policy, f"JetLanding - SAC {os.path.basename(args.checkpoint)}", args.seed, args.episodes)


if __name__ == "__main__":
    main()
