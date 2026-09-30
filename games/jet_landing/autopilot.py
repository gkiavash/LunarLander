"""Hand-written autopilot for JetLanding: a sanity check that the task is flyable, and a baseline for RL to beat.

    python -m games.jet_landing.autopilot                 # watch it fly random approaches
    python -m games.jet_landing.autopilot --runs 200      # headless: success rate and touchdown stats
"""
import argparse
import math
from collections import Counter

import numpy as np

from games.jet_landing.env import AGENT_HZ, APPROACH_SPEED_KT, CONTROL_LIMITS, GEAR_HEIGHT, GLIDE_SLOPE, make_env


class Autopilot:
    """Maps the env's runway-frame state dict to an action in [-1, 1]^4 (pitch, roll, rudder, throttle).

    Tuned on random approaches: lands at the aim point at ~4 ft/s in winds up to 20 kt.
    """

    FLARE_HEIGHT = 6.0        # metres of wheel height where the flare starts

    def __init__(self):
        self.sink_integral = 0.0

    def __call__(self, s):
        wheels = s["h"] - GEAR_HEIGHT

        # Lateral: aim the ground track back at the centreline, bank to turn onto that track, damp the roll rate.
        track_cmd = -math.atan(s["y"] / 1000)
        bank_limit = math.radians(25 if wheels > 30 else 5)
        bank_cmd = float(np.clip(2.5 * (track_cmd - s["track"]), -bank_limit, bank_limit))
        roll = 1.5 * (bank_cmd - s["roll"]) - 0.4 * s["p"]

        # Vertical: follow the glide path's sink rate plus a correction toward the path; flare near the ground.
        if wheels > self.FLARE_HEIGHT:
            vz_cmd = -math.tan(GLIDE_SLOPE) * s["vx"] - 0.2 * s["gs_error"]
        else:
            vz_cmd = -max(0.8, 0.5 * wheels)
        vz_cmd = float(np.clip(vz_cmd, -8.0, 2.0))
        # The F-16's stick commands g-load, so a proportional term alone leaves a steady sink-rate error;
        # the integral removes it. Negative stick = pull = climb.
        error = vz_cmd - s["vz"]
        self.sink_integral = float(np.clip(self.sink_integral + error / AGENT_HZ, -20, 20))
        pitch = -(0.1 * error + 0.02 * self.sink_integral)

        # Speed: throttle around the trimmed setting; idle just before touchdown.
        throttle = -0.3 + 0.08 * (APPROACH_SPEED_KT - s["airspeed_kt"]) if wheels > 5 else -1.0

        # pitch and roll above are stick deflections; the env's actions are fractions of CONTROL_LIMITS.
        return np.clip(np.array([pitch / CONTROL_LIMITS[0], roll / CONTROL_LIMITS[1], 0.0, throttle]), -1.0, 1.0)


def evaluate(runs, seed, **env_kwargs):
    env = make_env(**env_kwargs)
    outcomes, landings = Counter(), []
    for i in range(runs):
        env.reset(seed=seed + i)
        autopilot, done, info, ret = Autopilot(), False, {}, 0.0
        while not done:
            _, reward, terminated, truncated, info = env.step(autopilot(env.state))
            ret += reward
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
        if info["outcome"] == "landed":
            landings.append((info["sink_fps"], info["touchdown_x"], info["touchdown_y"], ret))
    print(f"{runs} approaches: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    if landings:
        sink, tx, ty, ret = np.array(landings).T
        print(f"landings: sink {sink.mean():.1f} fps (max {sink.max():.1f}), touchdown {tx.mean():.0f} m past threshold "
              f"(+-{tx.std():.0f}), centreline {np.abs(ty).mean():.1f} m (max {np.abs(ty).max():.1f}), return {ret.mean():.1f}")


def watch(seed):
    from games.jet_landing.watch import run_viewer

    def make_policy():
        autopilot = Autopilot()          # fresh integrator every episode
        return lambda _obs, state: autopilot(state)

    run_viewer(make_policy, "JetLanding - autopilot", seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=0, help="evaluate headless over this many approaches (0 = watch)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wind", type=float, default=10.0, help="knots")
    args = parser.parse_args()
    if args.runs:
        evaluate(args.runs, args.seed, max_wind_kt=args.max_wind)
    else:
        watch(args.seed)


if __name__ == "__main__":
    main()
