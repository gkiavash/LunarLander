"""LunarLander-v3: land between the flags. State is 8 numbers, 4 discrete engine actions."""
import gymnasium as gym

GAME = "lunar_lander"     # checkpoints/<GAME>/
ACTIONS = ["noop", "fire left", "fire main", "fire right"]
STATE_LABELS = ["x", "y", "vel x", "vel y", "angle", "ang vel", "left leg", "right leg"]


def make_env(render_mode=None):
    return gym.make("LunarLander-v3", render_mode=render_mode)
