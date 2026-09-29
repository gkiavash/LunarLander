"""CarRacing-v3 with discrete actions: a new random track every episode, 96x96 pixels in.

Reward is +1000/N per track tile reached and -0.1 per frame, so finishing faster scores higher.
"""
import gymnasium as gym
import numpy as np

GAME = "car_racing"       # checkpoints/<GAME>/
ACTIONS = ["noop", "right", "left", "gas", "brake"]
FPS = 50                  # CarRacing simulation rate, used to turn frames into lap seconds
ZOOM_FRAMES = 50          # the first second of every episode is a camera zoom-in animation


class CarRacingPreprocess(gym.Wrapper):
    """Skip the zoom-in, repeat actions, convert to grayscale, and cut hopeless episodes short."""

    def __init__(self, env, frame_skip, max_neg_steps):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.max_neg_steps = max_neg_steps
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (h, w), dtype=np.uint8)

    @staticmethod
    def gray(obs):
        return (obs @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(ZOOM_FRAMES):
            obs, *_ = self.env.step(0)
        self.frames = 0
        self.neg_steps = 0
        return self.gray(obs), info

    def step(self, action):
        total, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            self.frames += 1
            if terminated or truncated:
                break
        self.neg_steps = self.neg_steps + 1 if total < 0 else 0
        # Driving around on the grass teaches little; treat it as a time-limit cut, not a real ending.
        if self.neg_steps >= self.max_neg_steps and not terminated:
            truncated = True
        if info.get("lap_finished"):
            info["lap_time"] = self.frames / FPS
        return self.gray(obs), total, terminated, truncated, info


def make_env(render_mode=None, frame_skip=4, max_neg_steps=25):
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    return CarRacingPreprocess(env, frame_skip, max_neg_steps)
