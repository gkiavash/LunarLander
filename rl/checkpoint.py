"""Checkpoint files and device selection, shared by every game.

Each game keeps its files in checkpoints/<game>/:
    latest.pt  full training state (weights, optimizer, replay buffer, RNG, progress) for resuming
    best.pt    bare policy weights (DQN: Q-network, SAC: actor) from the best point in training, for watching
"""
import os
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint_paths(game):
    """(directory, latest.pt, best.pt) for a game, anchored at the project root so the working directory doesn't matter."""
    directory = os.path.join(PROJECT_ROOT, "checkpoints", game)
    return directory, os.path.join(directory, "latest.pt"), os.path.join(directory, "best.pt")


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(agent, progress, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "config": agent.cfg,
        **agent.state_dict(),          # networks and optimizers, named by the agent (e.g. q/target/optimizer for DQN)
        "buffer": agent.buffer.state_dict(),
        "progress": progress,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    # Write to a temp file first so a crash mid-save can't corrupt the existing checkpoint.
    tmp = path + ".tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def load_checkpoint(agent, path):
    # Load on CPU: the RNG state must stay a CPU tensor; load_state_dict copies weights to the agent's device.
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    agent.load_state_dict(checkpoint)
    agent.buffer.load_state_dict(checkpoint["buffer"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    return checkpoint["progress"]


def save_weights(network, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(network.state_dict(), path)


def load_weights(path, key="q"):
    """Policy weights from either a best.pt (bare weights) or a latest.pt (full checkpoint, read at `key`:
    "q" for DQN, "actor" for SAC)."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    return checkpoint[key] if key in checkpoint else checkpoint
