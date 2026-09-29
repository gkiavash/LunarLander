"""Checkpoint files and device selection, shared by every game.

Each game keeps its files in checkpoints/<game>/:
    latest.pt  full training state (weights, optimizer, replay buffer, RNG, progress) for resuming
    best.pt    bare Q-network weights with the best 100-episode average, for watching
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
        "q": agent.q.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
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
    agent.q.load_state_dict(checkpoint["q"])
    agent.target.load_state_dict(checkpoint["target"])
    agent.optimizer.load_state_dict(checkpoint["optimizer"])
    agent.buffer.load_state_dict(checkpoint["buffer"])
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch"])
    return checkpoint["progress"]


def save_weights(network, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(network.state_dict(), path)


def load_weights(path):
    """Q-network weights from either a best.pt (bare weights) or a latest.pt (full checkpoint)."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    return checkpoint["q"] if "q" in checkpoint else checkpoint
