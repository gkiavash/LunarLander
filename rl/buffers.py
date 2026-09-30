"""Replay buffers for off-policy algorithms such as DQN."""
import numpy as np
import torch


class ReplayBuffer:
    """Stores whole (s, a, r, s', done) transitions. Fine for small vector states.

    action_dim=None stores one discrete action index per transition (DQN); an int stores a float vector (SAC).
    """

    def __init__(self, capacity, state_dim, action_dim=None):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        if action_dim is None:
            self.actions = np.zeros(capacity, dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, done):
        i = self.pos
        self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.dones[i] = s, a, r, s2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_states[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def state_dict(self):
        n = self.size
        return {
            "states": self.states[:n].copy(), "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(), "next_states": self.next_states[:n].copy(),
            "dones": self.dones[:n].copy(), "pos": self.pos, "size": self.size,
        }

    def load_state_dict(self, d):
        n = d["size"]
        self.states[:n], self.actions[:n], self.rewards[:n] = d["states"], d["actions"], d["rewards"]
        self.next_states[:n], self.dones[:n] = d["next_states"], d["dones"]
        self.pos, self.size = d["pos"], n


class FrameReplayBuffer:
    """Stores every frame once; transitions refer to stacks of frames by id.

    Storing whole stacked states for s and s' would take 8x the memory. Frame ids only grow,
    and frame i lives in slot i % capacity, so a transition is still usable only while its
    oldest frame hasn't been overwritten.
    """

    def __init__(self, capacity, frame_shape, stack):
        self.capacity = capacity
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.n_frames = 0
        self.state_ids = np.zeros((capacity, stack), dtype=np.int64)
        self.next_ids = np.zeros((capacity, stack), dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add_frame(self, frame):
        self.frames[self.n_frames % self.capacity] = frame
        self.n_frames += 1
        return self.n_frames - 1

    def add(self, state_ids, a, r, next_ids, done):
        i = self.pos
        self.state_ids[i], self.actions[i], self.rewards[i], self.next_ids[i], self.dones[i] = state_ids, a, r, next_ids, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def stacked(self, ids):
        return self.frames[np.asarray(ids) % self.capacity]

    def sample(self, batch_size):
        oldest_alive = self.n_frames - self.capacity
        idx = np.random.randint(0, self.size, size=batch_size)
        stale = self.state_ids[idx, 0] < oldest_alive
        while stale.any():
            idx[stale] = np.random.randint(0, self.size, size=stale.sum())
            stale = self.state_ids[idx, 0] < oldest_alive
        return (
            torch.from_numpy(self.stacked(self.state_ids[idx])),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.stacked(self.next_ids[idx])),
            torch.from_numpy(self.dones[idx]),
        )

    def state_dict(self):
        n = min(self.n_frames, self.capacity)
        return {
            "frames": self.frames[:n].copy(), "n_frames": self.n_frames,
            "state_ids": self.state_ids[:self.size].copy(), "next_ids": self.next_ids[:self.size].copy(),
            "actions": self.actions[:self.size].copy(), "rewards": self.rewards[:self.size].copy(),
            "dones": self.dones[:self.size].copy(), "pos": self.pos, "size": self.size,
        }

    def load_state_dict(self, d):
        n, size = len(d["frames"]), d["size"]
        self.frames[:n], self.n_frames = d["frames"], d["n_frames"]
        self.state_ids[:size], self.next_ids[:size] = d["state_ids"], d["next_ids"]
        self.actions[:size], self.rewards[:size], self.dones[:size] = d["actions"], d["rewards"], d["dones"]
        self.pos, self.size = d["pos"], size
