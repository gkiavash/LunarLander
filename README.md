# RL playground

Reinforcement-learning agents for Gymnasium games. Shared algorithm code lives in `rl/`; each game gets its own
package under `games/`.

```
rl/                      game-independent building blocks
  dqn.py                 DQNAgent (epsilon-greedy act, learn with optional Double DQN, soft target update)
  networks.py            MLPQNetwork (vector states), AtariQNetwork (stacked frames)
  buffers.py             ReplayBuffer (vector states), FrameReplayBuffer (frames stored once)
  checkpoint.py          checkpoints/<game>/ paths, save/load full state or weights, device pick
  viewer.py              pygame window: game frame + info panel
games/
  lunar_lander/          env.py · train.py · watch.py · random_agent.py
  car_racing/            env.py · train.py · watch.py
checkpoints/<game>/      latest.pt (full training state, resumable) · best.pt (best weights)
.claude/docs/            explanations: RL vs. DL concepts, CarRacing design
```

## Run

From the project root, with the virtualenv active (`source .venv/bin/activate`):

```bash
python -m games.lunar_lander.train     # train (resumes from checkpoints/lunar_lander/latest.pt)
python -m games.lunar_lander.watch     # watch the best model
python -m games.lunar_lander.random_agent

python -m games.car_racing.train       # train on random tracks (resumes automatically)
python -m games.car_racing.watch --seed 42
```

Training scripts take `--episodes N` (target total) and `--fresh` (ignore the checkpoint). Ctrl+C saves before exiting.
In PyCharm, right-click a file and Run: the project root is on the path, so the imports resolve.

## Adding a game

1. `games/<name>/env.py`: `GAME = "<name>"`, action labels, and `make_env(render_mode=None)` that returns a
   Gymnasium env with any wrappers the game needs.
2. `games/<name>/train.py`: pick a network and buffer from `rl/`, build a `DQNAgent`, and write the episode loop
   (copy the closest existing game).
3. `games/<name>/watch.py`: load `best.pt` with `rl.checkpoint.load_weights` and draw with `rl.viewer.Viewer`.

## Adding an algorithm

Put it in `rl/<algorithm>.py` next to `dqn.py`, reusing `networks.py` / `checkpoint.py` where they fit.
On-policy methods such as PPO need a rollout buffer instead of a replay buffer. Then add a
`train_<algorithm>.py` in the game packages that should use it.
