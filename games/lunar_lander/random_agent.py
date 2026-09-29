"""Watch LunarLander with random actions, as a baseline.

    python -m games.lunar_lander.random_agent
"""
from games.lunar_lander.env import ACTIONS, STATE_LABELS, make_env
from rl.viewer import WHITE, Viewer, state_lines, step_lines

env = make_env(render_mode="rgb_array")
state, _ = env.reset()
viewer = Viewer(env, "LunarLander - random agent")

episode, step, episode_return = 1, 0, 0.0

for _ in range(10000):
    if not viewer.is_open():
        break

    action = env.action_space.sample()   # random action
    state, reward, terminated, truncated, _ = env.step(action)
    step += 1
    episode_return += reward

    viewer.draw(env.render(), step_lines(episode, step, action, reward, episode_return, ACTIONS) + [("", WHITE)]
                + state_lines(state, STATE_LABELS))

    if terminated or truncated:
        print(f"episode {episode}: return {episode_return:.1f} in {step} steps")
        state, _ = env.reset()
        episode, step, episode_return = episode + 1, 0, 0.0

env.close()
viewer.close()
