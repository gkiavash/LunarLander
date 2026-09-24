import gymnasium as gym

from viewer import WHITE, Viewer, state_lines, step_lines

env = gym.make("LunarLander-v3", render_mode="rgb_array")
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

    viewer.draw(env.render(), step_lines(episode, step, action, reward, episode_return) + [("", WHITE)] + state_lines(state))

    if terminated or truncated:
        print(f"episode {episode}: return {episode_return:.1f} in {step} steps")
        state, _ = env.reset()
        episode, step, episode_return = episode + 1, 0, 0.0

env.close()
viewer.close()
