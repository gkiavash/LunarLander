"""Pygame window showing a game frame (any env with render_mode="rgb_array") with a live info panel on the right."""
import pygame

WHITE = (230, 230, 230)
GREY = (140, 140, 150)
YELLOW = (240, 200, 90)
GREEN = (120, 220, 120)
RED = (230, 110, 110)

PANEL_W = 260
PANEL_BG = (20, 20, 28)


class Viewer:
    def __init__(self, env, title, fps=50):
        frame = env.render()
        self.game_h, self.game_w = frame.shape[:2]
        self.fps = fps
        pygame.init()
        self.screen = pygame.display.set_mode((self.game_w + PANEL_W, self.game_h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("menlo", 14)
        self.clock = pygame.time.Clock()

    def is_open(self):
        return not any(event.type == pygame.QUIT for event in pygame.event.get())

    def draw(self, frame, lines):
        self.screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        self.screen.fill(PANEL_BG, (self.game_w, 0, PANEL_W, self.game_h))
        for i, (text, color) in enumerate(lines):
            self.screen.blit(self.font.render(text, True, color), (self.game_w + 14, 12 + i * 17))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()


def step_lines(episode, step, action, reward, episode_return, actions):
    return [
        (f"episode    {episode}", WHITE),
        (f"step       {step}", WHITE),
        ("", WHITE),
        (f"action     {actions[action]}", YELLOW),
        (f"reward     {reward:+.2f}", GREEN if reward >= 0 else RED),
        (f"return     {episode_return:+.1f}", WHITE),
    ]


def state_lines(state, labels):
    return [("state", GREY)] + [(f"{label:<10} {value:+.3f}", WHITE) for label, value in zip(labels, state)]


def q_value_lines(q_values, action, actions):
    lines = [("action values", GREY)]
    for i, (name, value) in enumerate(zip(actions, q_values)):
        lines.append((f"{name:<10} {value:+7.2f}", YELLOW if i == action else WHITE))
    return lines
