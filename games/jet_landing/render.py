"""Lightweight 3D view of the approach, drawn with pygame into an RGB array (no display needed).

Top: chase camera behind the aircraft with ground grid, runway, glide-path gates and an F-16 wireframe.
Bottom: side profile (height vs. distance) and top-down view (centreline offset vs. distance), with the flown trail.
"""
import math

import numpy as np
import pygame

from games.jet_landing.env import AIM_POINT, RUNWAY_LENGTH, RUNWAY_WIDTH, glide_path_height

W, H_3D, H_2D = 720, 420, 180
FOCAL = 520
NEAR = 1.0
FAR = 12_000
CAM_BACK, CAM_UP, CAM_PITCH = 55.0, 12.0, math.radians(6)

SKY, GROUND, GRID = (118, 160, 205), (78, 110, 68), (92, 128, 80)
RUNWAY, MARK, GATE = (70, 70, 74), (235, 235, 235), (120, 230, 240)
JET, JET_EDGE, SHADOW = (170, 175, 185), (40, 40, 45), (55, 80, 50)
PANEL_BG, AXIS, PATH, TRAIL = (24, 26, 34), (90, 95, 110), (120, 230, 240), (250, 200, 90)
GOOD, BAD = (120, 220, 120), (235, 100, 100)

# F-16-ish outline in the body frame (x forward, y right, z down), metres.
_HALF_WING = [(1.5, 0, 0), (-2.5, 4.7, 0), (-4.0, 4.7, 0), (-4.0, 0, 0)]
_HALF_TAIL = [(-5.5, 0, 0), (-7.5, 2.8, 0), (-8.0, 2.8, 0), (-8.0, 0, 0)]
_FIN = [(-4.5, 0, 0), (-7.8, 0, -3.2), (-8.3, 0, -3.2), (-8.0, 0, 0)]
_FUSELAGE = [(7.5, 0, 0), (4.0, -0.6, 0), (-8.0, -0.6, 0), (-8.0, 0.6, 0), (4.0, 0.6, 0)]


def _mirror(poly):
    return [(x, -y, z) for x, y, z in poly]


JET_POLYS = [_FUSELAGE, _HALF_WING, _mirror(_HALF_WING), _HALF_TAIL, _mirror(_HALF_TAIL), _FIN]


def body_to_world(roll, pitch, heading):
    """Rotation from body axes to the runway frame with z up (x along runway, y right)."""
    cr, sr, cp, sp, ch, sh = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(heading), math.sin(heading)
    # Standard aerospace body -> NED (x fwd, y right, z down), then flip z to point up.
    r = np.array([
        [cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh],
        [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch],
        [-sp, sr * cp, cr * cp],
    ])
    return np.diag([1, 1, -1]) @ r


class Camera:
    def __init__(self, position, heading):
        self.pos = np.asarray(position, dtype=float)
        self.ch, self.sh = math.cos(heading), math.sin(heading)
        self.cp, self.sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
        self.cx, self.cy = W / 2, H_3D / 2

    def to_camera(self, p):
        """World point -> (forward, right, up) in camera axes."""
        d = np.asarray(p, dtype=float) - self.pos
        fwd = d[0] * self.ch + d[1] * self.sh
        right = -d[0] * self.sh + d[1] * self.ch
        up = d[2]
        return fwd * self.cp - up * self.sp, right, fwd * self.sp + up * self.cp

    def project(self, c):
        f, r, u = c
        # Clamp: points just past the near plane project far off-screen, which pygame can't take as ints.
        return (min(max(self.cx + FOCAL * r / f, -1e5), 1e5), min(max(self.cy - FOCAL * u / f, -1e5), 1e5))

    def horizon_y(self):
        return self.cy - FOCAL * math.tan(CAM_PITCH)

    def clip_polygon(self, cam_points):
        """Keep the part of a polygon in front of the near plane (one-plane Sutherland-Hodgman)."""
        out = []
        n = len(cam_points)
        for i in range(n):
            a, b = cam_points[i], cam_points[(i + 1) % n]
            a_in, b_in = a[0] >= NEAR, b[0] >= NEAR
            if a_in:
                out.append(a)
            if a_in != b_in:
                t = (NEAR - a[0]) / (b[0] - a[0])
                out.append(tuple(a[k] + t * (b[k] - a[k]) for k in range(3)))
        return out

    def polygon(self, surface, world_points, color, width=0):
        cam = self.clip_polygon([self.to_camera(p) for p in world_points])
        if len(cam) < 3 or min(c[0] for c in cam) > FAR:
            return
        pts = [self.project(c) for c in cam]
        if width == 0:
            pygame.draw.polygon(surface, color, pts)
        else:
            pygame.draw.lines(surface, color, True, pts, width)

    def line(self, surface, a, b, color, width=1):
        ca, cb = self.to_camera(a), self.to_camera(b)
        if ca[0] < NEAR and cb[0] < NEAR:
            return
        if ca[0] < NEAR or cb[0] < NEAR:
            t = (NEAR - ca[0]) / (cb[0] - ca[0])
            clipped = tuple(ca[k] + t * (cb[k] - ca[k]) for k in range(3))
            ca, cb = (clipped, cb) if ca[0] < NEAR else (ca, clipped)
        if min(ca[0], cb[0]) > FAR:
            return
        pygame.draw.line(surface, color, self.project(ca), self.project(cb), width)


class Renderer:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("menlo", 13)
        self.big_font = pygame.font.SysFont("menlo", 26, bold=True)
        self.surface = pygame.Surface((W, H_3D + H_2D))

    def draw(self, state, trail, outcome=None):
        self._draw_3d(state)
        self._draw_profiles(state, trail)
        if outcome:
            color = GOOD if outcome == "landed" else BAD
            text = self.big_font.render(outcome.upper(), True, color)
            self.surface.blit(text, text.get_rect(center=(W / 2, 40)))
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    # ---- 3D chase view --------------------------------------------------------------------------------------

    def _draw_3d(self, s):
        surf = self.surface.subsurface((0, 0, W, H_3D))
        jet = np.array([s["x"], s["y"], s["h"]])
        heading = s["heading"]
        cam = Camera(jet - CAM_BACK * np.array([math.cos(heading), math.sin(heading), 0]) + [0, 0, CAM_UP], heading)

        horizon = cam.horizon_y()
        surf.fill(SKY)
        pygame.draw.rect(surf, GROUND, (0, horizon, W, H_3D - horizon))

        # Ground grid every 500 m.
        for gx in range(-10_000, 4_001, 500):
            cam.line(surf, (gx, -4000, 0), (gx, 4000, 0), GRID)
        for gy in range(-4000, 4001, 500):
            cam.line(surf, (-10_000, gy, 0), (4000, gy, 0), GRID)

        # Runway surface and markings.
        hw = RUNWAY_WIDTH / 2
        cam.polygon(surf, [(0, -hw, 0), (RUNWAY_LENGTH, -hw, 0), (RUNWAY_LENGTH, hw, 0), (0, hw, 0)], RUNWAY)
        for yb in np.linspace(-hw + 3, hw - 3, 8):          # threshold bars
            cam.polygon(surf, [(6, yb - 1, 0), (36, yb - 1, 0), (36, yb + 1, 0), (6, yb + 1, 0)], MARK)
        for xd in range(60, int(RUNWAY_LENGTH) - 60, 60):    # centreline dashes
            cam.line(surf, (xd, 0, 0), (xd + 30, 0, 0), MARK, 2)
        for side in (-1, 1):                                 # aim-point blocks
            y0 = side * 7
            cam.polygon(surf, [(AIM_POINT - 25, y0 - 2.5, 0), (AIM_POINT + 25, y0 - 2.5, 0),
                               (AIM_POINT + 25, y0 + 2.5, 0), (AIM_POINT - 25, y0 + 2.5, 0)], MARK)
            cam.line(surf, (0, side * hw, 0), (RUNWAY_LENGTH, side * hw, 0), MARK, 1)

        # Glide-path gates every 500 m: fly through the middle of the boxes.
        for gx in range(-9000, int(AIM_POINT) - 400, 500):
            gh = glide_path_height(gx)
            cam.polygon(surf, [(gx, -40, gh - 20), (gx, 40, gh - 20), (gx, 40, gh + 20), (gx, -40, gh + 20)], GATE, 1)

        # Shadow on the ground, then the aircraft itself.
        rot = body_to_world(s["roll"], s["pitch"], heading)
        for poly in JET_POLYS:
            pts = [jet + rot @ np.array(p) for p in poly]
            cam.polygon(surf, [(p[0], p[1], 0.05) for p in pts], SHADOW)
        for poly in JET_POLYS:
            pts = [jet + rot @ np.array(p) for p in poly]
            cam.polygon(surf, pts, JET)
            cam.polygon(surf, pts, JET_EDGE, 1)

    # ---- 2D profiles ----------------------------------------------------------------------------------------

    def _draw_profiles(self, s, trail):
        half = W // 2
        side = self.surface.subsurface((0, H_3D, half, H_2D))
        top = self.surface.subsurface((half, H_3D, W - half, H_2D))
        x_min, x_max = -9000.0, RUNWAY_LENGTH + 200
        pad = 10

        def sx(x, width):
            return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

        # Side profile: height (0..550 m) against distance.
        side.fill(PANEL_BG)
        h_max = 550.0

        def side_pt(x, h):
            return sx(x, half), H_2D - pad - h / h_max * (H_2D - 2 * pad - 14)

        pygame.draw.line(side, AXIS, side_pt(x_min, 0), side_pt(x_max, 0))
        pygame.draw.line(side, MARK, side_pt(0, 0), side_pt(RUNWAY_LENGTH, 0), 4)
        pygame.draw.line(side, PATH, side_pt(x_min, glide_path_height(x_min)), side_pt(AIM_POINT, glide_path_height(AIM_POINT)))
        if len(trail) > 1:
            pygame.draw.lines(side, TRAIL, False, [side_pt(x, h) for x, _, h in trail], 2)
        pygame.draw.circle(side, TRAIL, side_pt(s["x"], s["h"]), 4)
        side.blit(self.font.render("side: height vs distance", True, AXIS), (pad, 4))

        # Top view: centreline offset (+-400 m, exaggerated) against distance.
        top.fill(PANEL_BG)
        y_max = 400.0
        tw = W - half

        def top_pt(x, y):
            return sx(x, tw), H_2D / 2 + 7 + y / y_max * (H_2D / 2 - pad - 7)

        for xd in range(int(x_min), 0, 200):
            pygame.draw.line(top, AXIS, top_pt(xd, 0), top_pt(xd + 100, 0))
        hw = RUNWAY_WIDTH / 2
        rw = [top_pt(0, -hw), top_pt(RUNWAY_LENGTH, -hw), top_pt(RUNWAY_LENGTH, hw), top_pt(0, hw)]
        pygame.draw.polygon(top, MARK, rw)
        if len(trail) > 1:
            pygame.draw.lines(top, TRAIL, False, [top_pt(x, y) for x, y, _ in trail], 2)
        pygame.draw.circle(top, TRAIL, top_pt(s["x"], s["y"]), 4)
        top.blit(self.font.render("top: offset vs distance", True, AXIS), (pad, 4))
