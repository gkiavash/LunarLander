"""F-16 approach and landing on JSBSim.

Every episode starts somewhere on a randomised final approach (distance, height, lateral offset, heading,
airspeed, wind) to a virtual runway on flat ground; the agent must follow the 3 degree glide path, flare and
touch down on the runway gently. Everything the agent sees is measured relative to the runway, so the runway's
compass heading (also random) doesn't matter.

Runway frame: x along the runway from the threshold (negative = before it), y to the right of the centreline,
h = height of the aircraft's centre of gravity above the ground. Metres and m/s unless a name says otherwise.
"""
import math

import gymnasium as gym
import jsbsim
import numpy as np

GAME = "jet_landing"      # checkpoints/<GAME>/
ACTIONS = ["pitch", "roll", "rudder", "throttle"]

FT = 0.3048
KT = 0.514444
EARTH_M_PER_DEG = 6_371_000 * math.pi / 180

RUNWAY_LENGTH = 2500.0
RUNWAY_WIDTH = 45.0
AIM_POINT = 300.0                     # touchdown target, metres past the threshold
GLIDE_SLOPE = math.radians(3.0)
GEAR_HEIGHT = 5.5 * FT                # CG height above ground with wheels on the ground
APPROACH_SPEED_KT = 155.0
RUNWAY_LAT, RUNWAY_LON = 37.0, -122.0

SIM_HZ = 120
AGENT_HZ = 10
MAX_TIME_S = 180.0

GEAR_UNITS = [f"gear/unit[{i}]/WOW" for i in range(3)]              # 0 nose, 1 left main, 2 right main
STRUCTURE_UNITS = [f"contact/unit[{i}]/WOW" for i in range(3, 10)]  # wingtips, tail, fins, intake, radome

# Fraction of full stick / rudder the agent's [-1, 1] actions map to. Full F-16 stick is a 9 g pull; an approach needs
# far less, and exploring with full authority throws the jet out of control within seconds (99 % of the autopilot's
# commands stay under 0.42 pitch and 0.17 roll).
CONTROL_LIMITS = (0.5, 0.5, 0.2)      # pitch, roll, rudder

STATE_LABELS = ["x km", "y m", "h m", "gs err m", "hdg err", "track err", "airspeed", "sink m/s",
                "roll", "pitch", "alpha", "p", "q", "r", "throttle"]


def glide_path_height(x):
    """CG height of the ideal 3 degree path at along-runway position x; flat (wheels on ground) past the aim point."""
    return max(math.tan(GLIDE_SLOPE) * (AIM_POINT - x), 0.0) + GEAR_HEIGHT


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class JetLandingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": AGENT_HZ}

    def __init__(self, render_mode=None, difficulty=1.0, max_wind_kt=10.0):
        """difficulty in (0, 1] scales how far out and how far off the ideal path episodes start (a curriculum knob)."""
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.max_wind_kt = max_wind_kt

        self.fdm = None

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(len(STATE_LABELS),), dtype=np.float32)
        # pitch stick, roll stick, rudder in [-1, 1] (scaled by CONTROL_LIMITS);
        # throttle [-1, 1] maps to idle..full military power (no afterburner)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.renderer = None

    # ---- geometry -------------------------------------------------------------------------------------------

    def _runway_frame(self):
        f = self.fdm
        d_north = (f["position/lat-geod-deg"] - RUNWAY_LAT) * EARTH_M_PER_DEG
        d_east = (f["position/long-gc-deg"] - RUNWAY_LON) * EARTH_M_PER_DEG * math.cos(math.radians(RUNWAY_LAT))
        c, s = math.cos(self.runway_heading), math.sin(self.runway_heading)
        v_north, v_east = f["velocities/v-north-fps"] * FT, f["velocities/v-east-fps"] * FT
        return {
            "x": d_north * c + d_east * s,
            "y": -d_north * s + d_east * c,
            "h": f["position/h-agl-ft"] * FT,
            "vx": v_north * c + v_east * s,
            "vy": -v_north * s + v_east * c,
            "vz": f["velocities/h-dot-fps"] * FT,
            "heading": wrap_angle(f["attitude/psi-rad"] - self.runway_heading),
            "roll": f["attitude/phi-rad"],
            "pitch": f["attitude/theta-rad"],
            "alpha": f["aero/alpha-rad"],
            "airspeed_kt": f["velocities/vc-kts"],
            "p": f["velocities/p-rad_sec"], "q": f["velocities/q-rad_sec"], "r": f["velocities/r-rad_sec"],
            "throttle": f["fcs/throttle-cmd-norm"] * 2,
        }

    def _observe(self):
        s = self.state = self._runway_frame()
        s["gs_error"] = s["h"] - glide_path_height(s["x"])
        s["track"] = math.atan2(s["vy"], max(s["vx"], 1.0))
        return np.array([
            s["x"] / 5000, s["y"] / 200, s["h"] / 300, s["gs_error"] / 30,
            s["heading"] / 0.3, s["track"] / 0.3, (s["airspeed_kt"] - APPROACH_SPEED_KT) / 30, s["vz"] / 10,
            s["roll"] / 0.5, s["pitch"] / 0.2, s["alpha"] / 0.2, s["p"], s["q"], s["r"], s["throttle"],
        ], dtype=np.float32)

    def _potential(self):
        """0 on the ideal approach, more negative the further off it. The reward includes its change each step."""
        s = self.state
        # Floating down the runway without touching down: free up to 100 m past the aim point, then it costs.
        # Without it the agent learned to skim just above the runway, where the glide-path term is ~0 and the
        # missed-approach penalty at the far end is ~300 steps away (heavily discounted).
        float_distance = max(0.0, s["x"] - AIM_POINT - 100) / 200
        terms = [abs(s["gs_error"]) / 30, abs(s["y"]) / 50, abs(s["heading"]) / 0.2,
                 abs(s["airspeed_kt"] - APPROACH_SPEED_KT) / 15, abs(s["roll"]) / 0.5, float_distance]
        weights = [1.0, 1.0, 0.5, 0.5, 0.3, 1.0]
        # The cap is high on purpose: with a low one (3), everything beyond ~150 m off course looked equally bad,
        # so there was no pull back, and the agent learned to escape sideways instead of attempting a landing.
        return -sum(w * min(t, 20.0) for w, t in zip(weights, terms))

    # ---- gym API --------------------------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # A trim can fail for an unlucky mix of start conditions; draw new ones until it succeeds.
        for _ in range(20):
            if self._start_approach(self.np_random):
                break
        else:
            raise RuntimeError("JSBSim could not trim any sampled approach")

        self.t = 0.0
        self.trail = []
        self.outcome = None
        obs = self._observe()
        self.prev_potential = self._potential()
        self.trail.append((self.state["x"], self.state["y"], self.state["h"]))
        return obs, {}

    def _start_approach(self, rng):
        """Place the aircraft on a random final approach, trimmed for a steady descent. Returns False if trim fails."""
        # A fresh aircraft every episode (2-3 ms): no leftover fuel burn, control inputs, wind or filter states.
        self.fdm = f = jsbsim.FGFDMExec(None)
        f.set_debug_level(0)
        f.load_model("f16")
        f.set_dt(1.0 / SIM_HZ)
        d = self.difficulty
        self.runway_heading = rng.uniform(0, 2 * math.pi)
        # From 400 m out (seconds before touchdown) at difficulty 0 up to 8 km at difficulty 1. Starting close matters:
        # from 2 km or more an untrained policy never survives to the runway, so it never sees a landing to learn from.
        x0 = -rng.uniform(400, 400 + 7600 * d)
        # Offsets from the ideal path grow with the distance available to fix them: none at 400 m, full from ~3.4 km.
        # (A start 400 m out and 150 m to the side can't be landed by anyone.)
        room = d * min(1.0, (-x0 - 400) / 3000)
        y0 = rng.uniform(-150, 150) * room
        h0 = glide_path_height(x0) + rng.uniform(-60, 60) * room
        heading0 = self.runway_heading + math.radians(rng.uniform(-15, 15) * room)
        c, s = math.cos(self.runway_heading), math.sin(self.runway_heading)
        north, east = x0 * c - y0 * s, x0 * s + y0 * c

        f["ic/terrain-elevation-ft"] = 0.0
        f["ic/lat-geod-deg"] = RUNWAY_LAT + north / EARTH_M_PER_DEG
        f["ic/long-gc-deg"] = RUNWAY_LON + east / (EARTH_M_PER_DEG * math.cos(math.radians(RUNWAY_LAT)))
        f["ic/h-agl-ft"] = h0 / FT
        f["ic/vc-kts"] = APPROACH_SPEED_KT + rng.uniform(-10, 10) * d
        f["ic/gamma-deg"] = -3.0
        f["ic/psi-true-deg"] = math.degrees(heading0) % 360
        f["ic/phi-deg"] = 0.0
        f.run_ic()
        f["propulsion/set-running"] = -1
        f["gear/gear-cmd-norm"] = 1.0
        # Trim for the steady descent so "stick centred" means "keep doing this"; the agent's commands add to it.
        try:
            f["simulation/do_simple_trim"] = 1
        except jsbsim.TrimFailureError:
            return False
        # Wind has to be applied after trimming (trim clears it); the aircraft feels it as a gust at the start.
        wind_speed, wind_dir = rng.uniform(0, self.max_wind_kt) * KT / FT, rng.uniform(0, 2 * math.pi)
        f["atmosphere/wind-north-fps"] = wind_speed * math.cos(wind_dir)
        f["atmosphere/wind-east-fps"] = wind_speed * math.sin(wind_dir)
        return True

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        f = self.fdm
        f["fcs/elevator-cmd-norm"] = a[0] * CONTROL_LIMITS[0]
        f["fcs/aileron-cmd-norm"] = a[1] * CONTROL_LIMITS[1]
        f["fcs/rudder-cmd-norm"] = a[2] * CONTROL_LIMITS[2]
        f["fcs/throttle-cmd-norm"] = (a[3] + 1) / 4      # 0..0.5; above 0.5 is afterburner on this model

        touched = False
        for _ in range(SIM_HZ // AGENT_HZ):
            f.run()
            if any(f[p] for p in GEAR_UNITS) or any(f[p] for p in STRUCTURE_UNITS):
                touched = True
                break
        self.t += 1.0 / AGENT_HZ

        obs = self._observe()
        s = self.state
        self.trail.append((s["x"], s["y"], s["h"]))
        potential = self._potential()
        # Plain difference, not the textbook gamma * potential - previous: since the potential is negative, the
        # gamma version pays a bonus of (1 - gamma) * |potential| every step, i.e. more for loitering further off
        # course. The plain difference sums to potential(end) - potential(start) over any episode.
        reward = potential - self.prev_potential
        self.prev_potential = potential

        terminated, truncated, info = False, False, {}
        if touched:
            terminated = True
            outcome, bonus = self._judge_touchdown(info)
            reward += bonus
        elif abs(s["roll"]) > math.radians(60) or s["alpha"] > math.radians(25):
            terminated, outcome = True, "lost control"
            reward -= 100
        elif s["x"] > RUNWAY_LENGTH or abs(s["y"]) > 1500 or s["h"] > 1000:
            # As bad as a crash: if giving up cost less, the agent would learn to leave rather than try.
            terminated, outcome = True, "missed approach"
            reward -= 100
        elif self.t >= MAX_TIME_S:
            truncated, outcome = True, "timeout"
        else:
            outcome = None

        if outcome:
            self.outcome = info["outcome"] = outcome
        return obs, float(reward), terminated, truncated, info

    def _judge_touchdown(self, info):
        """Grade the first ground contact. Returns (outcome, terminal reward)."""
        f, s = self.fdm, self.state
        sink_fps = -f["velocities/h-dot-fps"]
        info.update(sink_fps=sink_fps, touchdown_x=s["x"], touchdown_y=s["y"], roll_deg=math.degrees(s["roll"]))
        if any(f[p] for p in STRUCTURE_UNITS):
            return "crash (airframe hit ground)", -100.0
        if f[GEAR_UNITS[0]] and not (f[GEAR_UNITS[1]] or f[GEAR_UNITS[2]]):
            return "crash (nose gear first)", -100.0
        if sink_fps > 12 or abs(s["roll"]) > math.radians(12):
            return "crash (hard landing)", -100.0
        if not (0 <= s["x"] <= RUNWAY_LENGTH and abs(s["y"]) <= RUNWAY_WIDTH / 2 - 3):
            return "landed off runway", -60.0
        # A good landing: gentle, on the centreline, near the aim point.
        bonus = 100.0 - 4 * max(0.0, sink_fps - 3) - 30 * abs(s["y"]) / (RUNWAY_WIDTH / 2) \
            - 20 * min(1.0, abs(s["x"] - AIM_POINT) / 600)
        return "landed", bonus

    def render(self):
        if self.renderer is None:
            from games.jet_landing.render import Renderer
            self.renderer = Renderer()
        return self.renderer.draw(self.state, self.trail, self.outcome)

    def close(self):
        self.renderer = None


def make_env(render_mode=None, **kwargs):
    return JetLandingEnv(render_mode=render_mode, **kwargs)
