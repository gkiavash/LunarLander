# Jet landing: the math

An F-16 (JSBSim flight dynamics) on a random final approach learns to land with Soft Actor-Critic (SAC).
This page covers the formulas: task, inputs, outputs, reward, networks and losses. Design history, pitfalls and
how to run it are in [`.claude/docs/jet-landing.md`](../../.claude/docs/jet-landing.md).

```
python -m games.jet_landing.autopilot     # hand-written baseline
python -m games.jet_landing.train         # SAC
python -m games.jet_landing.watch
```

## 1. The task as an MDP

At every step t (0.1 s, 10 Hz):

```
state s_t → actor picks a_t → JSBSim simulates 0.1 s (12 physics steps at 120 Hz) → reward r_t, next state s_{t+1}
```

Goal: a policy π(a|s) that maximizes the **discounted return**

$$G_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k}, \qquad \gamma = 0.99$$

With γ = 0.99 the effective horizon is 1/(1−γ) = 100 steps ≈ **10 seconds**.

### Input: observation s ∈ ℝ¹⁵ (`env.py: _observe`)

Everything is measured in the **runway frame**:
- **x:** metres along the runway from the threshold (negative = before it).
- **y:** metres right of the centreline.
- **h:** centre-of-gravity height above ground.

The ideal glide path:

$$h_{gs}(x) = \max\big(\tan 3° \cdot (300 - x),\ 0\big) + 1.68\text{ m}$$

(300 m = aim point; 1.68 m = CG height with wheels on the ground.)

| # | Feature | Formula | Scale |
|---|---|---|---|
| 1 | along-track position | x | ÷ 5000 |
| 2 | centreline offset | y | ÷ 200 |
| 3 | height | h | ÷ 300 |
| 4 | glide-path error | h − h_gs(x) | ÷ 30 |
| 5 | heading error | ψ − ψ_runway | ÷ 0.3 rad |
| 6 | track error (ground path direction) | atan2(v_y, max(v_x, 1)) | ÷ 0.3 rad |
| 7 | airspeed error | kt − 155 | ÷ 30 |
| 8 | vertical speed | v_z | ÷ 10 m/s |
| 9–11 | roll φ, pitch θ, angle of attack α | | ÷ 0.5, 0.2, 0.2 rad |
| 12–14 | body rates p, q, r | | rad/s |
| 15 | current throttle | | 0..1 |

The divisions put every feature roughly in [−1, 1] (hand-made input normalization).

### Output: action a ∈ [−1, 1]⁴ (`env.py: step`)

| Action | Maps to | Why scaled |
|---|---|---|
| a₀ pitch | elevator = 0.5·a₀ | full stick = 9 g pull; an approach needs less than half |
| a₁ roll | aileron = 0.5·a₁ | |
| a₂ rudder | rudder = 0.2·a₂ | the autopilot never needs it |
| a₃ throttle | cmd = (a₃+1)/4 ∈ [0, 0.5] | 0.5 = full military power; above is afterburner |

The pitch stick doesn't move the elevator directly. The F-16's fly-by-wire system reads it as a **g-load
command** added to the trimmed setting, so a₀ = 0 means "keep descending as trimmed".

## 2. Reward

Dense **shaping** every step, plus a **terminal** grade on the last step:

$$r_t = \underbrace{\Phi(s_{t+1}) - \Phi(s_t)}_{\text{shaping}} \;+\; \underbrace{R_{\text{end}}}_{\text{last step only}}$$

### Shaping via a potential Φ (`env.py: _potential`)

Φ(s) is 0 on the ideal approach and more negative the further off it:

$$\Phi(s) = -\sum_i w_i \cdot \min(e_i,\ 20)$$

| Error term eᵢ | Formula | Weight wᵢ |
|---|---|---|
| glide path | \|h − h_gs(x)\| / 30 m | 1.0 |
| centreline | \|y\| / 50 m | 1.0 |
| heading | \|ψ_err\| / 0.2 rad | 0.5 |
| airspeed | \|kt − 155\| / 15 | 0.5 |
| bank | \|φ\| / 0.5 rad | 0.3 |
| floating | max(0, x − 400) / 200 m | 1.0 |

Example: 30 m below the glide path and 50 m left of the centreline gives Φ = −(1 + 1) = −2.

Over any episode the shaping **telescopes**:

$$\sum_{t=0}^{T-1} \big[\Phi(s_{t+1}) - \Phi(s_t)\big] = \Phi(s_T) - \Phi(s_0)$$

So its total depends only on where the episode ends compared with where it started, and zig-zagging can't farm it.
Each step still gets immediate feedback ("that move brought you closer to the path").

It is deliberately *not* the textbook `γ·Φ(s') − Φ(s)`. With Φ ≤ 0 that form pays an extra
`(1−γ)·|Φ(s')|` every step, a bonus that grows the further off course the aircraft is.

### Terminal reward R_end (`env.py: _judge_touchdown`)

| Outcome | Condition | R_end |
|---|---|---|
| **landed** | wheels first, on the runway, sink ≤ 12 ft/s, bank ≤ 12° | 100 − 4·max(0, sink − 3) − 30·\|y\|/22.5 − 20·min(1, \|x − 300\|/600) |
| landed off runway | wheels first, not on the runway | −60 |
| crash | airframe contact, nose gear first, sink > 12 ft/s, or bank > 12° | −100 |
| lost control | bank > 60° or angle of attack > 25° | −100 |
| missed approach | past the runway end, > 1.5 km aside, or > 1 km high | −100 |
| timeout | 180 s | 0, *truncated* (see §4) |

A perfect landing (≤ 3 ft/s, on the centreline, at the aim point) scores +100. Each penalty in the landing formula is
a partial deduction for sink rate, centreline offset and touchdown position.

## 3. Networks (`rl/networks.py`)

### Actor π_θ(a|s): `SquashedGaussianActor`

```
s (15) → Linear 256 → ReLU → Linear 256 → ReLU ─┬→ Linear → μ(s)        (4)
                                                └→ Linear → log σ(s)    (4), clamped to [−20, 2]
```

The actor outputs a **distribution**, not a single action:

$$u = \mu(s) + \sigma(s)\odot\varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I), \qquad a = \tanh(u) \in (-1, 1)^4$$

Log-probability of the squashed action (change of variables through tanh):

$$\log \pi(a|s) = \sum_{i=1}^{4}\Big[\log \mathcal{N}(u_i;\ \mu_i, \sigma_i) \;-\; \log\big(1 - \tanh^2 u_i\big)\Big]$$

The code uses the numerically stable form log(1 − tanh² u) = 2(log 2 − u − softplus(−2u)).
Training samples actions (exploration). Evaluation and `watch.py` use the deterministic a = tanh(μ(s)).

### Critics Q_φ₁, Q_φ₂: `TwinQCritic`

```
[s, a] (19) → Linear 256 → ReLU → Linear 256 → ReLU → Linear → Q (1)     ×2, independent
```

Q(s, a) is the expected future return after taking a in s and then following π. SAC uses the **smaller** of the two
estimates to counter overestimation. Each critic has a slowly trailing **target copy**:

$$\bar\phi \leftarrow (1-\tau)\,\bar\phi + \tau\,\phi, \qquad \tau = 0.005$$

## 4. SAC objective and losses (`rl/sac.py: learn`)

SAC maximizes return **plus an entropy bonus** (it's rewarded for staying random where it can afford to):

$$J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t \big(r_t + \alpha \, \mathcal{H}(\pi(\cdot|s_t))\big)\Big], \qquad \mathcal{H} = -\mathbb{E}[\log \pi]$$

Every gradient step samples 256 transitions (s, a, r, s′, d) from the replay buffer (capacity 1M) and makes three updates.

**① Critic loss:** regression onto a bootstrapped ("soft Bellman") target:

$$y = r + \gamma\,(1 - d)\Big(\min_{j=1,2} Q_{\bar\phi_j}(s', a') - \alpha \log \pi_\theta(a'|s')\Big), \quad a' \sim \pi_\theta(\cdot|s')$$

$$L_Q(\phi) = \mathbb{E}\big[(Q_{\phi_1}(s,a) - y)^2\big] + \mathbb{E}\big[(Q_{\phi_2}(s,a) - y)^2\big]$$

- **d = 1** only for real endings (landing, crash, lost control, missed approach): nothing comes after, so y = r.
- **A timeout has d = 0.** The episode was cut off but the situation still had a future
  (*terminated* vs. *truncated*).

**② Actor loss:** choose actions the critics rate highly, while staying random:

$$L_\pi(\theta) = \mathbb{E}_{s,\ \varepsilon}\Big[\alpha \log \pi_\theta(\tilde a|s) - \min_{j} Q_{\phi_j}(s, \tilde a)\Big], \qquad \tilde a = \tanh(\mu_\theta(s) + \sigma_\theta(s)\odot\varepsilon)$$

ã is a differentiable function of θ (the *reparameterization trick*, `rsample()`), so gradients flow from Q back
through the action into the actor.

**③ Temperature loss:** α is tuned so the policy's entropy stays near a target H̄:

$$L(\alpha) = -\,\mathbb{E}\big[\log\alpha \cdot (\log \pi_\theta(\tilde a|s) + \bar{\mathcal{H}})\big], \qquad \bar{\mathcal{H}} = -\dim(\mathcal{A}) = -4$$

More random than the target makes α shrink; less random makes it grow. The code optimizes log α so α stays positive.
This is the `alpha` column in the training log.

**Optimizer:** Adam, lr 3e-4, for all three. One gradient step per environment step, after 10,000 steps of purely
random actions.

## 5. Training loop (`train.py`)

**Episode start.** Difficulty d ∈ [0, 1] is set by the curriculum:

$$x_0 \sim -\mathcal{U}(400,\ 400 + 7600d), \qquad \text{room} = d\cdot\min\Big(1, \tfrac{-x_0-400}{3000}\Big)$$

$$y_0 \sim \mathcal{U}(\pm150)\cdot\text{room},\quad h_0 = h_{gs}(x_0) + \mathcal{U}(\pm60)\cdot\text{room},\quad \psi_0 = \psi_{rwy} + \mathcal{U}(\pm15°)\cdot\text{room}$$

Plus airspeed 155 ± 10d kt, wind 0–10 kt from a random direction, and a random runway heading (invisible to the agent,
since the observation is runway-relative). The aircraft starts trimmed for a steady 3° descent.

**Curriculum:** d += 0.1 once the last 50 episodes at the current level have a landing rate ≥ 70 %.

**Evaluation:** every 25 episodes the deterministic policy flies 20 fixed full-difficulty approaches.
`best.pt` = the actor with the highest mean return there.

## 6. Mapping to supervised learning

| Supervised learning | Here |
|---|---|
| input x | observation s (15 numbers) |
| model output | actor: a distribution over 4 controls; critics: one value each |
| labels y | *made by the model itself*: r + γ·min Q̄(s′, a′) − α log π (bootstrapping) |
| loss | critics: MSE to y; actor: α log π − Q (no labels at all) |
| dataset | replay buffer, filled by the agent's own flying and always changing |
| mini-batch | 256 random past transitions |
| validation metric | eval return and landing rate on 20 fixed approaches |

The key difference: **the actor has no labels**. Its loss is the critics' opinion, and the critics' labels come
from their own slow-moving copies. That circularity is why RL is so sensitive to reward design. An agent will
happily find ways to make these formulas look good without actually landing.

## 7. Constants at a glance

| Symbol | Value | Where |
|---|---|---|
| γ (discount) | 0.99 | `train.py: CONFIG` |
| τ (target update) | 0.005 | `train.py: CONFIG` |
| lr (all optimizers) | 3e-4 | `train.py: CONFIG` |
| batch / buffer | 256 / 1,000,000 | `train.py: CONFIG` |
| hidden units | 256 × 2 layers | `train.py: CONFIG` |
| H̄ (target entropy) | −4 | `rl/sac.py` |
| agent / physics rate | 10 Hz / 120 Hz | `env.py` |
| control limits (pitch, roll, rudder) | 0.5, 0.5, 0.2 | `env.py: CONTROL_LIMITS` |
| approach speed / glide slope / aim point | 155 kt / 3° / 300 m | `env.py` |
| runway | 2500 × 45 m | `env.py` |
