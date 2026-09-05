# U-collapse mechanism: why the identified loss zeroes the decision object (2026-09-05)

**Question (thread #1).** In the policy 2x2 (N=30), the reweighted identified
arms (`identified_hybrid`, `lagged_identified`) destroy the teacher-student
uncertainty object: `student_u_mean ~ -0.004` vs teacher `-50..-105`, u-rank
~noise. The mechanism note in
`research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md` described this as "local
u = w - g lands on the `u_min = 0` floor". That description is **wrong** for
the eval object (eval uses `u_min = -1e9`, so u is unclamped). This document
replaces it with the measured mechanism.

**Probe.** `udwm/scripts/probe_u_collapse.py` re-runs the exact 1800-step
policy protocol arm for one seed, adds per-training-call logging of the
identified loss components (teacher/student w, g, local u, EMA scales) and a
final eval-time decomposition of the local UBE object per state
(w_raw, w_deb, g, sigma_bar, coupling, u) plus the within-member spread of
next states across 8 paired pure-noise latents. Data:
`runs/probe_u_{variant}_s{seed}.json` for seeds 0-1 (CPU, default threads;
same protocol and teacher pairing as the N=30 study).

## Measured facts

### 1. The eval object on this benchmark is aleatoric-dominated: u ~= -g, g >> w

Under the post-training SAC critic (any arm), the teacher's local statistic
u = w - g is dominated by the within-member aleatoric term:

| arm (seed 0) | teacher w | teacher g | teacher u | g/w |
|---|---:|---:|---:|---:|
| identified_hybrid | 0.0067 | 80.1 | -80.1 | 1.2e4 |
| ordinary | 0.0079 | 76.3 | -76.3 | 9.7e3 |
| lagged_hybrid | 0.0075 | 79.9 | -79.9 | 1.1e4 |
| lagged_identified | 0.0038 | 84.0 | -84.0 | 2.2e4 |

Q(s', pi(s')) is extremely sensitive to the latent draw within a member
(teacher coupling ~ 0.9998: the latent-induced Q variation is shared across
members), while the members' means agree closely. So the "epistemic" term w
is ~1e-4 of the statistic that actually determines u. Any student that
preserves w but not g has u ~ w ~ 0.

### 2. The collapse is a collapse of g (aleatoric), not w (epistemic)

Identified arms (seed 0 and 1, both identical patterns):

| arm | teacher_u | student_u | teacher_w | student_w | teacher_g | student_g | teacher delta-spread | student delta-spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| identified_hybrid s0 | -78.8 | -0.0024 | 0.0067 | 0.0103 | 78.8 | 0.0127 | 0.37 | 0.0059 |
| identified_hybrid s1 | -72.5 | -0.0071 | 0.0038 | 0.0088 | 72.5 | 0.0160 | 0.36 | 0.0061 |
| lagged_identified s0 | -84.0 | -0.0107 | 0.0038 | 0.0050 | 84.0 | 0.0157 | 0.37 | 0.0078 |
| identified equal-weight s0 | -86.1 | -0.0049 | 0.0062 | 0.0084 | 86.1 | 0.0133 | 0.38 | 0.0060 |
| identified w-only s0 | -71.8 | -0.0030 | 0.0033 | 0.0069 | 71.8 | 0.0098 | 0.36 | 0.0059 |
| identified g-only s0 | -72.8 | -0.0020 | 0.0038 | 0.0074 | 72.8 | 0.0094 | 0.37 | 0.0056 |

delta-spread = mean over members of the std across 8 paired pure-noise latents
of the sampled next-state delta. The teacher's members sweep a broad state
region (spread ~0.36-0.38); under every identified configuration the 1-NFE
student's output is nearly latent-independent (spread ~0.006, ~60x smaller).
Consequently student g ~ 0.01 (vs teacher ~70-90) and student u = w - g ~ w ~ 0.
Student w still matches teacher w (w_rmse is the best in the table): the
epistemic half of the identified split *works*.

Contrast arms retain the spread and the object:

| arm | student_u | student g | student delta-spread |
|---|---:|---:|---:|
| ordinary s0 | -67.6 | 58.5 | 0.33 |
| hybrid s0 | -12.1 | 12.7 | 0.20 |
| lagged_hybrid s0 | -75.5 | 62.1 | 0.33 |

### 3. Not EMA lag, not the reweighting, not the u-floor

- **Not EMA lag:** `lagged_identified` (stationary target critic, value
  normalization, guard, 50-update warmup) collapses identically (u ~ -0.01,
  delta-spread 0.008). The pathology is present at the first eval (step 450)
  and persists; it does not require a moving value map.
- **Not the reweighting:** `identified` with equal weights (no EMA) collapses
  the same way (student_u -0.0049). Reweighting changes *which* half of the
  split is matched well, not *whether* g collapses.
- **Not the `u_min = 0` floor:** eval is unclamped; u_student ~ w - g with
  both terms ~1e-2 is genuinely tiny. (The training-side UBE clamp is a
  different object and plays no role in the eval endpoint.)
- **Not a w-target failure:** the epistemic term is matched to ~1e-5 loss
  (w_scale x MSE ~ 3.4 at the end); w_rmse is the best of all arms.

### 4. The mechanism: the aleatoric term cannot lift g; its per-update leverage is ~1e-6

Trajectories (identified_hybrid, both seeds): from the first model-training
calls to the last, student g / teacher g stays pinned at ~1e-3-1e-4. The
student's within-member value spread never develops, in every identified
configuration -- including the g-only arm where the aleatoric term is the ONLY
decision term at weight 1 with a loss of order 40-200.

Gradient post-mortem on the trained g-only end state (equal weights):
- aleatoric loss = 40 (huge mismatch), grad norm = 0.21 -- the corner is NOT
  a zero-gradient fixed point, and it is not cancelled by the member term
  (cos(member, aleatoric) ~ +0.04..+0.11).
- Matched-draw step response: one lr=1e-3 raw-SGD step along the aleatoric
  gradient changes the student's 2-latent g by **+2.3e-6**. The 2-latent
  estimator noise floor (same model, different draws) is ~2e-4 -- ~100x the
  per-step signal. A random control step of equal norm moves g by ~1e-8.
- To close the measured gap (student g ~0.002 to teacher's ~1.8 at
  train-time scale; ~80 at eval scale) would need ~1e6-1e7 such steps; the
  whole run performs 180 student updates, and individual update directions are
  dominated by estimation noise 100x above the signal.

The first-order aleatoric signal at the collapsed corner has essentially no
leverage on the object it is supposed to correct (d(g)/d(theta) is of order
~1e-2 per unit gradient, vs the ~1e4 ratio between g and w on this map). On
top of that, the configured EMA reweighting multiplies the aleatoric gradient
by g_scale ~ 0.36 while multiplying the epistemic gradient by
w_scale ~ 2.4e5 (lagged_identified: ~1e10 vs ~1.3e4) -- a ~1e5-1e6 relative
down-weight of the only term that could in principle lift g. But the g-only
cell shows the reweighting is not the primary cause: even at weight 1 the
aleatoric term cannot raise g within budget.

**One-line mechanism.** The identified loss matches the epistemic half of the
split to machine precision and leaves the aleatoric half at its
initialization level, because the aleatoric term's gradient has no effective
leverage on the student's latent-to-state sensitivity (measured ~1e-6 per
update, ~1e6-1e7 updates short of closing a g-gap of order 1-100); since the
decision object u = w - g is aleatoric-dominated on this benchmark
(teacher g/w ~ 1e4 under the SAC critic), u collapses to ~w ~ 0 and the rank
becomes noise. This is a property of the identified objective interacting with
an aleatoric-dominated value map -- not of critic nonstationarity, EMA lag, or
reweighting, and not of the identifiability argument itself (which is about
the loss target and stands as a theorem/toy result).

### 5. Open sub-thread (honest)

Why pure member-MSE (ordinary) and lagged_hybrid leave the student's t=1
(pure-noise) mapping latent-sensitive while every identified configuration
removes that sensitivity is not yet explained from first principles. The
identified loss acts on value statistics at training-time corruptions whose
noise magnitude never exceeds ~28% of the data scale
(`sqrt(1-alpha_bar)` max = 0.279 at t=7 of the 8-step schedule), so its
gradients shape the student's extrapolation at eval-time pure noise only
indirectly; member-MSE alone happens to leave a latent-sensitive
extrapolation, and the decision terms push the fitted map to the
latent-insensitive one. Suggested next probe (cheap): log per-update
||d(aleatoric)/d(theta)|| and s_g with a training variant that samples the
value map only at t_scaled=1 pure-noise corruptions, and check whether g then
rises.

## What this changes in the write-up

- Correct the "u_min=0 floor" sentence in `RESULTS-POLICY-2X2-30SEED-2026-09-03.md`.
- The identified arms fail in policy on this benchmark because the aleatoric
  half of the split cannot be lifted by the student in this value regime, not
  because of the weighting hole (which the reweighting fixed for w) and not
  because of nonstationarity (lagging does not help). Any fix must act on the
  aleatoric channel: sample the value map at eval-equivalent pure-noise
  latents in the loss, increase M further, or anchor the student's per-latent
  spread in state space.
- The fixed-map stress (value map ~O(1), w and g comparable) does not exhibit
  this failure mode; that is consistent: the pathology needs g >> w, which
  only the live-SAC value map produces at this scale.

## Addendum (2026-09-05): naive leverage fixes do not help

Registered experiment (`LEVERAGE-FIX-PREREGISTRATION-2026-09-05.md`, results
`RESULTS-LEVERAGE-FIX-2026-09-05.md`, data `runs/probe_u_lrscale_*.json`):
scaling the aleatoric loss by lambda_g in {1e2, 1e4} at clip 10/1e3 is inert
under Adam (A arms, g unchanged at ~0.013); giving the aleatoric-dominated
gradient a 10x-100x larger student learning rate drives g **down** to 0
(B arms: g 0.003 -> exactly 0.0 at lr=0.1, delta-spread 0.0000), replicated
on seeds 0-1. M=8 latents does not help. Leverage starvation is therefore not
solely a step-size artifact: the student parks at the g=0 non-negativity
boundary, where the within-member variance is exactly flat, because the
2-latent training-time estimator noise (~100x the coherent signal) drives the
walk into the boundary. The remaining lever is the objective's corruption
distribution (evaluate the w/g terms at pure-noise t_scaled=1 corruptions),
not its weights or step size. Thread closed as diagnosis + confirmed dead end
for the naive fix.
