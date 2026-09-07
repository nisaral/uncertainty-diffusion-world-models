# The identifiability frontier of the decision-aware distillation objective

**Date:** 2026-09-07. **Verification:** `theory/identifiability_frontier.py`
(deterministic, seeded; every check PASS). **Status:** formal statements with
proof sketches; the parts marked "measured" cite already-adjudicated docs.

This is the formal version of the claim in
`theory/distill_identifiability.py` and Section 2 of `PAPER-NARRATIVE.md`: not
only *which* functional of the teacher the single-latent objective matches,
but a precise statement of what is and is not identifiable from it, why the
M >= 2 fix identifies, why reweighting cannot rescue the M = 1 collapse, and
what this implies for gating imagined rollouts.

## Setup and notation

Fix a state-action pair and an ensemble of `N` members.  Write member next-value
means `mu*_i` (centred, `w* = (1/N) sum_i (mu*_i - mu_bar*)^2`), per-member
latent-conditional value variance `sigma_i^2`, and

    g* = (1/N) sum_i sigma_i^2,        Sigma_bar = Var( (1/N) sum_i eps_i ),

where `eps_i` is member `i`'s latent-driven value residual.  The local decision
statistic is `u* = w* - g*` (Luis local UBE).  Under a shared latent that
couples the members, the single-latent statistic the M = 1 objective matches
has population value

    S* = w* + (g* - Sigma_bar),        g* - Sigma_bar >= 0.

For the exchangeable model `Var(eps_i) = g*`, `Corr(eps_i, eps_j) = rho`
(`i != j`) used by the verification script,

    Sigma_bar = g* (rho + (1-rho)/N),
    S* = w* + g* (1-rho) (N-1)/N.

Conventions: `w`, `S` above are ddof-0 population quantities.  The code's
single-latent loss uses the ddof-1 member variance, which is a fixed factor
`N/(N-1)` larger in expectation; it cancels between teacher and student and
never changes a rank or a fibre argument.  It does inflate every reported
`w_rmse` by `N/(N-1) - 1 = 25%` at `N = 5`, which the paper must state once.

## Theorem 1 (the fibre; `u` is never sign-identified at M = 1)

**Statement.** Let the teacher be feasible with `S* > 0` and `rho < 1`.  The
zero-population-loss set of the M = 1 objective contains, for every
`alpha in (0, 1)`, the student with epistemic term `w_s = alpha S*` and
aleatoric term `g_s = (1-alpha) S* N / ((N-1)(1-rho))` (any student model that
can realise `(w_s, g_s, rho)`).  Along this family the decision statistic

    u_s = w_s - g_s = S* [ alpha - (1-alpha) N/((N-1)(1-rho)) ]

sweeps the interval `[-S* N/((N-1)(1-rho)), S*)`, which contains values of
both signs for every non-degenerate teacher.  In particular:

1. The M = 1 objective cannot identify the **magnitude** of `u*` (the fibre
   has length at least `2S*`).
2. The M = 1 objective cannot identify the **sign** of `u*`: the fibre always
   contains a student with `u_s > 0` (take `alpha -> 1`, `g_s -> 0`), and
   always contains students with `u_s` arbitrarily negative (`alpha -> 0`).

**Proof sketch.**  `S(w_s, g_s, rho) = w_s + g_s (1-rho)(N-1)/N`; substituting
the family gives `S = alpha S* + (1-alpha) S* = S*` identically, so every
member of the family matches the teacher's statistic exactly.  The endpoints
follow from the image of `alpha in (0,1)`.  The MC verification drives the
single-latent loss to `<= 1e-4` of `target^2` along the whole family for four
regime teachers (balanced, aleatoric-dominated, epistemic-dominated, and the
measured DelayedBimodal map with `w* = 1.5e-7`, `g* = 80.1`, `rho = 0.9998`),
and the swept `u_s` ranges are:

| teacher | `u*` | fibre `u_s` range | sign flip |
|---|---:|---:|---|
| balanced | -0.212 | [-0.624, +0.350] | yes |
| aleatoric-dominated | -3.999 | [-4.003, +1.601] | yes |
| epistemic-dominated | +1.322 | [-1.860, +1.339] | yes |
| measured DelayedBimodal | -80.102 | [-80.095, +0.005] | yes |

The measured-map row is the operational content: the collapsed student with
`g_s ~ 0` (the EMA arms' end state) sits on the fibre with `u_s > 0` while the
teacher's decision object is `u* ~ -80` - the objective cannot even tell that
the decision statistic changed sign.

## Theorem 2 (coupling erosion of the M = 1 statistic)

**Statement.**  The marginal sensitivity of the matched statistic to the
aleatoric channel is

    dS/dg = (1-rho)(N-1)/N  ->  0   as rho -> 1.

On an aleatoric-dominated, high-coupling value map the matched statistic is a
fixed fraction of the decision object:

    S* / |u*|  =  [ w* + g*(1-rho)(N-1)/N ] / |w* - g*|
               ~  (1-rho)(N-1)/N + w*/g*   (when g* >> w*).

**Proof sketch.**  Differentiate `S = w + g(1-rho)(N-1)/N`.  The verification
confirms the population identity `E[stat_ddof1] (N-1)/N = S*` at
`rho in {0, 0.5, 0.9, 0.9998}` to MC tolerance.

**Measured consequence.**  On the DelayedBimodal live-critic map
(`w* ~ 0.0067`, `g* ~ 80.1`, coupling `~ 0.9998`,
`research/U-COLLAPSE-MECHANISM-2026-09-05.md`) the attenuation is
`(1-rho)(N-1)/N = 1.6e-4`, so `S*/|u*| = 2.4e-4`: the M = 1 loss operates on a
quantity **~4000x smaller than the object a decision needs**.  Matching `S*`
to `1e-6` absolute error matches essentially nothing of `u*`.  This is the
formal reason the "decision-aware" term is not merely unidentified but, on
exactly the maps where `u` is aleatoric-dominated, **blind to the channel that
determines `u`**; the aleatoric share of `S*` at the measured coupling is 0.66
only because `g*/w* ~ 1.2e4` partially compensates the 1.6e-4 attenuation.

## Proposition 3 (the sticky boundary at `g = 0`)

**Statement.**  Let a student realise its aleatoric scale through a smooth map
`g_s(theta)` with `g_s = 0` at `theta = theta_0`.  For the separated mismatch
`L_g = (g_s - g_t)^2` the gradient at the collapse point vanishes:

    grad_theta L_g |_{theta_0} = 2 (g_s - g_t) grad_theta g_s = 0,

because `g_s(theta_0) = 0`, for **any** `g_t` and any smooth parameterisation
(linear amplitude, exp, softplus - all verified).  Just above the boundary the
force is linear in the scale: for `g_s = e^2`, `dL_g/de = 4 e (e^2 - g_t) ~
4 g_t e`, which vanishes as `e -> 0`.

Two consequences, both verified numerically and both consistent with the
leverage-fix study (`research/RESULTS-LEVERAGE-FIX-2026-09-05.md`):

1. **Multiplicative dials cannot create an escape force.**  Any
   `lambda(g_t) > 0` scales a gradient that is exactly zero at `g_s = 0` and
   `O(e)` just above it.  At the largest lambda actually tried in the
   leverage-fix A arms (`lambda_g = 1e4`), the escape force at `g_s ~ 0` is
   `~1e-3` of the per-state estimator noise floor (below), and under Adam a
   common loss scale is additionally inert.
2. **The estimator noise floor pins the boundary.**  Per-state
   `std(g_hat)/g*` is `0.76` at `M = 2`, `0.43` at `M = 4`, `0.28` at `M = 8`
   (verified, `g* = 80.1`, `rho = 0.3`; the independent-member analytic fit
   `~ sqrt(2/(N(M-1)))` recorded in `research/RESULTS.md` gives `0.63` at
   `M = 2`, and coupling inflates the mean over members, so the measured value
   is the right floor to use).  A gradient of order `4 g_t e` cannot compete
   with `~0.76 g*` of noise for any `e` below ~0.1.  Recovery therefore
   requires a term whose aleatoric gradient does
   **not** vanish at `g_s = 0`; in this repo that is the coupled
   member-MSE/anchor channel of the equal-weight (no-EMA) arm, which lifts
   the student's latent sensitivity through the shared-corruption next-state
   match (measured: equal-weight identified recovers `g` and `u`-rank at
   probe and policy scale, `research/RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md`).

The precise mechanism by which the EMA **epistemic up-weight** (not the
aleatoric down-weight) drives the collapse is an empirical attribution
(`research/RESULTS-CORRUPTION-2026-09-05.md`), not yet a theorem; the surface
facts above are necessary conditions for it, and Proposition 3 + Theorem 5
together explain why every purely multiplicative intervention failed.

## Theorem 4 (sufficiency of M >= 2, and the separation floor)

**Statement.**  Draw `M >= 2` shared latents per batch element and match the
debiased pair `(w_deb, g)` (the repo's `coupled_w_g`).  Both estimators are
unbiased for `(w*, g*)` (finite-M bias removed in closed form;
`theory/estimator_bias.py`), so in the population limit the M >= 2 loss

    L_2 = (w_s - w_t)^2 + (g_s - g_t)^2

has the **unique** minimiser `(w_s, g_s) = (w*, g*)`; hence `(w*, g*, u*)` are
identified.  Numerically: along the M = 1 fibre the M = 2 loss of the `w = 0`
collapse is `0.092` while the M = 1 loss is `~1e-33`; the M = 2 loss vanishes
only at the truth.

**Separation floor (design rule).**  In a single minibatch the teacher-side
estimators carry per-state variance `A_w/M`, `A_g/M` (`theory/estimator_variance.py`;
the `g` coefficient at `N = 5` is `2 g*^2/(M-1)/N` at the Gaussian law for
independent members, so `std(g_hat)/g* ~ sqrt(2/(N(M-1)))`; measured with
coupling `0.76 @ M = 2`).  Averaging over
`B` batch states reduces the floor by `sqrt(B)`.  For the loss to separate two
students whose targets differ by `(Delta_w, Delta_g)`, `M` must satisfy

    M  >~  (A_w + A_g) / (B (Delta_w^2 + Delta_g^2))

at the Gaussian law.  This is the quantitative reason the repo's `M = 2` is a
floor and not a luxury, and why claims about identified distillation must
report `M`; it also predicts that on kurtotic (rare-mode) value laws the floor
rises with the fourth moment (`kappa ~ 1/p`), so `M` must rise with the
rarity of the mode the diffusion model exists to represent.

## Theorem 5 (reweighting cannot rescue the M = 1 collapse)

**Statement (a).**  Any loss of the form

    L = (w_s - w_t)^2 * a(w_t) + (g_s - g_t)^2 * b(g_t),   a, b > 0,

has the same fibre as the unweighted M = 1 loss when `(w, g)` are matched from
one shared latent, because the fibre is a property of the pair of statistics,
not of their scale (Proposition 3 makes this local: at `g_s = 0` the aleatoric
gradient is zero under any `b`).  Reweighting changes *which half of the split
is matched well*, never *whether the fibre exists*.

**Statement (b, quantitative).**  Under the repo's inverse-variance EMA
(`TermScaleEMA`: weight `= 1/EMA(stat)^2`), the aleatoric-to-epistemic weight
ratio is

    g_scale / w_scale = (w* / g*)^2,

so on an aleatoric-dominated map (`g*/w* ~ 1.2e4` measured) the "balance"
correction down-weights the aleatoric term by **`~7e-9`** relative to the
epistemic term - the reweighted loss is, to eight orders of magnitude, the
`w`-only loss, in exactly the regime where `u ~ -g` is the decision object.
At parity (`w* = g*`) the ratio is 1 and the EMA arms still annihilate `g`
in the toy (`theory/ground_truth_w_g.py`), so (b) is a necessary aggravator,
not the whole cause; the whole cause includes Proposition 3's sticky boundary
and the measured estimator-noise floor.

**Verified numbers** (script Part 5): measured map `g_scale/w_scale = 7.0e-9`;
parity `1.0`; epistemic-dominated `1.0e4`.

## Theorem 6 (gating: percentile thresholds are scale-robust, absolute are not)

**Statement.**  Let the student's decision score be a strictly increasing
transform of the teacher's, `U_s = phi(U_t)`, with possibly collapsed scale
(e.g. `phi(x) = 1e-4 x`; this is the rank-preserving regime the combined-fix
arms occupy at `u`-rank ~ 0.95).  Percentile-threshold gating at fraction `p`
selects the set `{U > quantile(U, 1-p)}`, which is invariant to `phi`:

    {U_s > quantile(U_s, 1-p)} = {U_t > quantile(U_t, 1-p)}

identically for every strictly increasing `phi`.  Absolute-threshold gating
`{U > c}` is not invariant: for `phi(x) = 1e-4 x` and `c` at the teacher's
`p`-quantile, the student stops nothing.

**Proof.**  Quantiles are order statistics; strictly increasing maps preserve
order, hence preserve every threshold-crossing set defined by a quantile of
the same variable.  Verified numerically (20k states): under `1e-4` scale
collapse the percentile gate reproduces the teacher's refusal set with
agreement `0.9999` (ties aside) while the absolute gate stops `0.0000` of
states.

**Consequence for payoff experiments.**  A rank-preserving but scale-collapsed
student (exactly the profile the corrected arms achieve) can only gate
faithfully through **percentile** thresholds; absolute thresholds silently
disable the gate.  The repo's `u_gate` percentile default is therefore the
right protocol choice, and any payoff contrast must hold the gating rule fixed
across arms or it measures the threshold rule, not the uncertainty object.

## Theorem 7 (moving-target decomposition; the two fixes act on disjoint terms)

**Setup.**  Let the decision map's parameters evolve across a distillation
window; write the teacher's population decision statistics under the map at
reference time `t` as `z*_t = (w*_t, g*_t)` (the repo's `value_fn` is either
the live critic or the slow polyak target critic - the "lagged" arms), and the
student's statistics at the end of the window as `z_s`.

**Statement.**  For every reference time `tau` in the window the student's
residual against the eval-time map decomposes as

    ||z_s - z*_T||  <=  eps_opt(tau) + eps_drift(tau),
    eps_opt(tau)   = ||z_s - z*_tau||   (optimization gap at the map frozen at tau),
    eps_drift(tau) = ||z*_T - z*_tau||  (map drift from tau to the eval time T).

The inequality is a triangle inequality; the content is the two corollaries,
which say which fix removes which term, and the measured arm mapping below.

**C7.1 (equal-weight M >= 2 removes `eps_opt` at any fixed map; M = 1 does
not).**  Gradient flow on the equal-weight M >= 2 loss against a frozen map
converges to `(w*_tau, g*_tau)` from **any** start: the student's decision
statistic is initialization-independent (verified: 3,000-step flows from two
off-fibre starts both end at `u = -2.0000` vs teacher `u* = -2`).  Gradient
flow on the M = 1 single-latent loss against the **same** frozen map reaches
the `S`-fibre at zero loss, but *which* point on the fibre (hence `u_s`) is
initialization-dependent (verified: endpoints `u = +0.059` and `u = -5.676`
on the fibre `S* = 1.750`, `|uA - uB| = 5.7 > 0.2|u*|`).  Identifiability at a
fixed map is a property of the **loss**, not of the map: lagging alone
(`lagged_hybrid`, M = 1) reaches the measured parity (0.946) without any
loss-level guarantee - the parity is carried by the member/geometry anchors
and the initialization, and nothing in the M = 1 objective prevents a
different landing.

**C7.2 (lagging removes `eps_drift` over the window).**  With the map frozen
at `tau` and evaluated at `T = tau + drift`, the equal-weight M >= 2 student's
residual equals the map drift up to a vanishing optimization gap (verified on
200 random drift draws: median `|resid - drift| = 1.4e-15`, max optimization
gap `2.0e-15`, mean drift 0.653).  The lagged arm's residual is therefore
bounded by the map's own motion over the distillation window - the quantity
the target-critic update rate controls.

**Measured arm-to-term mapping** (N = 10, DelayedBimodal,
`research/RESULTS-COMBINED-FIX-POLICY-2026-09-07.md`; u-rank is the
scale-free adjudicated endpoint):

| arm | loss / map | term left open | u-rank (N=10) |
|---|---|---|---:|
| identified_eq | M>=2 equal weight / live | coupling + drift under the fast map | 0.877 (confirmed deficit vs ordinary, -0.057) |
| lagged_hybrid | M=1 / slow map | no loss-level guarantee (C7.1); parity by anchors + init | 0.946 |
| lagged_identified_eq | M>=2 equal weight / slow map | neither (both terms controlled) | 0.948 (top of table at parity) |

**Caveat on the w channel.**  The w-RMSE column of that table (identified_eq
0.922 vs lagged_identified_eq 0.283) is quoted in the record only as a
secondary endpoint: the lagged arms train with normalized value targets (a
registered knob, `normalize_values`), so the *absolute* w/g scale of lagged vs
live arms is confounded until a normalization-controlled check runs.  The
u-rank bars (scale-free under monotone score transforms, cf. Theorem 6) do not
carry that confound and are the adjudicated endpoints.  The fast-map coupling
mechanism itself (why the live-critic eq arm's u-rank deficit is confirmed
while the slow-map arm reaches parity) is open item 1 below.

## Proposition 8 (level-set geometry; the `epsilon`-identifiability gap)

**Setup.**  For a student pair `(w_s, g_s)` write the single-latent statistic
`S_s = w_s + c g_s` with `c = (1-rho)(N-1)/N`.  The M = 1 population risk
(the repo's `value_variance` term at one shared latent) depends on the pair
only through `S_s`; the M >= 2 equal-weight risk is
`R_2 = (w_s - w*)^2 + (g_s - g*)^2`.

**Statement.**
1. **M = 1 level sets are S-strips.**  For any risk level `eps` the M = 1
   sublevel set is `{|S_s - S*| <= sqrt(eps)}` (a strip of width
   `2 sqrt(eps)/sqrt(1 + c^2)` in the `(w, g)` plane); at `eps = 0` it is the
   fibre line `S_s = S*`.  Along that line the decision statistic
   `u = w - g` ranges over an interval of length at least `2 S*`
   (`u in [-S*/c, S*]` under `w, g >= 0`, verified: span `>= 2S*` and a sign
   flip for every regime teacher).  Worst-case decision error for a student
   that matches the teacher's statistic exactly is `>= S*` for every regime
   teacher (verified: measured DelayedBimodal `S* = 0.0128`, fibre span
   `80.1`, worst case `80.1`).
2. **M >= 2 level sets are disks.**  `{R_2 <= eps}` is a disk of radius
   `sqrt(eps)` centred on the truth, and since `u = w - g`,
   `|u_s - u*| <= sqrt(2 eps)` over the whole sublevel set, with the bound
   tight (attained at `dw = -dg`; verified: at `eps = 1e-6` the sampled max
   is `0.001414 = sqrt(2e-6)`).  Identifiability is therefore quantitative:
   the decision statistic is recovered with rate `sqrt(eps)` and coefficient
   `sqrt(2)`.
3. **The gap does not close.**  At matched population risk `eps -> 0` the
   M = 1 worst-case `u`-error stays `~` the fibre span (order `S*/c` on an
   aleatoric-dominated map) while the M >= 2 error is `<= sqrt(2 eps)`.
   Verified on the measured DelayedBimodal map: at
   `eps = 1.6e-14` the ratio is `>= 4.4e8`.  This is the formal content of
   "one number, two unknowns": no amount of optimization on the M = 1
   objective shrinks the decision-statistic uncertainty below the fibre, and
   every positive result about preserved `u` under M = 1 must therefore be
   attributed to anchors/initialization/regularization, never to the loss.

**Verification:** `theory/identifiability_frontier.py` Part 8 (deterministic,
P8a-P8c all PASS, 2026-09-07).  Recorded in
`research/RESULTS-THEORY-FRONTIER-2026-09-07.md`.

## Open items (conjectures, not theorems)

1. **Moving-map coupling.**  Why the live-critic equal-weight arm leaves a
   confirmed u-rank deficit (0.877 vs ordinary 0.934) and a w hole while the
   lagged-map version reaches parity is not closed by Theorem 7: the
   decomposition only covers the frozen-map case, and under a fast map the
   target statistics are partly a function of the student's own (still wrong)
   rollouts through the policy/buffer.  Candidate mechanisms: (i)
   train/eval distribution shift between the student's imagined states and the
   real-buffer eval states, amplified by the fast map; (ii) the registered
   `normalize_values` knob on the lagged arms.  Discriminator: a
   normalization-controlled eq-live vs eq-lagged w-RMSE contrast on
   DelayedBimodal (cheap, CPU), then the DMC `g*/w*` gate ratio.
2. **EMA-at-parity annihilation.**  Why EMA-both annihilates `g` even when
   `w* = g*` (weights equal) is only partly explained by Proposition 3; the
   measured driver is the epistemic **up**-weight and its interaction with
   Adam (empirical attribution in `RESULTS-CORRUPTION-2026-09-05.md`).
   A proof would need a model of the student's capacity competition between
   the `w` and `g` channels.
3. **Corruption-alignment.**  The claim that matching at the eval-time latent
   law is required for the aleatoric term to act on the object that decides
   the eval-time `g` is a mechanism statement; the corruption-probe verdict
   (`corruption` not binding once weights are corrected) bounds but does not
   close it.
4. **DMC regime.**  Whether the aleatoric-dominated, high-coupling regime
   (Theorem 2's aggravating condition) transfers to DMC is open; the DMC
   gate preregistration (`DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`) measures
   `g*/w*` under the DMC-trained critic before adjudicating arms.

**Closure notes (2026-09-07).**  Open item 1's discriminator ran: the G7
normalization-controlled contrast (`research/RESULTS-NORMALIZATION-CONTROL-
2026-09-07.md`, N=10, all four cells in one file) shows the w-RMSE gap and
the u-rank gap between the registered live-eq and lagged-eq configurations
are carried by candidate mechanism (ii) - the `normalize_values` knob - not
by the lag axis: within-cell lag contrasts are null (u_rank -0.0011 and
-0.0007, CIs include 0) while standardization alone moves u_rank +0.11
(10/10 in both cells). Candidate (i) (train/eval distribution shift under the
fast map) is not excluded by G7 but is not needed to explain the measured
gap at N=10 on DelayedBimodal. The measured arm-to-term mapping table in
Theorem 7 is re-read in the combined-fix record's G7 addendum: standardization
is the empirically active knob (and it acts on the live critic); Theorem 7
C7.2 remains a deterministic identity about the drift term, not an empirical
attribution. Open item 2 (EMA-at-parity) is now the only mechanism-level open
item with an unclosed empirical attribution (candidate: capacity competition
between the w and g channels under shared trunk parameters, with the epistemic
up-weight and Adam as aggravators; novel-gap G8).

## What the paper may cite from here

- Theorem 1: the M = 1 objective cannot identify the sign or magnitude of the
  local decision statistic - the formal content of the identifiability claim.
- Theorem 2: on aleatoric-dominated, high-coupling maps the matched statistic
  is `~1e-4` of the decision object; the "decision-aware" term is blind to the
  channel that determines `u` exactly where it matters.
- Proposition 3 + Theorem 5: why every multiplicative intervention failed and
  why the equal-weight fix needed a coupled, non-vanishing gradient channel.
- Theorem 4: sufficiency of M >= 2 and the `M`-vs-separation design rule.
- Theorem 6: percentile gating is the only scale-robust gating rule for
  rank-preserving students; fixes the payoff protocol.
- Theorem 7: the residual-vs-map decomposition - the equal-weight M >= 2 loss
  removes the optimization gap at any fixed map (C7.1, verified) and lagging
  removes the map-drift term (C7.2, verified) - is the formal content of
  "the two fixes compose".  C7.1's fibre-memory check is also the precise
  reason lagged-hybrid parity is not a loss-level guarantee.
- Proposition 8: the M = 1 objective's sublevel sets are S-strips, so its
  decision-statistic uncertainty does not shrink as the population risk goes
  to zero (worst-case error >= S*, sign flip), while the M >= 2 loss
  identifies with rate sqrt(eps) and tight coefficient sqrt(2) - the formal
  "can/cannot identify" statement of the paper's mechanism claim, with a
  quantitative separation that survives eps -> 0.
