# CRN-bias probe registration - Hypothesis A, value-reference consistency under paired M>=2 sampling (2026-09-09)

**Status: registered 2026-09-09, before any probe row runs and before any probe
harness code is written.** This probe is a mechanism-attribution study. It
NEVER re-adjudicates the DMC payoff bars or the DelayedBimodal transfer bars;
it tests the explanation proposed for the lag-axis asymmetry that the DMC
30-seed verdict left standing. Per project discipline: this document fixes the
endpoint set, direction, and adjudication rule first; the harness
(`udwm/scripts/probe_crn_bias.py`; implemented 2026-09-09) is implemented after registration and
before any row is read.

## 1. The anomaly this probe targets

The 30-seed DMC verdict (`research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md`)
confirmed, at N=30, the result that no prior explanation in the project
predicted: **lagging the critic helps the M=1 hybrid family and hurts the
M>=2 equal-weight identified family on the same drifting value map.**

- `lagged_hybrid - hybrid` on DMC/15k: +0.106 u_rank, CI [+0.059, +0.153], 22/30.
- `lagged_identified_eq - identified_eq` on DMC/15k: -0.073 u_rank,
  CI [-0.111, -0.036], 7/30 (reversal; the same contrast was +0.071 on
  DelayedBimodal N=10, where `lagged_identified_eq` tops the table at 0.948).
- Identifiability, EMA-starvation, and nonstationarity-alone each predict the
  two families should respond the SAME way to lagging (lagging is a property
  of the critic, not of the loss family). They do not. That asymmetry is the
  anomaly Hypothesis A is built to explain.

## 2. Hypothesis A (formal statement)

The M>=2 identified loss draws paired latent samples and computes a debiased
statistic from them - structurally a **coupled / common-random-number (CRN)
comparison** (Glasserman & Yao 1992, Management Science 38(6):884-908). CRN's
variance-reduction guarantee holds only when the paired evaluations occur
under similar/identical conditions. In this stack the "conditions" are the
value map used to score the paired draws:

- live-critic arms (`identified_eq`, `hybrid`) score both draws against the
  current critic Q_t;
- lagged-critic arms (`lagged_identified_eq`, `lagged_hybrid`) score them
  against the polyak target Q_{t-k} (k_eff ~ 1/(1-tau) = 100 critic updates
  at tau=0.01), which lags the live critic used at eval/gating time.

**Hypothesis A:** when the critic drifts, the lagged arms' paired evaluations
happen against a stale reference, so the CRN-style coupling guarantee is void
and a characterizable bias enters the student's (w, g) decomposition - a
mechanistically-detailed instance of ensemble head-collapse under an
over-shared/drifting target (Osband et al. 2016, arXiv:1602.04621). The M=1
`hybrid` family evaluates its single draw once and never needs a *paired*
comparison, so it structurally cannot suffer this failure mode - which is why
lagging helps it (fresh targets for a non-paired objective) while hurting the
paired family. Wu, Zheng, Zhang, Zhang & Wang (Management Science, DOI
10.1287/mnsc.2022.01205, online 2024-09-18) formalize exactly this case -
paired estimators biased by nonstationarity between paired observations - and
supply the bias-correction template the fix would adapt. EEDQN
(arXiv:2506.05716, 2025) documents that how ensemble uncertainty interacts
with recursive value learning "remains less well understood" - the live,
citable statement of the gap this probe addresses.

## 3. Confound control: the {live, lagged} x {norm, no-norm} 2x2

The registered DMC arms differ in TWO knobs at once:
`identified_eq` (live critic, unnormalized) vs `lagged_identified_eq`
(lagged target critic, value normalization + guard + warmup). Any
eq-vs-lagged_eq delta therefore conflates target staleness with value
normalization. The G7/G9 cells showed normalization alone moves
DelayedBimodal N=10 to the measured ceiling with the lag axis adding ~0; the
DMC normalization cells have never been run. **The probe therefore runs the
full identified-family 2x2** (cells already defined in
`udwm/scripts/run_delayed_bimodal_policy_ablation.py`):

| cell | critic for distill value ref | value normalization |
|---|---|---|
| A = `identified_eq` | live Q_t | off |
| B = `identified_eq_norm` | live Q_t | on |
| C = `lagged_identified_eq` | target Q_{t-k} | on |
| D = `lagged_identified_eq_nonorm` | target Q_{t-k} | off |

plus the hybrid-family lag pair (`hybrid`, `lagged_hybrid`) as the
single-evaluation control family and `identified_hybrid` (EMA) as the
collapse sanity arm. Lag is read **within a fixed normalization cell**
(C-B in the norm cell, D-A in the no-norm cell), never as a marginal contrast.

## 4. Probe measurements (instrumented, pre-committed definitions)

Logged per training step (info dict, every 100 steps) and at every full eval
checkpoint (3k/6k/9k/12k/15k at the DMC 15k budget; eval-side fields follow
the registered forward-fill convention - the last record at each milestone is
the fresh eval):

- **M1 critic drift rate** `D(t, k_eff)`: mean |Q_live(s,a) - Q_target(s,a)|
  over a fixed 1,024-state probe set drawn once from the real replay buffer
  (states only; actions from the frozen deterministic actor). k_eff = 100
  critic updates is the polyak lag; a secondary fast-drift proxy is the
  rolling mean per-update |Delta Q| (TD-scale), logged at the same frequency.
- **M2 pairing gap** `G_pair(t)`: the number of critic parameter updates
  between the teacher-side and student-side value evaluations of a paired
  draw inside the distill loss forward (a counter inside
  `udwm/models/consistency.py`'s paired path; expectation ~0 for all arms
  within a step), plus the arm's reference staleness k_eff. This is the
  "wall-clock/step gap between the two paired evaluations" measured directly.
- **M3 induced-bias proxy** `B_hat(t)`: re-score the student's imagined
  next-states from the last distill step under BOTH the reference critic
  (live Q_t for A/B, target Q_{t-k} for C/D) and the current live critic.
  Define per-arm `B_hat(t) = mean over probe states |mean_ij s_ij(Q_live) -
  mean_ij s_ij(Q_ref)|`, and the decomposed terms `B_w(t) = |w_hat(Q_live) -
  w_hat(Q_ref)|`, `B_g(t) = |g_hat(Q_live) - g_hat(Q_ref)|` from the same
  re-evaluation pass (the terms a Wu-style correction would subtract).
- **M4 u_rank / w_rmse / g ratio** at the same checkpoints (existing
  registered endpoints, unchanged definitions).

Default windows (pre-committed; changing them requires a new registration):
probe set = 1,024 states; paired latents M=2; 5 ensemble members; the
re-evaluation pass runs with the critic frozen and mirrors the distill-eval
batch structure exactly.

## 5. Endpoints and pre-registered predictions

Registered endpoints (direction fixed before any row exists):

- **E1 (primary, mechanism):** within each lagged identified cell on DMC/15k
  (C = lagged + norm, D = lagged + no-norm), u_rank fidelity at the final
  checkpoint is negatively associated with the measured staleness-induced
  term: Spearman rho( u_rank_corr(seed), B_hat(final, seed) ) across seeds is
  negative in both cells. In the live-critic cells (A, B) B_hat ~ 0 by
  construction, so no association is predicted there (registered as a
  within-cell null that the same statistic should return ~0). No pooling
  across cells is used - pooling could manufacture correlation from cell
  means.
- **E2 (decomposition):** within the no-norm cell, lagged < live for the
  identified family on DMC (D-A negative); within the norm cell, the same
  ordering (C-B negative) if staleness is the axis. The magnitude comparison
  |C-B| vs |D-A| is registered as a two-sided descriptive read with
  pre-committed interpretation: |C-B| >= |D-A| says normalization does not
  shelter the paired statistic from staleness; |C-B| < |D-A| says
  standardization absorbs part of the drift-induced distortion. E2 is a
  directional secondary read, not a bar.
- **E3 (control family):** the hybrid-family lag contrast is NOT predicted by
  B_hat: rho(hybrid-family lag benefit, B_hat) is registered as ~0 (no
  prediction of sign; the contrast itself stays whatever the data say).
- **E4 (placebo, analysis-side):** applying the Wu-style correction
  (subtract the M3 drift terms from the student u used to compute u_rank) to
  the hybrid-family logs changes their u_rank by ~0 (|delta| <= 0.03, effect
  size d_z < 0.2), while the same correction on the identified-family logs
  changes u_rank in the predicted direction (toward the live-critic ranking).
- **E5 (collapse sanity, not a bar):** `identified_hybrid` reproduces the EMA
  collapse (u_rank ~ 0, g ratio ~ 1e-5) under the probe config.

**Adjudication rules (house style):** E1 is read as SUPPORTED if the pooled
Spearman is negative with a bootstrap 95% CI excluding 0 (10^5 draws, rng
seed 0) at the n=10 diagnostic; anything else is INCONCLUSIVE, never
"refuted" at n=10. E4 placebo is read as CLEAN if the hybrid correction
changes u_rank by less than the identified-family correction on the same
seeds (paired comparison of |delta|) AND the identified correction moves in
the predicted direction. E2 is descriptive at n=10 and adjudicated only at a
registered n=30 extension.

## 6. Statistical commitments made before running

- **Power / MDE, computed from the measured DMC 30-seed within-seed SDs:**
  paired eq - lagged_eq delta has SD 0.108 (n=30); the minimum detectable
  mean delta at alpha=0.05 (two-sided), power 0.8, paired design is ~0.055 at
  n=30 and ~0.095 at n=10. The observed DMC reversal (-0.073) exceeds MDE(30)
  but not MDE(10): **an n=10 probe is underpowered for the E2 contrast and is
  diagnostic only by design.** The DMC eq - hybrid wash (-0.020, SD 0.138) is
  below MDE(30) ~ 0.071 - the pre-committed reading is "inconclusive, not
  null" for any eq-vs-hybrid statement at N=30, matching the paper's existing
  language. Any n=30 extension requires this registration's decision tree to
  fire first.
- **Effect sizes:** every headline contrast in the probe report carries a
  paired standardized effect (d_z = mean / SD of the paired deltas), not raw
  deltas only.
- **Multiple comparisons:** per-endpoint pre-registration is the stated
  mitigation (E1-E5 fixed above); no joint correction is applied across
  endpoints; any endpoint whose own rule is not met is reported
  inconclusive/exploratory. This is the paper's named methodological choice,
  not an implicit practice.
- **Seeds:** diagnostic run reuses the DMC sanity seeds 0-9 (all prior DMC
  rows exist on exactly these seeds, so 15k endpoints are comparable with
  zero re-runs). The n=30 extension (only if the decision tree fires) uses
  seeds 0-29.

## 7. Design table (what will run, after the harness exists)

| env | budget | arms | seeds | runner/config | expected file |
|---|---|---|---|---|---|
| DMC hopper-hop | 15,000 steps, eval_freq 3000 | A,B,C,D + hybrid, lagged_hybrid, identified_hybrid (7 arms) | 0-9 | `udwm/scripts/probe_crn_bias.py` (new, post-registration), config base `configs/dmc_hopper_probe.yaml` | `runs/crn_bias_probe_15k_n10_gpu.json` |
| DelayedBimodal (confirmatory, cheap) | 1,800 steps | same 7 arms | 0-9 | same harness, DelayedBimodal config | `runs/crn_bias_probe_db_n10.json` |

Estimated wall cost on the company VM (2 polite workers): ~5-7 h for the DMC
n=10 diagnostic including the M3 re-evaluation passes; DelayedBimodal is
minutes. No run starts before the harness exists and this registration is
committed.

## 8. Pre-committed decision rules (what each outcome means)

- **E1 SUPPORTED and E4 CLEAN:** Hypothesis A confirmed as the explanation
  for the lag asymmetry. Next registered step: implement the Wu-style
  correction inside `combine_coupled` as a new arm (`eq_crn`), pre-register
  its DMC adjudication (30 seeds, 15k), and update the paper's Section 7
  (target-staleness mechanism) with the measured B_hat curves.
- **E1 INCONCLUSIVE but E2 shows a within-cell normalization story (lag
  vanishes in one cell):** attribution corrected - the DMC lag-axis delta is
  (at least partly) a normalization artifact, not (only) CRN staleness; the
  combined-fix narrative is updated to separate the two knobs before any fix
  arm is registered.
- **E1 INCONCLUSIVE and E2 flat in both cells:** Hypothesis A as stated is
  not supported on DMC/15k; the lag asymmetry stands as an unexplained
  benchmark-specific empirical pattern, and the paper's Section 7 is written
  as an open problem rather than a mechanism.
- **E4 placebo fails** (correction moves hybrid as much as identified):
  the correction procedure itself is confounded; no `eq_crn` arm is
  registered until the correction is validated on a fixed-map toy first.

In all branches, this probe NEVER changes the 2026-09-08 DMC verdict; it only
determines which explanation the paper may assert for the reversal.

## 9. Citations carried by this registration

- Osband et al. 2016, Deep Exploration via Bootstrapped DQN, arXiv:1602.04621
  (head-diversity-collapse quote verified).
- Glasserman & Yao 1992, Management Science 38(6):884-908, DOI
  10.1287/mnsc.38.6.884 (CRN guarantees under similar conditions).
- Wu, Zheng, Zhang, Zhang & Wang, Nonstationary A/B Tests, Management
  Science, DOI 10.1287/mnsc.2022.01205 (bias under nonstationarity +
  correction template).
- EEDQN, arXiv:2506.05716 (2025) - gap statement.
- Bratley/Fox/Schrage, A Guide to Simulation: edition/year to be re-verified
  before the manuscript cites it (do not carry the "1986" shorthand into the
  paper unchecked).
- MACURA authorship correction (2026-09-09 bib fix) applies to any
  field-baseline prose: Frauenknecht, Eisele, Subhasish, Solowjow, Trimpe,
  ICML 2024, PMLR 235:13973-14005.
