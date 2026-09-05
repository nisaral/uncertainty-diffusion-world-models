# Paper narrative: one loss, two unknowns, and the object it does not preserve

Assembly draft (2026-09-05). Every number here is already adjudicated in a
per-endpoint-registered result doc; this file orders the results into one
argument and adds the statistics caveats the eventual paper must state.
Status: NOT a submission; no claims beyond the tables it cites.

One sentence: cheap distilled diffusion world models are the latency fix for
imagination, but the usual decision-aware distillation loss does not pin down
the epistemic object a decision needs -- and each correction that separates w
from g preserves one half: per-term EMA reweighting keeps the negligible w and
annihilates g, while equal weights keep g at the cost of an inflated w. Which
half matters is a property of the value map's g*/w* regime (Section 6).

---

## 1. Motivation

Diffusion world models (DIAMOND-style) are expressive enough to imagine from,
too slow for online rollouts. Distillation to a 1-NFE student is the standard
latency fix. Uncertainty-aware control does not only need the mean next state:
local-UBE rewards, pessimism, and "may this imagined transition train the
policy" depend on how ensemble members disagree *after* a value map
f(s') = Qbar(s', a'). Call that epistemic term w* and the mean within-member
spread g*; the local decision statistic is u = w* - g* (Luis UBE local
rewards).

**Contribution (honest).** Not a new world model, not a new UBE, not a
coverage guarantee, not policy SOTA. A falsifiable claim about the *loss*:
matching teacher-student value disagreement at a single shared diffusion
latent does not identify epistemic uncertainty; and on a live SAC critic the
EMA-style "correction" that separates w and g fails in policy while its
equal-weight counterpart recovers the decision object at probe scale, for a
mechanism we can now name (Section 6). The deliverable is the laboratory and
the negative tables, not a positive control result.

## 2. The identifiability problem

Writing member values as Y_i = mu_i + eps_i with eps_i correlated through a
shared latent, the repo's coupling identity (theory/distill_identifiability.py,
verified) gives

    E[Var_i(Y_i)] = w* + (g* - Sigma_bar),   g* - Sigma_bar >= 0.

The usual "decision-aware" term matches this one scalar, one equation in two
unknowns. Four exact students, including one with w* = 0, all match the
statistic to ~1e-6. Drawing M >= 2 latents and matching the debiased w and g
as two separate terms separates truth (loss ~0) from collapse (loss ~0.028).

Relation to prior art: this is the ensemble analogue of the variance-shrinkage
pathology Voelcker et al. (arXiv:2505.22772) prove for value-aware model
losses under sampled models (uncalibrated value-matching prefers low-variance
models). Their collapsing variance is one model's aleatoric spread; ours is
cross-member disagreement. We do not claim to have invented variance
shrinkage.

Nothing in this construction involves a critic: the degeneracy is a property
of the loss target, and a lagged target critic addresses a different problem
(nonstationarity). Result doc:
research/RESULTS-IDENTIFIABILITY-2026-08-29.md.

## 3. Fixed value map: the loss reallocates, it does not preserve

Small-lab stress (5 members, 8 paired latents, 1,024-state grid, fixed benign
value map, exact per-seed teacher checksum pairing). Hybrid (single-latent
value geometry + variance) vs ordinary member matching, N=50 (seeds 0-29
byte-identical to the N=30 file; extension registered before seeds 30-49 ran):

- w_rmse (uncertainty magnitude): better 49/50, mean delta -0.00283,
  95% [-0.00347, -0.00223].
- w_deb_rank_corr: better 38/50, mean +0.0449, 95% [+0.0231, +0.0665].
- top-decile recall: mean +0.023, 95% [+0.0012, +0.0456] excludes 0, but only
  27/50 seeds in the direction -> **reported inconclusive** under the
  pre-registered >= 35/50 bar. (The most tempting number in the project, and
  the discipline is: it is not confirmed.)
- g fidelity degrades (g_rmse worse, CI excludes 0): the loss reallocates the
  w/g split rather than preserving it.
- Identified (M>=2) vs hybrid on the fixed map: ranking wash at N=50, slightly
  worse magnitude at 2x sample cost. Theory predicted a stationary map would
  not discriminate; the table is not a win for identified distillation.

Docs: research/RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md and
research/RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md.

## 4. Live critic: the transfer claim is falsified

Same hybrid loss dropped into MBPO/SAC with the online critic as value map.
N=30 x 5 arms x 1,800 steps, exact teacher pairing:
hybrid vs ordinary u-rank worse 0/30 (delta -0.310, CI excludes 0), u-RMSE
worse 30/30, next-state MSE better 30/30, return inconclusive. MSE improves
while the decision object dies. "Drop decision-aware distillation into online
MBPO and uncertainty is preserved" is falsified at 30/30.
Docs: research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md,
research/RESULTS-POLICY-SCALE-2026-08-22.md (10-seed original).

## 5. Policy 2x2: what separates identifiability from nonstationarity

Arms {live-Q, lagged-Q} x {M=1 conflated, M>=2 identified}:

- lagged_hybrid - hybrid: u-rank +0.331, 30/30 -> lagging the value map fixes
  the single-latent collapse. u-rmse -126.5, 30/30.
- identified_hybrid - hybrid: -0.464, 0/30 (harm). lagged_identified - hybrid:
  -0.353, 1/30 (not a fix). Identified arms match w to machine precision
  (w_rmse 0.016, best in table) but collapse student u_mean to ~-0.004 vs
  teacher -50..-105: the propagated/local U field is ~0 everywhere, rank is
  noise. w-rank is the one endpoint identified arms improve (0.29 vs 0.22).
- Identified arms' return is best in table (-84.8 vs ordinary -104.4) but this
  is a secondary endpoint that does not adjudicate.
- (2026-09-05 correction: the identified rows in this table ran with the
  per-term EMA reweighting on; the collapse is attributed to that weighting,
  not to the identified target itself -- Section 6 and
  research/RESULTS-CORRUPTION-2026-09-05.md.)

Decision-tree outcome (registered in
research/DECISION-TREE-2X2-PREREGISTRATION.md): Row 1 - nonstationarity is the
dominant measured mechanism; the identifiability pathology stands as a
theoretical result (proof + toy) with no measured policy benefit here. The
payoff test should test lagging alone.

(2026-09-05 addendum: the identified rows in Section 5 ran EMA-reweighted.
The corrected-weight N=10 policy study
(research/RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md, registered before
running) shows equal-weight identified transfers to the live critic - u-rank
0.835, 9/10 >= 0.70, pre-registered bars A met; +0.214 vs hybrid (10/10),
+0.720 vs the EMA arms (10/10), -0.104 vs ordinary. "Identifiability has no
measured policy benefit" is superseded for the equal-weight arm; the EMA
normaliser, not the identified target, was the policy-scale pathology.)

## 6. Mechanism of the identified-arm policy collapse (thread #1; corrected 2026-09-05)

The prior 30-seed doc attributed the collapse to "u = w - g landing on the
u_min = 0 floor". That was wrong for the eval object (eval is unclamped), and
a config-override bug (fixed; registered in
research/CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md) meant every historical
"equal-weight / w-only / g-only / M=8" mechanism cell actually ran with the
per-term EMA reweighting on. The corrected picture
(research/RESULTS-CORRUPTION-2026-09-05.md, probe data in runs/probe_u_*.json):

1. On DelayedBimodal under the trained SAC critic the local object is
   aleatoric-dominated: teacher u ~ -g, with g ~ 73-90 vs w ~ 0.004-0.008
   (g/w ~ 1e4). Q(s', pi(s')) is hugely latent-sensitive; members agree in mean
   (coupling ~ 0.9998).
2. EMA-both and w-only-EMA collapse the student's aleatoric half: student g
   ~0.01-0.016 vs teacher ~70-90, latent-to-state spread ~0.006 vs teacher
   ~0.37 (ordinary and lagged_hybrid keep ~0.33), u-rank ~ noise. Student w
   still matches teacher w: the epistemic half works. Equal-weight identified
   (the corrected control) does the opposite: student g lifts to ~teacher
   within the 180-update run (u-rank 0.885/0.920 on seeds 0-1, schedule
   corruptions) and student w inflates ~50x -- the equal-weighting hole.
3. Driver and non-drivers: the collapse is the epistemic up-weight under the
   EMA normaliser (~1e5-1e6 over the aleatoric term) on an aleatoric-
   dominated map; the aleatoric down-weight is not required (w-only EMA
   collapses identically), and the corruption distribution of the decision
   terms is not the binding constraint (g recovers at schedule/maxt/pure
   alike). The measured ~1e-6 per-update "leverage" and ~100x estimator noise
   describe the EMA-down-weighted channel, not the raw aleatoric term.
4. Consequences: the identified loss is a correct fix for *identifiability*;
   the N=30 policy collapse was the EMA weighting, not the identified target.
   The operating envelope is a balance-window problem: per-channel estimator
   noise (std(g_hat)/g* ~ sqrt(2/(N(M-1))), std(w_hat)/w* ~ 1/sqrt(M)) sets
   how much one channel must be up-weighted before it dominates the shared
   parameters, and no single scalar weight recovers both w and g when
   g*/w* >> 1 (theory/identified_balance_window.py). The fixed map (w, g
   comparable) stays benign, which is why the fixed-map tables do not show the
   pathology. The registered DMC gate
   (research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md) is the out-of-sample
   test of this envelope.

(2026-09-05 policy-level confirmation: the corrected-weight N=10 study meets
both pre-registered bars - equal-weight identified u-rank 0.835 (9/10 >= 0.70),
+0.720 vs identified_hybrid (10/10), +0.214 vs hybrid (10/10), -0.104 vs
ordinary. Equal-weight transfer is real but partial: lagged_hybrid (0.955)
remains the best ranker and ordinary (0.939) still beats eq by ~0.10.)

## 7. Payoff test (#4): mechanism survives 2x horizon, return payoff does not

Registered test of lagging alone at 3,600 steps (2x the 2x2 horizon), N=30:
- P1 mechanism: lagged_hybrid u-rank 0.988 vs ordinary 0.986; beats hybrid
  +0.273, 30/30. Confirmed.
- P2 return payoff: mean delta -44.6 vs ordinary, 15/30 better,
  95% CI [-102.6, +10.2] -> not demonstrated (point estimate null-to-negative;
  the interval now excludes a large positive payoff).
- Caveat: absolute returns degrade on DelayedBimodal past ~1,800 steps (MBPO
  stack issue), so longer horizon did not give the better world models more
  room to pay off. A payoff claim needs a benchmark where longer rollouts
  compound (Section 9).

Doc: research/RESULTS-PAYOFF-LAGGED-2026-09-03.md (registration:
research/PAYOFF-LAGGED-PREREGISTRATION.md).

## 8. Statistics statement the paper must carry

We do not apply a joint multiple-comparisons correction across the endpoints
within any study (the N=50 table alone reports 11 endpoints). The mitigation
is per-endpoint pre-registration: endpoint set, direction, and adjudication
rule (bootstrap CI excludes 0 AND >= 70% of paired seeds, or the stated
variant-specific bar) were fixed in the result/registration doc *before* each
run was read, and every adjudicated row records wins/N with the interval. This
controls the cherry-picking failure mode (endpoints cannot be selected after
seeing results) but does not control the family-wise error rate across
endpoints; we therefore treat any single endpoint whose own registration is
not met as exploratory, and we call every such case inconclusive rather than
null. The 50-seed recall endpoint is the worked example: CI excludes 0 but the
seed bar is not met, so it is reported inconclusive, and any further extension
was declared to require a new registration before the run.

## 9. Limitations and future work

- What this is not: first uncertainty-gated imagination (MACURA, Nguyen,
  Kalweit), a new UBE recursion, calibrated/conformal coverage, policy SOTA,
  or a done story for the identified loss in policy.
- The identified-loss policy study at N=30 is not "finished" as a positive;
  it is finished as an adjudicated negative on this benchmark (2026-09-05:
  superseded for the equal-weight identified arm - the corrected-weight N=10
  study meets its pre-registered transfer bars; the EMA-variant negative
  stands and is now attributed to the EMA normaliser, see Section 6).
- Resolved sub-thread (mechanism): the "why does the identified loss remove
  latent sensitivity" question is a weighting question, not a corruption
  question -- equal-weight identified keeps the pure-noise mapping
  latent-sensitive (g recovers) while EMA-both/w-only remove it
  (research/RESULTS-CORRUPTION-2026-09-05.md).
- Open (balance window): the equal-weight hole on w (student w ~50x teacher at
  g*/w* ~ 1e4) has no measured scalar-weight fix that also keeps g; whether a
  dimensionless w-scale gate (~1e2-1e3, not the EMA normaliser) exists is
  unmeasured and would need a new registration before running.
- Next benchmark (modality decision, deferred): the payoff null at this scale
  is most plausibly a horizon problem, not a modality problem. The
  right-sized next step is a standard long-horizon low-dimensional task
  (e.g. a DMC/MuJoCo-scale environment where model error compounds over long
  rollouts), which also closes the repo's stated "no MuJoCo/DMC table"
  limitation. Pixels (image-scale diffusion) are explicitly future work and
  would only be justified after a null on the long-horizon benchmark - the
  GPU finding (research/RESULTS-GPU-VALIDATION-2026-09-05.md) says this lab's
  compute does not pay off below image scale. Any new benchmark requires a new
  registration before running, per the discipline used throughout.

## Artifact map

- Theory: theory/distill_identifiability.py; estimator bias derivations under
  theory/.
- Fixed map: research/RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md (runner:
  udwm/scripts/run_decision_distillation_stress.py, --device cpu|cuda).
- Policy 2x2: research/RESULTS-POLICY-2X2-30SEED-2026-09-03.md (runner:
  udwm/scripts/run_delayed_bimodal_policy_ablation.py,
  udwm/scripts/run_policy_2x2_split_seeds.py; summarizer:
  udwm/scripts/summarize_policy_2x2.py).
- Payoff: research/RESULTS-PAYOFF-LAGGED-2026-09-03.md.
- Mechanism/correction: research/U-COLLAPSE-MECHANISM-2026-09-05.md,
  research/RESULTS-CORRUPTION-2026-09-05.md (probe:
  udwm/scripts/probe_u_collapse.py; data runs/probe_u_*.json).
- Balance window: theory/identified_balance_window.py (+ toy re-adjudication
  of runs/ground_truth_w_g.json).
- DMC gate: research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md.
- Registration docs: research/DECISION-TREE-2X2-PREREGISTRATION.md,
  research/PAYOFF-LAGGED-PREREGISTRATION.md,
  research/CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md; N=50 amendment
  inside the 50-seed stress doc.
- Reproduce/GPU: REPRODUCE.md, RUN_ON_GPU.md.