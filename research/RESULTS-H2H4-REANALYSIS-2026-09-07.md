# Re-analysis: H2 (instability-conditional benefit) and H4 (tail metrics) from existing rows (2026-09-07)

**Status: zero-compute re-analysis of rows that already exist.** Hypotheses
H2 and H4 from `upgrade-plan-hypotheses-experiments-publication.md`. Data:
`runs/policy_corrected_weights_10seed.json` (DB policy, 1,800 steps, N=10),
`runs/dmc_payoff_10seed_gpu.json` (DMC sanity, 3,600 steps, N=10),
`runs/identified_stress_50seed.json` (fixed-map reference for what is NOT in
the data). Conventions: paired deltas over shared seeds, 20,000-draw
percentile bootstrap 95% CIs, wins/N. N=10 sanity conventions throughout:
only CI-excludes-0 plus >= 7/10 wins is called confirmed.

## What the logged columns are (before reading numbers)

- `u_rank_corr` = teacher-vs-student rank correlation of the local UBE object
  on eval states (distillation fidelity of the uncertainty object; the
  paper's headline metric).
- `selective_rank_corr` / `selective_recall_bad` = fitted student-side U-net
  score (sqrt(U)) on REAL-buffer states vs |TD residual| under the final
  critic (risk-flagging usefulness in the MBPO loop). This is a DIFFERENT
  construct from u_rank_corr: it is not "teacher-student ranking restricted
  to high-uncertainty states". Neither logged column tests H4's literal
  question (subset ranking); the nearest logged evidence is these.
- Fixed-map stress rows carry no `u_rank_corr` (they use w/g/conflated rank
  metrics on a ~O(1) value map), so the fixed-map studies cannot feed the
  H2 u-rank regression at all.

## H4: does the decision-relevant tail survive when aggregate u-rank does not?

Per-arm means (u_rank_corr / selective_recall_bad), N=10:

| cell | arm | u_rank_corr | selective_recall_bad | selective_rank_corr |
|---|---|---:|---:|---:|
| DB policy | ordinary | 0.939 | 0.276 | -0.103 |
| DB policy | hybrid | 0.621 | 0.324 | -0.111 |
| DB policy | lagged_hybrid | 0.955 | 0.356 | -0.077 |
| DB policy | identified_eq | 0.835 | 0.322 | -0.100 |
| DB policy | identified_hybrid (EMA) | 0.114 | 0.257 | -0.136 |
| DMC sanity | ordinary | 0.511 | 0.282 | -0.119 |
| DMC sanity | hybrid | 0.438 | 0.312 | -0.107 |
| DMC sanity | lagged_hybrid | 0.573 | 0.291 | -0.118 |
| DMC sanity | identified_eq | 0.429 | 0.282 | -0.130 |
| DMC sanity | identified_hybrid (EMA) | 0.010 | 0.260 | -0.146 |

Paired contrasts of note (mean delta, wins/N, bootstrap 95%):

| contrast | endpoint | DB policy | DMC sanity |
|---|---|---|---|
| eq - ordinary | u_rank | -0.104 (0/10, CI excl. 0) | -0.082 (2/10, CI excl. 0) |
| eq - ordinary | recall_bad | +0.046 (5/10, CI [-0.037,+0.140]) | +0.001 (6/10, CI [-0.024,+0.022]) |
| eq - hybrid | u_rank | +0.214 (10/10, CI excl. 0) | -0.008 (5/10) |
| eq - hybrid | recall_bad | -0.002 (4/10) | -0.030 (3/10, CI upper 0.0005) |
| eq - EMA | u_rank | +0.720 (10/10, CI excl. 0) | +0.419 (10/10, CI excl. 0) |
| eq - EMA | recall_bad | +0.065 (8/10, CI [-0.065,+0.174]) | +0.023 (5/10, CI [-0.022,+0.072]) |
| hybrid - ordinary | recall_bad | +0.048 (7/10, CI [-0.054,+0.138]) | +0.031 (7/10, CI excl. 0) |
| lagged_hybrid - ordinary | recall_bad | +0.080 (7/10, CI [-0.004,+0.163]) | +0.009 (7/10, CI [-0.011,+0.028]) |

Findings:

1. **On DMC, nothing separates arms on the tail, and the tail signal is
   degenerate for every arm.** selective_rank_corr is -0.11..-0.15 for all
   five arms and recall_bad is 0.26-0.31 for all arms - the mid-threshold
   gate flags ~37% of states and catches only 26-31% of worst-20% |TD|
   transitions (at/below the random-flagging baseline), so the fitted U-net
   is not positively risk-informative under any arm at the 3.6-episode
   budget. This is independent corroboration of the registered operating-
   point reading in `RESULTS-DMC-SANITY-2026-09-07.md`: at this budget even
   the end-to-end risk signal is floor-compressed, not merely the
   distillation-fidelity metric.
2. **On DB, u-rank fidelity and tail recall partially decouple, but not in
   the direction H4 hoped.** eq's confirmed aggregate deficit vs ordinary
   (-0.104) does NOT extend to tail recall (+0.046, 5/10, inconclusive);
   eq's confirmed aggregate win vs hybrid (+0.214) also does NOT extend to
   tail recall (-0.002). hybrid's confirmed u-rank dip vs ordinary (-0.318)
   coincides with recall_bad ABOVE ordinary (+0.048, 7/10) - the naive
   decision arm's distillation-fidelity collapse does not cost it end-to-end
   risk flagging. lagged_hybrid is best on both.
3. **H4 as stated is NOT confirmed.** No eq tail advantage is ever
   confirmed (CI rule); on DMC eq's aggregate wash is a tail wash too. The
   sharpest defensible statement from the logged columns is the negative
   one: aggregate u-rank fidelity and tail risk-flagging are partially
   decoupled, and the EMA collapse - the paper's main pathology - is
   real on u-rank but muted on tail recall (EMA recall 0.26 still near the
   other arms on both cells), so tail metrics would UNDERSTATE the pathology
   if used alone. A genuine H4 test needs the ranking restricted to
   high-uncertainty states, which the logged columns do not contain.

## H2: is eq's benefit conditional on naive-loss instability?

Available eq-hybrid paired cells: exactly two (DB policy N=10, DMC sanity
N=10). A regression across two cells cannot adjudicate; the fixed-map runs
do not contain eq-vs-hybrid u-rank rows (no u_rank_corr column; different
value regime). Reported below is everything the data supports.

Between-cell (the literal H2 regression):

| cell | hybrid u_rank mean (sd, range) | eq mean | eq - hybrid gap (95% CI) |
|---|---|---:|---:|
| DB policy 1,800 | 0.621 (0.090; 0.44-0.75) | 0.835 | +0.214 [+0.172, +0.260] |
| DMC sanity 3,600 | 0.438 (0.111; 0.27-0.59) | 0.429 | -0.008 [-0.088, +0.070] |

The cross-environment correlation runs the WRONG way at N=2 cells: hybrid
instability is if anything larger on DMC (sd 0.111 vs 0.090) where the eq
benefit vanishes. The between-cell driver is not naive-loss instability but
the operating-point/budget confound already registered for DMC (ordinary
itself drops from 0.939 to 0.511; eq tracks the floor down).

Within-cell (per-seed, the part that does replicate): the eq - hybrid gap
is negatively correlated with hybrid's own per-seed level in both cells -
Spearman(hybrid_level, gap) = -0.818 (p=0.004) on DB and -0.588 (p=0.074)
on DMC (Pearson -0.649, p=0.042). Decomposition by hybrid<median seeds:

| cell | weak-hybrid seeds | strong-hybrid seeds |
|---|---|---:|---:|
| DB policy | hybrid 0.555 -> eq 0.815, gap +0.260 | hybrid 0.686 -> eq 0.854, gap +0.168 |
| DMC sanity | hybrid 0.342 -> eq 0.401, gap +0.059 | hybrid 0.533 -> eq 0.458, gap -0.075 |

eq's cross-seed sd is below hybrid's on DB (0.074 vs 0.090) but not on DMC
(0.106 vs 0.111).

Reframed H2 (what the data supports):

- Within an environment, eq stabilizes exactly the seeds where the naive
  hybrid arm is weakest (supported in both cells; DB significant).
- Between environments, that stabilization disappears exactly when the whole
  measurement is floor-compressed (DMC), i.e., eq's benefit is conditional
  on the operating point supporting rank signal at all - which is the
  registered budget confound, not naive-loss instability per se.
- The literal H2 regression claim ("eq benefit scales with cross-seed
  variance of naive loss") is NOT supported by the two existing cells and
  would need a third/fourth cell at a healthy operating point (e.g., the
  DMC 15k budget probe or the 30-seed DMC adjudication, which contain
  hybrid + eq) before it can be either confirmed or closed.

## Files / doc updates

- This record; dated addenda in `RESULTS.md` and `PAPER-NARRATIVE.md`.
- No protocol or registration changes; no rows were run for this analysis.