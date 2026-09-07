# DMC gate preregistration: out-of-sample test of the balance-window theory (2026-09-05)

**Status: registered before any DMC-scale run.**  No dm_control/MuJoCo run has
happened; the environment is not installed on this machine (`gymnasium` only).
Runs go on a GPU host (Kaggle ships `dm_control`/`mujoco`); commands are in
`RUN_ON_GPU.md`; seed-parallel policy runs use
`udwm/scripts/run_policy_2x2_split_seeds.py --gpu-ids` (see `RUN_ON_GPU.md`).

**Why this study exists.** The DelayedBimodal policy 2x2 (N=30) falsified
"drop decision-aware distillation into online MBPO and uncertainty is
preserved" for the EMA-reweighted identified arm. The 2026-09-05 mechanism
correction (see `RESULTS-CORRUPTION-2026-09-05.md`) shows that collapse was a
**weighting** artifact, not an identifiability or corruption-distribution
limit: with per-term EMA reweighting the epistemic term is up-weighted
~1e5-1e6 relative to the aleatoric term and `g` is annihilated; with equal
weights `g` (and hence `u`, aleatoric-dominated) is recovered at probe scale.
The open question for a *payoff* claim is whether preserving the decision
object improves return on a benchmark with genuine long-horizon structure -
the deferred modality question. DMC (low-dimensional state vectors, long
episodes where model error compounds) is the right-sized move before pixels.

## Amendment 1 (2026-09-05, after the DelayedBimodal corrected-weight N=10 result)

The corrected-weight N=10 policy study
(`research/RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md`, registered before
running) landed Branch A on DelayedBimodal: equal-weight identified transfers
under the live critic (u-rank 0.835, 9/10 >= 0.70; +0.214 vs hybrid, 10/10;
+0.720 vs the EMA arms, 10/10; -0.104 vs ordinary and -0.120 vs
lagged_hybrid, 0/10). This is **partial transfer, not parity**. This
registration is amended accordingly, before any DMC run:

1. **Three conditions, not two.** The DMC comparison distinguishes (i) the
   baseline `ordinary`/`lagged_hybrid` (full u-rank ~0.94-0.96 on
   DelayedBimodal), (ii) the `identified_hybrid` EMA collapse control
   (expected u-rank ~ noise), and (iii) `identified_eq` as its own arm, the
   equal-weight partial-transfer candidate. `identified_eq` is expected to sit
   above `hybrid` and the EMA arms but below `ordinary`/`lagged_hybrid`;
   DMC adjudicates whether the partial-transfer story confirms (same ordering)
   or complicates (gap closes to parity, or transfer fails entirely).
2. **Return-payoff thread merged here.** Return has been null at every
   DelayedBimodal scale tested (3,600-step payoff, N=30; corrected-weight
   N=10: inconclusive everywhere). No further toy-scale return registration
   will be opened; DMC's long horizon is the same experiment that can resolve
   regime transfer and whether preserved u-rank cashes into return at once.
3. **Gate protocol concretized.** Run the pilot seeds {0, 1} with the
   `ordinary` arm only (same runner/protocol as the main study); per-seed
   gate ratio = `row.teacher_g_mean / row.teacher_w_mean` (live SAC critic,
   real-buffer eval states; reader `udwm/scripts/dmc_gate_ratio.py`).
   Median across the pilot seeds decides the regime below. Gate must pass
   before the full comparison is launched.

## Diagnostic gate (mandatory, first)

Before running the full comparison, measure the teacher's `g*/w*` ratio under
the DMC-trained SAC critic on real-buffer states (the repo's
`decompose_ube` / `evaluate_distillation_uncertainty` machinery, eval-time
pure-noise latents). Record the ratio per seed and its median across seeds.

Pre-committed conclusions per regime (decided before looking):
- **Aleatoric-dominated, `g*/w*` >> 1 (expect ~1e2-1e4 like DelayedBimodal):**
  the identified-arm results should reproduce the DelayedBimodal pattern -
  EMA-both identified collapses `u` (included as a mechanism-transfer control,
  expected negative), equal-weight identified recovers `u`-rank at the cost of
  the `w` hole. The *payoff* question then is: does recovered `u` improve
  return over ordinary/lagged-hybrid at long horizon?  Verdict on return,
  not on the mechanism (already adjudicated).
- **Balanced, `g*/w*` ~ O(1)-1e1:** the balance-window theory predicts both
  `w` and `g` recover under equal-weight identified (and, if within the
  window, under a moderately reweighted arm). Success condition pre-set:
  identified-eq `u_rank` >= 0.7 on >= 70% of seeds AND `next_state_mse` not
  degraded beyond 2x ordinary.
- **Epistemic-dominated, `g*/w*` << 1 (unlikely under a Q-value map):**
  equal-weight identified should match `w`; `u ~ w`; report as-is.

If the gate says aleatoric-dominated, the comparison arms carry the equal-
weight identified arm (not the EMA arm) as the uncertainty-preserving
candidate; the EMA arm remains as the control that reproduces the collapse.

## Protocol (registered; to be adapted only by a new registration)

- Environment: one DMC task with episode length >= 500 and a horizon where
  model error compounds (e.g. a locomotion task with default 1,000-step
  episode). `udwm/envs/registry.py` already routes `dm_control/<task>` ids
  through shimmy for generic 1-D Box obs/act spaces, but no DMC config +
  trainer smoke has run (no dm_control on this machine); the GPU-host smoke
  checklist in `RUN_ON_GPU.md` (Kaggle quickstart, step 3) is the blocking
  check before the gate.
- Arms (three conditions, Amendment 1): `ordinary`, `hybrid`,
  `lagged_hybrid` (baseline); `identified_hybrid` (EMA collapse control);
  `identified_eq` (equal-weight partial-transfer candidate, run as its own
  arm). 3,600 env steps (the payoff protocol), MBPO SAC.
- Seeds: 30 for verdicts; endpoints and thresholds identical to the
  DelayedBimodal payoff protocol (`return_mean` delta with paired 95% CI and
  per-seed sign counts; `u_rank_corr`; `u_rmse`; `w_rmse`; `next_state_mse`;
  selective recall/risk). `identified_eq` DMC bars mirror the DelayedBimodal
  study: u_rank >= 0.70 on >= 70% of seeds AND u_rank beats `hybrid` on
  >= 70% of seeds (confirm); the gap to `ordinary`/`lagged_hybrid` is
  reported as the partial-transfer magnitude.
- Pre-commitment: report every endpoint for every arm; no post-hoc arm
  addition. Multiple-comparisons statement as in `PAPER-NARRATIVE.md`.

## What each outcome means

- Balanced gate + identified-eq recovers rank and return improves: the
  boundary-condition theory transfers out of sample; the write-up becomes
  "identified decision distillation works when the aleatoric/epistemic scales
  are comparable (or when the map is aleatoric-dominated and equal weights are
  used to preserve u); the EMA normaliser is the failure mode to avoid."
- Aleatoric gate + equal-weight u-rank preserved but return still null: the
  DelayedBimodal payoff null was a horizon/benchmark issue, not a preservation
  issue; pixels stay future work.
- Aleatoric gate + u-rank also collapses under equal weight on DMC: the
  mechanism correction does not transfer; a new mechanism thread opens.

**Files.** Pre-registration: this file. Data:
`runs/dmc_payoff_*.json` (GPU/Kaggle host). Results doc:
`research/RESULTS-DMC-PAYOFF-*.md` (to be written after the gate).

---

## Addendum 3 (2026-09-07): combined-fix arm folded in; gate made an explicit axis

**Before any budget-probe or 30-seed adjudication row runs**, the DMC arm list
is amended, following the DelayedBimodal combined-fix N=10 result
(`research/RESULTS-COMBINED-FIX-POLICY-2026-09-07.md`):

1. **`lagged_identified_eq` is added as a primary arm.** It is the top
   DelayedBimodal configuration (u-rank 0.948, 10/10 >= 0.70; closes
   identified_eq's confirmed deficit vs ordinary to parity) and is the natural
   DMC headline arm for the "both known failure modes fixed" claim. The DMC
   arm list becomes: `ordinary`, `hybrid`, `lagged_hybrid`, `identified_hybrid`
   (EMA collapse control), `identified_eq`, `lagged_identified_eq`. The
   budget probe (15k steps, seeds 0-1) runs this full list so the
   budget-adequate comparison needs no second probe.
2. **Gating is an explicit axis, not a hidden default.** The base config has
   run `u_gate.mode: both` (percentile stop+weight, 0.85) from step 900 since
   creation; the DelayedBimodal tables are self-gated runs and the paper must
   say so (audit note: `research/RESULTS-THEORY-FRONTIER-2026-09-07.md`). DMC
   arms therefore gate identically (percentile, fixed rule) and the registered
   payoff question is read as "does preserved u-rank cash into return under
   self-gated imagination". A separate `gate-off` control arm (ordinary,
   `u_gate.mode: off`) is added so the payoff contrast can separate "the
   uncertainty object helps gating" from "gating itself hurts".
3. **Scale-robust gating rationale.** Theorem 6 of
   `research/proofs/identifiability-frontier.md`: percentile thresholds are
   invariant to strictly increasing score transforms, so a rank-preserving
   but scale-collapsed student gates identically to the teacher; absolute
   thresholds silently disable the gate. All gating in this study is
   percentile-based (config default), held fixed across arms.

No change to endpoints, seeds, or adjudication bars. The budget-probe
registration (`research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`) now
carries the amended arm list.

## Addendum 2 (2026-09-07): 10-seed sanity + gate recorded; no protocol change yet

The DMC 10-seed sanity + gate record exists
(`research/RESULTS-DMC-SANITY-2026-09-07.md`): gate regime
aleatoric-dominated (median g*/w* = 8,224); bar 1 replicates cross-environment
(eq - identified_hybrid u_rank +0.419, 10/10); bar 2 does not (eq 0/10 >=
0.70, mean 0.429); eq - hybrid is a wash (-0.008, 5/10), i.e. the
DelayedBimodal headline contrast does not transfer at sanity scale. Because
every arm loses ~half its u_rank level vs DelayedBimodal (ordinary 0.51) at a
3.6-episode budget (3,600 steps on 1,000-step episodes) with flat returns
everywhere, the sanity leaves an operating-point confound open. The registered
budget probe (`research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`,
15k = 15 episodes, seeds 0-1) decides whether the 30-seed adjudication runs at
3,600 steps as written or is amended (Amendment 2). No endpoint/arm/seed
change is made by this addendum; do not quote sanity rows as adjudicated.

## Addendum 4 (2026-09-07): the gate-off control arm is registered

Addendum 3 item 2 committed a separate `gate-off` control arm "so the payoff
contrast can separate 'the uncertainty object helps gating' from 'gating
itself hurts'". This addendum registers the arm, its config, and its
adjudication role before any payoff row runs. No endpoint, seed, or bar of
the six-arm comparison (Addendum 3) changes.

1. **Arm and override.** `ordinary_gate_off`: ordinary distillation (every
   decision weight 0) with `u_gate.mode: off`, applied as a root-level
   override (`VARIANTS["ordinary_gate_off"]._root`, added to
   `udwm/scripts/run_delayed_bimodal_policy_ablation.py` on 2026-09-07;
   verified: only `u_gate.mode` differs from `ordinary`, all other knobs
   identical). Same runner, same per-seed teacher checksum pairing, same
   eval endpoints, same seeds as the main comparison; rows written to a
   separate file (`runs/dmc_payoff_30seed_gpu_ctrl.json`) and never merged
   into the six-arm table.
2. **Role.** The six uncertainty arms answer "does preserved u-rank cash
   into return under self-gated imagination" (all gate identically:
   percentile 0.85 stop+weight from step 900). `ordinary_gate_off` answers
   the control question "does the gate itself move return for the *same*
   model arm" by toggling only the gate. It is a payoff contrast, not a
   mechanism arm: the identifiability mechanism is already adjudicated and
   is not re-opened here.
3. **Endpoints.** Primary: paired `final_return` delta
   (`ordinary_gate_off` - `ordinary`), bootstrap 95% CI + wins/N, same
   convention as every payoff contrast in this repo. Descriptive only
   (never adjudicative): `u_rank_corr` and `next_state_mse` on the gate-off
   arm - the gate changes which imagined rollouts enter the model/policy
   buffers, so the distillation environment (and therefore eval-side
   fidelity) is not identical-by-construction across the toggle.
4. **Pre-committed reading rule.** (i) CI excludes 0: gating has a
   measurable return effect at this budget, and the payoff headline is read
   conditionally (preserved u either offsets a gating penalty or compounds
   a gating gain - say which, with the same CI discipline). (ii) CI
   includes 0: gating is return-neutral at this budget and the registered
   u-rank-preservation story is the whole measurable payoff; report as
   such. (iii) The rule is applied to the registered payoff budget only -
   never to probe/sanity rows.
5. **Staging.** `dmc_payoff.sh` stage2b runs the control
   (`CTRL_VARIANTS=ordinary_gate_off`, `OUT_CTRL`); the Amendment-2 draft
   config `configs/dmc_hopper_payoff_30k.yaml` is pre-committed but is NOT
   runnable until the budget probe's decision rule records Amendment 2.

## Amendment 2 draft (2026-09-07): adequate-budget knobs, activation pending

Per the budget-probe registration's pre-committed rule, if the probe
(15k = 15 episodes, seeds 0-1) supports the budget confound, the 30-seed
adjudication budget is amended from 3,600 to >= 15,000 env steps and the
gate re-measured at the amended operating point before any 30-seed row runs.
Knobs are pre-committed now (nothing below changes until the probe verdict):
`configs/dmc_hopper_payoff_30k.yaml` - `total_env_steps: 30000` (30
episodes of the 1,000-step task, twice the DelayedBimodal-comparable 15),
`eval_freq: 3000` (10 in-training evals + final, bounded wall time), payoff
log/checkpoint dirs distinct from probe and sanity paths. A 15k payoff (if
chosen over 30k) uses `configs/dmc_hopper_probe.yaml` with the payoff arm
list. The registered decision rule and all six arms + the gate-off control
carry over unchanged.
