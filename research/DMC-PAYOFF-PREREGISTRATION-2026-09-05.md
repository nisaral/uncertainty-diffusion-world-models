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
- Arms: `ordinary`, `hybrid`, `lagged_hybrid`, `identified_hybrid` (EMA,
  expected collapse if gate is aleatoric), `identified_eq` (corrected-weight
  candidate). 3,600 env steps (the payoff protocol), MBPO SAC.
- Seeds: 30 for verdicts; endpoints and thresholds identical to the
  DelayedBimodal payoff protocol (`return_mean` delta with paired 95% CI and
  per-seed sign counts; `u_rank_corr`; `next_state_mse`; selective
  recall/risk).
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
