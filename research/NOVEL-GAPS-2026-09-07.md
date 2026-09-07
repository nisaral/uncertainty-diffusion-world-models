# Novel gaps beyond the current claim set (2026-09-07)

**Status:** proposal document, not results. Every item below is a falsifiable
experiment design (or a theory hook) that the current repo does not yet test.
None has run; none is adjudicated. Follow the standing rule: any run needs a
dated pre-registration before execution, and any write-up needs wins/N +
bootstrap CI. Cost column says whether it re-reads existing rows (no GPU),
needs a small CPU run, or needs the Kaggle DMC staging.

## G0 - learner-distill round-trip (does the u-signal loss compound?)

- Claim: distillation is a one-way map today (teacher -> 1-NFE student). If
  the student is distilled back into a multi-step teacher and re-distilled
  into a fresh student (a round trip), the decision-statistic `u`-signal loss
  either plateaus (bounded, one-shot) or compounds (each leg re-randomizes the
  fibre landing). The repo's Theorem 7 fibre-memory check predicts the M=1
  loss *re-introduces* initialization dependence on every leg even when the
  map is frozen, i.e. compounding; the equal-weight M>=2 loss should plateau.
- Why novel: nobody measures the uncertainty object across distillation
  chains; the literature tests one-shot student fidelity.
- Cost: small CPU runs on DelayedBimodal (existing runners, new chain logic).
- Theory hook: turns Theorem 7 C7.1 into a quantitative compounding claim
  (error after k legs vs k).

## G1 - Neyman-on-A adaptive MC allocation (allocate by variance, not by w)

- Claim: `theory/estimator_variance.py` derives the per-state MC budget rule
  `M_s proportional to sqrt(A_s)` (A = variance coefficient; kurtosis-driven).
  The repo has never run an adaptive-allocation arm. Hypothesis: allocating
  the fixed per-step MC budget by `sqrt(A_s)` reduces the worst-state
  `u`-error at equal total budget vs the current uniform-M allocation and vs
  allocation by `w` (the naive choice); and the adaptive gain concentrates on
  rare-mode states (kurtosis), exactly the states diffusion models exist for.
- Why novel: QU-SAC/UBE lines use uniform or analytic allocations; the
  variance-coefficient allocation with a closed-form A is the repo's own
  derivation, untested.
- Cost: CPU toy + one policy arm; reuses `toy_mc_ube_estimator.py`.
- Theory hook: finite-budget optimality is a Lagrange/Neyman statement; a
  strict improvement bound over uniform allocation under kurtotic laws is
  provable.

## G2 - time-shift cross-evaluation (out-of-domain without retraining)

- Claim: eval the student distilled at training step t against teacher
  members frozen at step t (matched, current practice) *and* against teacher
  members at step t + k and t - k (time-shifted). The u-rank decay under
  teacher time-shift measures how much of the preserved signal is
  student-loss carryover vs teacher-drift tracking. A large decay would mean
  the combined fix (lagged + equal weight) is tuned to the map cadence, not
  robust across cadences - a real deployment question for MBPO loops.
- Why novel: no prior work time-shifts the teacher reference in a distilled
  world-model UBE evaluation; all repo runs use matched-step teachers.
- Cost: no GPU - re-reads existing per-seed eval rows if teacher snapshots
  were kept; otherwise a small CPU run storing teacher snapshots.

## G3 - PCA-latent distillation line: orthogonal-residual penalty (confident-garbage probe)

- Claim: the PCA/SVD-latent distillation line (venue-verified in the
  2026-09-07 literature sweep; cite only with a confirmed source) matches the
  top principal components of the teacher's output distribution but drops the
  orthogonal residual. A student can then be confident on the projected
  subspace while the discarded residual carries the disagreement (the repo's
  `w` lives partly in the residual). Experiment: add an orthogonal-residual
  penalty to that family's loss on the repo's toy and measure whether
  `u`-rank of the projected student tracks the teacher or falls to the level
  the repo measures for the M=1 fibre (a "confident garbage" positive
  control).
- Why novel: the repo's w/g decomposition is exactly the tool that line
  lacks; the measurement (residual share of w) is new.
- Cost: CPU toy; reuses `ground_truth_w_g.py` structure.

## G4 - normalized vs unnormalized UBE gating (gate-mode arm axis)

- Claim: the repo gates by percentile threshold (Theorem 6: the only
  scale-robust rule for rank-preserving students). The open question is
  whether *normalizing* the score (per-state variance-normalized `u`) before
  an **absolute** threshold recovers scale-robustness at the cost of a
  distributional assumption. A DMC result from OpenAI's normalized ensemble
  Q-learning is a strong hint normalization changes behaviour - unverified,
  do not cite (see `papers/LITERATURE-VALIDATION-2026-09.md`). Experiment:
  gate-mode arm axis in the payoff design (percentile vs absolute-on-
  normalized), held fixed across uncertainty arms, DelayedBimodal first.
- Why novel: separates "what to gate on" (u preservation) from "how to gate"
  (threshold rule) - the paper currently only claims the former.
- Cost: one policy arm set on CPU; rides the existing payoff registration.

## G5 - teacher-relative oracle ranking tolerance (calibration vs capacity)

- Claim: u-rank on all states conflates calibration (is the student's rank
  right everywhere?) with capacity (can the student rank the genuinely
  uncertain states?). Restricting the evaluation to the top-tau states by
  *teacher* uncertainty (the oracle top-K) separates them: an arm that holds
  top-K u-rank but loses full-support u-rank is a calibration failure on
  low-uncertainty states; an arm that loses both is a capacity failure. H4
  (tail metrics) is the complementary bottom-tau check and was cleanly
  refuted - top-tau has not been run.
- Why novel: the selective-prediction literature (memo C5) supports exactly
  this split for classifiers; nobody has run it on distilled world-model
  students.
- Cost: no GPU - re-reads existing eval rows (rank endpoints are already
  logged per seed).

## G6 - full risk-coverage curves for gate claims

- Claim: the payoff story ("gating on the preserved u is safe") needs the
  risk-coverage curve the selective-prediction norm expects: refusal rate on
  the x-axis, imagined-rollout error/regret on the y-axis, one curve per arm.
  The repo logs `selective_recall_bad` / `selective_rank_corr` (two points);
  a full curve is the upgrade and would let the payoff contrast quote
  coverage-matched comparisons instead of a single percentile.
- Why novel: positions the payoff result against the El-Yaniv-Wiener norm the
  memo's protocol-alignment section already commits to.
- Cost: evaluator change + re-read of existing rollout logs; no new training.

## Which to run first (cheap-first ordering)

1. G5 and G6: zero-GPU re-reads of existing rows; H4 already cleared the
   bottom-tail question, so these close the evaluation design before the DMC
   payoff runs.
2. G0 and G1: small CPU runs on DelayedBimodal; each needs a dated
   pre-registration. G0 has a direct Theorem 7 hook.
3. G2, G3, G4: design-level until the DMC verdict lands; G4's gate-mode axis
   should be fixed in the DMC payoff registration before it runs, not after.
