# Hypothesis C registration - gradient interference scaling with task complexity (2026-09-10)

**Status: registered 2026-09-10.** Gate 1 (architecture coherence) and Gate 2
(Hypothesis-D desk check) were both answered *before* registration and are
reported below with their artifacts. The probe harness
(`udwm/scripts/probe_gradient_interference.py`) and its adjudicator
(`udwm/scripts/summarize_gradient_interference.py`) are implemented and
wiring-validated; no adjudicable row has been produced yet. This probe is a
mechanism-attribution study: it never re-adjudicates the DMC payoff bars or the
DelayedBimodal transfer bars.

This is the project's **third and final mechanism attempt** at explaining the
DMC transfer failure:

| # | Hypothesis | Outcome |
|---|---|---|
| A | CRN-bias / value-reference staleness under paired M>=2 sampling | **refuted cleanly** (DMC n=10 diagnostic, `research/RESULTS-CRN-BIAS-N10-2026-09-09.md`) |
| B | (return-payoff question) | deferred / gated, unchanged |
| **C** | **gradient interference between the uncertainty-matching and point-prediction terms scales with task complexity** | **this registration** |
| D | M=2 estimator variance scales with dimensionality | **desk check, not suggestive - not escalated** (section 3) |

**Pre-committed stop condition.** If C is refuted, the DMC transfer failure is
written up as a characterized-but-unexplained open question. There is no fourth
mechanism attempt; three well-grounded, falsifiable attempts (one confirmed
elsewhere in the project, two here) is the honest stopping point.

---

## 1. Gate 1 - architecture coherence (ANSWERED before registering)

Hypothesis C is only coherent if the uncertainty-matching terms and the
point-prediction term backprop through **shared** parameters. If the architecture
were separate heads on separate trunks, C would be moot and would be dropped.

**Answer: the parameters are shared. C is coherent.**

Artifact: `udwm/scripts/check_distill_param_sharing.py`; outputs
`runs/hypothesis_c_param_sharing_db.json`,
`runs/hypothesis_c_param_sharing_hopper.json`. Method: rebuild the exact model
from a saved checkpoint's `cfg`, load its weights, compute the identified loss
keeping each term live, backward each term separately, and intersect the
parameter sets that receive nonzero gradient (CPU, seconds; nothing trained).

Read from the real DelayedBimodal checkpoint
(`checkpoints/hf_diagnostic_delayedbimodal/identified_eq_seed0.pt`,
obs_dim 5 / action_dim 1 / x_dim 6, clean load: no missing or unexpected keys):

| term group | student param tensors receiving gradient |
|---|---|
| `member` (point prediction) | 50 |
| `epistemic_w` (uncertainty matching) | 50 |
| `aleatoric_g` (uncertainty matching) | 45 |
| `value_geometry` | 50 |
| **Jaccard(point, epistemic U aleatoric)** | **1.00** |
| verdict | **SHARED_TRUNK** |

The 50 tensors are the full student: 5 members x (`net.0/2/4` weights+biases,
`t_embed.0/2` weights+biases). There is **no separate point-prediction head**:
`ConsistencyStudent` is one `ConditionalDenoiser` per member, and every term is a
functional of the same `forward_member` output. Confirmed for the Hopper arm
topology (`configs/dmc_hopper_probe.yaml`, 15/4) for `identified_eq`,
`lagged_identified_eq` and `identified_hybrid` - Jaccard 1.00 in all three.

`ordinary` is not applicable by construction: every decision weight is zero, so
no uncertainty term exists to compete with (it is the zero-competition reference
arm, not a measured arm).

---

## 2. Hypothesis C (formal statement)

**Claim.** The identified loss's uncertainty-matching terms (`epistemic_w`,
`aleatoric_g`) and its point-prediction term (`member`) share the student trunk
(Gate 1). Their gradients may increasingly conflict as task complexity rises,
which would degrade the uncertainty-ranking sub-objective while leaving the
shared trunk's main-task performance intact - the asymmetric-degradation pattern
observed. This is an independent causal channel from Hypothesis A: it does not
depend on the M>=2 pairing mechanism that the placebo already disproved, and it
would explain why `ordinary` (no competing objective at all) wins outright on
the harder task independent of the lag axis.

**Grounding.**

1. Liu et al., *Measuring and Mitigating Interference in Reinforcement Learning*
   (PMLR 2023). Directly relevant, RL-native measurement methodology for
   destructive gradient interference between competing objectives sharing
   network parameters: (i) larger / more complex problems show measurably higher
   interference and degradation; (ii) target networks reduce interference
   magnitude. This registration adopts their measurement object - the conflict
   between the gradients of the competing objectives on the shared parameters at
   matched training checkpoints.
2. *Catastrophic Interference in Reinforcement Learning* (2021). Documents that
   interference worsens for high-dimensional, complex tasks relative to simple
   ones - the mapping from DelayedBimodal (5 obs / 1 action, smooth bimodal value
   landscape) to hopper-hop (15 obs / 4 action, contact dynamics, harder value
   landscape).

**Why this is a stronger starting point than A was.** It is a separate causal
channel, not a patched version of A. It predicts the observed asymmetry without
needing the pairing mechanism at all.

**Connection to a previously-shelved item.** This is not a new idea invented
under pressure: `research/NOVEL-GAPS-2026-09-07.md` G8 already specifies a
*capacity-competition* model on the student's shared trunk, and its stated
falsifiable prediction uses **student hidden width as the capacity knob**. G8
remains "the only mechanism-level open item with an unclosed empirical
attribution". Hypothesis C's endpoint E3 (section 7) is the gradient-level
measurement of exactly that shared-trunk competition.

---

## 3. Gate 2 - Hypothesis D desk check (ANSWERED before registering)

Hypothesis D (secondary, weaker grounding): the M>=2 debiased estimator's
practical variance at fixed M=2 scales unfavourably with state/action
dimensionality, swamping the correction's benefit on Hopper.

Per the plan this is a ~20-minute desk check, not a registered experiment.
Artifact: `theory/hypothesis_d_dimension_desk_check.py`; output
`runs/hypothesis_d_desk_check.json`.

The repo's own balance-window theory (`theory/identified_balance_window.py`)
predicts, for the Gaussian within-member case,

    std(g_hat)/g* ~ sqrt(2 / (N (M - 1)))

which **contains no obs_dim or action_dim term**. At the registered operating
point (N = model_batch_size = 128, M = 2) it gives **0.125**, identically on
DelayedBimodal (5/1) and DMC (15/4); the M-scaling at N=128 is
0.125 / 0.072 / 0.047 / 0.032 at M = 2 / 4 / 8 / 16.

A dimension-*independent* noise term cannot produce an environment-*specific*
deficit: the observed `identified_eq - ordinary` u-rank gaps are -0.104 on
DelayedBimodal and -0.246 on DMC.

**Verdict: NOT SUGGESTIVE.** The only surviving channel is non-Gaussian /
heavy-tailed constants in the O(1/sqrt(N)) rate, which the formula does not
model and which the repo's theory does not predict.

**Pre-committed consequence: Hypothesis D is not escalated to a registered run
on the strength of this check.** (It can be revisited only if E1/E2 of C leave a
residual that the dimensional-constants channel would explain - and that would be
a new registration, not a patch.)

---

## 4. Instrumentation and metric (fixed before any row is produced)

Harness: `udwm/scripts/probe_gradient_interference.py`. It reuses the exact
DelayedBimodal / DMC protocol and per-seed matched-teacher pairing used by every
other probe in this repo (`prepare_matched_teacher`, exact teacher checksum
pairing), and is **measurement-only**:

- The trainer's own update is unchanged. At each probe checkpoint the harness
  runs an *additional* forward/backward on `--probe-batches` freshly sampled
  real-replay minibatches and never calls an optimizer step.
- Live terms are exposed through a module-level `GRAD_PROBE` object in
  `udwm/models/consistency.py` (the same flag-guarded pattern as the existing
  `CRN_PROBE`); it is reset to `None` immediately after each measurement, so the
  training path is bit-identical when it is unset.
- Effective term weights are read from the arm's own config (the same
  multipliers training applies), so the measured gradients are the ones that
  actually compete.

**Metric (per probe checkpoint, per sampled minibatch).** With
`g_point = d(member)/d(theta_student)` and
`g_unc = d(sum_k w_k * unc_k)/d(theta_student)` (uncertainty terms of the arm,
weighted by the arm's own config multipliers):

    cos           = <g_point, g_unc> / (||g_point|| ||g_unc|| + 1e-12)
    interference  = -cos          (positive => conflict; negative => alignment)

Per seed, the reported interference is the mean over checkpoint steps **after
dropping the first 20% of checkpoints** (registered warmup rule). The critic's
`distill_value_warmup_updates` gate also makes early checkpoints structurally
non-applicable; those are recorded as `applicable: false`, never as zeros.

Arms: `identified_eq` (test), `hybrid` / `lagged_hybrid` (M=1 placebo family -
single-latent decision terms, never pair M>=2 draws), `ordinary` (reference,
not applicable). `lagged_identified_eq` is carried as a secondary read.

Output: one canonical JSON; `rows` = one record per (seed, variant) carrying the
`gi_checkpoints` trajectory plus the folded final-eval metrics
(`u_rank_corr`, `w_rmse`, `next_state_mse`, `return_mean`, ...). Per-(seed,
variant) partials are written atomically and merged additively under a lock -
the same race-safe pattern as `probe_crn_bias.py` (the 2026-09-09 shared-`--out`
data-loss race fix).

---

## 5. Registered endpoints

**E1 - environment contrast (primary).** Mean within-seed interference for
`identified_eq`, DMC/hopper-hop minus DelayedBimodal, two-sample bootstrap 95%
CI (10^5 draws, seed 0).

- *Prediction:* **DMC > DelayedBimodal** (more complex task -> more conflict).
- *Bar:* CI excludes zero **and** the delta is positive.
- *Placebo (P1):* the same contrast computed on `hybrid`. The M=1 family never
  pairs M>=2 draws, so it must show **no consistent env contrast**.

**E2 - within-environment dose-response (primary).** Across seeds within one
environment, the correlation between a seed's mean interference
(`identified_eq`) and that seed's u-rank gap
`u_rank(ordinary) - u_rank(identified_eq)`. Pearson primary (bootstrap CI),
Spearman reported as robustness.

- *Prediction:* **positive** (more interference -> larger eq deficit).
- *Bar:* CI excludes zero and r > 0.
- *Placebo (P2):* the same regression on `hybrid` must be null/near-zero.

**E3 - capacity axis (secondary, explicitly lower confidence; the G8 knob).**
Repeat the within-environment measurement at reduced student capacity
(`student_hidden_dims` `[32,32]` -> `[16,16]`) at fixed term weights.

- *Prediction:* interference **increases** as capacity shrinks (G8's stated
  falsifiable prediction is that collapse depth increases as capacity shrinks).
- *Bar:* same sign and CI rule as E2.
- *Why lower confidence:* G8's prediction is about collapse depth; the
  gradient-conflict version of it is an extrapolation, and the axis is a
  capacity knob rather than a task-complexity knob. E3 is reported as a
  secondary read and cannot on its own make C "confirmed".

**Falsification condition (stated now).** If E1's CI includes zero or the sign
is wrong, **or** E2 is uncorrelated, C is **refuted** and is reported exactly
that way - the same clean standard applied to Hypothesis A. A refuted C is a
result, not a failure to be softened.

---

## 6. Budget, seeds, and why

The probe reuses the already-established DMC protocol rather than re-deriving
one: `configs/dmc_hopper_probe.yaml` (hopper-hop, 1,000-step episodes, 15,000
env steps = 15 episodes, 5-member ensemble, `student_hidden_dims [32,32]`,
`model_batch_size 128`, `model_train_freq 100`, `eval_freq 3000`), exactly as
registered in
`research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md` (incl. Amendment 2) and used
by the 30-seed adjudication. DelayedBimodal reuses
`configs/delayed_bimodal_distill.yaml` (1,800-step registered protocol).

- **E1/E2 primary:** DMC n = 10 seeds, DelayedBimodal n = 10 seeds, probe-every
  300 steps, probe-batches 3. This matches the DMC 15k diagnostic budget that
  the 2026-09-08 verdict established is the minimum at which DMC arms separate
  (3.6k cannot adjudicate; 15k can).
- **Escalation to n = 30** (DMC only) is permitted **only** if E1 or E2 is
  directionally positive at n = 10 with a CI that excludes zero; the n = 30
  rerun then adjudicates. This is the same staged-probe-then-adjudicate
  discipline used for the DMC budget probe.
- **No GPU use is required for DelayedBimodal**; the DMC side needs a GPU
  window. This study must not displace the MACURA n=30 completion or the
  Walker2d budget decision, both of which are further along and more
  load-bearing.

---

## 7. Minimum detectable effect (item 5a - registered before the run)

Computed from the MDE machinery already in `udwm/scripts/effect_size_report.py`
(two-sided alpha 0.05, power 0.80).

**E1 (two-sample contrast, per-env n):** MDE = 3.962 * sigma / sqrt(n), where
sigma is the cross-seed SD of the interference endpoint.

| n per env | MDE (in sigma of the endpoint) |
|---|---|
| 10 | 1.25 |
| 20 | 0.89 |
| 30 | 0.72 |

**E2 (correlation, n seeds):** Fisher-z MDE, |r| detectable at 80% power.

| n seeds | MDE (|r|) |
|---|---|
| 10 | 0.79 |
| 20 | 0.59 |
| 30 | 0.49 |

Reference scale for the *effect the study is chasing*: the DMC u-rank contrast
`eq - ordinary` is -0.246 with cross-seed SD ~0.09 (from the 30-seed CIs). So a
n=10 E2 can only detect a strong dose-response (|r| >= 0.79). **This is
registered as a known limitation**: an n=10 null on E2 is *not* evidence that
the effect is absent, only that it is below the n=10 MDE. Escalation to n=30
(E2 MDE 0.49) is the pre-committed remedy. *(This is the same underpowered-n
trap that the CRN-bias E1 read fell into after the fact; here it is fixed before
the run.)*

---

## 8. Cheap-first-check feasibility (checked before committing GPU time)

- **Retrospective path covers DelayedBimodal only.** Final-checkpoint weights for
  six arms survive locally under
  `checkpoints/hf_diagnostic_delayedbimodal/` (ordinary, hybrid,
  lagged_hybrid, identified_eq, lagged_identified_eq, identified_hybrid), so
  Gate 1 and a single-checkpoint measurement are free (done: section 1).
- **No DMC checkpoints survive locally, and no per-step gradient logs exist for
  either environment** - the S1/S2/S3 rows are JSON metrics only. The
  registered E1/E2 therefore require an instrumented run; there is no
  retrospective shortcut for the cross-environment contrast.
- **Wiring smoke (NOT adjudicable, artifact removed).** A 600-step DelayedBimodal
  run (seed 0, probe-every 200, arms ordinary/identified_eq/hybrid) exercised the
  full path: teacher pairing exact (`exact_teacher_match: true`, checksum gap
  0.0), `ordinary` correctly not applicable, and two checkpoints each for
  `identified_eq` (cos +0.059 -> +0.171; interference -0.059 -> -0.171) and
  `hybrid` (cos -0.167 -> -0.061; interference +0.167 -> +0.061). **These numbers
  are wiring evidence only**: one seed, 600 steps, far below the registered 15k
  operating point, and the eq sign is opposite to the E1 prediction - which is
  exactly why a smoke must not be read as a result. The artifact file was
  deleted rather than left in `runs/`.
- Cost estimate: the measurement adds one forward/backward per probe checkpoint
  (50 checkpoints at probe-every 300 over 15k) on top of the normal run, so
  budget it at roughly the DMC 15k cost plus a small margin.

---

## 9. Sequencing and non-displacement

1. MACURA n=30 completion (resume-safe; seeds 0-3 already cached) - **first**,
   it strengthens the current DMC story and must not queue behind Walker2d.
2. Walker2d budget decision (needs `identified_eq` finished plus the budget
   call).
3. This probe's DMC side, in the next free GPU window - it is a mechanism read,
   not a payoff read, so it never blocks the above.
4. Only if C is confirmed (E1 or E2 bar met) does the correction idea from the
   plan's item 4 become buildable.

---

## 10. Explicit non-claims

- This registration does **not** claim C is true; it fixes how it will be
  measured and what would refute it.
- The Gate-1 sharing result is a statement about network topology, not about
  interference magnitude. Sharing makes interference *possible*; E1/E2 measure
  whether it is *real and complexity-scaled*.
- No DelayedBimodal or DMC transfer bar is re-opened by this document.
- Nothing here changes the 2026-09-08 DMC verdict.
