# CRN-bias probe on DMC at 15k, n=10: diagnostic readout (2026-09-09)

**Registration:** `research/CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md`
(Hypothesis A; section 3 cell map, section 5 endpoints E1-E5, section 6 power
commitments, section 8 decision tree; **Addendum 1** pre-committed 2026-09-09
*before* any DMC n=10 row was summarized: the step-residualized section-1c
read (A1.2) and the E1 power bounds (A1.3)). **Data:**
`runs/crn_bias_probe_15k_n10_gpu.json` (host), local mirror
`runs_local/runs/crn_bias_probe_15k_n10_gpu.json`; DelayedBimodal companion
rows `runs/crn_bias_probe_db_n10.json` (descriptive only, section 10).
**Host:** company VM (RTX 6000 Ada, polite workers, co-resident vLLM load
respected). **Status: DIAGNOSTIC by design** - the registration fixes n=10 as
underpowered for the E2 delta and informative for E1 only at |rho| >= ~0.79
(A1.3); this readout never re-adjudicates the DMC 30-seed verdict and never
mixes rows across files/budgets.

## Why this probe existed, restated

The DMC 30-seed verdict left one anomaly standing: on the same drifting value
map, lagging the critic **helps** the M=1 hybrid family
(`lagged_hybrid - hybrid` +0.106 u_rank, CI [+0.059, +0.153], 22/30) and
**hurts** the M>=2 equal-weight identified family
(`lagged_identified_eq - identified_eq` -0.073, CI [-0.111, -0.036], 7/30).
Identifiability, EMA-starvation, and nonstationarity alone each predict both
families should respond the same way to lagging (lagging is a property of the
critic, not of the loss). Hypothesis A proposed the missing mechanism: the
M>=2 loss evaluates *paired* latent draws against a (lagged) value reference -
a common-random-number comparison whose variance-reduction guarantee is void
under a drifting critic (Glasserman & Yao 1992; bias template from Wu et al.,
Management Science). If true, the paired statistic should accrue a bias term
that scales with the critic's drift rate, and the single-draw hybrid statistic
should show no such scaling because it never pairs draws.

This run measures that claim on DMC at the 15k budget, seeds 0-9, across the
registered {live, lagged} x {norm, no-norm} 2x2 plus the hybrid control family
and the EMA collapse sanity arm.

## Protocol and data integrity

| item | value |
|---|---|
| config | `configs/dmc_hopper_probe.yaml` (hopper-hop, 15,000 steps, eval_freq 3000) |
| cells | A `identified_eq` (live, no norm) - B `identified_eq_norm` (live, norm) - C `lagged_identified_eq` (lagged, norm) - D `lagged_identified_eq_nonorm` (lagged, no norm); + `hybrid`, `lagged_hybrid`, `identified_hybrid` (EMA) |
| seeds | 0-9 (same seeds as the DMC sanity and 30-seed files) |
| probe | 1,024 frozen probe states, M=2 paired latents, M3 re-evaluation pass at each checkpoint |
| driver | `udwm/scripts/probe_crn_bias.py` (per seed x variant rows, merged) |
| rows | 70 = 10 seeds x 7 arms; teacher_pairing 10/10 `exact_teacher_match: true`, `max_teacher_checksum_gap: 0.0` |
| checkpoint records | 6 per row: steps 3k/6k/9k/12k/15k plus the M3 re-evaluation record at 15k (the final-step pass) |

Merge history (context for the record): the first attempt at this run hit the
2026-09-09 shared-output merge race that silently deleted seeds 5-9 (same
failure class as the earlier `--set` precedence bug - last-writer-wins on a
shared `--out`). The probe runner's merge was permanently fixed to an additive,
lock-serialized union of disjoint per-worker output files, the missing seeds
were re-run into disjoint resume files (`..._resume_a.json`,
`..._resume_b.json`), and the canonical 70-row file was unioned from the
partials on 2026-09-09 evening IST. Row count is complete (seeds x arms
asserted after merge) and teacher pairing is exact for all 10 seeds.

Instrumentation sanity (visible in the data): the measured drift/B_hat terms
are exactly zero for the live-critic cells (reference = live critic, by
construction, as registered) and nonzero for the lagged cells - including
`lagged_hybrid`, an M=1 arm - so the instrument does separate the critic-gap
axis from the paired-draw-count axis, which is precisely the separation
Hypothesis A needs.

## Per-arm readout (final checkpoint, n = 10)

| arm (cell) | u_rank mean | u_rank median | n >= 0.70 | B_hat med | drift med | w_rmse med |
|---|---:|---:|---:|---:|---:|---:|
| A `identified_eq` (live, no norm) | 0.713 | 0.764 | 7/10 | 0 (by constr.) | 0 | 8.12e-06 |
| B `identified_eq_norm` (live, norm) | 0.620 | 0.598 | 2/10 | 0 (by constr.) | 0 | 5.00e-05 |
| C `lagged_identified_eq` (lagged, norm) | 0.691 | 0.705 | 6/10 | 0.00920 | 0.00871 | 2.87e-04 |
| D `lagged_identified_eq_nonorm` (lagged, no norm) | 0.768 | 0.751 | 8/10 | 0.00611 | 0.00586 | 2.23e-05 |
| `hybrid` (M=1) | 0.752 | 0.759 | 8/10 | 0 (by constr.) | 0 | 4.34e-06 |
| `lagged_hybrid` (M=1, lagged) | 0.866 | 0.870 | 10/10 | 0.00905 | 0.00980 | 1.36e-05 |
| `identified_hybrid` (EMA) | -0.092 | -0.097 | 0/10 | 0 (by constr.) | 0 | 1.95e-07 |

Probe-subset ordering (u-rank mean): lagged_hybrid (0.866) >
lagged_eq_nonorm (0.768) > hybrid (0.752) > identified_eq (0.713) >
lagged_identified_eq (0.691) > identified_eq_norm (0.620) > identified_hybrid
(-0.092).

## Cross-run consistency with the 30-seed verdict (seeds 0-9)

The probe rows and the 30-seed file share seeds, config base, and budget, so
the same-seed pairs are comparable as a cross-run reproducibility check only
(paired rows across files - never adjudication, house rule). Means on seeds
0-9 at the final checkpoint:

| arm | 30-seed file (seeds 0-9) | probe rows (seeds 0-9) |
|---|---:|---:|
| `identified_eq` | 0.746 (6/10 >= 0.70) | 0.713 (7/10) |
| `hybrid` | 0.741 (7/10) | 0.752 (8/10) |
| `lagged_hybrid` | 0.855 (9/10) | 0.866 (10/10) |
| `lagged_identified_eq` | 0.669 (5/10) | 0.691 (6/10) |
| `identified_hybrid` (EMA) | -0.045 (0/10) | -0.092 (0/10) |

Same relative ordering on every shared arm (lagged_hybrid > hybrid ~= eq >
lagged_eq > EMA); absolute deltas are within seed-level noise (0.01-0.05).
The probe's seven-arm config reproduces the adjudicated file's seeds-0-9
slice.

## E1 (primary, mechanism): Hypothesis A is NOT supported at n=10

Registered read: within each lagged identified cell, Spearman
rho(u_rank_corr(seed), B_hat(final, seed)) across seeds is predicted
**negative**; SUPPORTED only if negative with a bootstrap 95% CI excluding 0
(10^5 draws, rng seed 0); anything else is INCONCLUSIVE (never refuted at
n=10). Live-critic cells carry a registered within-cell null (B_hat ~ 0).

| cell | rho | 95% CI | read at n=10 |
|---|---:|---|---|
| C `lagged_identified_eq` (norm) | +0.018 | [-0.588, +0.770] | null point estimate; below the n=10 MDE (|rho| 0.785, A1.3) - inconclusive |
| D `lagged_identified_eq_nonorm` | +0.661 | [+0.018, +0.927] | CI excludes 0 in the POSITIVE direction - the opposite sign to the prediction; inconclusive as registered, and the only E1 signal that is not nothing |
| A / B live cells | B_hat mean 0.000 | - | registered within-cell null holds (B_hat ~ 0 by construction) |

What the numbers say:

- **The norm cell shows no association.** Final-checkpoint B_hat across the 10
  seeds spans only 0.0050-0.0259 (median 0.0092) while u_rank spans
  0.457-0.886; rho is +0.018 with a wide CI. With an n=10 MDE of 0.785, this
  is uninformative about any effect below ~0.8 - a null point estimate, not a
  null result.
- **The nonorm cell points the wrong way.** rho +0.661, CI excluding 0: seeds
  with a larger measured staleness term at the final checkpoint show *higher*
  u-rank fidelity, not lower - the opposite of the predicted
  drift-induced-bias penalty. It is not an artifact of the one degenerate seed
  (seed 5, final B_hat ~680 vs a cell median of 0.006): dropping seed 5 leaves
  rho at +0.533, and the Spearman (rank) form is scale-free by construction.
- **Magnitude caveat on that signal.** |rho| 0.661 sits below the registered
  n=10 MDE of 0.785, so by A1.3 it is a directional flag, not a measured
  effect. Under the same Fisher-z approximation as A1.3, |rho| = 0.661 needs
  only n ~ 16 for 80% power - so the *direction* question is cheap to
  adjudicate if it is pursued (section 11).

Bottom line for E1: neither lagged cell supports the prediction; the only
informative cell contradicts it in sign. Hypothesis A's primary endpoint is
not confirmed on DMC at n=10.

## E2 (decomposition; descriptive at n=10, adjudicated only at n=30)

Registered prediction: within a fixed normalization cell, lagging hurts the
paired family (D-A negative in the no-norm cell; C-B negative in the norm
cell) *if staleness is the axis*.

| contrast (u_rank) | DMC n=10 | DB n=10 |
|---|---:|---:|
| no-norm cell: D - A (`lagged_eq_nonorm - eq`) | +0.056 [-0.061, +0.197], 6/10 | +0.012 [-0.058, +0.074], 6/10 |
| norm cell: C - B (`lagged_eq - eq_norm`) | +0.071 [+0.011, +0.138], 6/10 | +0.004 [-0.013, +0.019], 6/10 |

What the numbers say:

- **The staleness prediction is not observed on DMC.** With normalization held
  fixed, lagging *helps* the paired family in both cells at n=10 (+0.056,
  +0.071) - the same direction as the hybrid family's lag benefit (+0.106 at
  DMC 30-seed) - and the norm-cell CI excludes 0. On the DB companion rows the
  same contrasts are flat (+0.004/+0.012, CIs include 0). Neither environment
  shows the lag-hurts-the-paired-family signature the CRN mechanism predicts,
  at n=10.
- **The 30-seed conflated reversal is not cleanly reproduced within cells.**
  The adjudicated reversal compared arms that differ in *two* knobs at once
  (lagged + normalization vs live + no normalization). Re-expressed on the
  probe rows, that conflated contrast is -0.022 [-0.139, +0.087], 4/10 at
  n=10 (vs -0.073, 7/30 at n=30): the direction matches but is small and
  descriptive, and it decomposes into a *helpful* lag knob and an
  environment-dependent normalization knob rather than a staleness penalty.
- **Normalization itself behaves differently by environment at n=10.** On DMC
  the live-critic norm cell B (0.620) sits *below* the no-norm cell A (0.713);
  on the DB companion rows the norm cell is the top cell (0.933 vs 0.827).
  Descriptive at n=10 and flagged, not adjudicated - but this is the knob
  whose cross-environment behavior most plausibly explains why the conflated
  contrast reversed between environments.

E2 is descriptive at n=10 by registration; the pre-committed reading is that
no staleness-attribution branch of the decision tree is supported by it.

## E3 and E5 (controls)

- **E3 (hybrid-family lag benefit vs B_hat):** rho = +0.309 at n=10 on DMC
  (registered read: ~0, no sign predicted). No bar is attached; an n=10 rank
  correlation has wide spread; nothing is claimed from it either way. DB
  companion: -0.115.
- **E5 (EMA collapse sanity):** `identified_hybrid` u_rank mean -0.092, 0/10
  >= 0.70 - the EMA collapse reproduces under the probe config, consistent
  with the DMC 30-seed control (-0.076, 0/30). The mechanism-transfer control
  story is intact: every equal-weight cell (A-D) clears the collapse arm by
  0.71-0.86 u_rank.

## Section-1c regression: raw slopes are a step-trend artifact; the residualized read is positive everywhere, including the placebo

Section 1c regresses per-seed estimation error e_u on the logged drift across
the checkpoint series. Two numbers matter: the raw pooled per-seed OLS slope
and - since Addendum 1 (A1.2) - the pooled partial slope of e_u on drift
*after removing each series' shared monotone trend on training step*. The raw
read is confounded: within every seed, drift falls with training step
(Spearman ~ -0.8 to -0.9 on the DB rows, A1.1) while e_u is flat-to-rising,
so the raw slope inherits the time trend and can flip sign without any
drift-driven bias. The DB raw slopes are negative in every lagged arm -
including the placebo - purely from this artifact.

| arm | raw slope DMC | partial slope DMC (A1.2 primary) | raw slope DB | partial slope DB |
|---|---:|---:|---:|---:|
| C `lagged_eq` (norm) | +4.910 [+2.995, +7.129] | +5.208 [+3.058, +7.753] (10/10 pos) | -1.335 [-2.163, -0.532] | +1.327 [-1.189, +3.943] (7/10 pos) |
| D `lagged_eq` (nonorm) | +5696 [+2.70, +17083] (skew) | +237.2, med 3.61, CI [+2.76, +705] (10/10 pos, skew) | -3.533 [-4.612, -2.600] | +1.798 [-0.016, +3.528] (9/10 pos) |
| `lagged_hybrid` (placebo) | +2.965 [+2.290, +3.796] | +2.961 [+2.330, +3.743] (10/10 pos) | -0.819 [-1.419, -0.184] | +1.394 [-1.622, +4.230] (6/10 pos) |
| `hybrid` (placebo) | unregressable (0/10) | unregressable (drift ~ 0 by construction) | unregressable | unregressable |

What the numbers say:

- **On DMC, e_u rises with drift after the step trend is removed - in every
  lagged arm, paired or not.** The norm cell (+5.208, CI excluding 0) and the
  *placebo* `lagged_hybrid` (+2.961, CI excluding 0) both show clean, tight
  positive partial slopes; the nonorm cell is positive but its pooled mean
  (+237) is driven by one skewed seed (per-seed median 3.61, all 10 seeds
  positive). Drift-sensitivity of estimation error is therefore real on DMC -
  but it is **not pairing-specific**: the M=1 placebo is about as sensitive as
  the norm paired cell. This is the same red flag the DB rows raised (weakly
  positive residualized means in all three arms there, +0.10/+0.15/+0.17,
  A1.1), and it undermines the clean pairing-causes-drift-sensitivity
  discriminator Hypothesis A required.
- **The raw slopes are not usable across environments.** DB raw slopes are
  negative and DMC raw slopes are positive, purely because the two maps place
  the step trend differently relative to e_u; the A1.2 residualized read is
  the only one with a fixed meaning, and it is weakly positive-but-inconclusive
  on DB (all CIs include 0) and cleanly positive on DMC.
- **Scale caution:** drift units are environment-specific (final-checkpoint
  drift medians ~0.006-0.010 on DMC vs ~0.95-1.46 on DB); slope magnitudes are
  comparable within an environment only. Keep the nonorm pooled slope to
  median/rank language.

## Cross-environment read (DMC vs DB, same probe, n=10 each)

DB companion rows (descriptive only - no endpoint is adjudicated on them at
n=10, but they are the same harness and the same 7 arms):

| arm | DMC u_mean (>= 0.70) | DB u_mean (>= 0.70) | DB B_hat med |
|---|---:|---:|---:|
| A `identified_eq` | 0.713 (7/10) | 0.827 (9/10) | 0 |
| B `identified_eq_norm` | 0.620 (2/10) | 0.933 (10/10) | 0 |
| C `lagged_identified_eq` | 0.691 (6/10) | 0.937 (10/10) | 1.19 |
| D `lagged_identified_eq_nonorm` | 0.768 (8/10) | 0.838 (9/10) | 1.42 |
| `hybrid` | 0.752 (8/10) | 0.623 (1/10) | 0 |
| `lagged_hybrid` | 0.866 (10/10) | 0.944 (10/10) | 0.965 |
| `identified_hybrid` (EMA) | -0.092 (0/10) | 0.098 (0/10) | 0 |

Cross-environment tension, measured: on DB the lagged cells sit at the top of
the table (C 0.937, lagged_hybrid 0.944 - consistent with the DB policy
adjudications where the lagged family tops the table); on DMC the lagged norm
cell (C, 0.691) sits *below* its unlagged siblings and the plain `hybrid`
outranks it. The reversal the 30-seed verdict documented is visible in the
probe rows too - but the drift magnitudes that would power Hypothesis A are
two orders of magnitude apart (final B_hat ~0.006-0.026 on DMC vs ~0.7-2.7 on
DB), and neither environment's E1 is supported at n=10. The DB rows also carry
the largest single measured drift in either file (seed 1 nonorm, B_hat 2.74),
and that seed has the *second-lowest* u_rank in its cell (0.710) - the one
per-seed point that faces the predicted direction, inside an overall null.

## Verdict: what this run does and does not support

**Hypothesis A does not confirm on DMC at n=10.** The three registered
evidences line up against it:

1. E1 is inconclusive in both lagged cells - null below MDE in the norm cell,
   and significant with the *wrong sign* in the nonorm cell (+0.661
   [+0.018, +0.927]; +0.533 with the degenerate seed dropped).
2. E2's staleness signature (lag hurts the paired family within a cell) is not
   observed: positive on DMC (+0.056/+0.071), flat on DB (+0.004/+0.012) -
   descriptive at n=10, but there is no drift toward the predicted direction.
3. The 1c drift-sensitivity is present in the lagged-hybrid placebo on DMC
   (+2.961, CI excluding 0) about as strongly as in the paired norm cell - so
   the mechanism's clean discriminator (M=1 arms should be insensitive) fails
   on DMC, matching the DB red flag.

Decision-tree mapping (registration section 8): no branch leads to the confirm
arm. E1 SUPPORTED + E4 CLEAN is not met (E1 is not supported; E4 is gated and
was never built). The two E1-INCONCLUSIVE branches condition on E2 outcomes
that are also not met (E2 is neither lag-vanishes-in-one-cell nor flat in both
cells on DMC at n=10). Net consequence, stated plainly:

- **The paper may not assert** the CRN/staleness mechanism as the explanation
  of the DMC lag-axis asymmetry on the current evidence. The anomaly stands as
  an unresolved, benchmark-specific pattern whose knob decomposition (lag vs
  normalization) is measured here only descriptively.
- **The mechanism-transfer story is untouched.** EMA collapse reproduces
  (E5: -0.092, 0/10 >= 0.70); every equal-weight cell clears the collapse arm;
  the DMC 30-seed verdict and its controls are unchanged by this probe (the
  registration never re-adjudicates them).
- **No `eq_crn` arm is registered and no Wu-style correction (E4) is built.**
  Both are pre-committed to stay unbuilt until the mechanism confirms.
- **What remains genuinely open:** whether the paired-family lag asymmetry is
  (a) a normalization-knob effect that the 30-seed conflated contrast
  misattributed to lagging, (b) environment-specific ordering noise, or
  (c) CRN staleness at a magnitude too small for this probe to separate -
  final-checkpoint B_hat on DMC is ~0.006-0.026 (vs ~0.7-2.7 on DB), and the
  drift that would power the mechanism mostly settles early in training.

## Next steps (all gated, none launched here)

- E1 direction adjudication, if pursued: the nonorm cell's positive rho (0.66)
  is cheap to adjudicate (n ~ 16 for 80% power, Fisher-z); confirming a
  *negative* association at the DB-observed magnitudes needs n ~ 38-64 (A1.3).
  Any such run needs its own registration.
- Section-1c reads are already on the pre-committed A1.2 residualized form; no
  re-read is needed.
- E4 (correction) and `eq_crn` remain unbuilt unless a confirmatory E1
  materializes at a registered n.
- The MACURA head-to-head and Walker2d staging are separate queued GPU work
  (Kaggle 2xT4 track) and are unaffected by this readout.

## Flags and caveats carried into the record

- n=10 is diagnostic by design (registration section 6); E2 is adjudicated
  only at n=30 and E1 is informative only for |rho| >= ~0.79 (A1.3). Every
  not-supported statement above is an n=10 statement, not a measured null.
- E1 uses the *final-checkpoint* B_hat, where drift on DMC has mostly settled
  (medians ~0.006-0.010) and the across-seed spread is narrow; the
  within-seed over-time signal (1c) and the across-seed final signal (E1) are
  different aggregations and should not be read as contradicting each other.
- DMC nonorm seed 5 shows a degenerate final B_hat (~680 vs cell median 0.006;
  drift ~588) - a numeric outlier, flagged and excluded from all magnitude
  claims; E1 rho is robust to dropping it (+0.533).
- The nonorm pooled partial slope (+237) is skew-dominated (per-seed median
  3.61); keep that cell to median/rank language.
- w_rmse remains rank-only (the collapsed EMA arm's w-scale instability, known
  from the 30-seed file, is present here too: EMA w_rmse median 1.95e-07); no
  w-scale claims are made.
- DB companion rows (`runs/crn_bias_probe_db_n10.json`) are descriptive: they
  carried the A1.1 step-trend discovery and provide the cross-environment read
  above, but no endpoint is adjudicated on them at n=10.
- Merge history is documented above; the 2026-09-09 data-loss race is
  permanently fixed (additive locked union + row-count assertion) and does not
  affect this file's completeness (70/70 rows, exact teacher pairing).

## Files / reproducibility

- Registration: `research/CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md`
  (Addendum 1 pre-committed before this readout).
- Verdict this probe refines: `research/RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md`.
- Data: `runs/crn_bias_probe_15k_n10_gpu.json` (70 rows; local mirror
  `runs_local/runs/crn_bias_probe_15k_n10_gpu.json`), partials
  `runs/crn_bias_probe_15k_n10_gpu_resume_{a,b}.json`, DB companion
  `runs/crn_bias_probe_db_n10.json`.
- Harness: `udwm/scripts/probe_crn_bias.py`. Adjudication:
  `python -m udwm.scripts.summarize_crn_bias --data runs/crn_bias_probe_15k_n10_gpu.json`
  (10^5-draw percentile bootstrap, rng seed 0; A1.2 residualized partial-slope
  read included; output reproduced above).
