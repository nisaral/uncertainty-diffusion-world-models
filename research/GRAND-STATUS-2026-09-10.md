# Grand status: every result so far, exact ledger (2026-09-10)

One document that says where the project stands after every run to date,
including the MACURA / Walker2d rows downloaded from the stopped Kaggle
session on 2026-09-10. Status tags are used throughout:

| tag | meaning |
|---|---|
| ADJUDICATED | registered bars read at the registered n; verdict stands |
| DIAGNOSTIC | reads that inform but never adjudicate (n below the registered bar, or probe config) |
| PARTIAL | registered study interrupted; only the listed seeds/arms exist |
| STAGED | preregistration + harness ready; no rows or no reported read yet |
| DEFERRED / GATED | registered deferral; runs only if its condition fires |
| SUPERSEDED | replaced by a later corrected run (kept for provenance) |

Data-locality note: every file analyzed here is the local mirror under
`runs_local/` (the VM repo is untouched; nothing on the VM was modified).
Row-level final endpoints (`u_rank_corr`, `w_rmse`, ...) are the adjudicated
fields the repo summarizers read; checkpoint series come from `eval_history`
and are approximate (see section 7).

## 1. New GPU data this pass (Kaggle 2xT4, stopped mid-run)

Downloaded files and what each contains:

| file | content | completeness |
|---|---|---|
| `runs_local/dmc_macura_gate_pilot.json` | DMC/hopper gate pilot, ordinary, seeds 0-1, 15k | complete |
| `runs_local/dmc_macura_probe_2seed_15k_gpu.json` | DMC 3-arm probe (ordinary, identified_eq, macura_gate), seeds 0-1, 15k | complete (n=2) |
| `runs_local/dmc_macura_30seed_15k_gpu_seed0..3.partial.json` | DMC 3-arm head-to-head, seeds 0-3, 15k | complete for seeds 0-3 (n=4) |
| `runs_local/dmc_macura_30seed_15k_gpu_seed4..5.partial.json` | DMC head-to-head, seeds 4-5 | ordinary only; eq + macura_gate not finished |
| `runs_local/walker_gate_pilot.json` | Walker2d gate pilot, ordinary, seeds 0-1, 15k | complete |
| `runs_local/walker2d_budget_probe_gpu_seed0..1.partial.json` | Walker2d budget probe, seeds 0-1, 15k | ordinary + lagged_hybrid only; identified_eq not finished |

Integrity: all files parse; every present row reports exact teacher pairing
(checksum gap 0.0, `exact_teacher_match: true`). Nothing below is
adjudicative - the MACURA n=4 and Walker n=2 reads are PARTIAL/DIAGNOSTIC by
definition and no registered bar is evaluated on them.

## 2. MACURA field baseline on DMC/hopper-hop (PARTIAL, n=4 of 30)

Registration: `research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md`
(MACURA = ensemble disagreement gating, Eq-4 `u_KL` implementation, reuse of
the existing teacher/student ensemble, deliberate Algorithm-2 deviations
documented). MACURA produces no w/g split, so u_rank is a distillation sanity
read only (macura_gate keeps ordinary distillation, so parity with ordinary
is expected by construction); the registered fair endpoints are rollout
gating quality and downstream return.

### 2a. Per-arm final eval, 15k (row-level finals; n = seeds with the arm)

| arm | u_rank mean | u_rank med | n >= 0.70 | w_rmse med | ns_mse mean | ret mean | ret med |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary (seeds 0-5) | 0.953 | 0.950 | 6/6 | 4.58e-05 | 1.901 | 0.0507 | 0.0319 |
| identified_eq (seeds 0-3) | 0.746 | 0.731 | 4/4 | 1.31e-05 | 1.740 | 0.0585 | 0.0002 |
| macura_gate (seeds 0-3) | 0.962 | 0.973 | 4/4 | 1.70e-04 | 1.879 | 0.0003 | 0.0003 |

### 2b. Paired contrasts (u_rank_corr; bootstrap 95% CI, 10^5 draws, rng 0)

| contrast | mean | 95% CI | wins |
|---|---:|---:|---:|
| identified_eq - ordinary | -0.219 | [-0.252, -0.189] | 0/4 |
| macura_gate - ordinary | -0.003 | [-0.022, +0.016] | 2/4 |
| macura_gate - identified_eq | +0.216 | [+0.173, +0.259] | 4/4 |

Return deltas (DESCRIPTIVE ONLY at 15k, deferred by the DMC floor rule):
macura_gate - ordinary final_return -0.045 [-0.111, +0.000], 1/4;
identified_eq - ordinary +0.013 [-0.111, +0.151], 2/4.

### 2c. The n=2 probe (earlier run, same seeds) cross-checks the n=4 read

Per-arm (final): ordinary 0.918 (2/2 >= 0.70), identified_eq 0.698 (1/2),
macura_gate 0.939 (2/2). Contrasts: eq - ordinary -0.220 [-0.317, -0.123]
0/2; macura_gate - eq +0.241 [+0.157, +0.326] 2/2; macura_gate - ordinary
u_rank +0.021 [+0.009, +0.034] 2/2. Same ordering as n=4; the two runs agree
to ~0.02-0.04 per arm (separate trainings on the same seeds/config).

### 2d. How it climbed, and whether MACURA actually gates

Checkpoint u-rank means (eval-history records at 3k/6k/9k/12k/15k, n=4):

| arm | 3k | 6k | 9k | 12k | 15k |
|---|---:|---:|---:|---:|---:|
| ordinary | 0.536 | 0.614 | 0.832 | 0.905 | 0.969 |
| macura_gate | 0.672 | 0.681 | 0.819 | 0.934 | 0.964 |
| identified_eq | 0.474 | 0.402 | 0.584 | 0.660 | 0.690 |

Gating behavior at the final record (mean over seeds 0-3): macura_gate stops
imagined rollouts at stopped_frac 0.158 vs ordinary 0.096 and identified_eq
0.111 (~1.6x ordinary), with imagine mean-weight 0.846 vs ~0.61 and its own
disagreement scale at imagine_mean_sqrt_u ~0.005 vs ~0.36 for the two
variance-based arms. So the MACURA gate is active and different from ordinary
self-gating, but rollout NFE / length_mean are identical across arms
(student_nfe 1, teacher_nfe 4, length_mean 1000) - at this one-step
imagination budget the gate changes which imagined states are kept, not the
sample count.

### 2e. Read (n=4, diagnostic)

1. **MACURA-gated ordinary distillation sits at parity with ordinary on
   u-rank** (-0.003, CI includes 0, 2/4) and both sit well above identified_eq
   (+0.216, 4/4) - reproducing the DMC 30-seed subset position of eq (the
   equal-weight identified arm trails ordinary by ~0.2 u-rank on hopper at
   15k). The field baseline does not change the distillation ordering.
2. **The registered payoff endpoints cannot be read at 15k on hopper-hop:**
   returns are floor-bound for every arm (macura_gate ret median 0.0003;
   ordinary ret mean 0.051 is driven by two non-floor seeds; the paired
   macura_gate - ordinary return delta is -0.045 with the CI upper edge at
   0.000, 1/4). This is the same floor the DMC 30-seed verdict documented -
   the MACURA-vs-ordinary payoff question needs either the full n=30 at a
   budget where returns separate, or the return-extension environment. No
   claim about MACURA's payoff is supported by these rows.
3. **Do not adjudicate at n=4.** Seeds 0-3 are complete; seeds 4-5 have
   ordinary only (0.925 / 0.934, both >= 0.70). Finishing the study needs
   eq + macura_gate on seeds 4-5 plus seeds 6-29 (resume-safe; see section 6).

## 3. Walker2d payoff staging (PARTIAL: gate pilot complete, budget probe n=2 of 3 arms)

Registration: `research/WALKER2D-PAYOFF-PREREGISTRATION-2026-09-09.md`.
Hypothesis-B motivation: Walker2d does not share hopper-hop's
zero-on-fall-without-reset structure, so the return-payoff question gets a
fairer test. Staged discipline (never skipped): gate check, then budget
probe, then sanity, then adjudication. The rows below cover gate + the start
of the budget probe only.

### 3a. Gate regime (ordinary, seeds 0-1, 15k)

| seed | teacher g* | teacher w* | g*/w* |
|---|---:|---:|---:|
| 0 | 14.641 | 0.00181 | 8111.28 |
| 1 | 0.072 | 0.00002 | 3609.12 |

Median g*/w* = 5,860 -> aleatoric-dominated regime, same branch as DMC
(Hopper medians measured 10k-42k across pilots). Two structural notes: the
Walker aleatoric w* is NOT identically ~1e-5 as on hopper (seed 0 w* 0.0018,
seed 1 2e-5), so the map is not as aleatoric-collapsed as hopper-hop; and the
teacher ensemble is a separate Walker-trained ensemble (initial checksum
~6064 vs ~5530 on hopper), so this is not the hopper teacher re-used.

### 3b. Budget probe, partial n=2 (ordinary + lagged_hybrid; identified_eq unfinished)

| arm | u_rank mean | u_rank med | n >= 0.70 | w_rmse med | ns_mse mean | ret mean | ret med |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 0.462 | 0.462 | 0/2 | 0.0797 | 30.92 | 32.65 | 32.65 |
| lagged_hybrid | 0.247 | 0.247 | 0/2 | 0.0167 | 31.76 | 27.55 | 27.55 |

Seed-level: ordinary u_rank 0.520 / 0.405 with final_return 37.97 / 27.34;
lagged_hybrid 0.191 / 0.303 with 31.03 / 24.06.

Paired contrasts: lagged_hybrid - ordinary u_rank_corr -0.215 [-0.329,
-0.102], 0/2; lagged_hybrid - ordinary final_return -5.11 [-6.93, -3.28],
0/2.

Checkpoint u-rank means (3k/6k/9k/12k/15k, n=2): ordinary 0.014 / 0.342 /
0.247 / 0.470 / 0.458; lagged_hybrid 0.112 / 0.222 / 0.313 / 0.399 / 0.230.

### 3c. Read (n=2, staging only - no bars, no adjudication)

1. **The return axis is usable on Walker2d - the Hypothesis-B premise holds
   in this pilot.** Returns are not floor-bound (ordinary 32.65, lagged_hybrid
   27.55, both seeds positive and non-degenerate). This is the first measured
   confirmation that Walker2d can carry a return-payoff question that
   hopper-hop at 15k cannot.
2. **15k is NOT an adequate operating point for Walker2d distillation
   fidelity.** Both arms sit far below the 0.70 bar (0/2) and below hopper's
   ~0.95 at the same budget; the checkpoint series is non-monotone and ends
   down (ordinary 0.470 -> 0.458 at 12k->15k; lagged_hybrid 0.399 -> 0.230),
   i.e. the arms have not converged into a stable ordering. Walker2d needs a
   higher-budget operating point (registered staging decision, e.g. a 30k
   sanity probe) before any arm comparison can adjudicate.
3. **identified_eq is unfinished for both seeds** (only ordinary and
   lagged_hybrid rows exist), so even the n=2 budget-probe arm set is
   incomplete. Nothing here supports a claim about the identified family on
   Walker2d.

## 4. Grand ledger - every study to date

### 4a. DelayedBimodal (toy env) line

| Study | Status | Verdict / read (one line) | Doc |
|---|---|---|---|
| Identifiability loss + ground-truth (w*, g*) recovery | ADJUDICATED | Estimator unbiased; M=1 walks the degenerate direction; equal-weighting hole + partial reweighting quantified | RESULTS-IDENTIFIABILITY-2026-08-29, RESULTS-GROUND-TRUTH-W-G-2026-09-01 |
| Large fixed-map stress N=20 / N=30 / N=50 | ADJUDICATED | Hybrid magnitude near-perfect; w-rank gain confirmed at 30 (22/30) and 50 (38/50); identified-vs-hybrid ranking wash at every n; top-decile recall inconclusive at 50 | RESULTS-STRESS-* (08-21, 09-03, 09-05) |
| Policy 2x2, N=30 (EMA-attributed) | ADJUDICATED | Lagging fixes the u-rank collapse 30/30; the identified rows were the EMA-reweighted variant (attribution corrected 09-05); equal-weight identified then transfers on the live critic (09-05, 0.835, 9/10) | RESULTS-POLICY-2X2-30SEED-2026-09-03, RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05 |
| Normalization control + G9 readjudication | ADJUDICATED | Standardization, not lagging, is the empirical driver on DB: eq_norm 0.948 (10/10) at/above ordinary 0.923; w-scale language dropped (rank-only) | RESULTS-NORMALIZATION-CONTROL-2026-09-07, RESULTS-NORMALIZATION-READJUDICATION-2026-09-07 |
| Combined-fix N=10 | ADJUDICATED | Registered bars fail (D-A, C-B w_rmse null); normalization alone moves u_rank +0.11 (10/10) | RESULTS-COMBINED-FIX-POLICY-2026-09-07 |
| CRN-bias DB companion rows | DIAGNOSTIC | Same harness as DMC probe; drift ~1.2 (100x DMC); E1 inconclusive both cells; step-trend artifact identified (A1.1); descriptive only | RESULTS-CRN-BIAS-DMC-2026-09-09 (cross-env table) |

### 4b. DMC / hopper-hop line (Amendment-2 budget: 15,000 steps)

| Study | Status | Verdict / read (one line) | Doc |
|---|---|---|---|
| Budget probe (3.6k vs 15k) | ADJUDICATED | Budget confound supported; baselines ~0.45-0.57 at 3.6k vs >= 0.70 at 15k; Amendment 2 activated (15k); gate re-measured aleatoric-dominated (median g*/w* 14,292) | RESULTS-DMC-BUDGET-PROBE-2026-09-08 |
| Sanity S1 (n=10) | DIAGNOSTIC | Probe ordering confirmed at n=10; EMA collapse reproduced; superseded by S2 for adjudication | RESULTS-DMC-SANITY-2026-09-07 |
| 30-seed adjudication S2 + gate-off S3 | ADJUDICATED | Mechanism transfers (EMA collapse 0/30 >= 0.70; eq - EMA +0.788 30/30); practical ordering does not (eq bars fail; lagged_eq below eq -0.073 7/30; ordinary tops 0.957 30/30); gating return-neutral at 15k (rule ii) | RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08 |
| CRN-bias probe n=10 | DIAGNOSTIC readout | Hypothesis A not supported: E1 norm null below MDE, nonorm +0.661 wrong sign; placebo (lagged_hybrid) drift-sensitive; EMA collapse reproduces; no eq_crn, E4 unbuilt | RESULTS-CRN-BIAS-DMC-2026-09-09 |
| MACURA head-to-head | PARTIAL (n=4 of 30) | macura_gate u-rank at parity with ordinary (-0.003), above identified_eq (+0.216, 4/4); returns floor-bound at 15k -> payoff question unreadable here; finish n=30 before any bar | GRAND-STATUS section 2 (this doc) |

### 4c. Cross-cutting items

| Item | Status | Note |
|---|---|---|
| Compute-normalized re-analysis | STAGED | Methods note + `reanalyze_compute_normalized.py` ready (zero GPU; re-indexes existing rows at fixed teacher-sample compute); no reported read yet |
| Effect sizes / power (5a) | PARTLY DONE | MDE/power tables landed in CRN Addendum A1.3 (pre-run); `effect_size_report.py` staged, not yet run on the adjudicated files |
| Conditional 30k return extension (DMC) | DEFERRED / GATED | Hypothesis: lagged_eq - ordinary return +0.101 [+0.016, +0.199] (descriptive only); runs only if the payoff question is still worth its compute |
| Hypothesis-A correction (eq_crn / E4) | NOT BUILT | Pre-committed gate: only after a confirmatory E1 at registered n |
| Hypothesis C (gradient interference) | REGISTERED 2026-09-10, gates pre-answered | Gate 1 architecture: `member` and `epistemic_w`/`aleatoric_g` share the trunk - Jaccard(point, uncertainty)=1.00 on the real DelayedBimodal checkpoint and on the Hopper (15/4) topology, so C is coherent. Probe + adjudicator implemented and wiring-validated; no adjudicable row yet (DMC side needs a GPU window). `HYPOTHESIS-C-PREREGISTRATION-2026-09-10` |
| Hypothesis D (M=2 variance vs dimensionality) | DESK CHECK: NOT SUGGESTIVE | The repo's balance-window prediction `sqrt(2/(N(M-1)))` is dimension-free (0.125 at N=128, M=2, identical on DelayedBimodal 5/1 and DMC 15/4), so it cannot produce an environment-specific gap (-0.104 DB vs -0.246 DMC). Not escalated to a run - pre-committed consequence |
| Visualization V1-V4 + repo main page | DONE (2026-09-09) | Encodes the DMC 30-seed verdict + S1 curves; not yet refreshed with CRN/MACURA/Walker numbers (correct: those are diagnostic/partial) |
| GitHub Pages | OPEN | Enable Pages at repo root on main so `visuals/v4-explorer.html` is live at the public URL |
| Manuscript / paper narrative | DRAFT | `research/PAPER-NARRATIVE.md` assembled; open items above decide the CRN chapter and the MACURA/Walker additions |

## 5. What is safe to assert now (paper-facing inventory)

SAFE (adjudicated, cross-environment where claimed):
- The identifiability claim and the EMA-weighting failure mechanism
  (fixed-map N=50; live-critic N=10; DMC 30-seed EMA collapse 0/30).
- The cross-environment sentence from the DMC verdict: the failure modes are
  general (EMA collapse reproduces; equal weights avoid it), but on a
  drifting-map benchmark at 15 episodes the plain self-gated ordinary arm is
  the top u-rank arm and the DB practical ordering does not transfer.
- Normalization (value standardization), not lagging, is the measured driver
  of the DB live-critic gain (G9); w-scale language is rank-only.

NOT YET SAFE (do not put in the manuscript as results):
- The CRN/staleness mechanism as the explanation of the DMC lag-axis
  asymmetry (Hypothesis A not supported at n=10; the lag asymmetry currently
  reads as an unresolved, benchmark-specific pattern with a normalization-knob
  alternative).
- Any MACURA baseline claim (n=4 partial; payoff endpoints floor-bound at
  15k on hopper; needs n=30 and/or a readable-return setting).
- Any Walker2d arm claim (staging only; 15k insufficient; identified_eq
  unfinished).

## 6. Exact next queue (priority order)

1. MACURA - finish the DMC 30-seed head-to-head (Kaggle, resume-safe:
   seeds 0-3 cached; eq + macura_gate needed on seeds 4-5, all three arms on
   6-29; est. ~9-12 h on 2xT4 at the observed ~23 min/seed wall). Then run
   the registered MACURA contrasts (return + rollout-gating quality) and the
   DMC-payoff summarizer with `--no-bars` (registered bars do not apply to
   this arm set).
2. Walker2d - finish the budget probe arm set (identified_eq on seeds 0-1),
   then make the staged budget decision. Pilot evidence says 15k is not the
   operating point: plan a higher-budget sanity (e.g. 30k) before any arm
   comparison, and confirm the return axis stays usable (it is not
   floor-bound at 15k - first positive signal for Hypothesis B).
3. Compute-normalized re-analysis + effect-size report - run locally now
   (zero GPU, scripts staged, methods note already pre-committed). This is
   free and closes the longest-open cross-cutting item.
4. CRN - no further run is recommended unless the normalization-vs-staleness
   question matters for the paper; if it does, register the n=30 2x2 E2
   adjudication (the DMC normalization cells have never been run at n=30 and
   are the live alternative explanation).
5. Visuals / README / manuscript refresh only after items 1-2 adjudicate;
   do not put diagnostic or partial numbers in the committed visual payloads.
6. Enable GitHub Pages at repo root on main (independent of the above).

**New (2026-09-10b) - Hypothesis C, the third and final mechanism attempt.**
Registered at `research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md`. Both gates are
already answered with artifacts (`udwm/scripts/check_distill_param_sharing.py`,
`theory/hypothesis_d_dimension_desk_check.py`): the identified loss terms share the
student trunk (E1/E2 are meaningful), and Hypothesis D's dimensional-constants channel
is not suggested. The DMC side of the probe needs a GPU window and **must not displace
items 1-2** - it is a mechanism read, not a payoff read. If C is refuted, the DMC
transfer failure is written up as a characterized-but-unexplained open question; there
is no fourth mechanism attempt.

## 7. Integrity and method notes for this read

- PARTIAL files are read arm-by-arm only; no cross-arm bar is evaluated on an
  incomplete arm set, and no partial file is ever mixed with an adjudicated
  file in one table.
- Hardware pools differ: all adjudications before 2026-09-09 ran on the
  company VM (RTX 6000 Ada); MACURA/Walker rows ran on Kaggle 2xT4. Same-seed
  cross-checks line up (seeds 0-3: macura-run ordinary 0.965 vs the 30-seed
  file 0.958, macura-run eq 0.746 vs 0.729; seeds 0-5 ordinary: 0.953 vs
  0.949), so the T4 rows are consistent with the Ada rows at the subset
  level; a formal cross-hardware statement is not needed because these rows
  are not adjudicating anything yet.
- Checkpoint u-rank series come from `eval_history` records, which
  forward-fill between evals; only the per-eval records are used above and the
  series are approximate. Row-level final fields are the adjudicated
  endpoints everywhere in this doc.
- Returns on hopper-hop at 15k are floor-bound for every arm measured
  (ordinary, eq, macura_gate): return means are driven by a few non-floor
  seeds. The DMC floor rule therefore applies to MACURA rows too, and the
  return axis on hopper stays deferred.
- All files and scripts used for this read are local under `runs_local/` and
  `_probe/` (gitignored); nothing was committed from the VM or Kaggle
  working directories.

## Files

- New partials analyzed here: `runs_local/dmc_macura_30seed_15k_gpu_seed*.partial.json`,
  `runs_local/dmc_macura_probe_2seed_15k_gpu.json`, `runs_local/dmc_macura_gate_pilot.json`,
  `runs_local/walker_gate_pilot.json`, `runs_local/walker2d_budget_probe_gpu_seed*.partial.json`.
- Merged local unions: `runs_local/runs/dmc_macura_30seed_15k_gpu_kaggle_partial.json`,
  `runs_local/runs/walker2d_budget_probe_gpu_kaggle_partial.json`.
- Adjudicators: `udwm/scripts/summarize_dmc_payoff.py` (MACURA contrast block),
  `udwm/scripts/summarize_walker2d_payoff.py`, `udwm/scripts/dmc_gate_ratio.py`.
