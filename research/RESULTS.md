# Result index

Read **identifiability** first. The two earlier tables are the same mechanism
under two value maps, not two competing theses.

| Study | What it shows | File |
|---|---|---|
| Identifiability of the distillation target | Single-latent hybrid matches \(w+(g-\bar\Sigma)\); zero-\(w^\star\) students get ~0 loss; \(M\ge2\) + split \((\hat w,\hat g)\) identifies. Fixed-map 20-seed (confirmed at 30): identified vs hybrid is a **wash**. Policy 2×2 adjudicated at N=30: lagging (not identifiability) is the measured fix -- [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md). | [RESULTS-IDENTIFIABILITY-2026-08-29.md](RESULTS-IDENTIFIABILITY-2026-08-29.md) |
| Large controlled stress (20 seeds, **fixed** map) | Hybrid RMSE 20/20; rank 13/20; recall inconclusive. `w_deb` up, `g` down: variance is reallocated. | [RESULTS-STRESS-LARGE-2026-08-21.md](RESULTS-STRESS-LARGE-2026-08-21.md) |
| Fixed-map stress, N=30 extension | 20-seed rows bit-identical to the 08-29 study (strict superset). Hybrid magnitude 29/30; w-rank gain now confirmed (22/30, CI excludes 0); `g` worse (reallocation); top-decile recall still inconclusive. Identified vs hybrid: ranking wash, slightly worse magnitude at 2x cost. Policy is the discriminating experiment (see N=30 policy row). | [RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md](RESULTS-STRESS-IDENTIFIED-30SEED-2026-09-03.md) |
| Fixed-map stress, N=50 precision extension | Registered before seeds 30..49 ran (seeds 0..29 byte-identical, strict superset). Magnitude 49/50; w-rank 38/50 (CI excludes 0); `g` reallocation stable. Top-decile recall: mean CI [+0.0012, +0.0456] just excludes 0 but only 27/50 seeds positive -- inconclusive by the pre-registered 70% seed bar. Identified-vs-hybrid wash unchanged at 50. | [RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md](RESULTS-STRESS-IDENTIFIED-50SEED-2026-09-05.md) |
| Policy-scale (10 seeds, **live** critic) | Hybrid MSE 10/10; uncertainty rank/RMSE **0/10**; return 3/10. | [RESULTS-POLICY-SCALE-2026-08-22.md](RESULTS-POLICY-SCALE-2026-08-22.md) |
| Earlier policy probe | Inconclusive / negative transfer | [RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md](RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md) |
| Earlier distillation table | Superseded by the 20-seed study | [RESULTS-DISTILLATION-2026-08-20.md](RESULTS-DISTILLATION-2026-08-20.md) |

| Ground-truth (w*, g*) recovery | First check of the identified loss against ANALYTIC (w*, g*): estimator unbiased; M=1 objective walks the degenerate direction from both inits; equal-weighting hole (w rank 0.40 when g* >> w*) and the partial reweighting fix quantified. | [RESULTS-GROUND-TRUTH-W-G-2026-09-01.md](RESULTS-GROUND-TRUTH-W-G-2026-09-01.md) |
| Policy 2×2, 5-seed preliminary | Hybrid u-rank collapse reproduces (-0.301, 0/5); lagged critic recovers u-rank at 5 seeds (leans nonstationarity-dominant); identified arm confounded by reweighting hole. Not a verdict. | [RESULTS-POLICY-2X2-2026-09-01.md](RESULTS-POLICY-2X2-2026-09-01.md) |
| Policy 2×2, N=30 adjudicated rerun | Pre-registered extension to 30 seeds (EMA-reweighted identified arms). **Row 1 of the decision tree**: lagging fixes the u-rank collapse (30/30, to ~ordinary); the identified loss does not (0/30, U collapses to ~0). Nonstationarity is the measured mechanism (2026-09-05 addendum: the identified rows were the EMA-reweighted variant - see the corrected-weight N-study row below; identifiability now has a measured policy benefit for the equal-weight arm). Return: identified arms best, but return does not adjudicate. | [RESULTS-POLICY-2X2-30SEED-2026-09-03.md](RESULTS-POLICY-2X2-30SEED-2026-09-03.md) |
| Payoff test #4 (lagging alone, 3,600 steps), N=30 | Mechanism survival confirmed (lagged u-rank 0.988 ~ ordinary 0.986 at N=30; vs hybrid +0.273, 30/30). Return payoff still not demonstrated: pooled delta -44.6, 15/30, CI [-102.6, +10.2] (MBPO returns drift down past ~1,800 steps on this benchmark). | [RESULTS-PAYOFF-LAGGED-2026-09-03.md](RESULTS-PAYOFF-LAGGED-2026-09-03.md) |
| U-collapse mechanism (thread #1) | Why the identified arms zero the U object in policy (2026-09-05 addendum: the leverage/estimator-noise numbers describe the EMA-down-weighted channel; corrected attribution in the Corruption-probe row). The local object is aleatoric-dominated (teacher g/w ~ 1e4 under the SAC critic); identified arms match w but leave g at initialization (student latent-to-state spread ~0.006 vs teacher ~0.37). Corrects the "u_min=0 floor" claim in the N=30 policy doc. | [U-COLLAPSE-MECHANISM-2026-09-05.md](U-COLLAPSE-MECHANISM-2026-09-05.md) |
| Leverage-fix experiment | Registered test of whether the aleatoric starvation is a step-size artifact (2026-09-05 addendum: a config-override bug made every arm run EMA-on; the verdict and the "corruption is the remaining lever" claim are superseded by the Corruption-probe row). Loss-scale scaling is inert under Adam; 10x-100x larger student LR drives g to exactly 0 at the flat g=0 boundary. | [RESULTS-LEVERAGE-FIX-2026-09-05.md](RESULTS-LEVERAGE-FIX-2026-09-05.md) |
| Corruption probe + weighting correction (2026-09-05) | Registered corruption-distribution test with the probe's config-override bug fixed. Verdict: corruption is NOT the binding constraint - equal-weight identified recovers g on the live critic (u-rank 0.885/0.920, 2 seeds; N=10 replication in the doc) while EMA-both and w-only EMA annihilate g to ~0.01 (driver: the epistemic up-weight ~1e5-1e6, not the aleatoric down-weight). `pure` decision corruptions additionally degrade next-state MSE. Supersedes the leverage-fix/mechanism rows above. | [RESULTS-CORRUPTION-2026-09-05.md](RESULTS-CORRUPTION-2026-09-05.md) |
| Corrected-weight policy N-study (N=10, 2026-09-05) | Equal-weight identified (identified_eq) transfers to the live critic at policy scale: u-rank mean 0.835, 9/10 >= 0.70 (pre-registered bars A met); beats hybrid +0.214 (10/10) and the EMA arms +0.720 (10/10); below ordinary -0.104 (0/10) and lagged_hybrid -0.120 (0/10). The N=30 "identified does not matter in policy" verdict is attributed to the EMA variant, not the identified target; the w_rmse hole persists (0.995 vs ordinary 0.497). | [RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md](RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md) |
| Estimator-noise scaling + balance window | Analytic fit std(g_hat)/g* ~ sqrt(2/(N(M-1))) and std(w_hat)/w* ~ 1.35@M=2 -> 0.22@M=32 set the window where the split (w_hat, g_hat) is learnable. Toy re-adjudication: equal-weight identified recovers g (g_rank 0.81-0.97) but leaves w at the hole (w_rank 0.40) when g* >> w*; EMA-both fixes w (0.97-0.99) and annihilates g (g_hat ~ 0.03-0.4 vs g* 46-3966). No single scalar weight recovers both at g* >> w*. | [theory/identified_balance_window.py](../theory/identified_balance_window.py) |
| Paper narrative assembly | Single-arc ordering of the result docs (identifiability -> fixed map N=50 -> live-critic falsification -> policy 2x2 -> payoff -> mechanism/correction -> balance window -> limitations), plus the statistics statement (no joint multiple-comparisons correction; per-endpoint pre-registration as the mitigation) and the future-work position (long-horizon low-dimensional benchmark before pixels). | [PAPER-NARRATIVE.md](PAPER-NARRATIVE.md) |

Estimator notes (not policy tables):
[ESTIMATOR-BIAS-FINDING.md](ESTIMATOR-BIAS-FINDING.md) (first moment, distribution-free),
[ESTIMATOR-VARIANCE-FINDING.md](ESTIMATOR-VARIANCE-FINDING.md) (second moment, kurtosis; why the model class can matter).

**Payoff test #4 (2026-09-03):** adjudicated at N=30 (3,600 steps, lagging
alone): mechanism survives (lagged u-rank 0.988 ~ ordinary 0.986; +0.273 vs
hybrid, 30/30); return payoff not demonstrated (delta -44.6, 15/30, CI
[-102.6, +10.2]). The open question for a payoff claim is the
benchmark/protocol (the deferred modality choice), not more steps on
DelayedBimodal: MBPO returns drift down past ~1,800 env steps here.

---

## Addendum 2026-09-07: DMC 10-seed sanity + budget probe

DMC/hopper-hop 10-seed sanity (GPU rows, N=10, NOT adjudication): gate regime
aleatoric-dominated (median g*/w* 8,224). The EMA-collapse mechanism
replicates cross-environment (eq - identified_hybrid u_rank +0.419, 10/10);
the arm ordering replicates (lagged_hybrid 0.573 > ordinary 0.511 > hybrid
0.438 ~= eq 0.429 > EMA 0.01); absolute levels do not (ordinary 0.51 here vs
0.94 on DelayedBimodal). The DelayedBimodal headline (eq > hybrid, +0.214,
10/10) is a wash on DMC (-0.008, 5/10); eq - ordinary and hybrid - ordinary
are confirmed-below on u_rank; the eq w hole does not replicate (eq w_rmse
better than ordinary, 9/10). Because 3,600 steps = 3.6 1,000-step episodes
and returns are flat everywhere, the sanity cannot distinguish "identified_eq
does not transfer" from "the u_rank measurement is degenerate at this
budget"; the 30-seed adjudication is held pending the registered budget probe
(15k = 15 episodes, seeds 0-1).
[RESULTS-DMC-SANITY-2026-09-07.md](RESULTS-DMC-SANITY-2026-09-07.md)

## Addendum 2026-09-07 (second): H1 combined fix + H2/H4 re-analysis

1. **H1: the {lag} x {equal-weight} cell ran (N=10, DelayedBimodal).**
   Registration `COMBINED-FIX-PREREGISTRATION-2026-09-07.md`; data
   `runs/policy_combined_fix_10seed.json` (fresh full 7-arm table, exact
   pairing, gap 0.0; the merge-with-existing design was abandoned after the
   registered bit-exactness gate failed). `lagged_identified_eq` is the top of the table: u-rank 0.948 (10/10 >= 0.70).
   Bar 2 met (vs identified_eq +0.071, 8/10, CI excludes 0); bar 3 met. Vs
   ordinary +0.014 (6/10, CI [-0.013, +0.047]) and vs lagged_hybrid +0.003
   (5/10, CI [-0.019, +0.022]) are PARITY, not confirmed exceedances - every
   top arm sits in the ceiling band. The confirmed non-ceiling effects are the
   w_rmse hole closing (0.92 -> 0.28 vs identified_eq, CI excludes 0) and the
   u_rmse improvement. eq's confirmed deficit vs ordinary (-0.057 here, -0.104
   in the 09-05 file) is fully closed by adding lag; partial -> full transfer
   on DelayedBimodal is supported at N=10. Bar 1 (exceeding lagged_hybrid) is
   a ceiling test on this benchmark and moves to DMC.
   [RESULTS-COMBINED-FIX-POLICY-2026-09-07.md](RESULTS-COMBINED-FIX-POLICY-2026-09-07.md)
2. **H2/H4 zero-compute re-analysis of existing rows.** H4: the logged
   selective columns do not show tail preservation for any arm - the fitted
   U-net risk signal is at/below chance everywhere (recall_bad 0.26-0.36 vs
   ~0.37 random baseline; rank corr negative for all arms) and no eq tail
   contrast confirms; the EMA collapse is muted on tail recall, so tail
   metrics alone would understate the pathology. H2: across the two existing
   eq-vs-hybrid cells the literal instability regression is refuted at N=2
   cells (DMC hybrid sd 0.111 > DB 0.090 with eq benefit ~0 vs +0.214); the
   replicating pattern is within-cell (eq's gap grows on seeds where hybrid
   is weak, both cells) and vanishes under DMC floor-compression - i.e., the
   conditional-benefit claim is operating-point-bound, consistent with the
   registered budget confound.
   [RESULTS-H2H4-REANALYSIS-2026-09-07.md](RESULTS-H2H4-REANALYSIS-2026-09-07.md)

## Addendum 2026-09-07 (third): identifiability-frontier formalization + DMC protocol

1. **Theory formalized (verified, not hypothesized).** The identifiability
   identity is upgraded to a theorem set with proofs and a deterministic
   numeric verification, all checks PASS
   (`theory/identifiability_frontier.py`):
   [proofs/identifiability-frontier.md](proofs/identifiability-frontier.md),
   [RESULTS-THEORY-FRONTIER-2026-09-07.md](RESULTS-THEORY-FRONTIER-2026-09-07.md).
   Headlines: T1 - the M=1 zero-loss fibre never identifies the sign of the
   decision statistic `u` (measured map: collapse student `u_s > 0` vs
   teacher `u* ~ -80` at ~zero loss); T2 - coupling erosion
   `dS/dg = (1-rho)(N-1)/N` puts the matched statistic at `S*/|u*| = 2.4e-4`
   on the measured map; P3 + T5 - the aleatoric boundary is sticky
   (gradient `O(g_s)`, per-state noise `0.76 g*` at M=2, no multiplicative
   dial escapes, EMA ratio = `(w*/g*)^2 = 7e-9`); T4 - M>=2 has a unique
   population minimiser and a separation floor `M >~ (A_w+A_g)/(B Delta^2)`;
   T6 - percentile gating is invariant to monotone score transforms,
   absolute-threshold gating is not.
2. **Protocol audit: the policy runs are self-gated.** `u_gate.mode: both`
   (percentile 0.85 stop+weight via the training-side UBE net) has been
   active from step 900 since `configs/delayed_bimodal_distill.yaml` was
   created; all arms gated identically so comparisons stand, but prose must
   not call these runs ungated and the null payoff reading applies to
   self-gated training. No table re-adjudicated; the paper must state the
   protocol as run.
3. **DMC arm list + gating axis amended (pre-compute).** `lagged_identified_eq`
   is folded into the DMC study as a primary arm and the budget probe runs the
   full 6-arm list at 15k steps; a `gate-off` ordinary control arm is added so
   the payoff contrast separates object quality from gating itself; all gating
   is percentile (T6). Registration:
   [DMC-PAYOFF-PREREGISTRATION-2026-09-05.md](DMC-PAYOFF-PREREGISTRATION-2026-09-05.md)
   (Addendum 3).
4. **Gate-off control arm registered + adequate-budget knobs pre-committed.**
   `ordinary_gate_off` (ordinary distillation, `u_gate.mode: off`) is a
   registered payoff control (Addendum 4) with its own adjudication rule;
   `configs/dmc_hopper_payoff_30k.yaml` stages the Amendment-2 budget (30k
   steps, `eval_freq: 3000`) but must not run before the budget probe's
   decision rule records Amendment 2.
5. **Combined-fix parity contrasts re-verified under one method.** The
   combined-vs-ordinary u-rank contrast now carries the identical-method
   paired bootstrap (wins/N + CI + P) recorded in the results doc addendum:
   +0.0142, 6/10, CI [-0.0132, +0.0464] - parity, not exceedance; correct
   phrasing is "met parity on a ceiling-bound benchmark".
6. **Literature metadata corrected (2026-09-07 second pass).** MACURA venue =
   ICML 2024 (not RLDM); Biased Dreams exact title "...in Latent Space
   Models"; GIRL venue unconfirmed (cite as arXiv); AAAI-Liu + ELVIS
   metadata confirmed. Recorded in
   `papers/LITERATURE-VALIDATION-2026-09.md`; entries in
   `papers/references.bib`. New gaps G7 (normalization-controlled w-channel
   contrast - closes the Theorem 7 caveat) and G8 (EMA-at-parity capacity-
   competition theory target) are in
   `research/NOVEL-GAPS-2026-09-07.md`.

## Addendum 2026-09-08: G7/G9 attribution + DMC budget-probe verdict

1. **G7 adjudicated (N=10, DelayedBimodal, one file): the w_rmse gap and the
   combined-fix Bar-2 u-rank gap are normalization-carried, not lag-carried.**
   Both pre-registered primary bars failed (D - A and C - B w_rmse CIs
   include 0); within-cell lag contrasts on u_rank are null (-0.0011,
   -0.0007) while `normalize_values` alone moves u_rank +0.11 (10/10, both
   cells) and w_rmse -0.8..-0.9 (10/10). Consequence: w-scale language is
   dropped (rank-only) and the combined-fix record is re-read via its G7
   addendum. [RESULTS-NORMALIZATION-CONTROL-2026-09-07.md]
   (RESULTS-NORMALIZATION-CONTROL-2026-09-07.md)
2. **G9 re-adjudication (N=10, in-file baselines): standardization closes
   the live-eq deficit on the LIVE critic.** `identified_eq_norm` 0.948
   (10/10 >= 0.70) sits at/above ordinary (0.923) and lagged_hybrid (0.927);
   mechanism sanity reproduces (eq > hybrid +0.214, 10/10); the lag null
   replicates (C - B, D - A). The corrected arm-to-term sentence: the
   equal-weight M>=2 loss + value-target standardization is the empirical
   fix; the slow target critic adds nothing measurable on DelayedBimodal at
   N=10 and its contribution is deferred to the DMC verdict.
   [RESULTS-NORMALIZATION-READJUDICATION-2026-09-07.md]
   (RESULTS-NORMALIZATION-READJUDICATION-2026-09-07.md)
3. **DMC budget probe verdict: confound supported -> Amendment 2 activated.**
   At 15k (seeds 0-1) baseline u_rank climbs to 0.925 (ordinary) / 0.856
   (lagged_hybrid) from 0.454/0.574 at 3.6k; identified_eq clears the bar on
   both seeds (0.766/0.755) and sits >= hybrid; EMA collapse control
   reproduces (0.132); lagged_identified_eq does NOT top the DMC table
   (0.634) - the 30-seed verdict is open; returns are floor-bound on seed 1
   at 15k (return thread deferred). Gate re-measured at 15k: g*/w* median
   14,292 (aleatoric-dominated). Amendment 2 sets the adjudication budget to
   15,000 steps; 3.6k DMC rows superseded.
   [RESULTS-DMC-BUDGET-PROBE-2026-09-08.md](RESULTS-DMC-BUDGET-PROBE-2026-09-08.md)

---

## Addendum 2026-09-08 (30-seed DMC verdict): mechanism transfers, ordering does not

30-seed x six-arm adjudication at 15k (Amendment 2 budget) completed
(`runs/dmc_payoff_30seed_15k_gpu.json` + `*_ctrl.json`, all seeds exact
teacher match). Registered bars for identified_eq NOT met (15/30 >= 0.70; eq -
hybrid -0.020 [-0.067, +0.030], 13/30 wash). Controls MET 30/30: EMA collapse
reproduces (mean -0.076, 0/30 >= 0.70; ordinary - EMA +1.034) and eq - EMA
+0.788. Combined-fix arm (lagged_identified_eq) confirmed below eq (-0.073,
CI excludes 0, 7/30) - the DelayedBimodal ordering reversal is real at N=30;
on DMC the plain self-gated ordinary arm tops the table (0.957, 30/30) and the
lag axis is loss-family-dependent (lagged_hybrid - hybrid +0.106, 22/30;
lagged_eq - eq -0.073). Gate-off control return-neutral at 15k (+0.025, CI
includes 0); return thread deferred per registration.
[RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md](RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08.md)

## Addendum 2026-09-09: next registered probe (Hypothesis A, CRN bias) - registered, NOT yet run

No experiment has run since the DMC 30-seed verdict. The next registered
probe - the CRN-bias / value-reference-consistency diagnostic that would
explain the lag-axis asymmetry (lagged_hybrid +0.106 vs lagged_eq -0.073 on
DMC) - is now preregistered, before any harness code or rows:
[CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md](CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md).
It runs the identified-family {live, lagged} x {norm, no-norm} 2x2 plus
hybrid-family controls and the EMA collapse arm, logs critic drift rate,
paired-evaluation gap, and a Wu-et-al-style induced-bias proxy, and commits
power/MDE numbers computed from the DMC 30-seed SDs (n=10 diagnostic is
underpowered for the eq-vs-lagged_eq contrast by design; n=30 only if the
decision tree fires). Also still open and untouched: compute-normalized
re-analysis, MACURA head-to-head baseline, and the conditional 30k return
extension - none have run.

## Addendum 2026-09-09 (evening): CRN-bias DMC n=10 diagnostic readout - Hypothesis A not supported at n=10

The diagnostic described above has since run and merged (same day; the probe's
Addendum 1 was pre-committed before the readout). Full numbers and analysis:
[RESULTS-CRN-BIAS-DMC-2026-09-09.md](RESULTS-CRN-BIAS-DMC-2026-09-09.md).
Verdict in one line: **Hypothesis A does not confirm on DMC at n=10** - E1 is
inconclusive in both lagged cells (norm cell +0.018 [-0.588, +0.770], null
below MDE; nonorm cell +0.661 [+0.018, +0.927], significant in the WRONG
direction and robust to the one degenerate seed, +0.533 with it dropped); E2
within-cell lag deltas are positive on DMC (+0.056/+0.071, descriptive only);
and the section-1c drift-sensitivity (A1.2 residualized read) is present in
the lagged_hybrid placebo (+2.961, CI excluding 0) about as strongly as in the
paired norm cell (+5.208) - so pairing-specific drift-bias is not supported
there either. Controls intact: EMA collapse reproduces (identified_hybrid
-0.092, 0/10 >= 0.70) and the probe rows reproduce the 30-seed file's
seeds-0-9 ordering on every shared arm. No `eq_crn` arm is registered; the
Wu-style correction (E4) is not built; the DMC 30-seed verdict is unchanged.
The DelayedBimodal companion rows (crn_bias_probe_db_n10.json) stay
descriptive: same harness, drift ~100x larger (B_hat ~1.2 vs ~0.009), E1
inconclusive there too. Separate GPU track still in flight: MACURA head-to-head
and Walker2d gate/budget probe staged on Kaggle 2xT4 (kaggle_macura_walker.sh);
compute-normalized re-analysis and the 30k return extension remain open.

## Addendum 2026-09-10: grand status ledger + first MACURA/Walker2d GPU rows (Kaggle partials)

Exact full status for every study to date is now in one place:
[GRAND-STATUS-2026-09-10.md](GRAND-STATUS-2026-09-10.md) (per-study ledger,
claims safe to assert, next queue, integrity notes). New this pass - the
stopped Kaggle session's rows were downloaded to `runs_local/` and read as
PARTIAL/DIAGNOSTIC only (no bars evaluated):

- MACURA DMC head-to-head is PARTIAL at n=4 of 30 (seeds 0-3 complete, three
  arms; seeds 4-5 have ordinary only). u-rank at parity with ordinary
  (macura_gate - ordinary -0.003, CI includes 0) and above identified_eq
  (+0.216, 4/4); returns floor-bound at 15k on hopper-hop, so the registered
  payoff endpoints are not readable yet. MACURA gate is active (stopped_frac
  0.158 vs ordinary 0.096 at final). Full 30-seed read still needed
  (~9-12 h on 2xT4, resume-safe).
- Walker2d staging: gate pilot aleatoric-dominated (median g*/w* 5,860; w*
  nonzero on seed 0 - structurally different from hopper), returns usable and
  NOT floor-bound (ordinary 32.65 vs lagged_hybrid 27.55 at n=2) - first
  positive signal for the Hypothesis-B premise. But 15k is not an adequate
  Walker operating point (u-rank 0.46/0.25, 0/2 >= 0.70, non-monotone ends);
  identified_eq arm unfinished. Higher-budget staging decision required
  before any arm comparison.
- Unchanged and still open: CRN Hypothesis A not supported at n=10, no
  eq_crn/E4; compute-normalized re-analysis + effect-size report staged but
  not run; 30k return extension gated; GitHub Pages not yet enabled.

## Addendum 2026-09-10 (b): Hypothesis C registered - gradient interference as the third mechanism attempt (both gates pre-answered)

Third and final mechanism attempt at the DMC transfer failure:
[HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md](HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md).
Two gates were answered *before* registration, each with a runnable artifact.
No verdict moved and no bar was re-opened.

- **Gate 1 (architecture coherence): SHARED TRUNK - C is coherent.**
  `udwm/scripts/check_distill_param_sharing.py` rebuilds the exact model from a
  saved checkpoint's `cfg`, backwards each identified-loss term separately, and
  intersects the parameter sets that receive nonzero gradient. On the real
  DelayedBimodal checkpoint (`checkpoints/hf_diagnostic_delayedbimodal/identified_eq_seed0.pt`):
  `member` (point prediction) and `epistemic_w`/`aleatoric_g` (uncertainty
  matching) all reach the same 50 student tensors; `Jaccard(point, uncertainty)
  = 1.00`; clean load (no missing/unexpected keys). Confirmed for the Hopper
  arm topology (15/4) on `identified_eq`, `lagged_identified_eq` and
  `identified_hybrid`. There is **no separate point-prediction head** - one
  `ConditionalDenoiser` per member, every term a functional of the same
  `forward_member` output. `ordinary` has no uncertainty term by construction
  (all decision weights zero), so it is not applicable, not measured.
- **Gate 2 (Hypothesis D desk check): NOT SUGGESTIVE - not escalated.**
  `theory/hypothesis_d_dimension_desk_check.py` shows the repo's own
  balance-window prediction `std(g_hat)/g* ~ sqrt(2/(N(M-1)))` is
  **dimension-free** at fixed (N, M): 0.125 at the registered operating point
  (N=128, M=2), identically on DelayedBimodal (5/1) and DMC (15/4). A
  dimension-independent noise term cannot produce an environment-specific gap
  (observed `eq - ordinary` u-rank -0.104 DB vs -0.246 DMC). The only surviving
  channel is non-Gaussian/heavy-tailed constants, which the formula does not
  model and the repo's theory does not predict.

Registered endpoints: **E1** environment contrast (`identified_eq` interference,
DMC - DB > 0), **E2** within-environment dose-response (interference vs the
per-seed `ordinary - eq` u-rank gap), **E3** capacity axis (student hidden
width - the previously shelved G8 knob; explicitly secondary/lower-confidence),
with `hybrid` placebos for E1/E2 and a pre-stated falsification condition.

Instrumentation is measurement-only: `udwm/scripts/probe_gradient_interference.py`
adds one forward/backward per probe checkpoint and never steps an optimizer
(`GRAD_PROBE` mirrors the existing `CRN_PROBE` flag, so the training path is
bit-identical when unset), plus
`udwm/scripts/summarize_gradient_interference.py` (bootstrap CIs + wins/N).
Wiring smoke (600-step DelayedBimodal, seed 0, artifact deleted rather than left
in `runs/`): teacher pairing exact (`exact_teacher_match: true`, gap 0.0),
`ordinary` correctly N/A, `identified_eq` cos +0.059 -> +0.171 and `hybrid` cos
-0.167 -> -0.061 over two checkpoints. **Smoke is not a result** - one seed, 600
steps, far below the registered 15k operating point, and the eq sign is opposite
the E1 prediction, which is precisely why it must not be read as one.

MDE registered before the run (item 5a): E1 two-sample MDE = 3.962 sigma/sqrt(n)
(1.25 sigma at n=10, 0.72 at n=30); E2 Fisher-z |r| MDE 0.79 at n=10, 0.49 at
n=30 - so an n=10 E2 null sits below the MDE and is not evidence of absence
(the underpowered-n trap the CRN-bias E1 read fell into after the fact).

DMC side needs a GPU window and must not displace the MACURA n=30 completion or
the Walker2d budget decision.

## Addendum 2026-09-10 (c): the two long-open free items are run - compute-normalized re-analysis + effect-size/power report

Both zero-GPU items from the master-plan-v3 script specs were run locally on the
already-adjudicated rows (no new training, no bar re-opened):

- [RESULTS-COMPUTE-NORMALIZED-AND-EFFECT-SIZE-2026-09-10.md](RESULTS-COMPUTE-NORMALIZED-AND-EFFECT-SIZE-2026-09-10.md)
- `udwm/scripts/reanalyze_compute_normalized.py` (item 2),
  `udwm/scripts/effect_size_report.py` (item 5a);
  registration: [COMPUTE-NORMALIZED-REANALYSIS-2026-09-09.md](COMPUTE-NORMALIZED-REANALYSIS-2026-09-09.md).

**Compute normalization (item 2).** Cost multipliers are read from
`udwm/models/consistency.py`, not assumed: `ordinary` 1x, hybrid family 5x,
identified family 10x teacher samples per model-train epoch. Re-indexing the DMC
30-seed u-rank curves against teacher-sample spend, the ordering does not change
and eq's deficit does not narrow - `ordinary` reaches 0.954 at 1,776u while
`identified_eq` needs 17,760u for 0.712 (0.040 vs 0.537 u_rank/1000u, ~13x).
DelayedBimodal shows the same shape. A true fixed-compute head-to-head is **not
readable** from these rows (every arm ran 15k env steps, so the common
teacher-sample domain is empty); the doc states that instead of extrapolating.

**Effect size / power (item 5a).** Additive to the raw delta + CI + wins/N:
`eq - ordinary` d_z = **-2.63** (0/30), about 5x the n=10 MDE and ~3x the n=30
MDE, so the DMC failure of the identified arm is not a power problem; `eq -
hybrid` d_z = -0.14 (13/30) recasts the registered Bar-2 wash on a standardized
scale; the mechanism contrast is the largest effect in the study (`ordinary -
EMA` +8.01, `eq - EMA` +5.33, both 30/30). The retroactive CRN-E1 MDE note
closes the item-5a gap for that study: n=10 could only detect ~0.08-0.12 on a
u_rank-scaled endpoint, which is why the CRN E1 readout was called uninformative
rather than negative.

No DMC/DB verdict, bar, or win/N count changes; both items are re-readings of
already-adjudicated rows.

