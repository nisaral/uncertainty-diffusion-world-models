# DMC 30-seed adjudication at 15k: verdict (2026-09-08)

**Registration:** `research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md` (Amendment 1;
Addenda 2-4; Amendment 2 activated 2026-09-08, 15,000-step budget). **Data:**
`runs/dmc_payoff_30seed_15k_gpu.json` (six arms x 30 seeds) and
`runs/dmc_payoff_30seed_15k_gpu_ctrl.json` (`ordinary_gate_off`, Addendum 4).
**Host:** company VM (RTX 6000 Ada, 2 polite GPU workers, no co-resident
interference; chain ran unattended 00:25 -> 19:30 IST under a restart-safe
watchdog with zero relaunches). **Integrity:** all 30 seeds x 6 arms report
exact teacher pairing (checksum gap 0.0, `exact_teacher_match: true`); the
30-seed file merged 17:26 IST, the gate-off control merged 19:30 IST. Local
copies are byte-identical to the host files (SHA-256 checked). The 10-seed
sanity (`runs/dmc_sanity_15k_gpu.json`) is diagnostic only and is superseded
by this file for adjudication (registered: never mix rows across files/budgets
in one table).

## Per-arm result at 15,000 steps (n = 30)

| arm | u_rank mean | u_rank median | n >= 0.70 | w_rmse (median) | next_state_mse | return mean | return median |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 0.957 | 0.960 | 30 | 3.31e-05 | 1.815 | 0.100 | 0.025 |
| lagged_hybrid | 0.837 | 0.863 | 27 | 3.40e-05 | 1.749 | 0.126 | 0.031 |
| hybrid | 0.731 | 0.764 | 21 | 2.57e-06 | 1.625 | 0.144 | 0.075 |
| identified_eq | 0.711 | 0.705 | 15 | 3.76e-06 | 1.621 | 0.056 | 0.013 |
| lagged_identified_eq | 0.638 | 0.622 | 8 | 3.43e-05 | 1.620 | 0.201 | 0.075 |
| identified_hybrid (EMA) | -0.076 | -0.084 | 0 | 1.59e-07 | 0.897 | 0.089 | 0.026 |

Ordering (u-rank): ordinary > lagged_hybrid > hybrid ~= identified_eq >
lagged_identified_eq > identified_hybrid. This is exactly the ordering seen at
every checkpoint from the n=2 budget probe through S1 (n=10) and the mid-run
reads (n=13/18/22/24) - no checkpoint reversed it.

## Registered bars (mirror the DelayedBimodal study)

| Bar | Result | Status |
|---|---|---|
| 1. identified_eq u_rank >= 0.70 on >= 21/30 seeds | 15/30 (mean 0.711) | **NOT MET** |
| 2. identified_eq beats hybrid on >= 21/30, CI excludes 0 | -0.020 [-0.067, +0.030], 13/30 | **NOT MET** (wash) |
| Mechanism-transfer control: identified_eq - EMA, CI excludes 0 | +0.788 [+0.736, +0.840], 30/30 | **MET** |
| EMA collapse control: identified_hybrid ~ noise, ordinary - EMA 30/30 | -0.076 mean, 0/30 >= 0.70; +1.034 [+0.988, +1.079] | **MET** |

## Paired contrasts (u_rank_corr; 10^5-draw percentile bootstrap 95% CI, RNG seed 0)

| Contrast | Mean delta | 95% CI | Wins |
|---|---:|---:|---:|
| eq - ordinary | -0.246 | [-0.278, -0.213] | 0/30 |
| eq - lagged_hybrid | -0.125 | [-0.169, -0.080] | 4/30 |
| eq - hybrid | -0.020 | [-0.067, +0.030] | 13/30 |
| eq - EMA | +0.788 | [+0.736, +0.840] | 30/30 |
| lagged_eq - ordinary | -0.320 | [-0.350, -0.289] | 0/30 |
| lagged_eq - hybrid | -0.093 | [-0.142, -0.043] | 8/30 |
| lagged_eq - eq (DB ordering reversal) | -0.073 | [-0.111, -0.036] | 7/30 |
| lagged_hybrid - hybrid (lag axis, hybrid family) | +0.106 | [+0.059, +0.153] | 22/30 |
| lagged_hybrid - ordinary | -0.121 | [-0.150, -0.092] | 0/30 |
| hybrid - ordinary | -0.226 | [-0.274, -0.183] | 0/30 |
| ordinary - EMA | +1.034 | [+0.988, +1.079] | 30/30 |
| hybrid - EMA | +0.807 | [+0.750, +0.866] | 30/30 |

## Verdict

**The mechanism story transfers; the DelayedBimodal practical ordering does
not.**

1. **EMA collapse reproduces across environments (30/30).** The identified
   EMA-reweighted control sits at noise (mean -0.076, 0/30 >= 0.70) and every
   arm beats it with CI excluding zero. The 2026-09-05 weighting attribution
   is not a DelayedBimodal artifact; it is measured on a second benchmark at
   the amended budget.
2. **The equal-weight identified arm does not reach its DelayedBimodal
   partial-transfer position on DMC.** Both registered bars fail: 15/30 seeds
   >= 0.70 (bar 1), and eq - hybrid is a wash (13/30, CI includes 0; on
   DelayedBimodal the same contrast was +0.214, 10/10). eq is 0.246 below
   ordinary (0/30) and 0.125 below lagged_hybrid (4/30). "Preserved u sits
   between the EMA arms and the baselines" holds only in the eq > EMA leg.
3. **The combined-fix arm (lagged_identified_eq) - top of the DelayedBimodal
   table - is DMC's worst mechanism arm after the EMA control.** It is below
   its equal-weight sibling (lagged_eq - eq = -0.073, CI excludes 0, 7/30),
   below hybrid (-0.093, 8/30), and 0.320 below ordinary (0/30). The
   DelayedBimodal ordering reversal flagged at n=2 is **confirmed at N=30
   with a tight CI**.
4. **The lag axis is loss-family-dependent on the drifting DMC map.** Lagging
   helps the plain hybrid family (lagged_hybrid - hybrid +0.106, 22/30, CI
   excludes 0) but hurts the equal-weight identified family (lagged_eq - eq
   -0.073, CI excludes 0). "Lagging fixes nonstationarity" is therefore not a
   uniform arm-level claim: its sign depends on which loss is being lagged,
   on the map where drift is actually measurable.
5. **On hopper-hop at 15 episodes, the plain self-gated baseline is the top
   arm.** ordinary (0.957, 30/30 >= 0.70) beats every uncertainty-object
   variant; even lagged_hybrid - the strongest single fix on DelayedBimodal -
   is 0.121 below ordinary (0/30) here.

## Gate-off control (Addendum 4 reading rule)

`ordinary_gate_off` - `ordinary` paired final_return: +0.025 [-0.057, +0.109],
15/30. CI includes 0 -> **rule (ii): gating is return-neutral at this budget,
and the registered u-rank-preservation story is the whole measurable payoff**
(descriptive only, as registered: gate_off u_rank 0.975 vs ordinary 0.957;
next_state_mse 1.753 vs 1.815 - never adjudicative).

## Return thread (registered deferral)

Returns stay floor-bound on hopper-hop at 15k (per-arm means 0.06-0.20,
medians 0.01-0.08; seed-level means dominated by a few non-floor seeds), so
no return endpoint is adjudicated at 15k (Amendment 2 item 5). Descriptive
lead for the conditional 30k extension: lagged_identified_eq - ordinary
final_return +0.101 [+0.016, +0.199] (15/30) is the only arm whose paired CI
excludes zero - flagged as a *hypothesis for the 30k run*, not a result. The
mechanism verdict above is negative for the DB transfer-ordering claims, so
per the preregistration the 30k return extension is only justified if the
positive-control payoff question (does preserved u improve control under an
adequate budget) is still worth its compute - see the decision-tree mapping
below.

## Mapping to the preregistered decision tree

The gate at the amended operating point is aleatoric-dominated (teacher
g*/w* median 14,292, measured on the probe ordinary rows). Of the three
pre-committed aleatoric-branch outcomes:

- "u-rank also collapses under equal weight on DMC" - **not observed** (eq
  0.711 mean, +0.788 vs the EMA control, 30/30);
- "equal-weight u-rank preserved but return null" - **partially observed and
  sharpened**: u-rank is preserved *relative to the collapse control* but
  does not clear the registered position bars (vs hybrid wash; 0.25 below
  ordinary); return is null/floor-bound at 15k;
- net consequence: the mechanism correction transfers its *control behavior*
  (no collapse; EMA control reproduces) but not its *practical ordering*.
  The DelayedBimodal closer "partial transfer becomes full transfer once both
  known failure modes are addressed" is a DelayedBimodal-scale statement; the
  DMC verdict is environment-dependence of the practical ordering, with the
  mechanism (EMA weighting annihilates the aleatoric channel; equal weights
  preserve it) confirmed cross-environment.

## What this does and does not change

- **Unchanged:** the headline mechanism claims (identifiability of the
  distillation target; the EMA weighting failure; Theorem 1/3/5/7/8 +
  Proposition 8), the DelayedBimodal adjudications as DelayedBimodal-scale
  results, the fixed-map N=50 stress, and the attribution reads at N=10.
- **Changed:** any sentence that presents the combined fix or the
  partial-transfer ordering as transferring out of sample. The honest
  cross-environment statement is: *the failure modes are general (EMA
  collapse reproduces 30/30; equal weights avoid it), but on a drifting-map
  benchmark at 15 episodes the plain self-gated baseline is the top u-rank
  arm and every candidate fix sits below it; the lag axis is loss-family-
  dependent there.*
- **Outreach/paper framing:** the DMC verdict is a clean, adjudicated result
  in either direction (bars fail with tight CIs; controls pass with 30/30) -
  it resolves the previously open "does it transfer" question with a precise
  negative for the ordering claims and a precise positive for the mechanism
  controls. That is the publishable form of the cross-environment chapter.

## Files / reproducibility

- Pre-registration: `research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`
  (Amendments 1-2, Addenda 2-4) and
  `research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`.
- Data: `runs/dmc_payoff_30seed_15k_gpu.json`,
  `runs/dmc_payoff_30seed_15k_gpu_ctrl.json`,
  `runs/dmc_sanity_15k_gpu.json` (diagnostic S1).
- Adjudication: `python -m udwm.scripts.summarize_dmc_payoff` (10^5-draw
  percentile bootstrap, RNG seed 0; output reproduced above). Chain launcher
  and per-stage logs archived on the host; watchdog log shows no relaunch.
