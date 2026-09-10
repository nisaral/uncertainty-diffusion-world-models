# MACURA field baseline on DMC/hopper-hop, 30 seeds @ 15k - adjudicated (2026-09-10)

**Registration:** `research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md`
(Eq-4 `u_KL` ensemble-disagreement gate; reuse of the existing teacher/student
ensemble; deliberate Algorithm-2 deviations documented in section 1.1).
**Data:** `runs/dmc_macura_30seed_15k_gpu.json` - 90 rows = 30 seeds x
{`ordinary`, `identified_eq`, `macura_gate`}, 37,234,730 bytes, sha256
`6d708716e51dce41652c36a775dff6f8` (local mirror byte-identical to the host
file). **Host:** company VM (RTX 6000 Ada). **Run:** launched 2026-09-10
17:54:53 IST, resuming seeds 0-3 from the stopped Kaggle session; merged and
exited `rc=0` at 23:15:28 IST. No relaunch, one worker per GPU-visible slot,
2 workers at ~544 MiB, co-resident vLLM untouched.
**Status: ADJUDICATED** at the registered n.

## 1. What this arm set can and cannot decide

`macura_gate` keeps ordinary distillation and swaps only the *gating signal*:
the rollout stop/weight rule is driven by MACURA's Eq-4 `u_KL` instead of the
learned UBE `u`. That makes two things true by construction:

- **`u_rank` is a distillation sanity read, not a payoff endpoint.** Because
  `macura_gate` distills the same target as `ordinary`, parity with `ordinary`
  is the expected outcome; a large deficit would have been the surprise.
- **The registered fair endpoints are rollout-gating quality and downstream
  return**, and they are read below.

The DMC-payoff registered bars (bar 1 / bar 2) are *Hopper-study* bars for the
identified-family arm set and do not apply here; the summarizer is run with
`--no-bars` and no bar is evaluated on this file.

## 2. Integrity

| check | value |
|---|---|
| rows | 90 = 30 seeds x 3 arms (asserted after merge) |
| teacher pairing | 30/30 seeds `exact_teacher_match: true`, `max_teacher_checksum_gap: 0.0`, `n_arms: 3` |
| merge | additive lock-serialized union; every requested `(seed, arm)` present |
| rows mixed across budgets/files | none |

## 3. Per-arm final eval (15k, n=30)

| arm | u_rank mean | u_rank med | n >= 0.70 | w_rmse med | ns_mse mean | ret mean | ret med |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordinary` | 0.961 | 0.969 | **30/30** | 4.04e-05 | 1.8154 | 0.1031 | 0.0552 |
| `macura_gate` | 0.953 | 0.956 | **30/30** | 1.24e-05 | 1.8228 | 0.1735 | 0.0299 |
| `identified_eq` | 0.706 | 0.717 | 18/30 | 9.68e-06 | 1.6463 | 0.1037 | 0.0172 |

## 4. Registered contrasts (paired, 10^5 bootstrap, rng 0)

| contrast | u_rank mean | 95% CI | wins | d_z |
|---|---:|---|---:|---:|
| `macura_gate - ordinary` | **-0.009** | [-0.017, -0.001] | 9/30 | -0.38 |
| `macura_gate - identified_eq` | **+0.246** | [+0.209, +0.285] | **30/30** | +2.27 |
| `identified_eq - ordinary` | **-0.255** | [-0.295, -0.217] | 0/30 | -2.30 |

Reading, precisely:

- **The MACURA signal does not beat plain self-gating.** `macura_gate -
  ordinary` is -0.009 u_rank. The paired CI excludes 0 (the per-seed delta SD
  is only 0.0235), but the magnitude is *smaller than the n=30 MDE of 0.012*
  and d_z is -0.38; both arms sit in the ceiling band (0.953 vs 0.961, both
  30/30 >= 0.70). This is **parity**, not an exceedance - the honest phrasing
  is "the field baseline matches, and does not improve on, the plain
  self-gated arm".
- **The field baseline clearly beats the identified arm** (+0.246, 30/30,
  d_z +2.27) - the same ordering the n=4 partial predicted (+0.216, 4/4).

## 5. Independent replication of the DMC verdict

This file is a **second, independent 30-seed run** of the identified-vs-ordinary
gap on hopper-hop at 15k, on a disjoint arm set, and it reproduces the
2026-09-08 verdict: `identified_eq - ordinary` = **-0.255** here vs **-0.246**
in `runs/dmc_payoff_30seed_15k_gpu.json` (0/30 in both). The per-arm means agree
to ~0.01-0.05 (`ordinary` 0.961 vs 0.957; `identified_eq` 0.706 vs 0.711). The
practical-ordering failure of the identified arm on DMC is therefore **not a
single-run artifact**.

## 6. Registered endpoint 1 - rollout-gating quality (descriptive at 15k)

Mean over 30 seeds at the final gating record, from the arms' own logs:

| arm | `imagine_stopped_frac` | `imagine_mean_weight` | `imagine_mean_sqrt_u` |
|---|---:|---:|---:|
| `ordinary` | 0.110 | 0.607 | 0.3670 |
| `identified_eq` | 0.117 | 0.604 | 0.3677 |
| `macura_gate` | **0.154** | **0.850** | **0.0051** |

The MACURA gate is **active and materially different**: it stops ~1.4x more
imagined rollouts than ordinary self-gating, keeps them at a much higher mean
weight (0.850 vs 0.607), and its disagreement scale is ~70x smaller
(`sqrt(u_KL)` 0.0051 vs the UBE arms' 0.367) - two different statistics on
different scales driving the same rule. More aggressive stopping did **not**
buy fidelity: u_rank parity or slightly below. Rollout NFE and `length_mean`
are identical across arms (one-step imagination budget), so the gate changes
*which* imagined states are kept, not the sample count.

## 7. Registered endpoint 2 - downstream return: still unreadable at 15k

| contrast | mean | 95% CI | wins |
|---|---:|---|---:|
| `macura_gate - ordinary` | +0.070 | [-0.040, +0.213] | 13/30 |
| `macura_gate - identified_eq` | +0.070 | [-0.044, +0.202] | 15/30 |
| `identified_eq - ordinary` | +0.001 | [-0.059, +0.060] | 12/30 |

Every CI includes 0. Per-arm return medians sit at the hopper floor (0.0172 /
0.0299 / 0.0552), so the DMC floor rule applies and the return thread stays
**deferred**, exactly as the DMC 30-seed verdict registered. `macura_gate` has
the highest return *mean* (0.1735 vs 0.1031) and the lowest *median* - a
few-non-floor-seeds pattern, not a signal. **No MACURA payoff claim is made.**

## 8. Verdict in one line

The MACURA `u_KL` field baseline gates correctly and lands **at parity with
plain self-gated distillation** on hopper-hop (-0.009 u_rank, 9/30, d_z -0.38),
**well above the identified arm** (+0.246, 30/30), and its registered payoff
endpoint **remains unreadable at 15k** (all return CIs include 0, medians at the
floor) - so the DMC distillation ordering is unchanged by the field baseline,
and the payoff question needs the return-extension environment.

## 9. Caveats carried into the record

- `macura_gate`'s two `u`-scale columns are not comparable to the UBE arms'
  (different statistic, different units); the comparison is stop behaviour and
  fidelity, never raw magnitudes.
- The published-MACURA absolute numbers are not reproduced and are not
  compared against anywhere (section 1.1 of the registration); the deliberate
  deviations (per-round kappa, shared `stop_percentile` 0.85, soft weights
  alongside the hard stop, terminating transition stored as terminal) are
  applied identically to every arm and sit outside the adjudicated contrasts.
- u_rank is ceiling-bound for `ordinary`/`macura_gate` (30/30 >= 0.70), so the
  -0.009 contrast is measured in a compressed band; it is a null, not a loss.

## 10. Files

- `runs/dmc_macura_30seed_15k_gpu.json` (90 rows, host) / `runs_local/` mirror
- `runs/dmc_macura_30seed_15k_gpu.summary.txt` (host-captured summarizer output)
- Registration: `research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md`
- Adjudicator: `udwm/scripts/summarize_dmc_payoff.py --no-ctrl --no-bars`;
  effect sizes `udwm/scripts/effect_size_report.py`
- Predecessor partial read: `research/GRAND-STATUS-2026-09-10.md` section 2
