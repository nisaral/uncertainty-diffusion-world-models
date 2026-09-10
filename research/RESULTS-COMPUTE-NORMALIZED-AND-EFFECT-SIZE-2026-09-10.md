# Compute-normalized re-analysis + effect-size/power report (2026-09-10)

Two long-outstanding zero-GPU items from the master-plan-v3 script specs (items
2 and 5a). Neither runs new training: both re-index or re-report the
already-adjudicated rows. **No verdict is changed and no bar is re-opened.**

- Item 2 registration: `research/COMPUTE-NORMALIZED-REANALYSIS-2026-09-09.md`
  (question and normalization rule fixed before results were looked at).
- Item 5a is a utility (`udwm/scripts/effect_size_report.py`), additive to the
  existing summarizers.

---

## A. Compute-normalized re-analysis

Script: `udwm/scripts/reanalyze_compute_normalized.py`.
Inputs: `runs/dmc_payoff_30seed_15k_gpu.json` (180 rows = 30 seeds x 6 arms),
`runs/policy_combined_fix_10seed.json` (DelayedBimodal).

### A1. Cost multipliers, derived from the loss code (not assumed)

Read from `udwm/models/consistency.py` (ensemble `N=5`, `M=2`); teacher-member
diffusion forwards per model-train **epoch**, and the critic `value_fn` state
evaluations that accompany them:

| arm | teacher fwd/epoch | value_fn calls | multiplier |
|---|---|---|---|
| `ordinary` | 1 | 0 | **1x** |
| `hybrid`, `lagged_hybrid` | 5 | 2 | **5x** |
| `identified_eq`, `identified_hybrid`, `lagged_identified_eq` | 10 | 4 | **10x** |

So at the same 15,000 env steps the identified arms spent **10x** the
teacher-sample budget of `ordinary` and **2x** the hybrid family's. The
"teacher-sample unit" below is one `ordinary` epoch's teacher cost (1 teacher +
1 student diffusion forward on a 128-state minibatch).

### A2. DMC / hopper-hop, 30 seeds, u_rank at matched eval milestones

| arm | final u_rank | cumulative spend | u_rank per 1000u | curve |
|---|---|---|---|---|
| `lagged_hybrid` | 0.852 | 8,880u | 0.096 | 0.524@1680 0.621@3480 0.706@5280 0.800@7080 0.852@8880 |
| `ordinary` | 0.954 | 1,776u | **0.537** | 0.574@336 0.605@696 0.794@1056 0.907@1416 0.954@1776 |
| `hybrid` | 0.718 | 8,880u | 0.081 | 0.493@1680 0.457@3480 0.513@5280 0.659@7080 0.718@8880 |
| `identified_eq` | 0.712 | 17,760u | 0.040 | 0.471@3360 0.429@6960 0.456@10560 0.609@14160 0.712@17760 |
| `lagged_identified_eq` | 0.624 | 17,760u | 0.035 | 0.532@3360 0.601@6960 0.638@10560 0.648@14160 0.624@17760 |
| `identified_hybrid` | -0.064 | 17,760u | -0.004 | 0.007@3360 -0.032@6960 -0.122@10560 -0.148@14160 -0.064@17760 |

### A3. DelayedBimodal, 10 seeds, same normalization

| arm | final u_rank | cumulative spend | u_rank per 1000u |
|---|---|---|---|
| `ordinary` | 0.939 | 192u | **4.890** |
| `lagged_hybrid` | 0.955 | 960u | 0.995 |
| `lagged_identified_eq` | 0.947 | 1,920u | 0.493 |
| `identified_eq` | 0.858 | 1,920u | 0.447 |
| `hybrid` | 0.629 | 960u | 0.655 |
| `identified_wonly` | 0.147 | 1,920u | 0.076 |
| `identified_hybrid` | 0.107 | 1,920u | 0.056 |

### A4. Verdict - the ordering does not change, and eq's deficit does not narrow

Re-indexed against teacher-sample spend rather than env steps:

- **DMC:** `ordinary` reaches 0.954 at 1,776u; `identified_eq` needs 17,760u to
  reach 0.712. On a per-unit basis eq is **~13x less compute-efficient than
  ordinary** (0.040 vs 0.537 u_rank/1000u) and ~2x less than `hybrid`.
- **DelayedBimodal:** the same direction. `identified_eq` (0.858 at 1,920u) does
  beat `hybrid` (0.629 at 960u) in *absolute* u_rank (the registered headline),
  but it spends 2x the compute to do it, so per-unit it sits **below** hybrid
  (0.447 vs 0.655). `ordinary` is ~10x more efficient than either.

So the pre-registered question - "does eq's apparent underperformance at a fixed
step count narrow when compared at fixed compute?" - is answered **no** on both
environments. Compute normalization, if anything, widens the gap, because the
identified objective pays 10x for its per-step u_rank.

**Important limitation, stated plainly:** the fixed-compute head-to-head the
registration actually asked for is **not readable from these rows.** All arms ran
the same *env-step* budget (15k), not the same *teacher-sample* budget, so the
common teacher-sample domain across all arms is empty (DMC: `ordinary` totals
1,776u while the decision arms' first eval milestone is 3,360u). This table is a
*re-indexing of the existing curves*, which is what the methods note registered;
a genuine fixed-compute comparison requires re-running `ordinary` to a matched
spend (~10x more env steps) and is not claimed here.

---

## B. Effect-size + power report (item 5a)

Script: `udwm/scripts/effect_size_report.py`; endpoint `u_rank_corr`, paired
per-seed deltas, 10^5 bootstrap draws, two-sided alpha 0.05, power 0.80. Additive
to the raw delta + CI + wins/N that `summarize_dmc_payoff.py` already reports.

### B1. Paired standardized effects, DMC 30 seeds

| contrast | mean | **d_z** | 95% CI | wins |
|---|---|---|---|---|
| `ordinary - EMA` | +1.034 | **+8.01** | [+0.988, +1.079] | 30/30 |
| `eq - EMA` | +0.788 | **+5.33** | [+0.736, +0.840] | 30/30 |
| `hybrid - EMA` | +0.807 | **+4.89** | [+0.750, +0.866] | 30/30 |
| `lagged_eq - ordinary` | -0.320 | **-3.74** | [-0.350, -0.289] | 0/30 |
| `eq - ordinary` | -0.246 | **-2.63** | [-0.278, -0.212] | 0/30 |
| `hybrid - ordinary` | -0.226 | -1.75 | [-0.274, -0.183] | 0/30 |
| `lagged_hybrid - ordinary` | -0.121 | -1.45 | [-0.150, -0.092] | 0/30 |
| `eq - lagged_hybrid` | -0.125 | -0.98 | [-0.170, -0.080] | 4/30 |
| `lagged_hybrid - hybrid` | +0.106 | +0.79 | [+0.059, +0.153] | 22/30 |
| `lagged_eq - eq` | -0.073 | -0.68 | [-0.111, -0.036] | 7/30 |
| `lagged_eq - hybrid` | -0.093 | -0.66 | [-0.142, -0.043] | 8/30 |
| `eq - hybrid` | -0.020 | **-0.14** | [-0.067, +0.030] | 13/30 |

### B2. Per-arm cross-seed reference variance (N=30)

| arm | mean | sd |
|---|---|---|
| `ordinary` | +0.9573 | 0.0180 |
| `lagged_hybrid` | +0.8367 | 0.0927 |
| `hybrid` | +0.7310 | 0.1361 |
| `identified_eq` | +0.7112 | 0.0892 |
| `lagged_identified_eq` | +0.6378 | 0.0854 |
| `identified_hybrid` | -0.0763 | 0.1224 |

### B3. MDE at planned seed counts (same endpoint variance)

| contrast | n=10 | n=30 |
|---|---|---|
| `eq - ordinary` | 0.083 | 0.048 |
| `lagged_eq - ordinary` | 0.076 | 0.044 |
| `lagged_hybrid - hybrid` | 0.118 | 0.068 |
| `eq - hybrid` | 0.122 | 0.071 |
| `eq - EMA` | 0.131 | 0.076 |

### B4. Reading

- **The headline deficit is enormous, not marginal.** `eq - ordinary` is
  d_z = **-2.63** - roughly 3x the n=30 MDE (0.048) and 5x the n=10 MDE (0.083).
  The DMC failure of the identified arm is not a power problem.
- **`eq - hybrid` is a genuine wash, now on a standardized scale:** d_z = -0.14
  with the CI straddling zero. This corroborates the registered Bar-2 "wash"
  verdict with a measure that does not depend on the raw-unit scale.
- **The mechanism contrast is the largest effect in the study:** `ordinary - EMA`
  d_z = +8.0 and `eq - EMA` d_z = +5.3, both 30/30. The confirmed mechanism
  (EMA-collapse) is an order of magnitude more decisive than any of the
  practical-ordering contrasts.
- **Retroactive power for the CRN-bias E1 read (closes the item-5a gap for that
  study).** With a comparable cross-seed delta SD (~0.09-0.13 on a u_rank-scaled
  endpoint), n=10 could only detect effects of ~0.08-0.12 and n=30 of ~0.05-0.07.
  The CRN E1 deltas were reported *below* that floor, which is exactly why the
  CRN readout called the n=10 result uninformative rather than negative. The
  same caution applies to the n=10 MDE table registered for Hypothesis C's E1/E2
  (`research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md`, section 7).

---

## Files

- `udwm/scripts/reanalyze_compute_normalized.py` (item 2, run 2026-09-10)
- `udwm/scripts/effect_size_report.py` (item 5a, run 2026-09-10)
- `research/COMPUTE-NORMALIZED-REANALYSIS-2026-09-09.md` (item 2 registration)
- Data: `runs/dmc_payoff_30seed_15k_gpu.json`, `runs/policy_combined_fix_10seed.json`
  (row files are git-ignored; both tools print their tables to stdout and write
  no artifact)
