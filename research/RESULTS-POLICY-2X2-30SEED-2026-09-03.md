# Policy 2x2, N=30 adjudicated rerun (2026-09-03)

**Data:** `runs/policy_identifiability_2x2_30seed.json` (written 2026-09-03).
**Registration:** `research/DECISION-TREE-2X2-PREREGISTRATION.md`, including the
N=30 extension amendment registered on 2026-09-03 **before** any verdict from
the 10-seed rerun was read or reported.
**Runner:** `udwm/scripts/run_delayed_bimodal_policy_ablation.py`; executed as
one subprocess per seed via `udwm/scripts/run_policy_2x2_split_seeds.py`
(per-seed batches, merged; exact teacher pairing holds by construction per
seed). 30 seeds x 5 arms x 1,800 env steps, DelayedBimodal-v0, CPU. All 30
seeds report `exact_teacher_match: true`, `max_teacher_checksum_gap: 0.0`.
**Summary tool:** `udwm/scripts/summarize_policy_2x2.py` (mean paired delta,
wins/N, 5,000-draw percentile bootstrap 95% CI).

## Verdict (pre-registered decision tree, adjudicated on the pooled N=30)

Primary endpoint `u_rank_corr`, paired per-seed deltas:

| Contrast | Mean delta | Seeds in direction | Bootstrap 95% | Verdict |
|---|---:|---:|---|---|
| `hybrid` - `ordinary` (reproduction) | -0.3100 | 0/30 | [-0.3401, -0.2808] | confirmed: collapse |
| `lagged_hybrid` - `hybrid` | +0.3310 | 30/30 | [+0.3048, +0.3578] | **confirmed: lagging fixes u-rank** |
| `identified_hybrid` - `hybrid` (live critic) | -0.4637 | 0/30 | [-0.5068, -0.4196] | confirmed: NOT a fix (harm) |
| `lagged_identified` - `hybrid` | -0.3533 | 1/30 | [-0.4120, -0.2965] | confirmed: not a fix |
| `lagged_identified` - `identified_hybrid` | +0.1104 | 20/30 | [+0.0538, +0.1675] | inconclusive (66.7% < 70% bar) |

**Decision tree outcome: Row 1.** `lagged_hybrid` is confirmed to restore the
teacher-student uncertainty rank to ~ordinary level (0.941 vs 0.920), and
`identified_hybrid` is **not** confirmed under the live critic -- in fact it is
confirmed-harmful (0/30, interval excludes zero). Nonstationarity is the
dominant measured mechanism; the identifiability pathology stands as a
theoretical result (proof + toy) with no measured policy benefit on this
benchmark. The payoff test (#4) should test lagging alone, not the identified
machinery.

## Per-arm means (n=30)

| arm | u_rank_corr | w_rank_corr | u_rmse | w_rmse | next_state_mse | final_return | selective_rank_corr | selective_recall_bad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 0.9197 | 0.2153 | 45.34 | 0.5705 | 0.2719 | -104.39 | -0.0912 | 0.3096 |
| hybrid | 0.6097 | 0.2244 | 160.91 | 0.0802 | 0.1902 | -116.32 | -0.1122 | 0.3155 |
| lagged_hybrid | **0.9408** | 0.1971 | **34.37** | 0.2529 | 0.2692 | -130.44 | -0.0291 | **0.3858** |
| identified_hybrid | 0.1461 | **0.2911** | 157.76 | **0.0162** | **0.1630** | **-84.80** | -0.1469 | 0.2319 |
| lagged_identified | 0.2564 | 0.2719 | 157.75 | 0.0158 | 0.1603 | -94.50 | -0.1099 | 0.2765 |

## What is confirmed at N=30 (secondary endpoints)

- **Hybrid vs ordinary (reproduction of the published 10-seed study).**
  u-rank collapse -0.310, 0/30 (published: -0.297, 0/10); u-rmse worse 30/30;
  next-state MSE better 30/30 (-0.0817); w-rmse better 30/30. The falsification
  ("MSE can improve while the decision object dies") is now 30/30, not 10/10.
- **Lagging fixes the collapse.** `lagged_hybrid` u-rank +0.331, 30/30,
  restoring rank to ~ordinary; u-rmse -126.5, 30/30. Costs: w-rmse worse 30/30
  (+0.173), next-state MSE worse 30/30 (+0.079), return inconclusive.
- **Identified arms (M>=2, EMA-reweighted) do not fix -- they destroy the U
  object.** `identified_hybrid` u-rank -0.464, 0/30; `lagged_identified` -0.353,
  1/30. `w_rmse` becomes tiny (0.016, best in table) and `next_state_mse` best,
  but `student_u_mean` collapses to ~-0.003 versus teacher -50..-105: the
  propagated U field is ~0 everywhere, so rank is noise. See mechanism note.
- **Return (secondary; does not adjudicate).** `identified_hybrid` return
  +31.5 vs hybrid (23/30, CI excludes 0) and best per-arm mean (-84.8 vs
  ordinary -104.4). `lagged_hybrid` return is not better than ordinary (-130.4).
  At this horizon nothing decision-related pays off in return; that is exactly
  why the payoff test is a separate (#4) question.
- **Selective.** `lagged_hybrid` improves selective rank/recall vs hybrid
  (CI excludes 0, 19/30 each -> inconclusive at the 70% bar); `identified`
  harms selective bad-recall vs hybrid (9/30, CI excludes 0) -- consistent with
  the U collapse.

## Mechanism note: why the reweighted identified arm collapses U

With `distill_reweight_ema: true`, the identified loss matches the local
epistemic and aleatoric statistics separately and (on this benchmark) very
hard: `w_rmse` 0.016 (teacher scale ~1e-4 epistemic vs ~0.15 aleatoric). The
local decision statistic is u = w - g; when g* >> w* the matched local u is
<= 0 and lands on the `u_min = 0` floor (implementation assumption I1), so the
propagated U collapses to ~0 and the teacher-student rank becomes noise. This
is an implementation-level failure mode of the identified arm as configured,
distinct from the identifiability argument (which is about the loss target, and
is proven in the toy). The ground-truth toy's P4 "w-ranking is robust to the
weighting" is toy-level: there the member term anchors on x0, while in the repo
w lives on values, so the anchor is weaker. State this caveat with the result;
it does not rescue the identified loss in policy on this benchmark.

## Registration hygiene

- N=30 extension registered (2026-09-03) before the pooled adjudication was
produced; thresholds fixed: CI excludes 0 AND >= 70% of paired seeds in the
predicted direction to confirm; >= 70% against to confirm-absent; else
inconclusive.
- The 10-seed slice (seeds 0-9, uniform thread config) is included in the
pooled file; seed 9 was re-run at the same thread setting as the rest of the
table (it had first run as a single-seed crash probe at default threads).
- All five arms ran per seed with one prepared teacher (checksum gap 0); no
arm dropped post-hoc.

## Correction (2026-09-05): the mechanism note above mis-described the eval object

The "lands on the `u_min = 0` floor" sentence above is **not** what the eval
endpoint measures: `evaluate_distillation_uncertainty` uses
`MCUBELocalRewards(u_min=-1e9, ...)`, so the reported `u` is unclamped. The
measured mechanism (probe data `runs/probe_u_*.json`, write-up
`U-COLLAPSE-MECHANISM-2026-09-05.md`) is: the local object on this benchmark is
aleatoric-dominated (teacher u ~ -g with g ~ 73-90 vs w ~ 0.005); identified
arms match the epistemic half (student w ~ teacher w, w_rmse best in table)
but leave the aleatoric half at initialization (student g ~ 0.01, student
latent-to-state spread ~0.006 vs teacher ~0.37), so u_student ~ w_student ~ 0.
The aleatoric term cannot lift g: measured per-update leverage ~2e-6, ~100x
below its 2-latent estimator noise and ~1e6-1e7 steps short of the gap, in a
run with 180 student updates; EMA reweighting then down-weights the aleatoric
gradient by a further ~1e5-1e6. Not EMA lag (lagged_identified collapses
identically), not the reweighting (equal-weight identified collapses), not the
u-floor. The verdicts of this table are unchanged.

## Addendum (2026-09-05): the identified rows were the EMA-reweighted variant

The identified arms in this table ran with the per-term EMA reweighting on
(config-override bug; see `CORRUPTION-PROBE-PREREGISTRATION-2026-09-05.md`
and `RESULTS-CORRUPTION-2026-09-05.md`). The postscript above was written
under that config and its "not the reweighting (equal-weight identified
collapses)" sentence is superseded. The registered corrected-weight policy
N-study (`research/RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05.md`, N=10)
shows the equal-weight identified arm transfers under the live critic:
u-rank 0.835 (9/10 >= 0.70, pre-registered bars A met), +0.214 vs hybrid
(10/10), +0.720 vs identified_hybrid (10/10), -0.104 vs ordinary. Row 1 of
the decision tree (lagging fixes the collapse) stands; the "identified loss
does not recover u-rank (0/30)" cell is specific to the EMA variant, and
the equal-weight identified arm does recover u-rank at N=10.
