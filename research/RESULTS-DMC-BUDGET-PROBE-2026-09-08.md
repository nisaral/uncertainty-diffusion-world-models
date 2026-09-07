# DMC budget probe: result and pre-committed verdict (2026-09-08)

**Registration:** `research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`
(registered before any probe row ran; diagnostic only, NEVER adjudicates the
DMC payoff bars). **Data:** `runs/dmc_budget_probe_gpu.json` (12 rows,
seeds {0, 1} x the six amended arms, 15,000 env steps, exact per-seed teacher
checksum pairing, gap 0.0). **Host:** company VM (RTX 6000 Ada, 2 polite GPU
workers ~544 MiB each, no interference with co-resident workloads). **Config:**
`configs/dmc_hopper_probe.yaml` (identical to the DMC config except
`eval_freq: 3000`). Same seeds as the 3,600-step sanity rows
(`runs/dmc_payoff_10seed_gpu.json`), so the paired budget read needs no
re-run of the 3.6k operating point.

## Per-arm result at 15,000 steps (n=2 per arm; diagnostic)

| arm | u_rank s0 | u_rank s1 | u_rank mean | u_rank @3.6k mean | final_return s0 | s1 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 0.977 | 0.874 | 0.925 | 0.454 | 1.2232 | 0.0002 |
| hybrid | 0.765 | 0.726 | 0.746 | 0.487 | 0.0922 | 0.0606 |
| lagged_hybrid | 0.956 | 0.756 | 0.856 | 0.574 | 0.0535 | 0.0000 |
| identified_hybrid (EMA) | -0.176 | 0.440 | 0.132 | 0.012 | 0.0952 | 0.0000 |
| identified_eq | 0.766 | 0.755 | 0.760 | 0.420 | 0.0840 | 0.0000 |
| lagged_identified_eq | 0.603 | 0.665 | 0.634 | (not run @3.6k) | 0.0003 | 0.0009 |

In-training curves (eval_history in the rows) show the climb: ordinary seed 0
u_rank 0.510 @ 3k -> 0.656 @ 6k -> 0.804 @ 12k -> 0.835 @ 15k in-training ->
0.977 final eval; returns follow the same trajectory on seed 0 only.

## Pre-committed verdict: budget confound SUPPORTED

Decision rule (registered before rows existed): "If mean(seeds 0,1) u_rank
at 15k is >= 0.70 for BOTH ordinary and lagged_hybrid AND/OR paired
final_return (15k - 3.6k) rises clearly on both seeds -> budget confound
supported". Both baseline arms clear the u_rank condition:

| arm | u_rank @3.6k (seeds 0-1) | u_rank @15k (seeds 0-1) | >= 0.70 @15k |
|---|---:|---:|---|
| ordinary | 0.454 | 0.925 | yes (both seeds) |
| lagged_hybrid | 0.574 | 0.856 | yes (both seeds) |

The 3,600-step operating point (3.6 episodes of a 1,000-step task) was the
confound: at episode-parity with the DelayedBimodal studies (15 episodes)
the baseline arms climb out of the ~0.5 floor to the 0.86-0.93 band. The
registered 30-seed adjudication is therefore amended (Amendment 2,
`research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`) and must not run at
3,600 steps.

## Diagnostic reads at the adequate budget (never adjudicative at n=2)

- **identified_eq climbs above the registered bar on both seeds** (0.766 /
  0.755, mean 0.760 vs 0.420 @3.6k) and sits at/above `hybrid` (0.746) on
  both seeds - the DelayedBimodal headline contrast (eq > hybrid) points in
  the right direction once the floor is gone (it was a wash at 3.6k).
- **EMA collapse control reproduces** (identified_hybrid mean 0.132, seed 0
  negative) - the mechanism's transfer control behaves at the adequate
  budget.
- **lagged_identified_eq (0.634) does NOT top the table on DMC** - below
  identified_eq (0.760), lagged_hybrid (0.856) and ordinary (0.925). On
  DelayedBimodal this arm was the top configuration; at n=2 on DMC the
  ordering inverts. The 30-seed verdict is therefore genuinely informative:
  arms separate once the floor is gone, and the combined-fix arm's DMC
  standing is an open adjudication question, not a foregone ceiling.
- **Returns are still floor-bound on seed 1 at 15k** (0.0000-0.0009 for five
  of six arms) - the return thread cannot be adjudicated at 15k on this task
  (hopper locomotion needs substantially more real experience than
  episode-parity with a 2-D toy; an MBPO-scale DMC budget is beyond this
  repo's registered protocol). The 30-seed adjudication answers the
  u_rank mechanism-transfer question; the return payoff question is
  conditional on that verdict and needs a longer-budget extension.
- **Gate re-measured at 15k** from the probe ordinary rows (same seeds, same
  eval states): teacher g*/w* = 10,973 (seed 0) / 17,612 (seed 1), median
  14,292 - aleatoric-dominated regime confirmed at the amended operating
  point (was 8,224 at 3.6k); the pre-committed aleatoric branch
  (EMA collapse expected; equal-weight partial transfer; payoff question
  open) carries over unchanged.

## Consequence

Amendment 2 of the DMC preregistration is activated (15,000-step budget for
the gate + sanity + 30-seed adjudication; 3,600-step DMC rows are
superseded for adjudication; return thread deferred to a longer-budget
extension conditional on the mechanism verdict). See the amendment record in
`research/DMC-PAYOFF-PREREGISTRATION-2026-09-05.md`.

## Files

- Pre-registration: `research/DMC-BUDGET-PROBE-PREREGISTRATION-2026-09-07.md`.
- Data: `runs/dmc_budget_probe_gpu.json`.
- Curves: `python -m udwm.scripts.print_eval_history --data runs/dmc_budget_probe_gpu.json`.
