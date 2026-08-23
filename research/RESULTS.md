# Result index

| Study | Status | File |
|---|---|---|
| Large controlled stress (20 seeds, fixed value map) | Hybrid RMSE 20/20; rank 13/20; recall inconclusive | [RESULTS-STRESS-LARGE-2026-08-21.md](RESULTS-STRESS-LARGE-2026-08-21.md) |
| Policy-scale delayed-bimodal (10 seeds, live critic) | Hybrid MSE 10/10; uncertainty rank/RMSE **0/10**; return 3/10 | [RESULTS-POLICY-SCALE-2026-08-22.md](RESULTS-POLICY-SCALE-2026-08-22.md) |
| Earlier policy probe | Inconclusive / negative transfer | [RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md](RESULTS-POLICY-DELAYED-BIMODAL-2026-08-21.md) |
| Earlier distillation table | Superseded by the 20-seed study | [RESULTS-DISTILLATION-2026-08-20.md](RESULTS-DISTILLATION-2026-08-20.md) |

**Read these two as a pair.** The first is the positive narrow result. The
second is the negative transfer result. The next method is lagged target-critic
distillation (`configs/lagged_target_distill.yaml`), not another ordinary vs
geometry Pendulum run.
