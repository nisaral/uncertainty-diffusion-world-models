# -*- coding: utf-8 -*-
"""Hypothesis D desk check: does the M=2 estimator variance blow up with dims?

Hypothesis D (secondary, weaker grounding) claims the M>=2 debiased estimator's
practical variance at fixed M=2 scales unfavourably with state/action
dimensionality, and that Hopper's (15/4) vs DelayedBimodal's (5/1) gap swamps
the correction's benefit.

Per the project plan this is a ~20-minute desk check, not a registered
experiment: plug the actual operating point into the repo's *own* measured
balance-window theory and see whether the predicted noise alone can explain the
environment-specific gap.  Escalate to a real probe only if it is suggestive.

The repo's formula (theory/identified_balance_window.py, docstring):

    std(g_hat)/g*  ~  sqrt(2 / (N (M - 1)))      (Gaussian within-member)
    std(w_hat)/w*  ~  sqrt(A_w / M) / w*         (A_w from ESTIMATOR-VARIANCE)

Neither expression contains obs_dim or action_dim: at fixed (N, M) the
predicted *relative* estimator noise is an environment-independent constant.
That is the whole desk check.  This script prints the numbers, contrasts them
with the observed environment-specific eq-vs-ordinary u-rank gaps, and writes
the verdict.

Run: python theory/hypothesis_d_dimension_desk_check.py  (CPU, <5 s)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Registered operating point: model_batch_size 128, M=2, in both configs
# (configs/delayed_bimodal_distill.yaml and configs/dmc_hopper_probe.yaml).
REGISTERED_N = 128
REGISTERED_M = 2

# Observed eq - ordinary u-rank gaps (headline, already adjudicated).
OBSERVED = {
    # DelayedBimodal, corrected-weight N=10 (RESULTS-CORRECTED-WEIGHT-POLICY-2026-09-05)
    "delayed_bimodal_eq_minus_ordinary": -0.104,
    # DMC/hopper-hop, 30-seed adjudication (RESULTS-DMC-30SEED-ADJUDICATION-2026-09-08)
    "dmc_eq_minus_ordinary": -0.246,
}

DIMS = {"delayed_bimodal": {"obs_dim": 5, "action_dim": 1},
        "dmc_hopper_hop": {"obs_dim": 15, "action_dim": 4}}


def main() -> int:
    rows = []
    for N in (32, 64, 128, 256, 512):
        for M in (2, 4, 8):
            rows.append({
                "N": N, "M": M,
                "pred_std_g_rel": math.sqrt(2.0 / (N * (M - 1))),
            })

    at_point = math.sqrt(2.0 / (REGISTERED_N * (REGISTERED_M - 1)))
    # The measured fit at M=2 from theory/identified_balance_window.py is 1.35
    # for w (w* << g* regime); the closed form above is the g channel.
    w_rel_measured_m2 = 1.35

    # Sensitivity: how much would dims have to enter to close the DMC gap?
    # Solve sqrt(2/(N_eff (M-1))) = |gap| with gap in *rank* units is not a
    # legitimate inversion (rank != channel noise), so we state it as: the
    # formula would need an N_eff inflation of (1/gap)^2 * at_point^2 to reach
    # the observed u-rank deficit, i.e. the dimensionality channel is not a
    # few-percent perturbation but an order-of-magnitude one, and it would have
    # to be present on DMC and absent on DB.

    payload = {
        "registered_operating_point": {"N": REGISTERED_N, "M": REGISTERED_M},
        "predicted_relative_g_noise_at_point": at_point,
        "measured_relative_w_noise_at_M2": w_rel_measured_m2,
        "formula_contains_dims": False,
        "dims": DIMS,
        "grid": rows,
        "observed_eq_minus_ordinary": OBSERVED,
        "verdict": (
            "NOT SUGGESTIVE. The repo's balance-window prediction is "
            "dimension-free at fixed (N, M): it predicts the same "
            f"{at_point:.3f} relative channel noise on DelayedBimodal (5/1) and "
            "DMC (15/4). A dimension-independent noise term cannot produce an "
            "environment-specific deficit (-0.104 DB vs -0.246 DMC), so the "
            "stated Hypothesis-D mechanism (dims swamp the correction) is not "
            "supported by the repo's own theory at the registered operating "
            "point. The only surviving channel is non-Gaussian / heavy-tailed "
            "constants in the O(1/sqrt(N)) rate, which this formula does not "
            "model; that would need a real probe and is NOT predicted. "
            "Pre-committed consequence: do not escalate Hypothesis D to a "
            "registered run on the strength of this check."
        ),
    }
    out = ROOT / "runs" / "hypothesis_d_desk_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"registered point N={REGISTERED_N} M={REGISTERED_M}")
    print(f"predicted std(g_hat)/g* = {at_point:.4f} (same on both envs)")
    print("M-scaling (N=128):", {M: round(math.sqrt(2.0 / (128 * (M - 1))), 4)
                                 for M in (2, 4, 8, 16)})
    print("observed eq - ordinary u-rank: DB %+.3f, DMC %+.3f"
          % (OBSERVED["delayed_bimodal_eq_minus_ordinary"],
             OBSERVED["dmc_eq_minus_ordinary"]))
    print("formula contains dims:", payload["formula_contains_dims"])
    print("VERDICT:", payload["verdict"].split(".")[0] + ".")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
