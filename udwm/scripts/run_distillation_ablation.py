"""Matched-compute ordinary versus geometry-preserving distillation study.

This is intentionally a model/uncertainty experiment before the expensive RL
study. Both variants get the same environment data, teacher pretraining budget,
student budget, architecture, and estimator budget. The only changed factor is
the ensemble-level preservation weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.eval.metrics import evaluate_distillation_uncertainty
from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def _make_cfg(base, variant: str, seed: int, steps: int):
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["mbpo"]["total_env_steps"] = int(steps)
    cfg["model"]["use_consistency_distill"] = True
    cfg["model"]["freeze_teacher"] = False
    cfg["model"]["preserve_distilled_uncertainty"] = True
    if variant == "ordinary":
        cfg["model"]["distill_mean_weight"] = 0.0
        cfg["model"]["distill_geometry_weight"] = 0.0
        cfg["model"]["distill_pairwise_weight"] = 0.0
    elif variant == "geometry":
        cfg["model"]["distill_mean_weight"] = 1.0
        cfg["model"]["distill_geometry_weight"] = 1.0
        cfg["model"]["distill_pairwise_weight"] = 1.0
    else:
        raise ValueError(variant)
    return cfg


def _run(base, variant: str, seed: int, collect_steps: int, teacher_epochs: int, student_epochs: int, m: int):
    cfg = _make_cfg(base, variant, seed, collect_steps)
    set_seed(seed)
    trainer = MBPOTrainer(cfg)
    for _ in range(collect_steps):
        trainer._env_step()

    # Stage 1: identical teacher optimization budget and optimizer scope.
    trainer.wm_opt = torch.optim.Adam(trainer.world_model.teacher.parameters(), lr=1e-3)
    for _ in range(teacher_epochs):
        trainer._train_world_model()

    teacher_state = copy.deepcopy(trainer.world_model.teacher.state_dict())
    trainer.world_model.freeze_teacher()
    trainer.wm_opt = torch.optim.Adam(trainer.world_model.student.parameters(), lr=1e-3)
    for _ in range(student_epochs):
        trainer._train_world_model()

    # Fixed, shared value map isolates transition uncertainty from SAC learning.
    def q_fn(obs, actions):
        return obs[..., :1] + 0.1 * actions.pow(2).sum(dim=-1, keepdim=True)

    def policy_fn(obs):
        return torch.zeros(obs.shape[0], trainer.action_dim, device=obs.device)

    report = evaluate_distillation_uncertainty(
        trainer.world_model, q_fn, policy_fn, trainer.real_buffer,
        batch_size=min(256, len(trainer.real_buffer)), n_batches=4, m_samples=m,
    )
    report.update({
        "variant": variant,
        "seed": int(seed),
        "collect_steps": int(collect_steps),
        "teacher_epochs": int(teacher_epochs),
        "student_epochs": int(student_epochs),
        "teacher_parameters": float(sum(p.numel() for p in trainer.world_model.teacher.parameters())),
        "teacher_frozen": all(not p.requires_grad for p in trainer.world_model.teacher.parameters()),
        "teacher_state_checksum": float(sum(v.float().abs().sum() for v in teacher_state.values())),
    })
    return report


def summarize(rows):
    """Return an evidence-oriented summary, without pretending to prove novelty."""
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["variant"]] = row
    comparisons = []
    for seed, pair in by_seed.items():
        if "ordinary" not in pair or "geometry" not in pair:
            continue
        ordinary, geometry = pair["ordinary"], pair["geometry"]
        comparisons.append({
            "seed": seed,
            "delta_u_rank_corr_geometry_minus_ordinary": geometry["u_rank_corr"] - ordinary["u_rank_corr"],
            "delta_w_rank_corr_geometry_minus_ordinary": geometry["w_rank_corr"] - ordinary["w_rank_corr"],
            "delta_u_rmse_geometry_minus_ordinary": geometry["u_rmse"] - ordinary["u_rmse"],
        })
    if not comparisons:
        return {"status": "incomplete", "comparisons": []}
    mean_rank = float(np.mean([x["delta_u_rank_corr_geometry_minus_ordinary"] for x in comparisons]))
    mean_rmse = float(np.mean([x["delta_u_rmse_geometry_minus_ordinary"] for x in comparisons]))
    # Tiny RMSE deltas are noise. Rank correlation is the primary decision metric.
    if mean_rank > 0.05 and mean_rmse <= 0.02:
        verdict = "supports_preservation_hypothesis"
    elif mean_rank < -0.05 and mean_rmse >= 0.02:
        verdict = "weakens_or_falsifies_preservation_hypothesis"
    else:
        verdict = "inconclusive_requires_more_seeds_or_better_metric"
    return {
        "status": "complete",
        "verdict": verdict,
        "mean_delta_u_rank_corr": mean_rank,
        "mean_delta_u_rmse": mean_rmse,
        "comparisons": comparisons,
        "interpretation": (
            "This is evidence about the empirical preservation hypothesis, not a proof of literature novelty. "
            "Control and matched-compute experiments remain necessary."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "consistency_distill.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--collect-steps", type=int, default=800)
    p.add_argument("--teacher-epochs", type=int, default=5)
    p.add_argument("--student-epochs", type=int, default=10)
    p.add_argument("--m-samples", type=int, default=4)
    p.add_argument("--out", default="runs/distillation_ablation.json")
    args = p.parse_args(argv)
    base = load_config(args.config)
    rows = []
    for seed in args.seeds:
        for variant in ("ordinary", "geometry"):
            row = _run(base, variant, seed, args.collect_steps, args.teacher_epochs, args.student_epochs, args.m_samples)
            rows.append(row)
            print(json.dumps(row, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "summary": summarize(rows)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
