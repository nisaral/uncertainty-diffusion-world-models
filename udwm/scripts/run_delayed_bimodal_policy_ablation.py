"""Matched policy-level test of ordinary, state, and hybrid distillation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


VARIANTS = {
    "ordinary": {
        "distill_decision_weight": 0.0,
        "distill_value_variance_weight": 0.0,
        "distill_hybrid_state_weight": 0.0,
        "distill_hybrid_pairwise_weight": 0.0,
        "distill_geometry_weight": 0.0,
        "distill_pairwise_weight": 0.0,
    },
    "state_geometry": {
        "distill_decision_weight": 0.0,
        "distill_value_variance_weight": 0.0,
        "distill_hybrid_state_weight": 0.0,
        "distill_hybrid_pairwise_weight": 0.0,
        "distill_geometry_weight": 1.0,
        "distill_pairwise_weight": 1.0,
    },
    "hybrid": {
        "distill_decision_weight": 1.0,
        "distill_value_variance_weight": 1.0,
        "distill_hybrid_state_weight": 1.0,
        "distill_hybrid_pairwise_weight": 1.0,
        "distill_use_target_critic": False,
        "distill_normalize_values": False,
        "distill_guard_enabled": False,
    },
    "lagged_hybrid": {
        "distill_decision_weight": 1.0,
        "distill_value_variance_weight": 1.0,
        "distill_hybrid_state_weight": 1.0,
        "distill_hybrid_pairwise_weight": 1.0,
        "distill_use_target_critic": True,
        "distill_normalize_values": True,
        "distill_guard_enabled": True,
        "distill_value_warmup_updates": 50,
    },
    # --- identifiability arms -------------------------------------------------
    # The 2x2 that separates nonstationarity from identifiability:
    #   {live-Q, lagged-Q} x {M=1 conflated, M>=2 identified}
    # `hybrid` and `lagged_hybrid` above are the M=1 column.
    "identified_hybrid": {
        # Live critic -- deliberately the FALSIFIED condition -- with the
        # identified objective. If uncertainty rank survives here, the collapse
        # was an identifiability failure, not a nonstationarity failure.
        # NOTE (2026-09-01): reweight_ema fixes the equal-weighting hole
        # (g dominates w by orders of magnitude); see theory/ground_truth_w_g.py.
        "distill_decision_weight": 1.0,
        "distill_value_variance_weight": 1.0,
        "distill_identified": True,
        "distill_m_latents": 2,
        "distill_aleatoric_weight": 1.0,
        "distill_reweight_ema": True,
        "distill_use_target_critic": False,
        "distill_normalize_values": False,
        "distill_guard_enabled": False,
    },
    "lagged_identified": {
        # Both corrections together: the intended method.
        "distill_decision_weight": 1.0,
        "distill_value_variance_weight": 1.0,
        "distill_identified": True,
        "distill_m_latents": 2,
        "distill_aleatoric_weight": 1.0,
        "distill_reweight_ema": True,
        "distill_use_target_critic": True,
        "distill_normalize_values": True,
        "distill_guard_enabled": True,
        "distill_value_warmup_updates": 50,
    },
}


def make_cfg(base, variant, seed, steps):
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["mbpo"]["total_env_steps"] = int(steps)
    epochs = int(cfg["mbpo"]["num_model_epochs"])
    warmup = int(cfg["mbpo"]["warmup_steps"])
    available_calls = (
        0 if int(steps) < warmup else
        1 + (int(steps) - warmup) // int(cfg["mbpo"]["model_train_freq"])
    )
    # Reserve at least one model-training call for student distillation.
    teacher_calls = max(1, min(3, available_calls - 1))
    cfg["model"]["distill_teacher_pretrain_updates"] = teacher_calls * epochs
    cfg["model"].update(VARIANTS[variant])
    return cfg


def summarize(rows):
    output = {}
    for variant in VARIANTS:
        rs = [r for r in rows if r["variant"] == variant]
        if not rs:
            continue
        output[variant] = {}
        for key in ("final_return", "selective_rank_corr", "selective_recall_bad", "next_state_mse"):
            vals = np.asarray([r.get(key, np.nan) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            output[variant][key] = {
                "mean": float(vals.mean()) if vals.size else None,
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "n": int(vals.size),
            }
    return output


def prepare_matched_teacher(base, seed: int, steps: int):
    """Collect one shared replay set and fit one teacher per seed."""
    cfg = make_cfg(base, "ordinary", seed, steps)
    cfg["mbpo"]["warmup_steps"] = min(int(cfg["mbpo"]["warmup_steps"]), int(steps))
    set_seed(seed)
    trainer = MBPOTrainer(cfg)
    collect_steps = max(int(steps), int(cfg["mbpo"]["model_batch_size"]))
    for _ in range(collect_steps):
        trainer._env_step()
    bs = int(cfg["mbpo"]["model_batch_size"])
    opt = torch.optim.Adam(trainer.world_model.teacher.parameters(), lr=1e-3)
    updates = int(cfg["model"].get("distill_teacher_pretrain_updates", 1))
    for _ in range(max(1, updates)):
        batch = trainer.real_buffer.sample(bs)
        loss = trainer.world_model.teacher.nll_loss(
            batch["obs"], batch["actions"], batch["next_obs"], batch["rewards"], batch["dones"]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return copy.deepcopy(trainer.real_buffer), copy.deepcopy(trainer.world_model.teacher.state_dict())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delayed_bimodal_distill.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--out", default="runs/delayed_bimodal_policy_ablation.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None,
                        help="override config device (cpu/cuda); default: config value")
    args = parser.parse_args(argv)
    base = load_config(args.config)
    if args.device is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        base["device"] = args.device
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.resume and out.exists():
        rows = json.loads(out.read_text(encoding="utf-8")).get("rows", [])
    completed = {(int(r["seed"]), r["variant"]) for r in rows}
    for seed in args.seeds:
        pending = [v for v in args.variants if (int(seed), v) not in completed]
        if not pending:
            continue
        prepared_buffer, prepared_teacher = prepare_matched_teacher(base, seed, args.steps)
        for variant in pending:
            cfg = make_cfg(base, variant, seed, args.steps)
            set_seed(seed)
            trainer = MBPOTrainer(cfg)
            trainer.real_buffer = copy.deepcopy(prepared_buffer)
            trainer.world_model.teacher.load_state_dict(prepared_teacher)
            trainer.world_model.freeze_teacher()
            trainer.wm_opt = torch.optim.Adam(trainer.world_model.student.parameters(), lr=1e-3)
            result = trainer.train()
            metrics = result["final_metrics"]
            row = {
                "seed": seed,
                "variant": variant,
                "final_return": result["final_eval_return"],
                "teacher_frozen": bool(getattr(trainer.world_model, "teacher_frozen", False)),
                "teacher_updates": int(cfg["model"].get("distill_teacher_pretrain_updates", 0)),
                "teacher_initial_checksum": getattr(trainer, "_teacher_initial_checksum", None),
                "teacher_final_checksum": (
                    trainer._parameter_checksum(trainer.world_model.teacher)
                    if hasattr(trainer.world_model, "teacher") else None
                ),
                **metrics,
            }
            rows.append(row)
            print(json.dumps(row))
            # Atomic-enough per-arm checkpoint: a timeout loses at most the
            # currently running arm, not all previously completed seeds.
            pairing = build_pairing(rows)
            payload = {"protocol": vars(args), "rows": rows, "teacher_pairing": pairing, "summary": summarize(rows)}
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload = {"protocol": vars(args), "rows": rows, "teacher_pairing": build_pairing(rows), "summary": summarize(rows)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def build_pairing(rows):
    pairing = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        checks = [r["teacher_final_checksum"] for r in rows if int(r["seed"]) == seed]
        pairing[str(seed)] = {
            "max_teacher_checksum_gap": float(max(checks) - min(checks)) if checks else None,
            "exact_teacher_match": bool(checks and max(checks) == min(checks)),
            "n_arms": len(checks),
        }
    return pairing


if __name__ == "__main__":
    main()
