"""Run Gap-3 ablations on one env and write a comparison table.

Variants (default):
  1. gaussian + UBE
  2. diffusion separate-R + UBE
  3. diffusion joint-R + UBE
  4. diffusion separate-R, no UBE
  5. gaussian, no UBE

Usage::

    python -m udwm.scripts.run_ablations --config configs/ablation_fast.yaml --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


DEFAULT_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "gaussian_ube",
        "model_type": "gaussian",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.5,
    },
    {
        "name": "gaussian_no_ube",
        "model_type": "gaussian",
        "joint_reward": False,
        "use_ube": False,
        "optimism_lambda": 0.0,
    },
    {
        "name": "diffusion_sepR_ube",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.5,
    },
    {
        "name": "diffusion_sepR_no_ube",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": False,
        "optimism_lambda": 0.0,
    },
    {
        "name": "diffusion_jointR_ube",
        "model_type": "diffusion",
        "joint_reward": True,
        "use_ube": True,
        "optimism_lambda": 0.5,
    },
]

# Novelty A+B factorial (use with configs/ab_novel.yaml)
AB_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "off",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.3,
        "u_gate_mode": "off",
        "adaptive_mc": False,
    },
    {
        "name": "A_ugate",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.3,
        "u_gate_mode": "both",
        "adaptive_mc": False,
    },
    {
        "name": "B_adapt_mc",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.3,
        "u_gate_mode": "off",
        "adaptive_mc": True,
    },
    {
        "name": "AB_both",
        "model_type": "diffusion",
        "joint_reward": False,
        "use_ube": True,
        "optimism_lambda": 0.3,
        "u_gate_mode": "both",
        "adaptive_mc": True,
    },
]


def apply_variant(cfg: Dict[str, Any], variant: Dict[str, Any], steps: int | None) -> Dict[str, Any]:
    c = deepcopy(cfg)
    c["model"]["type"] = variant["model_type"]
    c.setdefault("reward_term", {})["joint_with_diffusion"] = bool(variant.get("joint_reward", False))
    c["agent"]["use_ube"] = bool(variant["use_ube"])
    c["agent"]["optimism_lambda"] = float(variant.get("optimism_lambda", 0.0))
    if "u_gate_mode" in variant:
        c.setdefault("u_gate", {})["mode"] = variant["u_gate_mode"]
    if "adaptive_mc" in variant:
        c.setdefault("ube", {}).setdefault("adaptive_mc", {})["enabled"] = bool(variant["adaptive_mc"])
    if steps is not None:
        c["mbpo"]["total_env_steps"] = int(steps)
    c["mbpo"]["eval_freq"] = max(int(c["mbpo"]["total_env_steps"] // 4), 250)
    return c


def run_one(cfg: Dict[str, Any], name: str, seed: int) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    cfg["seed"] = seed
    set_seed(seed)
    print(f"\n===== {name} seed={seed} steps={cfg['mbpo']['total_env_steps']} =====")
    trainer = MBPOTrainer(cfg)
    result = trainer.train()
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = f"ablation_{name}_seed{seed}"
    trainer.save(ckpt_dir / f"{tag}.pt")
    metrics = result.get("final_metrics", {})
    row = {
        "name": name,
        "seed": seed,
        "final_return": result.get("final_eval_return", metrics.get("return_mean")),
        "return_std": metrics.get("return_std"),
        "next_state_mse": metrics.get("next_state_mse"),
        "reward_mse": metrics.get("reward_mse"),
        "u_corr": metrics.get("corr_std_abserr"),
        "mean_pred_std": metrics.get("mean_pred_std"),
        "mean_abs_td": metrics.get("mean_abs_err"),
        "steps": cfg["mbpo"]["total_env_steps"],
        "model": cfg["model"]["type"],
        "joint_reward": cfg["reward_term"]["joint_with_diffusion"],
        "use_ube": cfg["agent"]["use_ube"],
        "u_gate_mode": cfg.get("u_gate", {}).get("mode", "off"),
        "adaptive_mc": bool(cfg.get("ube", {}).get("adaptive_mc", {}).get("enabled", False)),
        "selective_rank_corr": metrics.get("selective_rank_corr"),
        "selective_over_rejection": metrics.get("selective_over_rejection"),
        "selective_recall_bad": metrics.get("selective_recall_bad"),
        "selective_risk_at_50": metrics.get("selective_risk_at_50"),
    }
    # learning curve points
    curve = [
        {"step": e.get("step"), "return_mean": e.get("return_mean")}
        for e in result.get("logs", [])
        if "return_mean" in e
    ]
    row["curve"] = curve
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Gap-3 ablation suite")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "ablation_fast.yaml"),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--suite",
        type=str,
        choices=["core", "ab"],
        default="core",
        help="core = Gaussian/diffusion/UBE; ab = U-gate × adaptive-MC factorial",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Subset of variant names (default: all in the suite)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    base = load_config(args.config)
    if args.device:
        base["device"] = args.device

    pool = AB_VARIANTS if args.suite == "ab" else DEFAULT_VARIANTS
    variants = pool
    if args.variants:
        allowed = set(args.variants)
        variants = [v for v in pool if v["name"] in allowed]
        if not variants:
            raise SystemExit(f"No matching variants. Known: {[v['name'] for v in pool]}")

    rows: List[Dict[str, Any]] = []
    for v in variants:
        cfg = apply_variant(base, v, args.steps)
        for seed in args.seeds:
            rows.append(run_one(cfg, v["name"], seed))

    out_dir = Path(args.out or base["paths"]["log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ablation_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # flat CSV without nested curve
    csv_path = out_dir / "ablation_results.csv"
    flat_keys = [
        "name",
        "seed",
        "final_return",
        "return_std",
        "next_state_mse",
        "reward_mse",
        "u_corr",
        "mean_pred_std",
        "mean_abs_td",
        "steps",
        "model",
        "joint_reward",
        "use_ube",
        "u_gate_mode",
        "adaptive_mc",
        "selective_rank_corr",
        "selective_over_rejection",
        "selective_recall_bad",
        "selective_risk_at_50",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # console summary
    print("\n========== ABLATION SUMMARY ==========")
    print(
        f"{'variant':16s} {'seed':>4s} {'return':>9s} {'over_rej':>9s} {'recall':>8s} {'rkcorr':>7s}"
    )
    for r in rows:
        print(
            f"{r['name']:16s} {r['seed']:4d} "
            f"{float(r['final_return'] or 0):9.2f} "
            f"{float(r.get('selective_over_rejection') or float('nan')):9.3f} "
            f"{float(r.get('selective_recall_bad') or float('nan')):8.3f} "
            f"{float(r.get('selective_rank_corr') or float('nan')):7.3f}"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
