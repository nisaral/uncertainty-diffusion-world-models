"""Train shared-core MBPO (+ diffusion WM + optional UBE) from a YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="UDWM shared-core MBPO trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override total env steps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["gaussian", "diffusion"], default=None)
    parser.add_argument("--joint-reward", action="store_true", help="Gap 1: denoise reward with state")
    parser.add_argument("--separate-reward", action="store_true", help="Force separate R/T head")
    parser.add_argument("--no-ube", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.model is not None:
        cfg["model"]["type"] = args.model
    if args.joint_reward:
        cfg["reward_term"]["joint_with_diffusion"] = True
    if args.separate_reward:
        cfg["reward_term"]["joint_with_diffusion"] = False
    if args.no_ube:
        cfg["agent"]["use_ube"] = False
    if args.steps is not None:
        cfg["mbpo"]["total_env_steps"] = args.steps

    set_seed(int(cfg["seed"]))
    print(
        "Config:",
        json.dumps(
            {
                k: cfg[k]
                for k in ("seed", "device", "env", "model", "reward_term")
            },
            indent=2,
        ),
    )

    trainer = MBPOTrainer(cfg)
    result = trainer.train()
    print("Final eval return:", result["final_eval_return"])
    print("Final metrics:", json.dumps(result.get("final_metrics", {}), indent=2))

    joint = bool(cfg.get("reward_term", {}).get("joint_with_diffusion", False))
    tag = f"{cfg['model']['type']}_{'jointR' if joint else 'sepR'}_seed{cfg['seed']}"
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"mbpo_{tag}.pt"
    trainer.save(out)
    print("Saved:", out)

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"log_{tag}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"logs": result["logs"], "final_metrics": result.get("final_metrics", {})}, f, indent=2)
    print("Logs:", log_path)


if __name__ == "__main__":
    main()
