"""Evaluate a checkpoint: return, world-model error, U calibration, throughput."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.eval.metrics import throughput_benchmark
from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--collect-steps", type=int, default=2000, help="Env steps to fill buffer for WM metrics")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    trainer = MBPOTrainer(cfg)

    ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.world_model.load_state_dict(ckpt["world_model"])
    trainer.agent.actor.load_state_dict(ckpt["actor"])
    trainer.agent.critic.load_state_dict(ckpt["critic"])
    if "u_net" in ckpt:
        trainer.u_net.load_state_dict(ckpt["u_net"])

    # Collect some real data for accuracy / calibration
    for _ in range(args.collect_steps):
        trainer._env_step()

    metrics = trainer.evaluate_full(n_episodes=args.episodes)
    thr = throughput_benchmark(
        trainer.world_model,
        trainer.obs_dim,
        trainer.action_dim,
        batch_size=128,
        n_repeats=15,
        sample_steps_list=[1, 2, 4, 8],
        device=trainer.device,
    )
    metrics["throughput"] = thr
    print(json.dumps(metrics, indent=2))

    out = Path(cfg["paths"]["log_dir"]) / "eval_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
