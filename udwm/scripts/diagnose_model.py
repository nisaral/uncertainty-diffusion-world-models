"""One-step world-model diagnostics on a trained checkpoint or fresh short train.

Writes JSON with per-dimension MSE and a few sample predictions for debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.eval.metrics import evaluate_world_model_accuracy
from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "smoke_train.yaml"))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--collect-steps", type=int, default=1500)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    trainer = MBPOTrainer(cfg)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.world_model.load_state_dict(ckpt["world_model"])
        print("Loaded", args.checkpoint)
    else:
        print("No checkpoint: collecting random data only for baseline diagnose")

    for _ in range(args.collect_steps):
        trainer._env_step()

    # optional short WM train if no ckpt
    if not args.checkpoint:
        for _ in range(5):
            trainer._train_world_model()

    metrics = evaluate_world_model_accuracy(
        trainer.world_model, trainer.real_buffer, batch_size=min(256, len(trainer.real_buffer))
    )

    # per-dim squared error on one batch
    batch = trainer.real_buffer.sample(min(256, len(trainer.real_buffer)))
    with torch.no_grad():
        obs, act = batch["obs"], batch["actions"]
        nxt = batch["next_obs"]
        if getattr(trainer.world_model, "model_type", "") == "gaussian":
            pred, _ = trainer.world_model.dynamics.sample_next(obs, act, deterministic=True)
        elif getattr(trainer.world_model, "joint_reward", False):
            pred, _ = trainer.world_model.dynamics.sample_next_with_reward(obs, act, deterministic=True)
        else:
            pred = trainer.world_model.dynamics.sample_next(obs, act, deterministic=True)
        per_dim = ((pred - nxt) ** 2).mean(0).cpu().numpy().tolist()
        samples = {
            "obs0": obs[0].cpu().numpy().tolist(),
            "act0": act[0].cpu().numpy().tolist(),
            "true_next0": nxt[0].cpu().numpy().tolist(),
            "pred_next0": pred[0].cpu().numpy().tolist(),
        }

    out = {
        "aggregate": metrics,
        "per_dim_mse": per_dim,
        "example": samples,
        "buffer_size": len(trainer.real_buffer),
    }
    out_path = Path(args.out or Path(cfg["paths"]["log_dir"]) / "model_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["aggregate"], indent=2))
    print("per_dim_mse", np.round(per_dim, 5).tolist())
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
