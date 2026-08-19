"""Evaluate teacher/student uncertainty preservation for a distilled checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.eval.metrics import evaluate_distillation_uncertainty
from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "consistency_distill.yaml"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--collect-steps", type=int, default=1500)
    p.add_argument("--train-world-model", type=int, default=0)
    p.add_argument("--m-samples", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    trainer = MBPOTrainer(cfg)
    if not hasattr(trainer.world_model, "teacher"):
        raise SystemExit("Config must enable model.use_consistency_distill")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.world_model.load_state_dict(ckpt["world_model"])
        if "actor" in ckpt:
            trainer.agent.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            trainer.agent.critic.load_state_dict(ckpt["critic"])
        if "u_net" in ckpt:
            trainer.u_net.load_state_dict(ckpt["u_net"])
    for _ in range(args.collect_steps):
        trainer._env_step()
    for _ in range(args.train_world_model):
        trainer._train_world_model()

    report = evaluate_distillation_uncertainty(
        trainer.world_model,
        q_fn=lambda o, a: trainer.agent.q_min(o, a),
        policy_fn=lambda o: trainer.agent.policy_tensor(o, deterministic=True),
        buffer=trainer.real_buffer,
        batch_size=args.batch_size,
        n_batches=args.batches,
        m_samples=args.m_samples,
    )
    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Wrote", out)


if __name__ == "__main__":
    main()
