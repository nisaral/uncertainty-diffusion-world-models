"""Print a selective-prediction report for √U vs |TD error|.

Honest output: ranking quality + what happens if you treat √U as a threshold.
Does not claim conformal coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.eval.selective import collect_score_and_error, selective_report
from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "ab_novel.yaml"))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--collect-steps", type=int, default=800)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    trainer = MBPOTrainer(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.world_model.load_state_dict(ckpt["world_model"])
        trainer.agent.actor.load_state_dict(ckpt["actor"])
        trainer.agent.critic.load_state_dict(ckpt["critic"])
        trainer.u_net.load_state_dict(ckpt["u_net"])

    for _ in range(args.collect_steps):
        trainer._env_step()
    if not args.checkpoint:
        for _ in range(3):
            trainer._train_world_model()

    pair = collect_score_and_error(trainer.agent, trainer.u_net, trainer.real_buffer)
    report = selective_report(pair["score"], pair["abs_td"])
    print(report["disclaimer"])
    print(f"n={report['n']}  rank_corr(√U, |TD|)={report['rank_corr_score_vs_error']:.3f}")
    print("\nRisk–coverage (keep lowest-√U fraction):")
    print(f"{'cov':>6} {'abstain':>8} {'risk':>10} {'abst_err':>10}")
    for row in report["risk_coverage"]:
        print(
            f"{row['coverage']:6.2f} {row['abstention_rate']:8.2f} "
            f"{row['risk_mean_abs_td']:10.4f} {row['abstained_mean_abs_td']:10.4f}"
        )
    print("\nIf √U is used as an *absolute* threshold (DIO-style over-rejection):")
    print(f"{'tau':>8} {'abstain':>8} {'recall':>8} {'over_rej':>8} {'prec':>8}")
    for row in report["threshold_sweep"]:
        print(
            f"{row['tau']:8.3f} {row['abstention_rate']:8.3f} "
            f"{row['recall_bad']:8.3f} {row['over_rejection']:8.3f} "
            f"{row['precision_abstain']:8.3f}"
        )

    out = Path(args.out or Path(cfg["paths"]["log_dir"]) / "selective_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nWrote", out)


if __name__ == "__main__":
    main()
