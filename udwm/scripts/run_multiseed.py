"""Run the same config over multiple seeds and aggregate metrics.

Example::

    python -m udwm.scripts.run_multiseed --config configs/smoke_train.yaml --seeds 0 1 2 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.rl.trainer import MBPOTrainer
from udwm.utils.config import load_config, set_seed


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std()) if len(arr) else float("nan"),
        "n": float(len(arr)),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "smoke_train.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    base = load_config(args.config)
    if args.device:
        base["device"] = args.device
    if args.steps is not None:
        base["mbpo"]["total_env_steps"] = int(args.steps)

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        cfg = json.loads(json.dumps(base))  # deep copy via json
        cfg["seed"] = int(seed)
        set_seed(seed)
        print(f"\n===== multiseed seed={seed} steps={cfg['mbpo']['total_env_steps']} =====")
        trainer = MBPOTrainer(cfg)
        result = trainer.train()
        metrics = result.get("final_metrics", {})
        row = {
            "seed": seed,
            "final_return": result.get("final_eval_return"),
            "return_std": metrics.get("return_std"),
            "next_state_mse": metrics.get("next_state_mse"),
            "reward_mse": metrics.get("reward_mse"),
            "u_corr": metrics.get("corr_std_abserr"),
            "steps": cfg["mbpo"]["total_env_steps"],
            "model": cfg["model"]["type"],
            "use_ube": cfg["agent"].get("use_ube", True),
            "joint_reward": cfg.get("reward_term", {}).get("joint_with_diffusion", False),
        }
        rows.append(row)
        tag = f"multiseed_{cfg['model']['type']}_seed{seed}"
        ckpt = Path(cfg["paths"]["checkpoint_dir"]) / f"{tag}.pt"
        trainer.save(ckpt)
        print("saved", ckpt)

    # aggregate
    keys = ["final_return", "next_state_mse", "reward_mse", "u_corr"]
    summary = {k: _mean_std([float(r[k]) for r in rows if r.get(k) is not None]) for k in keys}

    out_dir = Path(args.out or base["paths"]["log_dir"]) / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"runs": rows, "summary": summary, "config": base}
    out_path = out_dir / "multiseed_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n========== MULTI-SEED SUMMARY ==========")
    for k, v in summary.items():
        print(f"{k:16s}  mean={v['mean']:.4f}  std={v['std']:.4f}  n={int(v['n'])}")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
