"""Compare imagination throughput: Gaussian vs diffusion sample_steps (WIMLE critique)."""

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
from udwm.models.world_model import WorldModel
from udwm.utils.config import set_seed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)
    set_seed(0)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    results = {}
    for name, kwargs in [
        ("gaussian", dict(model_type="gaussian", joint_with_diffusion=False)),
        ("diffusion_separate_r", dict(model_type="diffusion", joint_with_diffusion=False)),
        ("diffusion_joint_r", dict(model_type="diffusion", joint_with_diffusion=True)),
    ]:
        wm = WorldModel.build(
            kwargs["model_type"],
            args.obs_dim,
            args.action_dim,
            ensemble_size=3,
            hidden_dims=(128, 128),
            diffusion_steps=10,
            sample_steps=4,
            joint_with_diffusion=kwargs["joint_with_diffusion"],
        ).to(device)
        results[name] = throughput_benchmark(
            wm,
            args.obs_dim,
            args.action_dim,
            batch_size=args.batch_size,
            n_repeats=20,
            sample_steps_list=[1, 2, 4, 8, 10],
            device=device,
        )

    print(json.dumps(results, indent=2))
    out = ROOT / "runs" / "throughput_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
