"""Does a shared DDIM latent actually couple ensemble members? The gating experiment.

`research/COUPLED-MC-UBE-PROPOSAL.md` rests on one empirical claim: feeding the
same x_T to independently initialised diffusion members yields *pointwise*
similar next states, not merely equal in distribution. If true, the finite-M bias
and variance of the ensemble variance `w` both shrink for free, since

    bias(w_hat) = (g* - mean(Sigma)) / M = mean_ij Var(Y_i - Y_j) / (2 N^2 M).

If false, mean(Sigma) ~ g*/N, coupling buys nothing, and the proposal collapses to
"common random numbers, as usual". This script decides which, and it is cheap.

Reports, on real buffer states:
  * corr(Y_i, Y_j)  under shared vs independent latents  (the headline number)
  * `coupling` in [0,1]: 0 = independent, 1 = members move together
  * the realised bias of w_hat against a high-M reference, both estimators
  * sd(w_hat), which is the quantity that would let M be cut

Run::

    python -m udwm.scripts.measure_coupling --config configs/smoke_train.yaml \
        --checkpoint runs/.../world_model.pt
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

from udwm.rl.trainer import MBPOTrainer
from udwm.uncertainty.mc_ube import MCUBELocalRewards
from udwm.utils.config import load_config, set_seed


def _mean_offdiag_corr(q: torch.Tensor) -> float:
    """Mean off-diagonal correlation of Y across members, from [N,M,B,1] samples."""
    n = q.shape[0]
    if n < 2:
        return float("nan")
    y = q.squeeze(-1).permute(2, 0, 1)                 # [B, N, M]
    y = y - y.mean(dim=-1, keepdim=True)
    sd = y.pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)  # [B, N]
    c = (y @ y.transpose(1, 2)) / (sd.unsqueeze(2) * sd.unsqueeze(1))
    off = ~torch.eye(n, dtype=torch.bool, device=c.device)
    return float(c[:, off].mean())


@torch.no_grad()
def measure(trainer, m: int, batch_size: int, m_ref: int) -> dict:
    dyn = trainer.world_model.dynamics
    if not hasattr(dyn, "sample_next_coupled"):
        raise SystemExit(
            f"{type(dyn).__name__} has no shared-latent sampler; this experiment "
            "only applies to implicit (diffusion/consistency) ensembles."
        )
    agent = trainer.agent
    batch = trainer.real_buffer.sample(min(batch_size, len(trainer.real_buffer)))
    obs, act = batch["obs"], batch["actions"]

    def q_fn(o, a):
        return agent.q_min(o, a)

    def policy_fn(o):
        return agent.policy_tensor(o, deterministic=True)

    est = MCUBELocalRewards(u_min=-1e9, debias=True, m_samples=m)

    def sample_q(n_samples: int, coupled: bool) -> torch.Tensor:
        s, _ = dyn.sample_next_coupled(obs, act, m=n_samples, coupled=coupled)
        n, ms, b, o_dim = s.shape
        flat = s.reshape(n * ms * b, o_dim)
        return q_fn(flat, policy_fn(flat)).reshape(n, ms, b, 1)

    # high-M independent reference for w*: bias -> 0 as M grows
    ref = est.combine_coupled(sample_q(m_ref, coupled=False))
    w_ref = ref["w"]

    out = {"m": m, "m_ref": m_ref, "batch": int(obs.shape[0])}
    for name, coupled in (("independent", False), ("coupled", True)):
        q = sample_q(m, coupled)
        naive = q.mean(dim=1).var(dim=0, unbiased=False)
        cc = est.combine_coupled(q)
        out[name] = {
            "corr_across_members": _mean_offdiag_corr(q),
            "coupling": float(cc["coupling"].mean()),
            "w_bias_naive": float((naive - w_ref).mean()),
            "w_bias_corrected": float((cc["w"] - w_ref).mean()),
            "w_sd": float(cc["w"].std()),
            "u_mean": float(cc["u"].mean()),
            "g_mean": float(cc["g"].mean()),
        }
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(ROOT / "configs" / "smoke_train.yaml"))
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--collect-steps", type=int, default=1500)
    p.add_argument("--wm-train-iters", type=int, default=20)
    p.add_argument("--m", type=int, default=8, help="samples/member under test")
    p.add_argument("--m-ref", type=int, default=256, help="samples/member for the w* reference")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    trainer = MBPOTrainer(cfg)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.world_model.load_state_dict(ckpt["world_model"])
        print("Loaded", args.checkpoint)

    for _ in range(args.collect_steps):
        trainer._env_step()
    if not args.checkpoint:
        # Untrained members are unrelated random functions and will show ~zero
        # coupling for reasons that say nothing about the claim. Train briefly.
        for _ in range(args.wm_train_iters):
            trainer._train_world_model()

    res = measure(trainer, args.m, args.batch_size, args.m_ref)

    ind, cou = res["independent"], res["coupled"]
    print(json.dumps(res, indent=2))
    print()
    print(f"corr across members: independent={ind['corr_across_members']:+.4f}  "
          f"coupled={cou['corr_across_members']:+.4f}")
    print(f"coupling (0=indep, 1=locked): {cou['coupling']:+.4f}")
    print(f"sd(w): independent={ind['w_sd']:.6f}  coupled={cou['w_sd']:.6f}  "
          f"ratio={cou['w_sd'] / max(ind['w_sd'], 1e-12):.3f}")
    print()
    if cou["corr_across_members"] > 0.3:
        print("VERDICT: shared latents do couple the members. The proposal's "
              "enabling claim holds; proceed to measuring MSE vs M.")
    else:
        print("VERDICT: shared latents do NOT meaningfully couple the members. "
              "The coupling story does not survive; treat COUPLED-MC-UBE-PROPOSAL "
              "as refuted on this environment and say so.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("Wrote", out_path)


if __name__ == "__main__":
    main()
