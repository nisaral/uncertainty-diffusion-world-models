"""Mechanism probe for the identified_hybrid U collapse (thread #1).

Runs one seed x one variant through the exact policy-ablation protocol, then
dumps:
  - the full per-log-step training trajectory (w/g/u means, EMA scales,
    loss terms, eval endpoints at eval_freq);
  - a final eval-time decomposition of the local UBE object on real-buffer
    states: per-state w_raw / w_deb / g / sigma_bar / coupling / u for the
    teacher and the 1-NFE student, plus the within-member spread of next
    states and of mapped values across paired latents.

Nothing here changes training math; it only logs and measures. CPU-only by
default (device override is accepted but this lab is CPU-bound).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from udwm.models.consistency import identified_decision_distill_loss
from udwm.rl.trainer import MBPOTrainer
from udwm.scripts.run_delayed_bimodal_policy_ablation import (
    VARIANTS,
    make_cfg,
    prepare_matched_teacher,
)
from udwm.uncertainty.mc_ube import MCUBELocalRewards
from udwm.utils.config import load_config, set_seed


@torch.no_grad()
def decompose_ube(distilled, q_fn, policy_fn, buffer, batch_size=256, n_batches=4,
                  m_samples=8):
    """Per-state decomposition of the local UBE object (unclamped u = w - g)."""
    est = MCUBELocalRewards(u_min=-1e9, m_samples=max(2, int(m_samples)), debias=True)
    n = distilled.teacher.ensemble_size
    m = max(2, int(m_samples))
    rows = []
    for _ in range(int(n_batches)):
        batch = buffer.sample(min(int(batch_size), len(buffer)))
        obs, act = batch["obs"], batch["actions"]
        latents = torch.randn(m, obs.shape[0], distilled.teacher.x_dim, device=obs.device)
        tq, sq = [], []
        t_delta_spread, s_delta_spread = [], []
        for i in range(n):
            t_i, s_i = [], []
            t_d, s_d = [], []
            for j in range(m):
                t_raw = distilled.teacher._ddim_sample_member(
                    i, obs, act, deterministic=False, x_T=latents[j])
                t_delta, _ = distilled.teacher._unpack_x(t_raw)
                t_x = obs + t_delta
                s_x, _ = distilled.student.sample_next(
                    distilled.teacher, obs, act, member=i, x_T=latents[j])
                t_d.append(t_delta)
                s_d.append(s_x - obs)
                t_i.append(q_fn(t_x, policy_fn(t_x)))
                s_i.append(q_fn(s_x, policy_fn(s_x)))
            tq.append(torch.stack(t_i, dim=0))       # [M,B,1]
            sq.append(torch.stack(s_i, dim=0))
            t_delta_spread.append(torch.stack(t_d, dim=0))  # [M,B,D]
            s_delta_spread.append(torch.stack(s_d, dim=0))
        tq_all = torch.stack(tq, dim=0)              # [N,M,B,1]
        sq_all = torch.stack(sq, dim=0)
        t_delta = torch.stack(t_delta_spread, dim=0)  # [N,M,B,D]
        s_delta = torch.stack(s_delta_spread, dim=0)
        for tag, vals in (("teacher", tq_all), ("student", sq_all)):
            out = est.combine_coupled(vals)
            mu = vals.mean(dim=1)                     # [N,B,1]
            w_raw = mu.var(dim=0, unbiased=False)     # [B,1]
            d = vals - vals.mean(dim=1, keepdim=True)
            spread = d.pow(2).sum(dim=1) / max(m - 1, 1)  # [N,B,1]
            rows.append({
                "tag": tag,
                                "w_raw": w_raw.reshape(-1).cpu().numpy(),
                "w": out["w"].reshape(-1).cpu().numpy(),
                "g": out["g"].reshape(-1).cpu().numpy(),
                "sigma_bar": out["sigma_bar"].reshape(-1).cpu().numpy(),
                "coupling": out["coupling"].reshape(-1).cpu().numpy(),
                "u": out["u"].reshape(-1).cpu().numpy(),
                "member_q_spread": spread.mean(dim=0).reshape(-1).cpu().numpy(),
            })
        t_ds = t_delta.std(dim=1, unbiased=False).mean(dim=0).mean(dim=-1)  # mean over members, per state -> scalar mean over batch
        s_ds = s_delta.std(dim=1, unbiased=False).mean(dim=0).mean(dim=-1)
        rows.append({"tag": "teacher", "delta_spread": t_ds.mean().item(), "delta_spread_state_mean": t_ds.cpu().numpy()})
        rows.append({"tag": "student", "delta_spread": s_ds.mean().item(), "delta_spread_state_mean": s_ds.cpu().numpy()})
    return rows


def summarize_decomp(rows):
    """Collapse probe rows into JSON-serialisable per-tag means/stats."""
    tags = ["teacher", "student"]
    by_tag = {t: {k: np.concatenate([r[k] for r in rows if r["tag"] == t and k in r and r[k].ndim > 0])
                  for k in ("w_raw", "w", "g", "sigma_bar", "coupling", "u", "member_q_spread")}
             for t in tags}
    out = {}
    for t in tags:
        stats = {}
        for k, arr in by_tag[t].items():
            stats[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "q05": float(np.quantile(arr, 0.05)),
                "q50": float(np.quantile(arr, 0.50)),
                "q95": float(np.quantile(arr, 0.95)),
            }
        ds = [r["delta_spread"] for r in rows if r.get("tag") == t and "delta_spread" in r]
        if ds:
            stats["delta_spread_mean"] = float(np.mean(ds))
        out[t] = stats
    return out


def gradient_postmortem(trainer, seed_batch=1):
    """Per-term gradient norms/alignments on a fresh batch at the end state.

    Recomputes the student distillation loss once (identified arms) and
    backpropagates each term separately to measure (a) whether the aleatoric
    term has any first-order signal at the collapsed corner, and (b) whether
    the epistemic and aleatoric gradients point against each other.
    """
    wm = trainer.world_model
    if not (getattr(wm, "identified", False) and wm.teacher_frozen):
        return None
    wm.eval()
    value_fn = trainer._make_distill_value_fn()
    bs = int(trainer.mbcfg["model_batch_size"])
    reports = []
    for _ in range(int(seed_batch)):
        batch = trainer.real_buffer.sample(bs)
        parts = identified_decision_distill_loss(
            wm.student, wm.teacher, batch["obs"], batch["actions"],
            batch["next_obs"], value_fn, batch["rewards"],
            m_latents=wm.m_latents,
            value_weight=1.0, variance_weight=1.0, aleatoric_weight=1.0,
            normalize_values=wm.normalize_values, reweight=False,
            corruption=getattr(wm, "corruption", "schedule"),
            reweight_w_only=getattr(wm, "reweight_w_only", False),
        )
        report = {}
        for name in ("member", "value_geometry", "epistemic_w", "aleatoric_g"):
            term = parts.get(name)
            if term is None:
                continue
            wm.zero_grad()
            term.backward(retain_graph=True)
            norm = sum(
                p.grad.detach().norm().item() ** 2
                for p in wm.student.parameters() if p.grad is not None
            ) ** 0.5
            report[name] = {"grad_norm": float(norm), "loss": float(term.item())}
        grads = {}
        for name in ("member", "epistemic_w", "aleatoric_g"):
            if name in report:
                wm.zero_grad()
                parts[name].backward(retain_graph=True)
                flat = torch.cat(
                    [p.grad.detach().reshape(-1)
                     for p in wm.student.parameters() if p.grad is not None]
                )
                grads[name] = flat
        for a, b in (("member", "epistemic_w"), ("member", "aleatoric_g"),
                     ("epistemic_w", "aleatoric_g")):
            if a in grads and b in grads:
                ga, gb = grads[a], grads[b]
                report[f"cos_{a}_{b}"] = float(
                    (ga * gb).sum() / (ga.norm() * gb.norm() + 1e-12)
                )
        wm.zero_grad()
        reports.append(report)
    wm.train()
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="identified_hybrid", choices=sorted(VARIANTS))
    parser.add_argument("--config", default="configs/delayed_bimodal_distill.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", default=None,
                        help="comma-separated model-config overrides, e.g. "
                             "distill_reweight_ema=false,distill_m_latents=4")
    parser.add_argument("--save", default=None,
                        help="optional checkpoint path for the trained trainer")
    parser.add_argument("--student-lr", type=float, default=1e-3,
                        help="Adam learning rate for the student optimizer")
    args = parser.parse_args(argv)
    base = load_config(args.config)
    if args.device is not None:
        base["device"] = args.device
    overrides = {}
    if args.set:
        for item in args.set.split(","):
            key, value = item.split("=", 1)
            v = value.strip().lower()
            if v in ("true", "false"):
                parsed = v == "true"
            else:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
            overrides[key.strip()] = parsed
    steps = int(args.steps)
    prepared_buffer, prepared_teacher = prepare_matched_teacher(base, args.seed, steps)
    cfg = make_cfg(base, args.variant, args.seed, steps)
    # --set must win over the VARIANTS defaults, so apply it AFTER make_cfg.
    cfg["model"].update(overrides)
    set_seed(args.seed)
    trainer = MBPOTrainer(cfg)
    trainer.real_buffer = copy.deepcopy(prepared_buffer)
    trainer.world_model.teacher.load_state_dict(prepared_teacher)
    trainer.world_model.freeze_teacher()
    trainer.wm_opt = torch.optim.Adam(
        trainer.world_model.student.parameters(), lr=args.student_lr
    )
    result = trainer.train()
    q_fn = trainer.agent.q_min
    policy_fn = lambda states: trainer.agent.policy_tensor(states, deterministic=True)
    decomp = decompose_ube(trainer.world_model, q_fn, policy_fn, trainer.real_buffer)
    grad_post = gradient_postmortem(trainer)
    if args.save:
        trainer.save(args.save)
    payload = {
        "protocol": vars(args),
        "final_metrics": result["final_metrics"],
        "logs": result["logs"],
        "decomp_summary": summarize_decomp(decomp),
        "gradient_postmortem": grad_post,
        "student_scale_ema": (
            trainer.world_model.scale_ema.snapshot()
            if getattr(trainer.world_model, "scale_ema", None) is not None else None
        ),
    }
    out = Path(args.out or f"runs/probe_u_collapse_{args.variant}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["final_metrics"], indent=2))
    print(json.dumps(payload["decomp_summary"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
