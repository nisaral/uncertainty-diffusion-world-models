"""Controlled stress test for decision-relevant uncertainty distillation.

The synthetic transition law is multimodal and the evaluation grid includes
sparsely observed states. Teacher members use independent bootstrap datasets.
Three matched students compare ordinary member matching, state-space geometry,
and direct downstream value-disagreement preservation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from udwm.eval.metrics import _rank_corr
from udwm.models.consistency import (
    ConsistencyStudent,
    decision_preserving_distill_loss,
    uncertainty_preserving_distill_loss,
)
from udwm.models.diffusion_dynamics import DiffusionDynamicsEnsemble
from udwm.utils.config import set_seed


def decision_value(next_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Nonlinear consequence map used for both training and evaluation."""
    x, context = next_obs[..., :1], next_obs[..., 1:2]
    return torch.tanh(6.0 * (x - 0.35 * context)) + 0.35 * x.square() - 0.1 * actions.square()


def make_dataset(seed: int, n: int, device: torch.device):
    rng = np.random.default_rng(seed)
    # Most observations cover the center; a small tail creates sparse regions.
    center = rng.uniform(-1.0, 1.0, size=(n, 1))
    tail = rng.uniform(-2.5, 2.5, size=(n, 1))
    use_tail = rng.random((n, 1)) < 0.12
    x = np.where(use_tail, tail, center)
    context = rng.choice([-1.0, 1.0], size=(n, 1))
    action = rng.uniform(-1.0, 1.0, size=(n, 1))
    mode_prob = 1.0 / (1.0 + np.exp(-(1.7 * x - 1.2 * action + 0.8 * context)))
    mode = np.where(rng.random((n, 1)) < mode_prob, 1.0, -1.0)
    branch = mode * (0.45 + 0.35 * np.abs(action))
    next_x = x + 0.35 * action + branch + rng.normal(0.0, 0.035, size=(n, 1))
    next_context = 0.92 * context + 0.08 * mode
    obs = np.concatenate([x, context], axis=1).astype(np.float32)
    nxt = np.concatenate([next_x, next_context], axis=1).astype(np.float32)
    act = action.astype(np.float32)
    return tuple(torch.as_tensor(v, device=device) for v in (obs, act, nxt))


def train_teacher(seed: int, obs, actions, next_obs, ensemble_size: int, updates: int, batch_size: int):
    set_seed(seed)
    teacher = DiffusionDynamicsEnsemble(
        obs_dim=2, action_dim=1, ensemble_size=ensemble_size,
        hidden_dims=(64, 64), diffusion_steps=8, sample_steps=8,
    ).to(obs.device)
    teacher.update_stats(obs, next_obs)
    optimizers = [torch.optim.Adam(member.parameters(), lr=1e-3) for member in teacher.members]
    generators = [np.random.default_rng(seed * 1009 + 37 * i) for i in range(ensemble_size)]
    n = obs.shape[0]
    for _ in range(updates):
        for i, optimizer in enumerate(optimizers):
            idx = generators[i].integers(0, n, size=batch_size)
            loss = teacher.diffusion_loss_member(i, obs[idx], actions[idx], next_obs[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return teacher


def train_student(teacher, variant: str, seed: int, obs, actions, next_obs, updates: int, batch_size: int, shape_weight: float = 0.001):
    # Resetting to the same seed gives all arms identical initialization and
    # minibatch/noise streams; only the objective changes.
    set_seed(seed + 50_000)
    student = ConsistencyStudent(2, 2, 1, teacher.ensemble_size, hidden_dims=(16, 16)).to(obs.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 70_000)
    for _ in range(updates):
        idx = rng.integers(0, obs.shape[0], size=batch_size)
        if variant == "ordinary":
            parts = uncertainty_preserving_distill_loss(
                student, teacher, obs[idx], actions[idx], next_obs[idx],
                mean_weight=0.0, geometry_weight=0.0, pairwise_weight=0.0,
            )
        elif variant == "state_geometry":
            parts = uncertainty_preserving_distill_loss(
                student, teacher, obs[idx], actions[idx], next_obs[idx],
                mean_weight=1.0, geometry_weight=1.0, pairwise_weight=1.0,
            )
        elif variant == "decision_geometry":
            parts = decision_preserving_distill_loss(
                student, teacher, obs[idx], actions[idx], next_obs[idx], decision_value,
                value_weight=1.0, variance_weight=1.0,
            )
        elif variant == "decision_shape":
            parts = decision_preserving_distill_loss(
                student, teacher, obs[idx], actions[idx], next_obs[idx], decision_value,
                value_weight=1.0, variance_weight=1.0, variance_shape_weight=shape_weight,
            )
        elif variant == "decision_hybrid":
            parts = decision_preserving_distill_loss(
                student, teacher, obs[idx], actions[idx], next_obs[idx], decision_value,
                value_weight=1.0, variance_weight=1.0,
                state_geometry_weight=1.0, state_pairwise_weight=1.0,
            )
        else:
            raise ValueError(variant)
        optimizer.zero_grad()
        parts["total"].backward()
        optimizer.step()
    return student


@torch.no_grad()
def evaluate(teacher, student, seed: int, m_samples: int, grid_size: int):
    rng = np.random.default_rng(seed + 90_000)
    x = rng.uniform(-2.75, 2.75, size=(grid_size, 1)).astype(np.float32)
    context = rng.choice([-1.0, 1.0], size=(grid_size, 1)).astype(np.float32)
    action = rng.uniform(-1.0, 1.0, size=(grid_size, 1)).astype(np.float32)
    obs = torch.as_tensor(np.concatenate([x, context], axis=1), device=next(teacher.parameters()).device)
    actions = torch.as_tensor(action, device=obs.device)
    generator = torch.Generator(device=obs.device).manual_seed(seed + 110_000)
    latents = torch.randn(m_samples, grid_size, teacher.x_dim, generator=generator, device=obs.device)
    teacher_values, student_values, state_errors = [], [], []
    for i in range(teacher.ensemble_size):
        tv, sv = [], []
        for j in range(m_samples):
            raw = teacher._ddim_sample_member(i, obs, actions, deterministic=False, x_T=latents[j])
            t_next = obs + teacher._unpack_x(raw)[0]
            s_next, _ = student.sample_next(teacher, obs, actions, member=i, x_T=latents[j])
            tv.append(decision_value(t_next, actions))
            sv.append(decision_value(s_next, actions))
            state_errors.append((s_next - t_next).square().mean(dim=-1))
        teacher_values.append(torch.stack(tv).mean(dim=0))
        student_values.append(torch.stack(sv).mean(dim=0))
    t_values = torch.stack(teacher_values)
    s_values = torch.stack(student_values)
    t_w = t_values.var(dim=0, unbiased=True).reshape(-1).cpu().numpy()
    s_w = s_values.var(dim=0, unbiased=True).reshape(-1).cpu().numpy()
    cutoff = np.quantile(t_w, 0.9)
    predicted_cutoff = np.quantile(s_w, 0.9)
    top_recall = float(np.mean(s_w[t_w >= cutoff] >= predicted_cutoff))
    return {
        "n": int(grid_size),
        "w_rank_corr": _rank_corr(t_w, s_w),
        "w_rmse": float(np.sqrt(np.mean((t_w - s_w) ** 2))),
        "w_mae": float(np.mean(np.abs(t_w - s_w))),
        "top_decile_recall": top_recall,
        "member_value_rmse": float(torch.sqrt((t_values - s_values).square().mean()).item()),
        "paired_state_mse": float(torch.cat(state_errors).mean().item()),
        "teacher_w_mean": float(np.mean(t_w)),
    }


def summarize(rows):
    variants = sorted({row["variant"] for row in rows})
    summary = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {}
        for metric in ("w_rank_corr", "w_rmse", "top_decile_recall", "member_value_rmse", "paired_state_mse"):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[variant][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            }
    ordinary = {row["seed"]: row for row in rows if row["variant"] == "ordinary"}
    paired = {}
    for variant in variants:
        if variant == "ordinary":
            continue
        selected = [row for row in rows if row["variant"] == variant and row["seed"] in ordinary]
        for metric in ("w_rank_corr", "w_rmse", "top_decile_recall"):
            deltas = np.asarray(
                [row[metric] - ordinary[row["seed"]][metric] for row in selected], dtype=float
            )
            if deltas.size == 0:
                continue
            paired[f"{variant}_minus_ordinary_{metric}"] = {
                "mean": float(deltas.mean()),
                "std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "se": float(deltas.std(ddof=1) / np.sqrt(deltas.size)) if deltas.size > 1 else 0.0,
                "wins": int(np.sum(deltas > 0)) if metric != "w_rmse" else int(np.sum(deltas < 0)),
                "n": int(deltas.size),
                "values": [float(v) for v in deltas],
            }
    return {"by_variant": summary, "paired": paired}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-size", type=int, default=2000)
    parser.add_argument("--teacher-updates", type=int, default=120)
    parser.add_argument("--student-updates", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--m-samples", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=768)
    parser.add_argument("--shape-weight", type=float, default=0.001)
    parser.add_argument(
        "--variants", nargs="+",
        default=["ordinary", "state_geometry", "decision_geometry", "decision_hybrid"],
    )
    parser.add_argument("--out", default="runs/decision_distillation_stress.json")
    args = parser.parse_args(argv)
    device = torch.device("cpu")
    rows = []
    for seed in args.seeds:
        obs, actions, next_obs = make_dataset(seed, args.data_size, device)
        teacher = train_teacher(seed, obs, actions, next_obs, args.ensemble_size, args.teacher_updates, args.batch_size)
        teacher_state = copy.deepcopy(teacher.state_dict())
        checksum = float(sum(v.float().abs().sum() for v in teacher_state.values()))
        for variant in args.variants:
            teacher.load_state_dict(teacher_state)
            student = train_student(
                teacher, variant, seed, obs, actions, next_obs,
                args.student_updates, args.batch_size, args.shape_weight,
            )
            report = evaluate(teacher, student, seed, args.m_samples, args.grid_size)
            report.update({"seed": seed, "variant": variant, "teacher_checksum": checksum})
            rows.append(report)
            print(json.dumps(report))
    payload = {"protocol": vars(args), "rows": rows, "summary": summarize(rows)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
