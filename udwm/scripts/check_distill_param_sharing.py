# -*- coding: utf-8 -*-
"""Hypothesis C gate: do the identified loss terms share parameters?

Hypothesis C (research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md) claims the
uncertainty-matching terms and the point-prediction term of the identified
distill objective compete for the *same* parameters, so their gradients can
interfere as task complexity rises.  That claim is only coherent if the two
term groups backprop through a shared trunk.  This script answers the gate
question empirically rather than by reading the source.

Method (no training, CPU, seconds):

  1. Rebuild the exact model either from a saved checkpoint's ``cfg`` (and load
     its ``world_model`` weights) or, with ``--config`` + explicit dims, from a
     freshly initialised model of the same topology.
  2. Construct a batch and compute the identified distill loss, keeping each
     term as a separate live tensor.
  3. Backward each term on its own and record every parameter tensor that
     receives a nonzero gradient, plus the Jaccard overlap between the term
     groups.

If the point-prediction and uncertainty-matching groups touch a disjoint
parameter set, Hypothesis C is dropped (reported as such and not softened).

The result is a statement about the *network topology*, which is identical for
DelayedBimodal and Hopper (same ``student_hidden_dims`` and one
``ConditionalDenoiser`` trunk per member), so it is verified on both.

Run:
    python -m udwm.scripts.check_distill_param_sharing \
        --checkpoint checkpoints/hf_diagnostic_delayedbimodal/identified_eq_seed0.pt \
        --out runs/hypothesis_c_param_sharing.json
    python -m udwm.scripts.check_distill_param_sharing \
        --config configs/dmc_hopper_probe.yaml --obs-dim 15 --action-dim 4 \
        --arms identified_eq --out runs/hypothesis_c_param_sharing_hopper.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from udwm.models.consistency import identified_decision_distill_loss
from udwm.models.world_model import WorldModel
from udwm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]

# Per-arm knobs that decide which identified terms carry weight (mirrors
# udwm/scripts/run_delayed_bimodal_policy_ablation.py VARIANTS).
ARMS = {
    'ordinary': dict(decision_weight=0.0, variance_weight=0.0, aleatoric_weight=0.0,
                     identified=False, normalize=False, reweight=False,
                     reweight_w_only=False, corruption='schedule'),
    'identified_eq': dict(decision_weight=1.0, variance_weight=1.0, aleatoric_weight=1.0,
                          identified=True, normalize=False, reweight=False,
                          reweight_w_only=False, corruption='schedule'),
    'lagged_identified_eq': dict(decision_weight=1.0, variance_weight=1.0,
                                 aleatoric_weight=1.0, identified=True, normalize=True,
                                 reweight=False, reweight_w_only=False,
                                 corruption='schedule'),
    'identified_hybrid': dict(decision_weight=1.0, variance_weight=1.0,
                              aleatoric_weight=1.0, identified=True, normalize=False,
                              reweight=True, reweight_w_only=False, corruption='schedule'),
}


def build_from_cfg(cfg, obs_dim, action_dim, arm_overrides=None):
    mcfg = dict(cfg['model'])
    rcfg = cfg['reward_term']
    if arm_overrides:
        mcfg.update(arm_overrides)
    return WorldModel.build(
        model_type=mcfg['type'],
        obs_dim=int(obs_dim),
        action_dim=int(action_dim),
        ensemble_size=int(mcfg['ensemble_size']),
        hidden_dims=tuple(mcfg['hidden_dims']),
        student_hidden_dims=(tuple(mcfg['student_hidden_dims'])
                             if mcfg.get('student_hidden_dims') else None),
        diffusion_steps=int(mcfg.get('diffusion_steps', 10)),
        beta_start=float(mcfg.get('beta_start', 1e-4)),
        beta_end=float(mcfg.get('beta_end', 2e-2)),
        sample_steps=int(mcfg.get('sample_steps', 4)),
        reward_hidden=tuple(rcfg.get('hidden_dims', [128, 128])),
        joint_with_diffusion=bool(rcfg.get('joint_with_diffusion', False)),
        use_consistency_distill=bool(mcfg.get('use_consistency_distill', False)),
        preserve_distilled_uncertainty=bool(mcfg.get('preserve_distilled_uncertainty', False)),
        distill_mean_weight=float(mcfg.get('distill_mean_weight', 1.0)),
        distill_geometry_weight=float(mcfg.get('distill_geometry_weight', 1.0)),
        distill_pairwise_weight=float(mcfg.get('distill_pairwise_weight', 1.0)),
        distill_decision_weight=float(mcfg.get('distill_decision_weight', 0.0)),
        distill_value_variance_weight=float(mcfg.get('distill_value_variance_weight', 0.0)),
        distill_hybrid_state_weight=float(mcfg.get('distill_hybrid_state_weight', 0.0)),
        distill_hybrid_pairwise_weight=float(mcfg.get('distill_hybrid_pairwise_weight', 0.0)),
        distill_normalize_values=bool(mcfg.get('distill_normalize_values', False)),
        distill_guard_enabled=bool(mcfg.get('distill_guard_enabled', False)),
        distill_guard_min_corr=float(mcfg.get('distill_guard_min_corr', 0.0)),
        distill_guard_max_scale=float(mcfg.get('distill_guard_max_scale', 8.0)),
        distill_identified=bool(mcfg.get('distill_identified', False)),
        distill_m_latents=int(mcfg.get('distill_m_latents', 2)),
        distill_aleatoric_weight=float(mcfg.get('distill_aleatoric_weight', 1.0)),
        distill_reweight_ema=bool(mcfg.get('distill_reweight_ema', False)),
        distill_reweight_floor=float(mcfg.get('distill_reweight_floor', 1e-6)),
        distill_reweight_w_only=bool(mcfg.get('distill_reweight_w_only', False)),
        distill_corruption=str(mcfg.get('distill_corruption', 'schedule')),
        freeze_teacher=bool(mcfg.get('freeze_teacher', False)),
    )


def infer_dims(state_dict):
    """Recover (obs_dim, action_dim, x_dim) from parameter shapes.

    ``teacher.delta_mean`` has length obs_dim (the reward channel is separate).
    ``student.members.*.net.4`` is the output layer with width x_dim.
    """
    obs_dim = int(state_dict['teacher.delta_mean'].shape[0])
    x_dim = int(state_dict['student.members.0.net.4.weight'].shape[0])
    in_dim = int(state_dict['student.members.0.net.0.weight'].shape[1])
    t_embed = int(state_dict['student.members.0.t_embed.0.weight'].shape[0])
    action_dim = in_dim - x_dim - obs_dim - t_embed
    return obs_dim, action_dim, x_dim


def term_grad_params(student, term):
    """Names of student parameters receiving nonzero grad from ``term``."""
    student.zero_grad(set_to_none=True)
    term.backward(retain_graph=True)
    hit = []
    for name, p in student.named_parameters():
        if p.grad is not None and bool((p.grad != 0).any()):
            hit.append(name)
    return set(hit)


def run_case(model, cfg, arm, obs_dim, action_dim, batch, seed):
    for p in model.teacher.parameters():
        p.requires_grad_(False)
    student = model.student
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(batch, obs_dim, generator=g)
    actions = torch.rand(batch, action_dim, generator=g) * 2.0 - 1.0
    next_obs = obs + 0.1 * torch.randn(batch, obs_dim, generator=g)

    def value_fn(states, _actions):
        flat = states.reshape(-1, states.shape[-1])
        return flat.sum(-1, keepdim=True).reshape(*states.shape[:-1], 1)

    kw = ARMS[arm]
    parts = identified_decision_distill_loss(
        student, model.teacher, obs, actions, next_obs, value_fn,
        rewards=torch.zeros(batch, 1),
        m_latents=int(cfg['model'].get('distill_m_latents', 2)),
        value_weight=kw['decision_weight'],
        variance_weight=kw['variance_weight'],
        aleatoric_weight=kw['aleatoric_weight'],
        state_geometry_weight=0.0,
        state_pairwise_weight=0.0,
        normalize_values=kw['normalize'],
        reweight=kw['reweight'],
        reweight_w_only=kw['reweight_w_only'],
        corruption=kw['corruption'],
    )
    groups = {
        'point_prediction': parts['member'],
        'epistemic_w': parts['epistemic_w'],
        'aleatoric_g': parts['aleatoric_g'],
        'value_geometry': parts['value_geometry'],
    }
    hits = {k: term_grad_params(student, v) for k, v in groups.items()}
    point, unc = hits['point_prediction'], hits['epistemic_w'] | hits['aleatoric_g']
    inter, union = point & unc, point | unc
    return {
        'arm': arm,
        'identified_active': bool(kw['identified']),
        'grad_param_counts': {k: len(v) for k, v in hits.items()},
        'shared_trunk_params': sorted(point),
        'intersection_point_vs_uncertainty': sorted(inter),
        'jaccard_point_vs_uncertainty': (len(inter) / len(union)) if union else 0.0,
        'shared_trunk': bool(inter),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=(
        'checkpoints/hf_diagnostic_delayedbimodal/identified_eq_seed0.pt'))
    ap.add_argument('--config', default=None,
                    help='config to build from; default uses the checkpoint cfg')
    ap.add_argument('--obs-dim', type=int, default=None)
    ap.add_argument('--action-dim', type=int, default=None)
    ap.add_argument('--arms', nargs='+', default=['identified_eq'], choices=list(ARMS))
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs/hypothesis_c_param_sharing.json')
    args = ap.parse_args(argv)

    ck = None
    if args.checkpoint and args.checkpoint.lower() != 'none':
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if args.config is not None:
        cfg = load_config(args.config)
    elif ck is not None:
        cfg = dict(ck['cfg'])
    else:
        raise SystemExit('need --config or an existing --checkpoint')

    if ck is not None and args.obs_dim is None:
        obs_dim, action_dim, x_dim = infer_dims(ck['world_model'])
        load_sd = ck['world_model']
        source = str(ckpt_path)
    else:
        obs_dim = int(args.obs_dim)
        action_dim = int(args.action_dim)
        joint = bool(cfg['reward_term'].get('joint_with_diffusion', False))
        x_dim = obs_dim + (1 if joint else 0)
        load_sd, source = None, 'random-init'

    results = []
    for arm in args.arms:
        model = build_from_cfg(cfg, obs_dim, action_dim, arm_overrides=ARMS[arm])
        missing = unexpected = []
        if load_sd is not None:
            missing, unexpected = model.load_state_dict(load_sd, strict=False)
        model.eval()
        case = run_case(model, cfg, arm, obs_dim, action_dim, args.batch, args.seed)
        case['missing_keys'] = list(missing)
        case['unexpected_keys'] = list(unexpected)
        results.append(case)

    payload = {
        'source': source,
        'config_env': cfg['env']['id'],
        'obs_dim': obs_dim, 'action_dim': action_dim, 'x_dim': x_dim,
        'student_hidden_dims': cfg['model'].get('student_hidden_dims'),
        'results': results,
        'verdict': ('SHARED_TRUNK - Hypothesis C coherent'
                    if all(r['shared_trunk'] for r in results)
                    else 'DISJOINT - Hypothesis C moot, drop it'),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
