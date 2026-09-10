# -*- coding: utf-8 -*-
"""Gradient-interference probe (Hypothesis C) - registered 2026-09-10.

Registration: research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md.

Claim under test: the identified loss's point-prediction term (``member``) and
its uncertainty-matching terms (``epistemic_w`` + ``aleatoric_g``) backprop
through one shared student trunk (verified: see
``udwm/scripts/check_distill_param_sharing.py``), and their gradients interfere
more as task complexity rises.  Liu et al., "Measuring and Mitigating
Interference in Reinforcement Learning" (PMLR 2023) supply the methodology:
measure the conflict between the gradients of the competing objectives on the
shared parameters at matched training checkpoints.

Measurement (definitions fixed by the registration, section 4):

  For each probe checkpoint (every ``--probe-every`` env steps) and for each of
  ``--probe-batches`` freshly sampled real-replay minibatches:

    g_point = d(member) / d(theta_student)
    g_unc   = d(sum_k w_k * unc_k) / d(theta_student)      (arm's unc terms)
    cos     = <g_point, g_unc> / (||g_point|| ||g_unc|| + eps)
    interference = -cos                      (positive => conflict)

  Effective weights ``w_k`` are read from the arm's own config (the same
  multipliers training applies), so the measured gradients are the ones that
  actually compete.  The probe is measurement-only: it runs an *additional*
  forward/backward on a fresh batch after the trainer's own update and never
  steps an optimizer.  ``GRAD_PROBE`` is reset to None immediately afterwards,
  so the training path is untouched.

  ``ordinary`` has no uncertainty terms by construction (all decision weights
  are zero), so it is recorded as ``applicable: false`` and serves as the
  zero-competition reference in the arm list.

Arms: ``identified_eq`` is the test arm; ``hybrid``/``lagged_hybrid`` are the
M=1 placebo family (single-latent decision terms, no M>=2 pairing).

Output: one canonical JSON per run; ``rows`` = one record per (seed, variant)
with ``checkpoints`` and a folded final-eval metrics block.  Per-(seed,
variant) partials are written atomically and merged additively under a lock
(the same race-safe pattern as probe_crn_bias.py).

Examples:
  CPU smoke (DelayedBimodal, seconds):
    python -m udwm.scripts.probe_gradient_interference \
        --config configs/delayed_bimodal_distill.yaml \
        --seeds 0 --variants identified_eq ordinary hybrid \
        --steps 600 --probe-every 200 --out runs/gi_smoke_db.json
  DMC n=10 diagnostic (GPU, after the registration's decision tree fires):
    python -m udwm.scripts.probe_gradient_interference \
        --config configs/dmc_hopper_probe.yaml \
        --seeds 0 1 2 3 4 5 6 7 8 9 --steps 15000 --probe-every 300 \
        --out runs/gi_probe_15k_n10_gpu.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from udwm.models import consistency as C
from udwm.rl.trainer import MBPOTrainer
from udwm.scripts.run_delayed_bimodal_policy_ablation import (
    VARIANTS,
    make_cfg,
    prepare_matched_teacher,
)
from udwm.utils.config import load_config, set_seed

ROOT = Path(__file__).resolve().parents[2]

# Registered arm list for the Hypothesis-C probe.
PROBE_ARMS = [
    "ordinary",              # reference: no competing objective (N/A)
    "identified_eq",         # test arm (live critic, equal-weight identified)
    "lagged_identified_eq",  # combined fix
    "hybrid",                # placebo: M=1 decision terms, no M>=2 pairing
    "lagged_hybrid",         # placebo: lagged M=1
]

# Which returned loss term belongs to which group.  Keys not listed here are
# ignored by the probe.
TERM_GROUP = {
    "member": "point",
    "epistemic_w": "uncertainty",
    "aleatoric_g": "uncertainty",
    "value_variance": "uncertainty",
    "variance_shape": "uncertainty",
    "mean": "uncertainty",
    "geometry": "uncertainty",
    "pairwise": "uncertainty",
    "value_geometry": "geometry",
    "state_geometry": "geometry",
    "state_pairwise": "geometry",
}

# Config key that supplies the effective multiplier for each returned term
# (mirrors how DistilledWorldModel trains the term).
TERM_WEIGHT_KEY = {
    "member": None,                        # always weight 1
    "epistemic_w": "distill_value_variance_weight",
    "aleatoric_g": "distill_aleatoric_weight",
    "value_geometry": "distill_decision_weight",
    "value_variance": "distill_value_variance_weight",
    "variance_shape": "distill_variance_shape_weight",
    "state_geometry": "distill_hybrid_state_weight",
    "state_pairwise": "distill_hybrid_pairwise_weight",
    "mean": "distill_mean_weight",
    "geometry": "distill_geometry_weight",
    "pairwise": "distill_pairwise_weight",
}


def term_weight(mcfg, term) -> float:
    key = TERM_WEIGHT_KEY.get(term)
    if key is None:
        return 1.0
    return float(mcfg.get(key, 0.0))


def flat_student_grad(student, tensor) -> torch.Tensor:
    """Gradient of ``tensor`` w.r.t. all student parameters, flattened."""
    student.zero_grad(set_to_none=True)
    tensor.backward(retain_graph=True)
    chunks = []
    for p in student.parameters():
        g = p.grad
        chunks.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(chunks)


class GradientInterferenceProbeTrainer(MBPOTrainer):
    """MBPOTrainer + a measurement-only gradient-interference hook.

    The hook runs immediately after the trainer's own world-model update at
    each probe checkpoint.  It never calls an optimizer step and always resets
    ``consistency.GRAD_PROBE`` to None.
    """

    def __init__(self, cfg, probe_every=300, probe_batches=3, arm="", seed_id=0):
        super().__init__(cfg)
        self.probe_every = int(probe_every)
        self.probe_batches = int(probe_batches)
        self.arm = arm
        self.seed_id = int(seed_id)
        self.gi_checkpoints = []

    def _train_world_model(self):
        info = super()._train_world_model()
        if self.probe_every > 0 and self.total_steps % self.probe_every == 0:
            rec = self._measure_interference()
            if rec is not None:
                rec["step"] = int(self.total_steps)
                self.gi_checkpoints.append(rec)
        return info

    def _measure_interference(self):
        bs = int(self.cfg["mbpo"]["model_batch_size"])
        if len(self.real_buffer) < bs:
            return None
        value_fn = self._make_distill_value_fn()
        mcfg = self.cfg["model"]
        wm = self.world_model
        student = wm.student
        cos_list, pn_list, un_list = [], [], []
        term_values = {}
        applicable = False
        for _ in range(self.probe_batches):
            batch = self.real_buffer.sample(bs)
            C.GRAD_PROBE = {}
            try:
                with torch.enable_grad():
                    wm.train_loss(
                        batch["obs"], batch["actions"], batch["next_obs"],
                        batch["rewards"], batch["dones"], value_fn=value_fn,
                    )
                parts = C.GRAD_PROBE.get("parts")
            finally:
                C.GRAD_PROBE = None
            if parts is None:
                student.zero_grad(set_to_none=True)
                break
            point = parts.get("member")
            unc_terms = [k for k, v in parts.items()
                         if TERM_GROUP.get(k) == "uncertainty" and term_weight(mcfg, k) != 0.0]
            if point is None or not unc_terms:
                student.zero_grad(set_to_none=True)
                break
            applicable = True
            unc = None
            for k in unc_terms:
                t = term_weight(mcfg, k) * parts[k]
                unc = t if unc is None else unc + t
            g_point = flat_student_grad(student, point)
            g_unc = flat_student_grad(student, unc)
            student.zero_grad(set_to_none=True)
            denom = (g_point.norm() * g_unc.norm()).clamp_min(1e-12)
            cos = float((g_point @ g_unc) / denom)
            cos_list.append(cos)
            pn_list.append(float(g_point.norm()))
            un_list.append(float(g_unc.norm()))
            for k, v in parts.items():
                try:
                    term_values[k] = float(v.detach().mean())
                except Exception:
                    pass
        if not applicable:
            return {"applicable": False, "reason": "no uncertainty term active",
                    "arm": self.arm}
        cos_mean = float(np.mean(cos_list))
        return {
            "applicable": True,
            "arm": self.arm,
            "cos_point_unc": cos_mean,
            "interference": -cos_mean,
            "cos_per_batch": cos_list,
            "grad_norm_point": float(np.mean(pn_list)),
            "grad_norm_unc": float(np.mean(un_list)),
            "unc_terms": sorted(unc_terms),
            "term_values": term_values,
        }


def partial_path(out: Path, seed: int, variant: str) -> Path:
    return out.with_name("%s_s%d_%s.partial.json" % (out.stem, seed, variant))


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu()) if x.dim() == 0 else x.detach().cpu().tolist()
    return x


def run_one(args, base, seed, variant, out, prepared_buffer, prepared_teacher):
    cfg = make_cfg(base, variant, seed, args.steps)
    if args.student_hidden:
        # E3 capacity axis (G8 knob): override the student width at fixed weights.
        cfg['model']['student_hidden_dims'] = [int(w) for w in args.student_hidden]
    set_seed(seed)
    trainer = GradientInterferenceProbeTrainer(
        cfg, probe_every=args.probe_every, probe_batches=args.probe_batches,
        arm=variant, seed_id=seed)
    trainer.real_buffer = copy.deepcopy(prepared_buffer)
    trainer.world_model.teacher.load_state_dict(prepared_teacher)
    trainer.world_model.freeze_teacher()
    trainer.wm_opt = torch.optim.Adam(trainer.world_model.student.parameters(), lr=1e-3)
    result = trainer.train()
    final = result["final_metrics"]
    if trainer.gi_checkpoints:
        last = trainer.gi_checkpoints[-1]
        for key in ("u_rank_corr", "w_rmse", "next_state_mse", "return_mean",
                    "teacher_u_mean", "student_u_mean",
                    "teacher_w_mean", "student_w_mean",
                    "teacher_g_mean", "student_g_mean"):
            if key in final:
                last[key] = float(final[key])
        trainer.gi_checkpoints[-1] = last
    row = {
        "seed": int(seed),
        "variant": variant,
        "final_return": float(result["final_eval_return"]),
        "teacher_frozen": bool(getattr(trainer.world_model, "teacher_frozen", False)),
        "teacher_final_checksum": (
            trainer._parameter_checksum(trainer.world_model.teacher)
            if hasattr(trainer.world_model, "teacher") else None),
        "final_metrics": _json_safe(final),
        "gi_checkpoints": _json_safe(trainer.gi_checkpoints),
    }
    p = partial_path(out, int(seed), variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"seed": seed, "variant": variant, "ok": True,
                      "final_return": row["final_return"],
                      "n_checkpoints": len(trainer.gi_checkpoints)}))
    return row


def build_pairing(rows):
    pairing = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        checks = [r["teacher_final_checksum"] for r in rows if int(r["seed"]) == seed]
        if checks:
            pairing[str(seed)] = {
                "max_teacher_checksum_gap": (
                    float(max(checks) - min(checks)) if len(checks) > 1 else 0.0),
                "exact_teacher_match": all(abs(c - checks[0]) < 1e-9 for c in checks),
                "n_arms": len(checks),
            }
    return pairing


def _read_row_map(out):
    if not out.exists():
        return {}
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
        return {(int(r["seed"]), r["variant"]): r for r in payload.get("rows", [])}
    except (ValueError, KeyError, TypeError):
        return {}


def _protocol_mismatch(out, protocol):
    try:
        old = json.loads(out.read_text(encoding="utf-8")).get("protocol", {})
    except ValueError:
        return True
    keys = ("config", "steps", "probe_every", "probe_batches", "student_hidden",
            "registration")
    return any(old.get(k) != protocol.get(k) for k in keys)


def merge_out_additive(out, added_rows, expected, protocol, variants, *,
                       keep_partials=False, delete_partials=True):
    """Union added rows into the canonical --out under an advisory lock.

    Same race-safe pattern as probe_crn_bias.py: read-existing -> union ->
    validate completeness -> atomic rename, so concurrent finishers sharing one
    --out cannot overwrite each other's rows.
    """
    added = {(int(r["seed"]), r["variant"]): r for r in added_rows}
    lock_fd = None
    try:
        import fcntl
        lock_fd = open(out.with_name(out.name + ".lock"), "a+")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass
    try:
        merged = _read_row_map(out)
        if merged and _protocol_mismatch(out, protocol):
            print("[gi] FATAL: existing rows in {} are from a different protocol; "
                  "refusing to union. Use a fresh --out.".format(out.name))
            return False
        merged.update(added)
        missing = sorted(expected - set(merged))
        rows = [merged[k] for k in sorted(merged)]
        if missing:
            shown = ", ".join("s{}:{}".format(s, v) for s, v in missing[:8])
            more = " ..." if len(missing) > 8 else ""
            print("[gi] FATAL: expected {} (seed, variant) rows, found {}; "
                  "{} missing: {}{}".format(len(expected), len(rows),
                                            len(missing), shown, more))
            return False
        payload = {"protocol": protocol, "rows": rows,
                   "teacher_pairing": build_pairing(rows)}
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, out)
        print("[gi] merged {} rows -> {} (expected {})".format(
            len(rows), out, len(expected)))
        if delete_partials and not keep_partials:
            for seed in sorted({int(s) for s, _ in expected}):
                for variant in variants:
                    partial_path(out, seed, variant).unlink(missing_ok=True)
            print("[gi] removed partial files (--keep-partials to retain)")
        return True
    finally:
        if lock_fd is not None:
            try:
                import fcntl
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_fd.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/delayed_bimodal_distill.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--variants", nargs="+", choices=PROBE_ARMS, default=PROBE_ARMS)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--out", default="runs/gi_probe.json")
    p.add_argument("--device", default=None)
    p.add_argument("--probe-every", type=int, default=300,
                   help="measure at every this many env steps (registered: 300)")
    p.add_argument("--probe-batches", type=int, default=3,
                   help="fresh minibatches averaged per checkpoint (registered: 3)")
    p.add_argument("--student-hidden", type=int, nargs="+", default=None,
                   help="E3 capacity axis: override student hidden width "
                        "(e.g. --student-hidden 16 16); default keeps the config")
    p.add_argument("--merge", action="store_true",
                   help="only merge existing per-(seed,variant) partials")
    p.add_argument("--keep-partials", action="store_true")
    args = p.parse_args(argv)

    base = load_config(args.config)
    if args.device is not None:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        base["device"] = args.device
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.merge:
        for seed in args.seeds:
            for variant in args.variants:
                pth = partial_path(out, seed, variant)
                if pth.exists():
                    rows.append(json.loads(pth.read_text(encoding="utf-8")))
                else:
                    print("[merge] missing partial for seed {} {}".format(seed, variant))
    else:
        for seed in args.seeds:
            pending = [v for v in args.variants
                       if not partial_path(out, seed, v).exists()]
            if pending:
                prepared_buffer, prepared_teacher = prepare_matched_teacher(
                    base, seed, args.steps)
            else:
                prepared_buffer = prepared_teacher = None
            for variant in args.variants:
                pth = partial_path(out, seed, variant)
                if pth.exists():
                    rows.append(json.loads(pth.read_text(encoding="utf-8")))
                    print("[skip] seed {} {} already done".format(seed, variant))
                    continue
                rows.append(run_one(args, base, seed, variant, out,
                                    prepared_buffer, prepared_teacher))

    if not rows and not out.exists():
        raise SystemExit("no rows produced")
    dedup = {}
    for r in rows:
        dedup[(int(r["seed"]), r["variant"])] = r
    rows = list(dedup.values())
    protocol = {
        "config": args.config,
        "steps": int(args.steps),
        "probe_every": int(args.probe_every),
        "probe_batches": int(args.probe_batches),
        "student_hidden": ([int(w) for w in args.student_hidden]
                           if args.student_hidden else None),
        "registration": "research/HYPOTHESIS-C-PREREGISTRATION-2026-09-10.md",
        "driver": "probe_gradient_interference.py (per seed x variant partials, additive locked merge)",
        "device": base.get("device"),
    }
    expected = {(s, v) for s in args.seeds for v in args.variants}
    ok = merge_out_additive(out, rows, expected, protocol, args.variants,
                            keep_partials=args.keep_partials,
                            delete_partials=not args.merge)
    if not ok:
        raise SystemExit("[gi] FATAL: canonical output incomplete; partials retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
