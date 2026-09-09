# -*- coding: utf-8 -*-
"""CRN-bias probe harness (Hypothesis A) - registered 2026-09-09.

Registration: research/CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md. This
script INSTRUMENTS the exact DMC/DelayedBimodal protocol (same configs, same
per-seed matched teacher, same make_cfg arm definitions) and adds the
registered probe measurements; it changes no training math.

Measurements (definitions fixed by the registration, section 4):

  M1 critic drift rate D(t): mean |Q_live - Q_ref| over a fixed probe-state
     set drawn once from the real replay buffer (states only; actions from
     the frozen deterministic actor). Q_ref is the reference critic used by
     the arm's distill loss (live critic for the live-critic arms, the polyak
     target Q_{t-k} for the lagged arms); Q_live is the live critic used at
     eval/gating time. k_eff = (1-tau)/tau is the polyak staleness in critic
     updates.
  M2 pairing gap G_pair(t): the number of critic parameter updates between
     the teacher-side and student-side value evaluations of a paired draw
     inside the identified loss forward. Both paired evaluations happen in
     ONE forward against the SAME frozen reference closure, so G_pair is
     structurally 0 for every arm; the per-forward teacher/student Q means
     and |Q_t - Q_s| of the paired evals are recorded anyway (measurement
     per the registration), via the flag-guarded hook in
     udwm/models/consistency.py (_crn_record / CRN_PROBE).
  M3 induced-bias proxy B_hat(t): re-score the student's (and teacher's)
     imagined next-states from a paired M-sample pass under BOTH the
     reference critic and the live critic, then
        B_hat(t) = mean over probe states |mean_ij s_ij(Q_live)
                                             - mean_ij s_ij(Q_ref)|
     with B_w/B_g/B_u the same delta on the per-state debiased w/g/u, and
     e_w/e_g/e_u the student-vs-teacher estimation errors under the live
     critic (the terms the summarizer regresses on the logged drift).
  M4 u_rank / w_rmse / g ratio at the same checkpoints are the existing
     evaluate_full endpoints (unchanged definitions); they are folded into
     the same checkpoint rows.

Output: one canonical JSON per run, rows = one record per (seed, variant):
  - "checkpoints": [{step, drift, b_hat, b_w, b_g, e_*, u_rank_corr, ...}]
    one row per full eval (eval_freq milestones + the final 10-episode eval;
    the LAST record at a duplicated step is the fresh eval, matching the
    registered forward-fill convention);
  - "steps": [{step, crn_*} ...] one row per model-train call (the M2
    pairing-gap logs + per-batch measured w/g means are folded in from the
    trainer info dict).

Resume/parallel: per-(seed, variant) partial files are written atomically;
launch several processes with disjoint --seeds/--variants against the same
--out and finish with --merge (no git or shared-state needed).

Examples (smoke, CPU, DelayedBimodal):
    python -m udwm.scripts.probe_crn_bias --config configs/delayed_bimodal_distill.yaml \
        --seeds 0 1 --variants identified_eq lagged_identified_eq hybrid \
        --steps 600 --probe-states 96 --probe-batch 96 \
        --out runs/crn_bias_smoke_db.json
Full DMC n=10 diagnostic (GPU, after registration's decision tree fires):
    python -m udwm.scripts.probe_crn_bias --config configs/dmc_hopper_probe.yaml \
        --seeds 0 1 2 3 4 5 6 7 8 9 --steps 15000 \
        --out runs/crn_bias_probe_15k_n10_gpu.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from udwm.models import consistency as C
from udwm.models.consistency import lagged_target_value_fn
from udwm.rl.trainer import MBPOTrainer
from udwm.scripts.run_delayed_bimodal_policy_ablation import (
    make_cfg,
    prepare_matched_teacher,
)
from udwm.uncertainty.mc_ube import MCUBELocalRewards
from udwm.utils.config import load_config, set_seed

ROOT = Path(__file__).resolve().parents[2]

# The registered 2x2 + control arms (research/CRN-BIAS-PROBE-PREREGISTRATION).
PROBE_ARMS = [
    "identified_eq",            # A: live critic, no normalization
    "identified_eq_norm",       # B: live critic, normalized values
    "lagged_identified_eq",     # C: lagged target critic, normalized
    "lagged_identified_eq_nonorm",  # D: lagged target critic, no normalization
    "hybrid",                   # M=1 single-evaluation control (live)
    "lagged_hybrid",            # M=1 single-evaluation control (lagged)
    "identified_hybrid",        # EMA collapse sanity arm
]

CHECKPOINT_KEYS = [
    "drift", "drift_rel", "b_hat", "b_w", "b_g", "b_u",
    "e_u", "e_w", "e_g", "e_u_ref", "q_scale",
    "u_rank_corr", "w_rmse", "teacher_u_mean", "student_u_mean",
    "teacher_w_mean", "student_w_mean", "teacher_g_mean", "student_g_mean",
]


def _value_fn_live(agent):
    """q_min under a stop-gradient deterministic actor (the eval/gating critic)."""

    def value_fn(states, _actions):
        flat = states.reshape(-1, states.shape[-1])
        with torch.no_grad():
            action = agent.actor.deterministic(flat)
        return agent.q_min(flat, action).reshape(*states.shape[:-1], 1)

    return value_fn


class CRNBiasProbeTrainer(MBPOTrainer):
    """MBPOTrainer subclass adding the registered M1/M2/M3 instrumentation."""

    def __init__(self, cfg, probe_states=1024, probe_batch=128, probe_m=2,
                 arm=None, seed_id=None):
        super().__init__(cfg)
        self._arm = arm or str(cfg["model"].get("variant", "?"))
        self._seed_id = int(seed_id if seed_id is not None else cfg.get("seed", 0))
        self._probe_states = int(probe_states)
        self._probe_batch = max(16, int(probe_batch))
        self._probe_m = max(2, int(probe_m))
        self._probe_obs = None
        self._probe_act = None
        self._use_target = bool(self.cfg["model"].get("distill_use_target_critic", False))
        tau = max(float(self.cfg["agent"].get("tau", 0.01)), 1e-6)
        self._k_eff = float((1.0 - tau) / tau)
        self.crn_steps = []     # one row per model-train call (M2)
        self.crn_checkpoints = []  # one row per full eval (M1/M3 + M4)
        self._est = MCUBELocalRewards(u_min=-1e9, m_samples=max(2, self._probe_m),
                                      debias=True)

    # -- M2: fold the per-forward pairing log of the last distill call ------
    def _train_world_model(self):
        had = C.CRN_PROBE
        C.CRN_PROBE = {"last": None}
        try:
            info = super()._train_world_model()
        finally:
            last = C.CRN_PROBE.get("last") if C.CRN_PROBE is not None else None
            C.CRN_PROBE = had
        if last is not None:
            row = {"step": int(self.total_steps), "variant": self._arm,
                   "seed": self._seed_id}
            for key in ("q_teacher_mean", "q_student_mean", "q_teacher_absmean",
                        "q_student_absmean", "gap_critic_updates",
                        "n_states", "m_latents"):
                if key in last:
                    row[key] = last[key]
            for key in ("distill_teacher_w_mean", "distill_student_w_mean",
                        "distill_teacher_g_mean", "distill_student_g_mean",
                        "distill_teacher_local_u_mean", "distill_student_local_u_mean",
                        "wm_loss"):
                if key in info:
                    row[key] = float(info[key])
            self.crn_steps.append(row)
            info["crn_pair_gap"] = float(last["gap_critic_updates"])
            info["crn_q_pair_mean"] = float(
                abs(last["q_teacher_mean"] - last["q_student_mean"])
            )
        return info

    # -- M1 + M3: drift + bias proxy at every full eval ---------------------
    def evaluate_full(self, n_episodes: int = 10):
        out = super().evaluate_full(n_episodes=n_episodes)
        step = int(self.total_steps)
        probe = self._probe_pass(step)
        if probe is not None:
            # M4 (existing endpoints) from the same full eval, folded into the
            # checkpoint row so the summarizer reads one record per step.
            for key in ("u_rank_corr", "w_rmse", "teacher_u_mean", "student_u_mean",
                        "teacher_w_mean", "student_w_mean",
                        "teacher_g_mean", "student_g_mean"):
                if key in out:
                    probe[key] = float(out[key])
            self.crn_checkpoints.append(probe)
            for key in CHECKPOINT_KEYS:
                if key in probe and key not in ("u_rank_corr", "w_rmse",
                                                "teacher_u_mean", "student_u_mean",
                                                "teacher_w_mean", "student_w_mean",
                                                "teacher_g_mean", "student_g_mean"):
                    out[f"crn_{key}"] = probe[key]
        return out

    def _probe_set(self):
        """Fixed probe states/actions drawn once from the real replay buffer."""
        if self._probe_obs is None:
            buf = self.real_buffer
            n = min(int(self._probe_states), len(buf))
            self._probe_obs = torch.as_tensor(
                np.ascontiguousarray(buf.obs[:n]), device=self.device).clone()
            self._probe_act = torch.as_tensor(
                np.ascontiguousarray(buf.actions[:n]), device=self.device).clone()
        return self._probe_obs, self._probe_act

    @torch.no_grad()
    def _probe_pass(self, step):
        buf = self.real_buffer
        if buf is None or len(buf) < max(64, self._probe_batch):
            return None
        obs_all, act_all = self._probe_set()
        q_live = _value_fn_live(self.agent)
        q_ref = q_live if not self._use_target else lagged_target_value_fn(
            self.agent.critic_target, self.agent.actor)

        # M1: deterministic-actor actions on the fixed states, both critics.
        with torch.no_grad():
            a_det = self.agent.actor.deterministic(obs_all)
            q_l = q_live(obs_all, a_det)
            q_r = q_ref(obs_all, a_det)
        drift = float((q_l - q_r).abs().mean())
        q_scale = float(q_l.abs().mean())
        drift_rel = float(drift / (q_scale + 1e-12))

        # M3: re-score paired teacher/student next states under both critics.
        n = self.world_model.ensemble_size
        m = self._probe_m
        x_dim = self.world_model.teacher.x_dim
        acc = {k: 0.0 for k in
               ("b_hat", "b_w", "b_g", "b_u", "e_u", "e_w", "e_g", "e_u_ref")}
        n_states = 0
        b = self._probe_batch
        teacher, student = self.world_model.teacher, self.world_model.student
        for start in range(0, obs_all.shape[0], b):
            obs = obs_all[start:start + b]
            act = act_all[start:start + b]
            tq_l, tq_r, sq_l, sq_r = [], [], [], []
            for _j in range(m):
                z = torch.randn(obs.shape[0], x_dim, device=obs.device)
                t_l, t_r, s_l, s_r = [], [], [], []
                for i in range(n):
                    t_x = teacher._ddim_sample_member(
                        i, obs, act, deterministic=False, x_T=z)
                    t_next = obs + teacher._unpack_x(t_x)[0]
                    s_next, _r = student.sample_next(
                        teacher, obs, act, member=i, x_T=z)
                    t_l.append(q_live(t_next, None))
                    t_r.append(q_ref(t_next, None))
                    s_l.append(q_live(s_next, None))
                    s_r.append(q_ref(s_next, None))
                tq_l.append(torch.stack(t_l, dim=0))   # [N,B,1]
                tq_r.append(torch.stack(t_r, dim=0))
                sq_l.append(torch.stack(s_l, dim=0))
                sq_r.append(torch.stack(s_r, dim=0))
            tq_l = torch.stack(tq_l, dim=1)            # [N,M,B,1]
            tq_r = torch.stack(tq_r, dim=1)
            sq_l = torch.stack(sq_l, dim=1)
            sq_r = torch.stack(sq_r, dim=1)
            t_live = self._est.combine_coupled(tq_l)
            t_ref = self._est.combine_coupled(tq_r)
            s_live = self._est.combine_coupled(sq_l)
            s_ref = self._est.combine_coupled(sq_r)
            def flat(x):
                return x.reshape(-1).cpu().numpy()
            b_hat = np.abs(flat(sq_l.mean(dim=(0, 1))) - flat(sq_r.mean(dim=(0, 1))))
            b_w = np.abs(flat(s_live["w"]) - flat(s_ref["w"]))
            b_g = np.abs(flat(s_live["g"]) - flat(s_ref["g"]))
            b_u = np.abs(flat(s_live["u"]) - flat(s_ref["u"]))
            e_u = np.abs(flat(s_live["u"]) - flat(t_live["u"]))
            e_w = np.abs(flat(s_live["w"]) - flat(t_live["w"]))
            e_g = np.abs(flat(s_live["g"]) - flat(t_live["g"]))
            e_u_ref = np.abs(flat(s_ref["u"]) - flat(t_ref["u"]))
            nb = int(obs.shape[0])
            for key, arr in (("b_hat", b_hat), ("b_w", b_w), ("b_g", b_g),
                             ("b_u", b_u), ("e_u", e_u), ("e_w", e_w),
                             ("e_g", e_g), ("e_u_ref", e_u_ref)):
                acc[key] += float(arr.sum())
            n_states += nb
        if n_states == 0:
            return None
        probe = {"step": step, "variant": self._arm, "seed": self._seed_id,
                 "k_eff": self._k_eff, "drift": drift,
                 "drift_rel": drift_rel, "q_scale": q_scale,
                 "n_probe_states": n_states}
        for key in acc:
            probe[key] = acc[key] / float(n_states)
        return probe


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


def partial_path(out: Path, seed: int, variant: str) -> Path:
    return out.with_name(f"{out.stem}_s{seed}_{variant}.partial.json")


def run_one(args, base, seed, variant, out, prepared_buffer, prepared_teacher):
    cfg = make_cfg(base, variant, seed, args.steps)
    set_seed(seed)
    trainer = CRNBiasProbeTrainer(
        cfg, probe_states=args.probe_states, probe_batch=args.probe_batch,
        probe_m=args.probe_m, arm=variant, seed_id=seed)
    trainer.real_buffer = copy.deepcopy(prepared_buffer)
    trainer.world_model.teacher.load_state_dict(prepared_teacher)
    trainer.world_model.freeze_teacher()
    trainer.wm_opt = torch.optim.Adam(trainer.world_model.student.parameters(), lr=1e-3)
    result = trainer.train()
    final = result["final_metrics"]
    # Final fresh-eval checkpoint (the 10-episode evaluate_full at the end of
    # train()): fold M4 endpoints into the last probe pass it recorded.
    if trainer.crn_checkpoints:
        last_cp = trainer.crn_checkpoints[-1]
        for key in ("u_rank_corr", "w_rmse", "teacher_u_mean", "student_u_mean",
                    "teacher_w_mean", "student_w_mean",
                    "teacher_g_mean", "student_g_mean"):
            if key in final:
                last_cp[key] = float(final[key])
        trainer.crn_checkpoints[-1] = last_cp
    row = {
        "seed": int(seed),
        "variant": variant,
        "final_return": float(result["final_eval_return"]),
        "teacher_frozen": bool(getattr(trainer.world_model, "teacher_frozen", False)),
        "teacher_updates": int(cfg["model"].get("distill_teacher_pretrain_updates", 0)),
        "teacher_initial_checksum": getattr(trainer, "_teacher_initial_checksum", None),
        "teacher_final_checksum": (
            trainer._parameter_checksum(trainer.world_model.teacher)
            if hasattr(trainer.world_model, "teacher") else None
        ),
        "final_metrics": _json_safe(final),
        "crn_checkpoints": _json_safe(trainer.crn_checkpoints),
        "crn_steps": _json_safe(trainer.crn_steps),
    }
    p = partial_path(out, int(seed), variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps({"seed": seed, "variant": variant, "ok": True,
                      "final_return": row["final_return"],
                      "n_checkpoints": len(trainer.crn_checkpoints),
                      "n_steps": len(trainer.crn_steps)}))
    return row


def build_pairing(rows):
    pairing = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        checks = [r["teacher_final_checksum"] for r in rows if int(r["seed"]) == seed]
        if checks:
            pairing[str(seed)] = {
                "max_teacher_checksum_gap": float(max(checks) - min(checks)) if len(checks) > 1 else 0.0,
                "exact_teacher_match": all(abs(c - checks[0]) < 1e-9 for c in checks),
                "n_arms": len(checks),
            }
    return pairing


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/delayed_bimodal_distill.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--variants", nargs="+", choices=PROBE_ARMS, default=PROBE_ARMS)
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--out", default="runs/crn_bias_probe.json")
    p.add_argument("--device", default=None)
    p.add_argument("--probe-states", type=int, default=1024,
                   help="size of the persistent probe set (registered: 1024)")
    p.add_argument("--probe-batch", type=int, default=128)
    p.add_argument("--probe-m", type=int, default=2,
                   help="paired latents in the M3 re-scoring pass (registered: 2)")
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
                    print(f"[merge] missing partial for seed {seed} {variant}")
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
                    print(f"[skip] seed {seed} {variant} already done")
                    continue
                rows.append(run_one(args, base, seed, variant, out,
                                    prepared_buffer, prepared_teacher))

    if not rows:
        raise SystemExit("no rows produced")
    # dedupe: last wins per (seed, variant)
    dedup = {}
    for r in rows:
        dedup[(int(r["seed"]), r["variant"])] = r
    rows = [dedup[k] for k in sorted(dedup)]
    payload = {
        "protocol": {
            "config": args.config,
            "seeds": list(args.seeds),
            "steps": int(args.steps),
            "variants": list(args.variants),
            "probe_states": int(args.probe_states),
            "probe_m": int(args.probe_m),
            "registration": "research/CRN-BIAS-PROBE-PREREGISTRATION-2026-09-09.md",
            "device": args.device if args.device else base.get("device", "cpu"),
            "driver": "probe_crn_bias.py (per seed x variant partials, merged)",
        },
        "rows": rows,
        "teacher_pairing": build_pairing(rows),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[probe] merged {len(rows)} rows -> {out}")
    if not args.keep_partials and not args.merge:
        for seed in args.seeds:
            for variant in args.variants:
                partial_path(out, seed, variant).unlink(missing_ok=True)
        print("[probe] removed partial files (--keep-partials to retain)")


if __name__ == "__main__":
    main()
