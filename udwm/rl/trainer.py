from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import trange

from udwm.data.audit_log import GateAuditLog
from udwm.data.replay_buffer import ReplayBuffer
from udwm.eval.metrics import (
    benchmark_distilled_sampling,
    evaluate_distillation_uncertainty,
    evaluate_policy_return,
    evaluate_uncertainty_calibration,
    evaluate_world_model_accuracy,
)
from udwm.eval.selective import collect_score_and_error, selective_report
from udwm.models.world_model import WorldModel
from udwm.rl.sac import SACAgent
from udwm.rl.u_gated_imagination import u_gated_rollout
from udwm.uncertainty.adaptive_mc import AdaptiveMCUBELocalRewards
from udwm.uncertainty.baselines import one_step_state_disagreement
from udwm.uncertainty.mc_ube import MCUBELocalRewards, UNetwork, ube_loss
from udwm.utils.torch_utils import get_device


class MBPOTrainer:
    """MBPO-style loop with optional U-gated imagination (A) and adaptive MC-UBE (B)."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.get("device", "cpu"))
        self.seed = int(cfg.get("seed", 0))

        from udwm.envs.registry import make_env, space_info

        env_id = cfg["env"]["id"]
        self.env = make_env(env_id, seed=self.seed)
        self.eval_env = make_env(env_id, seed=self.seed + 1)
        info = space_info(self.env)
        self.obs_dim = info["obs_dim"]
        self.action_dim = info["action_dim"]
        self.action_low = info["action_low"]
        self.action_high = info["action_high"]

        mcfg = cfg["model"]
        rcfg = cfg["reward_term"]
        self.world_model = WorldModel.build(
            model_type=mcfg["type"],
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            ensemble_size=int(mcfg["ensemble_size"]),
            hidden_dims=tuple(mcfg["hidden_dims"]),
            student_hidden_dims=(tuple(mcfg["student_hidden_dims"]) if mcfg.get("student_hidden_dims") else None),
            diffusion_steps=int(mcfg.get("diffusion_steps", 10)),
            beta_start=float(mcfg.get("beta_start", 1e-4)),
            beta_end=float(mcfg.get("beta_end", 2e-2)),
            sample_steps=int(mcfg.get("sample_steps", 4)),
            reward_hidden=tuple(rcfg.get("hidden_dims", [128, 128])),
            joint_with_diffusion=bool(rcfg.get("joint_with_diffusion", False)),
            use_consistency_distill=bool(mcfg.get("use_consistency_distill", False)),
            preserve_distilled_uncertainty=bool(mcfg.get("preserve_distilled_uncertainty", False)),
            distill_mean_weight=float(mcfg.get("distill_mean_weight", 1.0)),
            distill_geometry_weight=float(mcfg.get("distill_geometry_weight", 1.0)),
            distill_pairwise_weight=float(mcfg.get("distill_pairwise_weight", 1.0)),
            distill_decision_weight=float(mcfg.get("distill_decision_weight", 0.0)),
            distill_value_variance_weight=float(mcfg.get("distill_value_variance_weight", 0.0)),
            distill_hybrid_state_weight=float(mcfg.get("distill_hybrid_state_weight", 0.0)),
            distill_hybrid_pairwise_weight=float(mcfg.get("distill_hybrid_pairwise_weight", 0.0)),
            distill_normalize_values=bool(mcfg.get("distill_normalize_values", False)),
            distill_guard_enabled=bool(mcfg.get("distill_guard_enabled", False)),
            distill_guard_min_corr=float(mcfg.get("distill_guard_min_corr", 0.0)),
            distill_guard_max_scale=float(mcfg.get("distill_guard_max_scale", 8.0)),
            distill_identified=bool(mcfg.get("distill_identified", False)),
            distill_m_latents=int(mcfg.get("distill_m_latents", 2)),
            distill_aleatoric_weight=float(mcfg.get("distill_aleatoric_weight", 1.0)),
            distill_reweight_ema=bool(mcfg.get("distill_reweight_ema", False)),
            distill_reweight_floor=float(mcfg.get("distill_reweight_floor", 1e-6)),
            distill_reweight_w_only=bool(mcfg.get("distill_reweight_w_only", False)),
            distill_corruption=str(mcfg.get("distill_corruption", "schedule")),
            freeze_teacher=bool(mcfg.get("freeze_teacher", False)),
        ).to(self.device)

        self.distill_two_stage = bool(mcfg.get("distill_two_stage", False))
        self.distill_freeze_after = int(mcfg.get("distill_freeze_after_steps", 0))
        self.distill_teacher_pretrain_updates = mcfg.get("distill_teacher_pretrain_updates")
        if self.distill_teacher_pretrain_updates is not None:
            self.distill_teacher_pretrain_updates = int(self.distill_teacher_pretrain_updates)
        self.distill_grad_clip = float(mcfg.get("distill_grad_clip", 10.0))
        # Number of successful teacher optimizer updates.  This guards against
        # freezing the teacher before it has received any gradient update when
        # the first model-training call happens at the warmup boundary.
        self._teacher_updates = 0
        self._critic_updates = 0
        self._teacher_checksum_before_freeze = None
        self._teacher_checksum_after_freeze = None
        self._teacher_initial_checksum = (
            self._parameter_checksum(self.world_model.teacher)
            if hasattr(self.world_model, "teacher") else None
        )
        if self.distill_two_stage and hasattr(self.world_model, "teacher"):
            self.wm_opt = torch.optim.Adam(self.world_model.teacher.parameters(), lr=1e-3)
        else:
            self.wm_opt = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

        acfg = cfg["agent"]
        self.agent = SACAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            gamma=float(acfg["gamma"]),
            tau=float(acfg["tau"]),
            actor_lr=float(acfg["actor_lr"]),
            critic_lr=float(acfg["critic_lr"]),
            alpha_lr=float(acfg["alpha_lr"]),
            auto_alpha=bool(acfg["auto_alpha"]),
            target_entropy=acfg.get("target_entropy"),
            hidden_dims=tuple(acfg["hidden_dims"]),
            action_low=self.action_low,
            action_high=self.action_high,
            optimism_lambda=float(acfg.get("optimism_lambda", 0.0)),
            use_ube=bool(acfg.get("use_ube", True)),
        )

        ucfg = cfg["ube"]
        self.u_net = UNetwork(self.obs_dim, self.action_dim, tuple(ucfg["hidden_dims"])).to(
            self.device
        )
        self.agent.attach_u_net(self.u_net, lr=float(acfg["critic_lr"]))

        # --- Novelty B: adaptive MC budget ---
        acfg_mc = ucfg.get("adaptive_mc", {}) or {}
        if bool(acfg_mc.get("enabled", False)):
            self.mc_ube: MCUBELocalRewards = AdaptiveMCUBELocalRewards(
                u_min=float(ucfg.get("u_min", 0.0)),
                m_min=int(acfg_mc.get("m_min", 2)),
                m_max=int(acfg_mc.get("m_max", ucfg.get("m_samples", 8))),
                m_probe=acfg_mc.get("m_probe"),
                refine_frac=float(acfg_mc.get("refine_frac", 0.5)),
                w_refine_percentile=float(acfg_mc.get("w_refine_percentile", 0.5)),
                k_min=acfg_mc.get("k_min", mcfg.get("sample_steps")),
                k_max=acfg_mc.get("k_max"),
                enabled=True,
                sample_split=bool(acfg_mc.get("sample_split", True)),
            )
        else:
            self.mc_ube = MCUBELocalRewards(
                u_min=float(ucfg.get("u_min", 0.0)),
                m_samples=int(ucfg.get("m_samples", 8)),
            )

        # --- Novelty A: U-gated imagination ---
        self.u_gate_cfg = cfg.get("u_gate", {}) or {}
        self.u_gate_mode = str(self.u_gate_cfg.get("mode", "off")).lower()
        self.u_gate_enable_after = int(self.u_gate_cfg.get("enable_after_steps", 0))
        self.prioritized_model = bool(self.u_gate_cfg.get("prioritized_model_sampling", True))
        audit_path = self.u_gate_cfg.get("audit_path")
        if audit_path is None:
            audit_path = str(Path(cfg.get("paths", {}).get("log_dir", "runs")) / "gate_audit.jsonl")
        self.audit = GateAuditLog(
            path=audit_path,
            enabled=bool(self.u_gate_cfg.get("audit", True)) and self.u_gate_mode != "off",
        )

        mbcfg = cfg["mbpo"]
        self.real_buffer = ReplayBuffer(
            int(mbcfg["buffer_size"]), self.obs_dim, self.action_dim, self.device
        )
        self.model_buffer = ReplayBuffer(
            int(mbcfg["model_buffer_size"]), self.obs_dim, self.action_dim, self.device
        )
        self.mbcfg = mbcfg
        self.acfg = acfg
        self.ucfg = ucfg

        self._obs, _ = self.env.reset(seed=self.seed)
        self.total_steps = 0
        self.logs: list = []

        self._horizon_start = int(mbcfg.get("rollout_length_start", mbcfg["rollout_length"]))
        self._horizon_end = int(mbcfg["rollout_length"])
        self._horizon_warmup = int(mbcfg.get("horizon_curriculum_steps", 0))

    def _current_horizon(self) -> int:
        if self._horizon_warmup <= 0 or self._horizon_end <= self._horizon_start:
            return self._horizon_end
        t = min(max(self.total_steps - int(self.mbcfg["warmup_steps"]), 0), self._horizon_warmup)
        frac = t / max(self._horizon_warmup, 1)
        h = self._horizon_start + frac * (self._horizon_end - self._horizon_start)
        return max(1, int(round(h)))

    def _train_world_model(self) -> Dict[str, float]:
        if len(self.real_buffer) < self.mbcfg["model_batch_size"]:
            return {}
        epochs = int(self.mbcfg["num_model_epochs"])
        bs = int(self.mbcfg["model_batch_size"])
        losses = []
        last_out: Dict[str, torch.Tensor] = {}
        reached_teacher_budget = (
            self.distill_teacher_pretrain_updates is not None
            and self._teacher_updates >= self.distill_teacher_pretrain_updates
        )
        reached_legacy_step = (
            self.distill_teacher_pretrain_updates is None
            and self.total_steps >= self.distill_freeze_after
        )
        if (
            self.distill_two_stage
            and hasattr(self.world_model, "teacher")
            and not self.world_model.teacher_frozen
            and self._teacher_updates > 0
            and (reached_teacher_budget or reached_legacy_step)
        ):
            self._teacher_checksum_before_freeze = self._parameter_checksum(self.world_model.teacher)
            self.world_model.freeze_teacher()
            self.wm_opt = torch.optim.Adam(self.world_model.student.parameters(), lr=1e-3)
        for _ in range(epochs):
            batch = self.real_buffer.sample(bs)
            if (
                self.distill_two_stage
                and hasattr(self.world_model, "teacher")
                and not self.world_model.teacher_frozen
            ):
                # Independent bootstrap resamples preserve epistemic diversity
                # between ensemble members instead of fitting every member to
                # the identical minibatch realization.
                self.world_model.teacher.update_stats(
                    batch["obs"], batch["next_obs"], batch["rewards"]
                )
                member_losses = []
                for i in range(self.world_model.teacher.ensemble_size):
                    idx = torch.randint(0, bs, (bs,), device=self.device)
                    member_losses.append(
                        self.world_model.teacher.diffusion_loss_member(
                            i, batch["obs"][idx], batch["actions"][idx],
                            batch["next_obs"][idx], batch["rewards"][idx],
                        )
                    )
                teacher_loss = torch.stack(member_losses).mean()
                out = {
                    "total": teacher_loss,
                    "dynamics": teacher_loss.detach(),
                    "reward": torch.zeros((), device=self.device),
                }
            else:
                value_fn = self._make_distill_value_fn()
                if hasattr(self.world_model, "teacher"):
                    out = self.world_model.train_loss(
                        batch["obs"], batch["actions"], batch["next_obs"],
                        batch["rewards"], batch["dones"], value_fn=value_fn,
                    )
                else:
                    out = self.world_model.train_loss(
                        batch["obs"], batch["actions"], batch["next_obs"],
                        batch["rewards"], batch["dones"],
                    )
            self.wm_opt.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.distill_grad_clip)
            self.wm_opt.step()
            if (
                self.distill_two_stage
                and hasattr(self.world_model, "teacher")
                and not self.world_model.teacher_frozen
            ):
                self._teacher_updates += 1
            losses.append(float(out["total"].item()))
            last_out = out
        info = {"wm_loss": float(np.mean(losses))}
        if hasattr(self.world_model, "teacher_frozen"):
            info["teacher_frozen"] = float(self.world_model.teacher_frozen)
            info["teacher_updates"] = float(self._teacher_updates)
            info["teacher_checksum"] = self._parameter_checksum(self.world_model.teacher)
            if self._teacher_checksum_before_freeze is not None:
                info["teacher_checksum_before_freeze"] = float(self._teacher_checksum_before_freeze)
        for key in (
            "distill_member", "distill_mean", "distill_geometry", "distill_pairwise",
            "distill_value_geometry", "distill_value_variance",
            "distill_state_geometry", "distill_state_pairwise",
            "distill_guard_fired", "distill_decision_scale",
            "distill_epistemic_w", "distill_aleatoric_g",
            "distill_teacher_w_mean", "distill_student_w_mean",
            "distill_teacher_g_mean", "distill_student_g_mean",
            "distill_teacher_local_u_mean", "distill_student_local_u_mean",
            "distill_w_scale", "distill_g_scale",
        ):
            if key in last_out:
                info[key] = float(last_out[key].item())
        return info

    def _make_distill_value_fn(self):
        """Lagged target-critic map, or None until the critic has been updated.

        Online live-Q distillation is the condition the policy-scale study
        falsified. Default is the SAC target critic plus a stop-gradient
        policy. Decision terms are withheld until ``distill_value_warmup_updates``.
        """
        if float(self.cfg["model"].get("distill_decision_weight", 0.0)) <= 0.0:
            return None
        warmup = int(self.cfg["model"].get("distill_value_warmup_updates", 0))
        if self._critic_updates < warmup:
            return None
        use_target = bool(self.cfg["model"].get("distill_use_target_critic", False))
        from udwm.models.consistency import lagged_target_value_fn

        critic = self.agent.critic_target if use_target else self.agent.critic
        if not use_target:
            # Live critic is the falsified path; keep it available as a control.
            def value_fn(states, _actions):
                frozen = list(self.agent.actor.parameters()) + list(self.agent.critic.parameters())
                flags = [p.requires_grad for p in frozen]
                for p in frozen:
                    p.requires_grad_(False)
                try:
                    flat = states.reshape(-1, states.shape[-1])
                    action = self.agent.actor.deterministic(flat)
                    q = self.agent.critic.min_q(flat, action)
                    return q.reshape(*states.shape[:-1], 1)
                finally:
                    for p, flag in zip(frozen, flags):
                        p.requires_grad_(flag)
            return value_fn
        return lagged_target_value_fn(critic, self.agent.actor)

    @staticmethod
    def _parameter_checksum(module) -> float:
        """Deterministic scalar fingerprint for paired-teacher diagnostics."""
        return float(sum(p.detach().float().abs().sum().item() for p in module.parameters()))

    def _imagine(self) -> Dict[str, float]:
        if len(self.real_buffer) < self.mbcfg["model_batch_size"]:
            return {}
        n = int(self.mbcfg["rollouts_per_step"])
        h = self._current_horizon()
        batch = self.real_buffer.sample(n)
        obs = batch["obs"]

        def policy_fn(o: torch.Tensor) -> torch.Tensor:
            return self.agent.policy_tensor(o, deterministic=False)

        info: Dict[str, float] = {"imagine_horizon": float(h)}
        # Warmup U-net before gating so random U does not kill all rollouts
        gate_mode = self.u_gate_mode
        if self.total_steps < self.u_gate_enable_after:
            gate_mode = "off"
        with torch.no_grad():
            score_source = str(self.u_gate_cfg.get("score", "ube")).lower()
            score_fn = None
            if score_source == "one_step_state":
                score_fn = lambda o, a: one_step_state_disagreement(
                    self.world_model,
                    o,
                    a,
                    m_samples=int(self.u_gate_cfg.get("one_step_m_samples", 4)),
                )
            roll = u_gated_rollout(
                self.world_model,
                policy_fn,
                self.u_net,
                obs,
                horizon=h,
                mode=gate_mode,
                stop_threshold=self.u_gate_cfg.get("stop_threshold"),
                stop_percentile=float(self.u_gate_cfg.get("stop_percentile", 0.85)),
                weight_beta=float(self.u_gate_cfg.get("weight_beta", 1.0)),
                min_weight=float(self.u_gate_cfg.get("min_weight", 0.05)),
                use_sqrt=bool(self.u_gate_cfg.get("use_sqrt", True)),
                score_fn=score_fn,
            )

        b, hh = roll["obs"].shape[0], roll["obs"].shape[1]
        n_added = 0
        for t in range(hh):
            w = roll["weights"][:, t, 0].cpu().numpy()
            keep = w > float(self.u_gate_cfg.get("min_store_weight", 1e-4))
            if not np.any(keep):
                continue
            self.model_buffer.add_batch(
                roll["obs"][keep, t].cpu().numpy(),
                roll["actions"][keep, t].cpu().numpy(),
                roll["rewards"][keep, t].cpu().numpy(),
                roll["next_obs"][keep, t].cpu().numpy(),
                roll["dones"][keep, t].cpu().numpy(),
                weights=w[keep],
            )
            n_added += int(keep.sum())

        info["imagine_added"] = float(n_added)
        info["imagine_stopped_frac"] = float(roll["stopped_frac"].item())
        info["imagine_mean_weight"] = float(roll["weights"].mean().item())
        info["imagine_mean_sqrt_u"] = float(roll["sqrt_u"].mean().item())
        if float(roll["stop_threshold_used"].item()) >= 0:
            info["imagine_stop_thresh"] = float(roll["stop_threshold_used"].item())
        self.audit.record_batch(
            step=self.total_steps,
            obs=roll["obs"],
            actions=roll["actions"],
            sqrt_u=roll["sqrt_u"],
            weights=roll["weights"],
            dones=roll["dones"],
            stop_threshold=float(roll["stop_threshold_used"].item()),
            mode=gate_mode,
        )
        info.update({f"audit_{k}": v for k, v in self.audit.summary().items()})
        return info

    def _mixed_batch(self) -> Dict[str, torch.Tensor]:
        bs = int(self.acfg["batch_size"])
        real_ratio = float(self.mbcfg["real_ratio"])
        n_real = int(bs * real_ratio)
        n_model = bs - n_real
        use_prio = self.prioritized_model and self.u_gate_mode in ("weight", "both")
        if len(self.model_buffer) < max(n_model, 1) or n_model == 0:
            return self.real_buffer.sample(bs)
        if n_real == 0:
            return self.model_buffer.sample(bs, prioritized=use_prio)
        real = self.real_buffer.sample(n_real)
        model = self.model_buffer.sample(n_model, prioritized=use_prio)
        return {k: torch.cat([real[k], model[k]], dim=0) for k in real if k in model}

    def _update_ube(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if not self.acfg.get("use_ube", True):
            return {}
        if self.agent.u_opt is None:
            return {}

        def q_fn(o, a):
            return self.agent.q_min(o, a)

        def policy_fn(o):
            return self.agent.policy_tensor(o, deterministic=False)

        local = self.mc_ube.estimate(
            self.world_model, q_fn, policy_fn, batch["obs"], batch["actions"]
        )
        with torch.no_grad():
            next_a = policy_fn(batch["next_obs"])
            next_u = self.u_net(batch["next_obs"], next_a)
        targets = self.mc_ube.ube_targets(
            local["u"], next_u, self.agent.gamma, batch["dones"]
        )
        loss = ube_loss(self.u_net, batch["obs"], batch["actions"], targets)
        self.agent.u_opt.zero_grad()
        loss.backward()
        self.agent.u_opt.step()
        out = {
            "ube_loss": float(loss.item()),
            "u_mean": float(local["u"].mean().item()),
            "w_mean": float(local["w"].mean().item()),
            "g_mean": float(local["g"].mean().item()),
        }
        if "m_mean" in local:
            out["mc_m_mean"] = float(local["m_mean"].item())
        if "refine_frac_used" in local:
            out["mc_refine_frac"] = float(local["refine_frac_used"].item())
        return out

    def _env_step(self) -> float:
        if self.total_steps < int(self.mbcfg["warmup_steps"]):
            action = self.env.action_space.sample()
        else:
            action = self.agent.act(self._obs, deterministic=False)
            action = np.clip(action, self.action_low, self.action_high)

        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = float(terminated or truncated)
        self.real_buffer.add(self._obs, action, float(reward), next_obs, done)
        self._obs = next_obs
        self.total_steps += 1
        if done:
            self._obs, _ = self.env.reset()
        return float(reward)

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 5) -> float:
        stats = evaluate_policy_return(
            self.agent,
            self.eval_env,
            n_episodes=n_episodes,
            seed=self.seed,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        return float(stats["return_mean"])

    @torch.no_grad()
    def evaluate_full(self, n_episodes: int = 10) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(
            evaluate_policy_return(
                self.agent,
                self.eval_env,
                n_episodes=n_episodes,
                seed=self.seed,
                action_low=self.action_low,
                action_high=self.action_high,
            )
        )
        if len(self.real_buffer) >= 64:
            out.update(
                evaluate_world_model_accuracy(
                    self.world_model,
                    self.real_buffer,
                    batch_size=min(256, len(self.real_buffer)),
                )
            )
            out.update(
                evaluate_uncertainty_calibration(
                    self.agent,
                    self.u_net,
                    self.real_buffer,
                    batch_size=min(256, len(self.real_buffer)),
                )
            )
            if hasattr(self.world_model, "teacher") and self.world_model.teacher_frozen:
                out.update(evaluate_distillation_uncertainty(
                    self.world_model,
                    self.agent.q_min,
                    lambda states: self.agent.policy_tensor(states, deterministic=True),
                    self.real_buffer,
                    batch_size=min(128, len(self.real_buffer)),
                    n_batches=1,
                    m_samples=2,
                ))
                out.update(benchmark_distilled_sampling(
                    self.world_model, self.real_buffer,
                    batch_size=min(64, len(self.real_buffer)), repeats=3,
                ))
            pair = collect_score_and_error(
                self.agent, self.u_net, self.real_buffer, batch_size=min(256, len(self.real_buffer))
            )
            sel = selective_report(pair["score"], pair["abs_td"])
            out["selective_rank_corr"] = float(sel["rank_corr_score_vs_error"])
            # snapshot mid-coverage risk (keep 50%)
            for row in sel["risk_coverage"]:
                if abs(row["coverage"] - 0.5) < 0.08:
                    out["selective_risk_at_50"] = float(row["risk_mean_abs_td"])
                    break
            # over-rejection at middle tau
            sweep = sel["threshold_sweep"]
            if sweep:
                mid = sweep[len(sweep) // 2]
                out["selective_over_rejection"] = float(mid["over_rejection"])
                out["selective_recall_bad"] = float(mid["recall_bad"])
        return out

    def train(self, total_env_steps: Optional[int] = None) -> Dict[str, Any]:
        total = int(total_env_steps or self.mbcfg["total_env_steps"])
        eval_freq = int(self.mbcfg["eval_freq"])
        log_freq = int(self.mbcfg["log_freq"])
        model_freq = int(self.mbcfg["model_train_freq"])
        agent_updates = int(self.mbcfg["agent_updates_per_step"])
        warmup = int(self.mbcfg["warmup_steps"])

        pbar = trange(total, desc="env_steps")
        recent_r = 0.0
        last_info: Dict[str, float] = {}
        for _ in pbar:
            recent_r = self._env_step()

            if self.total_steps >= warmup and self.total_steps % model_freq == 0:
                last_info.update(self._train_world_model())
                last_info.update(self._imagine())

            if self.total_steps >= warmup and len(self.real_buffer) >= self.acfg["batch_size"]:
                for _u in range(agent_updates):
                    batch = self._mixed_batch()
                    last_info.update(self.agent.update(batch))
                    self._critic_updates += 1
                    if self.acfg.get("use_ube", True) and _u == 0:
                        last_info.update(self._update_ube(batch))

            if self.total_steps % log_freq == 0:
                pbar.set_postfix(
                    r=f"{recent_r:.2f}",
                    wm=last_info.get("wm_loss", 0),
                    ube=last_info.get("ube_loss", 0),
                    stop=last_info.get("imagine_stopped_frac", 0),
                    m=last_info.get("mc_m_mean", 0),
                )
                self.logs.append({"step": self.total_steps, **last_info})

            if self.total_steps % eval_freq == 0 and self.total_steps > 0:
                full = self.evaluate_full(n_episodes=5)
                last_info.update(full)
                pbar.write(
                    f"step={self.total_steps} "
                    f"return={full.get('return_mean', 0):.1f} "
                    f"s_mse={full.get('next_state_mse', float('nan')):.4f} "
                    f"stop={last_info.get('imagine_stopped_frac', 0):.2f} "
                    f"m={last_info.get('mc_m_mean', 0):.1f}"
                )
                self.logs.append({"step": self.total_steps, **full})

        final_stats = self.evaluate_full(n_episodes=10)
        return {
            "final_eval_return": final_stats.get("return_mean", 0.0),
            "final_metrics": final_stats,
            "logs": self.logs,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "u_net": self.u_net.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
