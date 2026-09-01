from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from udwm.models.consistency import ConsistencyStudent, DistilledWorldModel
from udwm.models.diffusion_dynamics import DiffusionDynamicsEnsemble
from udwm.models.gaussian_ensemble import GaussianEnsemble
from udwm.models.reward_term import RewardTerminationModel


class WorldModel(nn.Module):
    """Unified interface over Gaussian or diffusion dynamics + reward/term.

    Modes
    -----
    - gaussian: PETS-style ensemble (state+reward in one NLL)
    - diffusion + separate R/T: DIAMOND-style (Gap 1 baseline)
    - diffusion + joint_reward: reward denoised with Δs (Gap 1 contribution)
    """

    def __init__(
        self,
        dynamics: nn.Module,
        reward_model: Optional[RewardTerminationModel] = None,
        model_type: str = "diffusion",
        joint_reward: bool = False,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.reward_model = reward_model
        self.model_type = model_type
        self.joint_reward = joint_reward

    @property
    def ensemble_size(self) -> int:
        return int(getattr(self.dynamics, "ensemble_size", 1))

    def train_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.model_type == "gaussian":
            dyn_loss = self.dynamics.nll_loss(obs, actions, next_obs, rewards)
            r_loss = torch.tensor(0.0, device=obs.device)
            total = dyn_loss
        elif self.joint_reward:
            # reward inside diffusion loss; term head inside dynamics.nll_loss
            dyn_loss = self.dynamics.nll_loss(obs, actions, next_obs, rewards, dones)
            r_loss = torch.tensor(0.0, device=obs.device)
            total = dyn_loss
        else:
            dyn_loss = self.dynamics.nll_loss(obs, actions, next_obs, rewards, dones)
            if self.reward_model is not None:
                r_loss = self.reward_model.loss(obs, actions, rewards, dones)
            else:
                r_loss = torch.tensor(0.0, device=obs.device)
            total = dyn_loss + r_loss
        return {"total": total, "dynamics": dyn_loss.detach(), "reward": r_loss.detach()}

    @torch.no_grad()
    def rollout(
        self,
        obs: torch.Tensor,
        policy_action_fn,
        horizon: int,
        member: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
        o = obs
        done = torch.zeros(o.shape[0], 1, device=o.device)
        for _ in range(horizon):
            a = policy_action_fn(o)
            if self.model_type == "gaussian":
                o2, r = self.dynamics.sample_next(o, a, member=member)
                d = torch.zeros_like(r)
            elif self.joint_reward:
                o2, r = self.dynamics.sample_next_with_reward(o, a, member=member)
                d = self.dynamics.predict_done(o, a)
            else:
                o2 = self.dynamics.sample_next(o, a, member=member)
                if self.reward_model is not None:
                    r, d = self.reward_model.predict(o, a)
                else:
                    r = torch.zeros(o.shape[0], 1, device=o.device)
                    d = torch.zeros_like(r)
            active = 1.0 - done
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r * active)
            next_list.append(o2)
            done_list.append(torch.clamp(done + d * active, 0, 1))
            o = o2
            done = done_list[-1]
        return {
            "obs": torch.stack(obs_list, 1),
            "actions": torch.stack(act_list, 1),
            "rewards": torch.stack(rew_list, 1),
            "next_obs": torch.stack(next_list, 1),
            "dones": torch.stack(done_list, 1),
        }

    @staticmethod
    def build(
        model_type: str,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int,
        hidden_dims,
        student_hidden_dims=None,
        diffusion_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        sample_steps: int = 4,
        reward_hidden=(128, 128),
        joint_with_diffusion: bool = False,
        use_consistency_distill: bool = False,
        preserve_distilled_uncertainty: bool = False,
        distill_mean_weight: float = 1.0,
        distill_geometry_weight: float = 1.0,
        distill_pairwise_weight: float = 1.0,
        distill_decision_weight: float = 0.0,
        distill_value_variance_weight: float = 0.0,
        distill_hybrid_state_weight: float = 0.0,
        distill_hybrid_pairwise_weight: float = 0.0,
        distill_normalize_values: bool = False,
        distill_guard_enabled: bool = False,
        distill_guard_min_corr: float = 0.0,
        distill_guard_max_scale: float = 8.0,
        distill_identified: bool = False,
        distill_m_latents: int = 2,
        distill_aleatoric_weight: float = 1.0,
        distill_reweight_ema: bool = False,
        distill_reweight_floor: float = 1e-6,
        freeze_teacher: bool = False,
    ):
        if model_type == "gaussian":
            dyn = GaussianEnsemble(obs_dim, action_dim, ensemble_size, hidden_dims)
            return WorldModel(dyn, reward_model=None, model_type="gaussian", joint_reward=False)

        dyn = DiffusionDynamicsEnsemble(
            obs_dim,
            action_dim,
            ensemble_size,
            hidden_dims,
            diffusion_steps,
            beta_start,
            beta_end,
            sample_steps,
            joint_reward=joint_with_diffusion,
        )
        if use_consistency_distill:
            x_dim = obs_dim + (1 if joint_with_diffusion else 0)
            student = ConsistencyStudent(
                x_dim, obs_dim, action_dim, ensemble_size,
                hidden_dims if student_hidden_dims is None else student_hidden_dims,
            )
            model = DistilledWorldModel(
                dyn,
                student,
                preserve_uncertainty=preserve_distilled_uncertainty,
                mean_weight=distill_mean_weight,
                geometry_weight=distill_geometry_weight,
                pairwise_weight=distill_pairwise_weight,
                decision_weight=distill_decision_weight,
                value_variance_weight=distill_value_variance_weight,
                hybrid_state_weight=distill_hybrid_state_weight,
                hybrid_pairwise_weight=distill_hybrid_pairwise_weight,
                normalize_values=distill_normalize_values,
                guard_enabled=distill_guard_enabled,
                guard_min_corr=distill_guard_min_corr,
                guard_max_scale=distill_guard_max_scale,
                identified=distill_identified,
                m_latents=distill_m_latents,
                aleatoric_weight=distill_aleatoric_weight,
                reweight_ema=distill_reweight_ema,
                reweight_floor=distill_reweight_floor,
            )
            if freeze_teacher:
                model.freeze_teacher()
            return model

        if joint_with_diffusion:
            return WorldModel(dyn, reward_model=None, model_type="diffusion", joint_reward=True)

        rmodel = RewardTerminationModel(obs_dim, action_dim, reward_hidden)
        return WorldModel(dyn, reward_model=rmodel, model_type="diffusion", joint_reward=False)
