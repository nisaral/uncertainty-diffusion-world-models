# Full-Scale Delayed-Bimodal Policy Study (2026-08-22)

## Protocol

Ten seeds (0-9), 1,800 environment steps per arm, and three matched arms:
ordinary member distillation, state geometry, and the decision-aware hybrid.
One teacher was pretrained per seed and loaded exactly into all arms; every
seed has zero teacher-checksum gap. The runner checkpoints each completed arm
in `runs/delayed_bimodal_policy_paired_3seed_1800.json`.

## Paired hybrid versus ordinary results

| Metric | Mean paired delta | Hybrid wins |
|---|---:|---:|
| Final return | -25.77 | 3/10 |
| Next-state MSE | -0.08255 | 10/10 |
| Teacher-student U rank correlation | -0.2968 | 0/10 |
| Teacher-student U RMSE | +180.40 | 0/10 |
| Selective rank correlation | -0.0316 | 6/10 |
| Selective bad-transition recall | +0.0003 | 6/10 |

The hybrid's state prediction error improved on every seed, but its uncertainty
fidelity degraded on every seed. This is a direct, reproducible counterexample
to the stronger claim that adding a learned downstream value-disagreement loss
automatically preserves uncertainty inside an online policy learner.

State geometry was less damaging but still not a reliable control improvement:
mean return delta -15.25, uncertainty-rank delta +0.0066, and selective-rank
delta +0.1782.

## Interpretation

The theory is not universally refuted. The large controlled prediction study
supports a narrower result: with a fixed, correctly specified decision map,
decision-aware distillation improves uncertainty magnitude fidelity. The policy
study falsifies the transfer claim under a changing learned SAC critic. The
critic-derived value map is nonstationary and poorly calibrated early in
training; using it directly as a distillation target can optimize state error
while destroying uncertainty geometry.

## Revised thesis

The publishable claim should be conditional:

> Decision-aware distillation can preserve downstream uncertainty under a fixed
> or separately validated value map. Naively coupling the loss to an online
> learned critic is unsafe and can systematically collapse uncertainty fidelity.

This negative transfer result is itself a useful contribution and motivates the
next method: lagged/target-critic value distillation with target normalization,
stop-gradient teacher values, and an explicit uncertainty-preservation guard.
