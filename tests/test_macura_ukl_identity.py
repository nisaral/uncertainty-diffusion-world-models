"""Independent checks for the MACURA u_KL plugin identity (Eq. 4 of
arXiv:2405.19014).

The plugin identity documented in udwm/uncertainty/macura_baseline.py and
research/MACURA-BASELINE-PREREGISTRATION-2026-09-09.md is

    u_KL = sum_e KL(pi_e || p_bar) = E * JSD(pi_1, ..., pi_E),

where pi_e are the per-member kernel-plugin categoricals over the pooled
Monte Carlo support and p_bar = (1/E) sum_e pi_e. These tests check the
identity three independent ways: (a) a hand-worked 2-member categorical
example with precomputed values; (b) E=3 kernel-plugin categoricals built
in-test from synthetic member draws; (c) the shipped
udwm.uncertainty.macura_baseline.ukl_disagreement on a fake world model
with deterministic draws, compared against the in-test replication.

Run: python -m pytest tests/test_macura_ukl_identity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udwm.uncertainty.macura_baseline import ukl_disagreement  # noqa: E402


def _entropy(q):
    q = q.clamp_min(1e-12)
    return -(q * torch.log(q)).sum(dim=-1)


def _kl(p, q):
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)


def test_hand_checked_two_member_identity():
    """2-member, 3-category example, values worked by hand.

    pi1 = (0.5, 0.3, 0.2), pi2 = (0.1, 0.4, 0.5),
    p_bar = (0.30, 0.35, 0.35).
    H(pi1) = 1.0296530141, H(pi2) = 0.9433483923, H(p_bar) = 1.0960673284
    JSD = H(p_bar) - (H(pi1) + H(pi2)) / 2 = 0.1095666253
    KL(pi1||p_bar) = 0.0972444503, KL(pi2||p_bar) = 0.1218888002
    u_KL = 0.2191332505 == 2 * JSD = 0.2191332505.
    """
    # float64: the hand values below were computed in double precision.
    pi1 = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    pi2 = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float64)
    p_bar = (pi1 + pi2) / 2.0
    u_kl_direct = float(_kl(pi1, p_bar) + _kl(pi2, p_bar))
    jsd = float(_entropy(p_bar) - (_entropy(pi1) + _entropy(pi2)) / 2.0)
    u_kl_from_jsd = 2.0 * jsd
    assert abs(u_kl_direct - 0.2191332505) < 1e-9
    assert abs(u_kl_from_jsd - 0.2191332505) < 1e-9
    assert abs(u_kl_direct - u_kl_from_jsd) < 1e-12
    assert abs(jsd - 0.1095666253) < 1e-9


def _member_clouds(n, m, b, d, obs, shift):
    """Deterministic per-member draw clouds replicable outside the module.

    Returns [n, m, B, D]. Member i is offset by i * shift; draw j adds a
    deterministic (j + 1) * 0.01 offset on the last feature so within-member
    spread is nonzero (kernel bandwidth is well defined).
    """
    clouds = []
    for i in range(n):
        cloud = obs.unsqueeze(0).expand(m, b, d).clone()
        cloud = cloud + i * shift
        j = torch.arange(1, m + 1, dtype=cloud.dtype).view(m, 1, 1)
        off = torch.zeros(1, 1, d)
        off[0, 0, -1] = 1.0
        cloud = cloud + (j * 0.01) * off
        clouds.append(cloud)
    return torch.stack(clouds, dim=0)


def _plugin_categoricals(clouds, eps=1e-12):
    """Replicate the module kernel-plugin categoricals from raw clouds.

    clouds: [n, m, B, D]. Returns pi: [n, B, S] over the pooled support.
    """
    n, m, b, d = clouds.shape
    device = clouds.device
    # per-state bandwidth: mean over members of median within-member
    # pairwise distance, matching macura_baseline.py.
    medians = []
    for i in range(n):
        x = clouds[i]  # [m, B, D]
        d2 = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)  # [m, m, B]
        tri = torch.triu(torch.ones(m, m, device=device, dtype=torch.bool),
                         diagonal=1)
        pairs = d2[tri]  # [P, B]
        if pairs.shape[0] > 0:
            medians.append(pairs.median(dim=0).values)
    h2 = torch.stack(medians, dim=0).mean(dim=0).clamp_min(eps).pow(2)
    support = clouds.permute(2, 0, 1, 3).reshape(b, n * m, -1)  # [B, S, D]
    cats = []
    for i in range(n):
        xi = clouds[i].permute(1, 0, 2)  # [B, m, D]
        d2 = (support.unsqueeze(2) - xi.unsqueeze(1)).pow(2).sum(dim=-1)
        k = torch.exp(-d2 / (2.0 * h2[:, None, None]))
        p = k.mean(dim=-1)
        p = p / (p.sum(dim=-1, keepdim=True) + eps)
        cats.append(p)
    return torch.stack(cats, dim=0)  # [n, B, S]


def test_plugin_categoricals_satisfy_identity_e3():
    """Identity holds on E=3 kernel-plugin categoricals from real draws."""
    torch.manual_seed(0)
    n, m, b, d = 3, 5, 4, 3
    obs = torch.randn(b, d)
    clouds = _member_clouds(n, m, b, d, obs, shift=0.3)
    pi = _plugin_categoricals(clouds)  # [n, B, S]
    p_bar = pi.mean(dim=0)
    u_kl = _kl(pi, p_bar.unsqueeze(0).expand_as(pi)).sum(dim=0)  # [B]
    jsd = _entropy(p_bar) - _entropy(pi).mean(dim=0)  # [B]
    assert torch.allclose(u_kl, n * jsd, atol=1e-9)
    assert bool((u_kl >= 0).all())
    assert bool(torch.isfinite(u_kl).all())


class _FakeDynamics:
    def __init__(self, shift):
        self.shift = shift

    def sample_next_multi(self, obs, actions, m, member):
        # Member i clouds = obs + i * shift + per-draw offsets, matching
        # the in-test replication for the same member index.
        cloud = _member_clouds(1, int(m), obs.shape[0], obs.shape[1], obs,
                               self.shift)[0]
        return cloud + member * self.shift


class _FakeWM:
    def __init__(self, n, shift):
        self.ensemble_size = n
        self.dynamics = _FakeDynamics(shift)


def test_ukl_disagreement_matches_in_test_replication():
    """Shipped module output equals the independent in-test computation."""
    torch.manual_seed(1)
    n, m, b, d = 3, 5, 4, 3
    obs = torch.randn(b, d)
    act = torch.randn(b, 2)
    shift = 0.25
    wm = _FakeWM(n, shift)
    got = ukl_disagreement(wm, obs, act, m_samples=m).squeeze(-1)
    clouds = _member_clouds(n, m, b, d, obs, shift)
    pi = _plugin_categoricals(clouds)
    p_bar = pi.mean(dim=0)
    expect = _kl(pi, p_bar.unsqueeze(0).expand_as(pi)).sum(dim=0)
    assert torch.allclose(got, expect, atol=1e-6)
    # and the documented identity E * JSD reproduces the module output.
    jsd = _entropy(p_bar) - _entropy(pi).mean(dim=0)
    assert torch.allclose(got, n * jsd, atol=1e-6)

