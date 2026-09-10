# -*- coding: utf-8 -*-
"""Sign/direction convention for the Hypothesis-C gradient-interference metric.

Guards the two ways a metric-direction bug could silently invalidate the
registered E1/E2 read:

  1. The signed convention itself:
         cos = <g_point, g_unc> / (||g_point|| ||g_unc||)
         interference = -cos     (positive => conflict, negative => alignment)
     pinned with CONSTRUCTED gradients whose cosine is known exactly by
     construction (aligned = +1, opposed = -1, orthogonal = 0), the same
     ground-truth discipline used for the identifiability estimator.

  2. The adjudicator's direction: E1 must be DMC - DB regardless of the order
     the `--data` flags are given, and E2's sign must be "more interference ->
     larger eq deficit".

Run: python -m pytest tests/test_gradient_interference_metric.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from udwm.models import consistency as C                      # noqa: E402
from udwm.models.world_model import WorldModel                # noqa: E402
from udwm.scripts.probe_gradient_interference import (        # noqa: E402
    flat_student_grad,
    interference_from_parts,
    interference_from_terms,
)


@pytest.fixture()
def probe_tmp():
    """Workspace-local scratch dir (greatest portability across sandboxes)."""
    d = Path(ROOT) / "_probe" / "pytest_tmp"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------
# 1. the signed convention, on constructed gradients
# --------------------------------------------------------------------------

def test_interference_is_positive_for_opposed_gradients():
    m = nn.Linear(3, 1, bias=False)
    cos, interference, n_p, n_u = interference_from_terms(
        m, m.weight.sum(), -m.weight.sum())
    assert cos == pytest.approx(-1.0, abs=1e-6)
    assert interference == pytest.approx(+1.0, abs=1e-6)   # conflict
    assert n_p > 0 and n_u > 0


def test_interference_is_negative_for_aligned_gradients():
    m = nn.Linear(3, 1, bias=False)
    cos, interference, _, _ = interference_from_terms(
        m, m.weight.sum(), m.weight.sum())
    assert cos == pytest.approx(1.0, abs=1e-6)
    assert interference == pytest.approx(-1.0, abs=1e-6)   # alignment


def test_interference_is_zero_for_orthogonal_gradients():
    m = nn.Linear(3, 2, bias=False)
    cos, interference, _, _ = interference_from_terms(
        m, m.weight[0].sum(), m.weight[1].sum())
    assert cos == pytest.approx(0.0, abs=1e-6)
    assert interference == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# 2. the production path: train_loss + GRAD_PROBE + interference_from_parts
# --------------------------------------------------------------------------

def _tiny_world_model():
    torch.manual_seed(0)
    wm = WorldModel.build(
        model_type="diffusion", obs_dim=4, action_dim=1, ensemble_size=2,
        hidden_dims=(8, 8), student_hidden_dims=(4, 4), diffusion_steps=4,
        sample_steps=2, joint_with_diffusion=True, use_consistency_distill=True,
        preserve_distilled_uncertainty=True, distill_decision_weight=1.0,
        distill_value_variance_weight=1.0, distill_identified=True,
        distill_m_latents=2, distill_aleatoric_weight=1.0,
    )
    for p in wm.teacher.parameters():
        p.requires_grad_(False)
    return wm


def _value_fn(states, _actions):
    flat = states.reshape(-1, states.shape[-1])
    return flat.sum(-1, keepdim=True).reshape(*states.shape[:-1], 1)


def _live_parts(wm, seed):
    torch.manual_seed(seed)
    b = 6
    obs = torch.randn(b, 4)
    actions = torch.rand(b, 1) * 2 - 1
    next_obs = obs + 0.1 * torch.randn(b, 4)
    C.GRAD_PROBE = {}
    try:
        wm.train_loss(obs, actions, next_obs, torch.zeros(b, 1),
                      torch.zeros(b, 1), value_fn=_value_fn)
        return C.GRAD_PROBE.get("parts")
    finally:
        C.GRAD_PROBE = None


def test_probe_record_equals_independent_cosine_and_flips_sign():
    wm = _tiny_world_model()
    parts = _live_parts(wm, 1)
    assert parts is not None, "the loss must expose live parts to the probe"

    mcfg = {"distill_value_variance_weight": 1.0, "distill_aleatoric_weight": 1.0}
    rec = interference_from_parts(wm.student, parts, mcfg)
    assert rec is not None
    assert rec["unc_terms"] == ["aleatoric_g", "epistemic_w"]

    # independent recomputation from the same live parts
    gp = flat_student_grad(wm.student, parts["member"])
    gu = flat_student_grad(wm.student, parts["epistemic_w"] + parts["aleatoric_g"])
    cos_ref = float((gp @ gu) / (gp.norm() * gu.norm()).clamp_min(1e-12))
    assert rec["cos"] == pytest.approx(cos_ref, abs=1e-6)
    assert rec["interference"] == pytest.approx(-rec["cos"], abs=1e-12)


def test_ordinary_has_no_uncertainty_term_and_is_not_applicable():
    wm = _tiny_world_model()
    parts = _live_parts(wm, 2)
    assert parts is not None
    rec = interference_from_parts(
        wm.student, parts,
        {"distill_value_variance_weight": 0.0, "distill_aleatoric_weight": 0.0})
    assert rec is None


def test_term_groups_are_the_registered_ones():
    from udwm.scripts.probe_gradient_interference import TERM_GROUP
    assert TERM_GROUP["member"] == "point"
    assert TERM_GROUP["epistemic_w"] == "uncertainty"
    assert TERM_GROUP["aleatoric_g"] == "uncertainty"
    assert TERM_GROUP["value_variance"] == "uncertainty"


# --------------------------------------------------------------------------
# 3. adjudicator direction
# --------------------------------------------------------------------------

def _write_probe(path: Path, interference_by_seed, gap_by_seed, env_shift):
    rows = []
    for seed, (interf, gap) in enumerate(zip(interference_by_seed, gap_by_seed)):
        for arm, value in (("ordinary", 0.95), ("identified_eq", 0.95 - gap),
                           ("hybrid", 0.70)):
            cps = []
            if arm != "ordinary":
                for i in range(10):
                    cps.append({"step": 300 * (i + 1), "applicable": True,
                                "interference": float(interf + env_shift),
                                "cos": float(-(interf + env_shift)),
                                "unc_terms": ["value_variance"]})
            rows.append({"seed": seed, "variant": arm, "final_return": 0.0,
                         "final_metrics": {"u_rank_corr": value},
                         "gi_checkpoints": cps})
    path.write_text(json.dumps({"protocol": {"steps": 15000}, "rows": rows}),
                    encoding="utf-8")


def test_e1_contrast_is_dmc_minus_db_even_when_flags_are_reversed(probe_tmp):
    from udwm.scripts.summarize_gradient_interference import main as summ_main
    db, dmc = probe_tmp / "db.json", probe_tmp / "dmc.json"
    seeds = list(range(6))
    _write_probe(db, [0.10 + 0.01 * s for s in seeds], [0.30] * 6, 0.0)
    _write_probe(dmc, [0.40 + 0.01 * s for s in seeds], [0.50] * 6, 0.0)
    out = probe_tmp / "sum.json"
    # deliberately reversed: db before dmc
    summ_main(["--data", "db=%s" % db, "--data", "dmc=%s" % dmc,
               "--out", str(out)])
    rep = json.loads(out.read_text(encoding="utf-8"))
    c = rep["endpoints"]["E1_env_contrast"]["identified_eq"]["contrast"]
    assert c["minus"] == "dmc_minus_db"
    assert c["delta"] > 0, "DMC must be the minuend"


def test_e2_sign_is_more_interference_more_deficit(probe_tmp):
    from udwm.scripts.summarize_gradient_interference import main as summ_main
    db, dmc = probe_tmp / "db.json", probe_tmp / "dmc.json"
    seeds = list(range(8))
    interf = [0.05 * s for s in seeds]                 # monotone in seed
    gap = [0.20 + 0.10 * s for s in seeds]             # monotone in seed
    _write_probe(db, interf, gap, 0.0)
    _write_probe(dmc, interf, gap, 0.10)
    out = probe_tmp / "sum.json"
    summ_main(["--data", "db=%s" % db, "--data", "dmc=%s" % dmc,
               "--out", str(out)])
    rep = json.loads(out.read_text(encoding="utf-8"))
    e2 = rep["endpoints"]["E2_dose_response"]["identified_eq"]["db"]
    assert e2["n"] == 8
    assert e2["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert e2["spearman"] == pytest.approx(1.0, abs=1e-6)
