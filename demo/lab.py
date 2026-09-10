"""CPU toy lab: four distillation losses on a 2-D (w, g) field.

This is a laboratory for the *loss*, not a control agent. Grid cells are
states. The teacher has analytic (w*, g*) with g* >> w* (aleatoric-dominated,
the measured DelayedBimodal / hopper regime). Four students are trained with
the four losses from the paper:

  ordinary       — match member means; the split is not a training target
  hybrid         — match the mixed scalar S = w + (1-rho) g  (M=1 fibre)
  identified_eq  — match (w, g) with equal weights
  identified_ema — match (w, g) with inverse-variance EMA weights

Run::

    python -m demo.lab
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

N_GRID = 24
N_MEM = 5
RHO = 0.8
STEPS = 280
SEED = 0
GATE_Q = 0.85


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def teacher_fields(n: int = N_GRID) -> Dict[str, np.ndarray]:
    ys, xs = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    # Epistemic ridge through the origin (the "unseen branch").
    w = 0.008 + 0.35 * np.exp(-(xs ** 2) / 0.08)
    # Aleatoric-dominated map, g*/w* ~ 1e2–1e4 as in the paper.
    g = 18.0 + 55.0 * (0.5 + 0.5 * ys)
    u = w - g
    S = w + (1.0 - RHO) * g
    return {"x": xs, "y": ys, "w": w, "g": g, "u": u, "S": S}


@dataclass
class ArmState:
    key: str
    w: np.ndarray
    g: np.ndarray
    ranks: List[float]
    g_ratio: List[float]


def _init_student(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w = np.exp(rng.normal(-2.2, 0.25, size=(n, n)))
    g = np.exp(rng.normal(-0.4, 0.25, size=(n, n)))
    return w, g


def train(steps: int = STEPS, n: int = N_GRID, seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed)
    t = teacher_fields(n)
    tw, tg, tu, tS = t["w"], t["g"], t["u"], t["S"]
    lr = 0.12

    arms: Dict[str, ArmState] = {}
    for key in ("ordinary", "hybrid", "identified_eq", "identified_ema"):
        w, g = _init_student(n, rng)
        arms[key] = ArmState(key, w, g, [], [])

    ema_w, ema_g = float(tw.mean()), float(tg.mean())
    beta = 0.99

    for step in range(steps):
        # ordinary: member-mean matching → w tracks teacher; g is a side
        # effect of the residual, so it slowly copies teacher g. Ranking holds.
        a = arms["ordinary"]
        a.w += lr * (tw - a.w)
        a.g += 0.35 * lr * (tg - a.g)

        # hybrid: match the mixed scalar S. Degenerate direction prefers
        # shrinking w (Voelcker-style variance shrinkage + fibre).
        a = arms["hybrid"]
        S_s = a.w + (1.0 - RHO) * a.g
        err = S_s - tS
        a.w -= lr * err
        a.g -= lr * (1.0 - RHO) * err
        a.w = np.maximum(a.w - 0.015 * lr * a.w, 1e-6)  # fibre walk toward w→0
        a.g = np.maximum(a.g, 1e-6)

        # identified equal-weight
        a = arms["identified_eq"]
        a.w += lr * (tw - a.w)
        a.g += lr * (tg - a.g)

        # identified EMA: inverse-variance weights 1/ema^2. On this map
        # (w*/g*)^2 ~ 1e-6 so the g term is numerically dead.
        a = arms["identified_ema"]
        ema_w = beta * ema_w + (1.0 - beta) * float(tw.mean())
        ema_g = beta * ema_g + (1.0 - beta) * float(tg.mean())
        ww = 1.0 / max(ema_w, 1e-6) ** 2
        wg = 1.0 / max(ema_g, 1e-6) ** 2
        # Normalise so the larger weight is O(1); g weight stays ~ (w/g)^2.
        scale = max(ww, wg)
        a.w += lr * (ww / scale) * (tw - a.w)
        a.g += lr * (wg / scale) * (tg - a.g)
        a.g = np.maximum(a.g, 1e-6)

        if step % 4 == 0 or step == steps - 1:
            for arm in arms.values():
                u = arm.w - arm.g
                arm.ranks.append(_spearman(tu, u))
                arm.g_ratio.append(float((arm.g.mean() + 1e-12) / (tg.mean() + 1e-12)))

    def pack(arm: ArmState) -> Dict:
        u = arm.w - arm.g
        tau_t = float(np.quantile(np.abs(tu), GATE_Q))
        tau_s = float(np.quantile(np.abs(u), GATE_Q))
        t_stop = np.abs(tu) >= tau_t
        s_stop = np.abs(u) >= tau_s
        over_keep = (~s_stop) & t_stop
        under_keep = s_stop & (~t_stop)
        agree_keep = (~s_stop) & (~t_stop)
        agree_stop = s_stop & t_stop
        return {
            "key": arm.key,
            "w": arm.w.tolist(),
            "g": arm.g.tolist(),
            "u": u.tolist(),
            "ranks": arm.ranks,
            "g_ratio": arm.g_ratio,
            "u_rank_final": arm.ranks[-1],
            "g_ratio_final": arm.g_ratio[-1],
            "gate": {
                "over_keep_frac": float(over_keep.mean()),
                "under_keep_frac": float(under_keep.mean()),
                "agree_keep_frac": float(agree_keep.mean()),
                "agree_stop_frac": float(agree_stop.mean()),
                "map": np.where(
                    over_keep, 3, np.where(under_keep, 2, np.where(agree_stop, 1, 0))
                ).tolist(),
            },
        }

    return {
        "n": n,
        "steps": steps,
        "teacher": {"w": tw.tolist(), "g": tg.tolist(), "u": tu.tolist()},
        "arms": {k: pack(v) for k, v in arms.items()},
        "note": "Toy (w,g) field, not DelayedBimodal pixels. Rank-only.",
    }


def main() -> None:
    out = train()
    for key, arm in out["arms"].items():
        print(
            f"{key:16s}  u-rank={arm['u_rank_final']:+.3f}  "
            f"g_ratio={arm['g_ratio_final']:.3f}  "
            f"over-keep={arm['gate']['over_keep_frac']:.2f}"
        )


if __name__ == "__main__":
    main()
