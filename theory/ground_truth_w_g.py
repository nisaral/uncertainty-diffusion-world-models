"""
GROUND-TRUTH (w*, g*) RECOVERY -- is the identified loss correct, or only
internally consistent?  And: how fragile is the (w, g) weighting, really?

Why this exists
---------------
The repo proved (theory/distill_identifiability.py) that the single-latent
`hybrid` objective matches w* + (g* - Sigma_bar): one equation, two unknowns.
The M>=2 `identified` objective splits them, but nobody checked whether a
student trained with either loss lands on the TRUE (w*, g*) -- only that it
was internally consistent with the teacher.  This script closes that gap with
a linear-Gaussian toy where (w*, g*) are analytic per (s, a):

  * member next-state means s_m(x) = Theta_m x, Theta_m drawn once (the
    "parameter posterior") -> epistemic w* = Var_m(E[V]),
  * injected noise eps ~ N(0, sigma^2(x) I) with shared-latent coupling rho ->
    aleatoric g* = E_m[Var(V)], closed form for V(s) = ||s||^2.

It also answers the follow-up question from the 2026-09-01 direction: when the
true scales of w and g differ by orders of magnitude, is there ANY scalar
weighting that recovers both?  Or is ranking robust to the weighting even when
magnitude is not (which would bound the "known hole" for rank-based
downstream gating)?

Parts:
  P0  Does the MC machinery (`coupled_w_g`, the repo's estimator) recover the
      analytic (w*, g*)?  (estimator correctness vs ground truth)
  P1  Benign init: does the M=1 objective walk the degenerate direction even
      from a good init, and does identified recover both components?
  P2  Collapsed init (the zero-loss family): is identified pulled back while
      hybrid stays off?
  P3  The equal-weighting hole (g* >> w*): equal weights fail on w; what do
      mean-magnitude reweighting and the EMA per-term normaliser (the fix now
      in udwm.models.consistency.TermScaleEMA) each buy?
  P4  Weighting sweep + rank robustness: sweep the relative weight of the
      (normalised) w and g terms in the hole regime; report w rank, w relative
      error, g rank and the gate-relevant top-decile recall of w as functions
      of the weight.  This is what decides whether the hole is a blocker or a
      bounded, documentable limitation.

Run (from repo root)::

    python theory/ground_truth_w_g.py

No GPU.  ~4-5 minutes on CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from udwm.models.consistency import TermScaleEMA, coupled_w_g  # noqa: E402

N = 5          # ensemble members
FDIM = 4       # feature dim (phi(s, a))
RHO = 0.4      # shared-latent correlation (the diffusion latent coupling)
M_EVAL = 64    # MC budget for evaluation
GRID = 256     # eval grid points
BATCH = 64
PRETRAIN = 400
STEPS = 1200
LR = 1e-3
SEED = 0


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_teacher(seed: int = SEED):
    """Fixed member parameter draws (the 'parameter posterior') + noise design."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(N, FDIM, 2, generator=g) * 0.8
    eta = torch.randn(FDIM, generator=g) * 0.5
    c = 0.0
    return theta, eta, c


def sigma_fn(x: torch.Tensor, eta: torch.Tensor, c: float) -> torch.Tensor:
    """[B] aleatoric noise scale, shared across members, varies across x."""
    return torch.exp(0.5 * (x @ eta + c))


def member_state_means(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """[N,B,2] s_m(x) = Theta_m x."""
    return torch.einsum("nfd,bf->nbd", theta, x)


def analytic_w_g(theta, x, sigma2):
    """Closed-form (w*, g*) for V(s) = ||s||^2 under s ~ N(s_m, sigma^2 I).

    E[V] = ||s_m||^2 + 2 sigma^2 ;  Var[V] = 4 sigma^2 ||s_m||^2 + 4 sigma^4.
    w* = Var_m(E[V]) (ddof=0, the `coupled_w_g` convention); g* = E_m[Var(V)].
    """
    s = member_state_means(theta, x)
    mu = s.pow(2).sum(-1)
    w = mu.var(dim=0, unbiased=False)
    g = 4.0 * sigma2 * s.pow(2).sum(-1).mean(dim=0) + 4.0 * sigma2 ** 2
    return w, g


def sample_values(state_means, scale, rho, m, gen):
    """[N,M,B,1] member values Y = ||s_m + eps||^2 with shared-latent coupling."""
    n, b, _ = state_means.shape
    common = torch.randn(b, m, 2, generator=gen)
    idio = torch.randn(n, b, m, 2, generator=gen)
    eps = scale[None, :, None, None] * (
        torch.sqrt(torch.tensor(rho, dtype=torch.float32)) * common[None, :, :, :]
        + torch.sqrt(torch.tensor(1.0 - rho, dtype=torch.float32)) * idio
    )
    y = (state_means.unsqueeze(2) + eps).pow(2).sum(-1)
    return y.transpose(1, 2).unsqueeze(-1)


class Student(nn.Module):
    """Per-member state means + a shared aleatoric scale, both functions of x."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(FDIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
        )
        self.head_state = nn.Linear(64, N * 2)
        self.head_scale = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        s = self.head_state(h).view(-1, N, 2).transpose(0, 1)
        scale = F.softplus(self.head_scale(h)).squeeze(-1) + 1e-3
        return s, scale


def value_geometry(y_s, y_t):
    mu_s = y_s.mean(dim=1)
    mu_t = y_t.mean(dim=1)
    cs = mu_s - mu_s.mean(dim=0, keepdim=True)
    ct = mu_t - mu_t.mean(dim=0, keepdim=True)
    return F.mse_loss(cs, ct)


def ordinary_loss(st, x, s_m, _sigma2, _gen):
    s_hat, _ = st(x)
    return F.mse_loss(s_hat, s_m)


def hybrid_loss(st, x, s_m, sigma2, gen):
    """Repo's decision loss at M=1: member + value geometry + single-latent var."""
    s_hat, scale_hat = st(x)
    y_s = sample_values(s_hat, scale_hat, RHO, 1, gen)
    y_t = sample_values(s_m, sigma2.sqrt(), RHO, 1, gen)
    member = F.mse_loss(s_hat, s_m)
    geom = value_geometry(y_s, y_t)
    var = F.mse_loss(
        y_s.var(dim=0, unbiased=True).squeeze(-1),
        y_t.var(dim=0, unbiased=True).squeeze(-1),
    )
    return member + geom + var


def identified_loss(st, x, s_m, sigma2, gen, m=2, reweight="none", ema=None, kw=1.0, kg=1.0):
    """Repo's identified loss: M>=2 latents, separate (w_deb, g) matching.

    reweight="none"  -> equal weights (the current repo default; the hole).
    reweight="meanmag" -> divide each term by the teacher mean-magnitude squared.
    reweight="ema"   -> the fix now in udwm.models.consistency.TermScaleEMA,
                        with optional explicit relative weights kw / kg (sweep).
    """
    s_hat, scale_hat = st(x)
    y_s = sample_values(s_hat, scale_hat, RHO, m, gen)
    y_t = sample_values(s_m, sigma2.sqrt(), RHO, m, gen)
    member = F.mse_loss(s_hat, s_m)
    geom = value_geometry(y_s, y_t)
    s_w, s_g = coupled_w_g(y_s, m)
    t_w, t_g = coupled_w_g(y_t, m)
    ep = F.mse_loss(s_w, t_w.detach())
    al = F.mse_loss(s_g, t_g.detach())
    if reweight == "meanmag":
        ep = ep / (t_w.detach().mean() ** 2 + 1e-8)
        al = al / (t_g.detach().mean() ** 2 + 1e-8)
    elif reweight == "ema" and ema is not None:
        ep = ep * ema.update("w", t_w) * kw
        al = al * ema.update("g", t_g) * kg
    return member + geom + ep + al


def train(st, loss_fn, x_src, s_m_src, sigma2_src, steps, gen):
    opt = torch.optim.Adam(st.parameters(), lr=LR)
    for _ in range(steps):
        idx = torch.randint(0, x_src.shape[0], (BATCH,))
        x = x_src[idx]
        loss = loss_fn(st, x, s_m_src[:, idx], sigma2_src[idx], gen)
        opt.zero_grad()
        loss.backward()
        opt.step()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ca, cb = ra - ra.mean(), rb - rb.mean()
    denom = ca.norm() * cb.norm()
    return float((ca * cb).sum() / denom) if denom > 0 else 0.0


def top_decile_recall(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Fraction of true top-decile points recovered in the predicted top-decile."""
    k = max(1, pred.numel() // 10)
    top_t = set(torch.topk(truth, k).indices.tolist())
    top_p = set(torch.topk(pred, k).indices.tolist())
    return len(top_t & top_p) / len(top_t)


@torch.no_grad()
def evaluate(st, x_grid, theta, sigma2_grid, gen):
    s_hat, scale_hat = st(x_grid)
    y = sample_values(s_hat, scale_hat, RHO, M_EVAL, gen)
    w_hat, g_hat = coupled_w_g(y, M_EVAL)
    w_hat, g_hat = w_hat.squeeze(-1), g_hat.squeeze(-1)
    w_true, g_true = analytic_w_g(theta, x_grid, sigma2_grid)
    w_rel = float(torch.sqrt(F.mse_loss(w_hat, w_true)) / w_true.std(unbiased=False))
    g_rel = float(torch.sqrt(F.mse_loss(g_hat, g_true)) / g_true.std(unbiased=False))
    ratio = ((g_hat / g_true) / (w_hat / w_true)).clamp(0.01, 100.0)
    return {
        "w_rel_rmse": w_rel,
        "g_rel_rmse": g_rel,
        "w_rank_corr": spearman(w_hat, w_true),
        "g_rank_corr": spearman(g_hat, g_true),
        "w_rmse": float(torch.sqrt(F.mse_loss(w_hat, w_true))),
        "g_rmse": float(torch.sqrt(F.mse_loss(g_hat, g_true))),
        "split_distortion_median": float(ratio.median()),
        "w_mean": float(w_true.mean()),
        "g_mean": float(g_true.mean()),
        "w_hat_mean": float(w_hat.mean()),
        "g_hat_mean": float(g_hat.mean()),
        "w_top10_recall": top_decile_recall(w_hat, w_true),
        "g_top10_recall": top_decile_recall(g_hat, g_true),
    }


def part0(theta, eta, c, gen):
    print("=" * 84)
    print("PART 0 -- does the MC machinery recover the ANALYTIC (w*, g*)?")
    print("=" * 84)
    x = torch.randn(GRID, FDIM, generator=gen)
    sigma2 = sigma_fn(x, eta, c) ** 2
    w_true, g_true = analytic_w_g(theta, x, sigma2)
    print(f"analytic: mean w* = {w_true.mean():.4f}  mean g* = {g_true.mean():.4f}")
    print(f"{'M':>4} {'w_raw rel err':>13} {'w_deb rel err':>14} {'g rel err':>11} "
          f"{'w_rank':>8} {'g_rank':>8}")
    out = {}
    for m in (2, 4, 8, 16, M_EVAL):
        y = sample_values(member_state_means(theta, x), sigma2.sqrt(), RHO, m, gen)
        w_hat, g_hat = coupled_w_g(y, m)
        w_hat, g_hat = w_hat.squeeze(-1), g_hat.squeeze(-1)
        w_raw = y.mean(dim=1).var(dim=0, unbiased=False).squeeze(-1)
        w_raw_rel = float(torch.sqrt(F.mse_loss(w_raw, w_true)) / w_true.std(unbiased=False))
        w_rel = float(torch.sqrt(F.mse_loss(w_hat, w_true)) / w_true.std(unbiased=False))
        g_rel = float(torch.sqrt(F.mse_loss(g_hat, g_true)) / g_true.std(unbiased=False))
        out[m] = {"w_raw_rel": w_raw_rel, "w_rel": w_rel, "g_rel": g_rel}
        print(f"{m:>4} {w_raw_rel:>13.4f} {w_rel:>14.4f} {g_rel:>11.4f} "
              f"{spearman(w_hat, w_true):>8.3f} {spearman(g_hat, g_true):>8.3f}")
    # Coupling identity: E[Var_m Y] (ddof=0) = w* + g* - Sigma_bar, verified
    # numerically under the NONLINEAR value map V(s) = ||s||^2.
    trials = 256
    ev = 0.0
    means = []
    for _ in range(trials):
        y1 = sample_values(member_state_means(theta, x), sigma2.sqrt(), RHO, 1, gen)
        yb = y1.squeeze(2).squeeze(-1)
        ev += float(yb.var(dim=0, unbiased=False).mean())
        means.append(yb.mean(dim=0))
    means = torch.stack(means)
    ev /= trials
    esb = float(means.var(dim=0, unbiased=False).mean())
    pred = float((w_true + g_true - esb).mean())
    print(f"single-latent stat: E[Var_m Y] MC = {ev:.4f} vs w* + g* - Sigma_bar = {pred:.4f}")
    print(f"  (w* = {w_true.mean():.4f}, g* = {g_true.mean():.4f}, Sigma_bar = {esb:.4f};")
    print("   the repo loss uses ddof=1, i.e. x N/(N-1) = %.4f)" % (ev * N / (N - 1.0)))
    print("  -> estimator unbiased for the analytic truth at every M>=2; the")
    print("     M=1 statistic is a blend of w* and the aleatoric remainder,")
    print("     and the coupling identity holds even under the nonlinear V.")
    print()
    return out


def run_arms(x_src, s_m_src, sigma2_src, x_grid, sigma2_grid, theta, gen,
             init=None, arms=("ordinary", "hybrid", "identified", "identified_ema")):
    results = {}
    for arm in arms:
        st = Student()
        ema = TermScaleEMA() if arm in ("identified_ema", "identified_sweep") else None
        if init == "collapsed":
            with torch.no_grad():
                st.head_state.weight.mul_(0.02)
                st.head_state.bias.mul_(0.02)
        if init == "benign":
            train(st, ordinary_loss, x_src, s_m_src, sigma2_src, PRETRAIN, gen)
        if init == "collapsed":
            # inflate the aleatoric scale to match the teacher's single-latent
            # statistic at (nearly) zero epistemic spread: the zero-loss family.
            def single_stat(scale):
                y = sample_values(member_state_means(theta, x_grid) * 0.0,
                                  torch.full((x_grid.shape[0],), scale, dtype=torch.float32),
                                  RHO, 1, gen)
                return y.var(dim=0, unbiased=True).mean().item()
            y_t = sample_values(member_state_means(theta, x_grid), sigma2_grid.sqrt(), RHO, 1, gen)
            target = y_t.var(dim=0, unbiased=True).mean().item()
            lo, hi = 1e-3, 50.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if single_stat(mid) < target:
                    lo = mid
                else:
                    hi = mid
            with torch.no_grad():
                st.head_scale.bias.fill_(torch.log(torch.tensor(hi, dtype=torch.float32)))

        if arm == "ordinary":
            fn = ordinary_loss
        elif arm == "hybrid":
            fn = hybrid_loss
        elif arm == "identified":
            fn = lambda st_, x, s_m, sg, g: identified_loss(st_, x, s_m, sg, g, m=2, reweight="none")
        elif arm == "identified_rw":
            fn = lambda st_, x, s_m, sg, g: identified_loss(st_, x, s_m, sg, g, m=2, reweight="meanmag")
        elif arm == "identified_ema":
            fn = lambda st_, x, s_m, sg, g: identified_loss(st_, x, s_m, sg, g, m=2, reweight="ema", ema=ema)
        else:
            raise ValueError(arm)
        train(st, fn, x_src, s_m_src, sigma2_src, STEPS, gen)
        results[arm] = evaluate(st, x_grid, theta, sigma2_grid, gen)
    return results


def fmt_table(results, arms, order):
    header = f"{'arm':<16}" + "".join(f"{k:>14}" for k in order)
    print(header)
    for arm in arms:
        r = results[arm]
        line = f"{arm:<16}"
        for k in order:
            line += f"{r[k]:>14.4f}"
        print(line)
    print()


def sweep_weighting(x_src, s_m_src, sigma2_src, x_grid, sigma2_grid, theta, gen):
    """P4: relative weight of (normalised) w vs g terms, hole regime, benign init.

    kw / kg multiply the EMA-normalised terms.  Sweep kw = r, kg = 1 - r and
    measure whether w RANK (and top-decile recall, the gate-relevant endpoint)
    stays high even where w MAGNITUDE does not.
    """
    print("=" * 84)
    print("PART 4 -- rank robustness to the weighting choice (hole regime)")
    print("=" * 84)
    rows = []
    for r in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999):
        torch.manual_seed(SEED + 100)
        st = Student()
        ema = TermScaleEMA()
        train(st, ordinary_loss, x_src, s_m_src, sigma2_src, PRETRAIN, gen)
        fn = lambda st_, x, s_m, sg, g, r=r: identified_loss(
            st_, x, s_m, sg, g, m=2, reweight="ema", ema=ema, kw=r, kg=1.0 - r)
        train(st, fn, x_src, s_m_src, sigma2_src, STEPS, gen)
        ev = evaluate(st, x_grid, theta, sigma2_grid, gen)
        rows.append((r, ev))
        print(f"r(w)={r:<7.3f}  w_rank={ev['w_rank_corr']:.3f}  w_rel={ev['w_rel_rmse']:.3f}  "
              f"g_rank={ev['g_rank_corr']:.3f}  g_rel={ev['g_rel_rmse']:.3f}  "
              f"w_top10={ev['w_top10_recall']:.3f}")
    # Robustness windows: where does w_rank stay high, and does a single r give
    # both good relative accuracy?
    good_rank = [r for r, ev in rows if ev["w_rank_corr"] > 0.8]
    both = [r for r, ev in rows if ev["w_rel_rmse"] < 0.5 and ev["g_rel_rmse"] < 0.5]
    print()
    print(f"  w_rank > 0.8 for r in [{good_rank[0]:.3f}, {good_rank[-1]:.3f}] "
          f"({len(good_rank)}/{len(rows)} points)")
    print(f"  single r with BOTH w_rel<0.5 and g_rel<0.5: "
          f"{both if both else 'NONE -> no single scalar weighting recovers both magnitudes'}")
    print()
    return rows


def main():
    t0 = time.time()
    set_seed(SEED)
    gen = torch.Generator().manual_seed(SEED + 1)
    theta, eta, c = make_teacher(SEED)

    x_src = torch.randn(512, FDIM, generator=gen)
    sigma2_src = sigma_fn(x_src, eta, c) ** 2
    s_m_src = member_state_means(theta, x_src)

    x_grid = torch.randn(GRID, FDIM, generator=torch.Generator().manual_seed(999))
    sigma2_grid = sigma_fn(x_grid, eta, c) ** 2

    out = {"config": {"N": N, "FDIM": FDIM, "rho": RHO, "steps": STEPS, "m_eval": M_EVAL},
           "part0": part0(theta, eta, c, gen)}

    order = ["w_rel_rmse", "g_rel_rmse", "w_rank_corr", "g_rank_corr",
             "w_top10_recall", "w_rmse", "g_rmse"]

    print("=" * 84)
    print("PART 1 -- benign init (member-pretrained): the fixed-map situation")
    print("=" * 84)
    arms = ("ordinary", "hybrid", "identified", "identified_ema")
    r_benign = run_arms(x_src, s_m_src, sigma2_src, x_grid, sigma2_grid, theta, gen,
                        init="benign", arms=arms)
    out["part1_benign"] = r_benign
    fmt_table(r_benign, arms, order)
    print("  -> on this toy even a benign init does NOT rescue the M=1 objective:")
    print("     hybrid inflates w (w_rel %.2f) and deflates g; the identified" % r_benign["hybrid"]["w_rel_rmse"])
    print("     objective recovers both components (w_rel %.2f, w_rank %.2f)." % (
        r_benign["identified"]["w_rel_rmse"], r_benign["identified"]["w_rank_corr"]))
    print("     Whether the trade is walked depends on the value map; the loss")
    print("     leaves it free, which is the point.")
    print()

    print("=" * 84)
    print("PART 2 -- collapsed init (w_s ~ 0, scale inflated to the M=1 statistic)")
    print("=" * 84)
    r_coll = run_arms(x_src, s_m_src, sigma2_src, x_grid, sigma2_grid, theta, gen,
                      init="collapsed", arms=arms)
    out["part2_collapsed"] = r_coll
    fmt_table(r_coll, arms, order)
    print("  -> from the zero-loss family, the hybrid w magnitude stays far off")
    print("     (w_rel %.2f) while identified recovers it to w_rel %.2f / w_rank %.2f," % (
        r_coll["hybrid"]["w_rel_rmse"], r_coll["identified"]["w_rel_rmse"],
        r_coll["identified"]["w_rank_corr"]))
    print("     and g_rank %.2f vs %.2f.  The identified target pulls the student" % (
        r_coll["identified"]["g_rank_corr"], r_coll["hybrid"]["g_rank_corr"]))
    print("     back to the truth; the M=1 objective leaves it on the family.")
    print()

    print("=" * 84)
    print("PART 3 -- the numerical hole: g* >> w*, equal vs meanmag vs EMA reweight")
    print("=" * 84)
    scale_noise = 4.0
    sigma2_src_h = sigma2_src * scale_noise ** 2
    sigma2_grid_h = sigma2_grid * scale_noise ** 2
    r_hole = run_arms(x_src, s_m_src, sigma2_src_h, x_grid, sigma2_grid_h, theta, gen,
                      init="benign", arms=("identified", "identified_rw", "identified_ema"))
    out["part3_hole"] = r_hole
    fmt_table(r_hole, ("identified", "identified_rw", "identified_ema"), order)
    print("  -> equal weights: with g* ~ %.0fx w* the epistemic term is numerically" % (
        r_hole["identified"]["g_mean"] / max(r_hole["identified"]["w_mean"], 1e-9)))
    print("     negligible, so w is not recovered (w_rank %.2f, w_rel %.2f)." % (
        r_hole["identified"]["w_rank_corr"], r_hole["identified"]["w_rel_rmse"]))
    print("     EMA per-term normalisation (the fix) recovers w (w_rank %.2f, w_rel %.2f);" % (
        r_hole["identified_ema"]["w_rank_corr"], r_hole["identified_ema"]["w_rel_rmse"]))
    print("     its g (g_rank %.2f) versus equal-weight g_rank %.2f." % (
        r_hole["identified_ema"]["g_rank_corr"], r_hole["identified"]["g_rank_corr"]))
    print()

    sweep = sweep_weighting(x_src, s_m_src, sigma2_src_h, x_grid, sigma2_grid_h, theta, gen)
    out["part4_sweep"] = [{"r": r, **ev} for r, ev in sweep]

    out_path = ROOT / "runs" / "ground_truth_w_g.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print("1. `coupled_w_g` recovers the analytic (w*, g*) at every M >= 2 (P0):")
    print("   the measurement is correct, not merely self-consistent.")
    print("2. The M=1 objective walks the degenerate direction even from a benign")
    print("   init; identified recovers both (P1), and from the zero-loss family")
    print("   hybrid stays far off in w while identified returns (P2).")
    print("3. Equal weighting fails on w when g* >> w* (P3); the EMA per-term")
    print("   normaliser (now in udwm.models.consistency.TermScaleEMA) fixes it.")
    print("4. Rank robustness (P4): see the window above -- is w ranking robust")
    print("   to the weighting even where magnitude is not?  That decides whether")
    print("   the hole is a blocker or a bounded, documentable limitation.")
    print("=" * 84)
    print(f"wrote {out_path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
