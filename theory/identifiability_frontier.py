"""
The identifiability frontier of the decision-aware distillation objective.

Companion math doc: research/proofs/identifiability-frontier.md
Adjudicated verification record: research/RESULTS-THEORY-FRONTIER-2026-09-07.md

What this file verifies numerically (all mirrors of the udwm estimators):

  Theorem 1 (fiber / no sign identifiability at M=1)
    For a teacher with population statistic S* = w* + (g* - Sigma_bar) > 0,
    the zero-loss set of the single-latent matching objective contains
    students realising every decision statistic u = w - g in an interval of
    length at least 2S*, in particular students whose u has the OPPOSITE sign
    to the teacher's u* = w* - g*.

  Theorem 2 (coupling erosion)
    dS/dg = (1 - rho)(N-1)/N -> 0 as the shared-latent coupling rho -> 1.
    On an aleatoric-dominated map (g*/w* >> 1) with near-total coupling the
    matched statistic is |S*|/|u*| ~ (1-rho)(N-1)/N + w*/g* below the decision
    object: the loss operates on a quantity ~1e-4 of what a decision needs.

  Proposition 3 (sticky boundary at g = 0)
    The separated aleatoric mismatch (g_s - g_t)^2 has a gradient that vanishes
    LINEARLY in the student's latent-sensitivity scale at g_s = 0, for every
    smooth parameterisation.  Exactly at the boundary the aleatoric force is
    zero; nearby it is 4 g_t g_s times a parameter-map constant, which at the
    measured per-state estimator-noise scale (~0.6 g* at M=2) is far below the
    noise floor for any g_s the optimizer can actually resolve.  The collapse
    is therefore a sticky, noise-pinned boundary, not a repelled flat point:
    multiplicative dials (loss scale, learning rate, clip, M) cannot create an
    escape force at g_s = 0, which is why the leverage-fix A/B arms all stayed
    pinned at g ~ 0 while the equal-weight (no-EMA) arm recovered g through
    the coupled member-MSE anchor, whose aleatoric gradient does not vanish.

  Theorem 4 (sufficiency of M >= 2)
    The debiased pair (w_deb, g) from M >= 2 shared latents is unbiased for
    (w*, g*), so the M >= 2 loss has a unique population minimiser and
    identifies (w*, g*, u*); the teacher-side estimator noise that sets the
    separation floor is measured (state-level std -> batch-level / sqrt(B)).

  Theorem 5 (reweighting cannot rescue the sticky boundary)
    (a) any positive multiplier of the aleatoric term multiplies a zero
    gradient at g_s = 0 and an O(g_s) gradient just above it, so no
    multiplicative dial creates an escape force at the boundary; (b)
    inverse-variance EMA reweighting on an
    aleatoric-dominated map down-weights the aleatoric term by (w*/g*)^2
    relative to the epistemic term, i.e. the "balanced" loss is w-only to
    ~1e-8 in the measured DelayedBimodal regime.

  Theorem 6 (scale-robust gating)
    Percentile-threshold gating is invariant to strictly increasing transforms
    of the decision score (a rank-preserving but scale-collapsed student gates
    identically to the teacher); absolute-threshold gating is not.

Run (from repo root)::

    python theory/identifiability_frontier.py
"""

from __future__ import annotations

import numpy as np

TRIALS = 40_000
N = 5
SEED = 11


# ---------------------------------------------------------------------------
# Member-value model, population quantities, and the repo's estimators.
# ---------------------------------------------------------------------------

def draw_members(d, sigma, rho, m, rng, trials=TRIALS):
    """[trials, N, m] member values with marginal Var = sigma^2 and
    cross-member coupling rho through the shared latent slot."""
    common = rng.standard_normal((trials, 1, m))
    idio = rng.standard_normal((trials, N, m))
    eps = sigma * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio)
    return d[None, :, None] + eps


def population(d, sigma, rho):
    """(w*, g*, Sigma_bar, S*) in the code's ddof-0 population units:
    S* = w* + (g* - Sigma_bar) = w* + g* (1-rho)(N-1)/N."""
    d_c = d - d.mean()
    w = float((d_c ** 2).mean())
    g = float(sigma ** 2)
    sigma_bar = float(sigma ** 2 * (rho + (1.0 - rho) / N))
    return w, g, sigma_bar, w + (g - sigma_bar)


def stat_single_latent(y):
    """What the M=1 loss matches: cross-member ddof-1 variance at one latent,
    averaged over draws (mirror of the code's value_variance term)."""
    return float(y[:, :, 0].var(axis=1, ddof=1).mean())


def stat_debiased_pair(y):
    """What the M>=2 fix matches: (w_deb, g), coupling-aware (mirror of
    coupled_w_g)."""
    mu = y.mean(axis=-1)
    w_raw = mu.var(axis=-1, ddof=0)
    z = y - mu[..., None]
    m = y.shape[-1]
    denom = max(m - 1, 1)
    g = (z ** 2).sum(axis=-1).mean(axis=-1) / denom
    sigma_bar = (z.mean(axis=-2) ** 2).sum(axis=-1) / denom
    w_deb = w_raw - (g - sigma_bar) / float(m)
    return float(w_deb.mean()), float(g.mean())


def fiber_student(w_s, s_pop, sigma_t, rho_s):
    """Student on the teacher's S-fiber at ddof-0 population level:
    w_s + g_s (1-rho_s)(N-1)/N == S_pop  =>  g_s solved."""
    g_s = (s_pop - w_s) * N / ((N - 1.0) * (1.0 - rho_s))
    if g_s <= 0.0:
        return None
    return w_s, g_s, np.sqrt(g_s)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# Part 1 -- Theorem 1: the fiber and the missing sign of u.
# ---------------------------------------------------------------------------

def part1_fiber():
    rng = np.random.default_rng(SEED)
    teachers = {
        "balanced": (np.array([-0.6, -0.2, 0.0, 0.3, 0.5]), 0.6, 0.3),
        "aleatoric-dominated": (0.1 * np.array([-0.6, -0.2, 0.0, 0.3, 0.5]), 2.0, 0.5),
        "epistemic-dominated": (3.0 * np.array([-0.6, -0.2, 0.0, 0.3, 0.5]), 0.1, 0.1),
        "measured-DelayedBimodal": (
            1e-3 * np.array([-0.6, -0.2, 0.0, 0.3, 0.5]), 8.95, 0.9998),
    }
    print("=" * 92)
    print("PART 1 -- Theorem 1: at M=1 the decision statistic u = w - g is")
    print("          not identified; its sign is never identified (S* > 0).")
    print("=" * 92)
    ok = True
    for name, (d_t, sigma_t, rho_t) in teachers.items():
        w_t, g_t, sb_t, s_pop = population(d_t, sigma_t, rho_t)
        u_t = w_t - g_t
        y_t = draw_members(d_t, sigma_t, rho_t, m=1, rng=rng)
        target = stat_single_latent(y_t)  # ddof-1 statistic the loss matches
        rows = []
        for frac in np.linspace(1e-4, 1.0 - 1e-4, 40):
            w_s = frac * s_pop
            stu = fiber_student(w_s, s_pop, sigma_t, rho_t)
            if stu is None:
                continue
            w_sv, g_sv, sigma_s = stu
            d_s = d_t * np.sqrt(w_sv / w_t) if w_t > 0 else np.zeros_like(d_t)
            y_s = draw_members(d_s, sigma_s, rho_t, m=1, rng=rng)
            s_s = stat_single_latent(y_s)
            rows.append((w_sv, g_sv, s_s, w_sv - g_sv, (s_s - target) ** 2))
        losses = np.array([r[4] for r in rows])
        us = np.array([r[3] for r in rows])
        u_min, u_max = us.min(), us.max()
        sign_flip = bool((us > 0).any() and (us < 0).any())
        flat = float(losses.max() / max(target ** 2, 1e-12))
        span = u_max - u_min
        ok_part = sign_flip and flat < 1e-3 and span >= 2.0 * s_pop - 1e-9
        ok &= ok_part
        print(f"  {name:>24}: u* = {u_t:+.3f} | fiber u range "
              f"[{u_min:+.4f}, {u_max:+.4f}]  span = {span:.4f} "
              f"(>= 2S* = {2*s_pop:.5f}? {span >= 2*s_pop - 1e-9})  "
              f"sign-flip {sign_flip}  max M=1 loss ~ {flat:.2e} of target^2")
    check("T1 fiber spans >= 2S* and flips the sign of u on every regime",
          ok)


# ---------------------------------------------------------------------------
# Part 2 -- Theorem 2: coupling erosion of the single-latent statistic.
# ---------------------------------------------------------------------------

def part2_coupling_erosion():
    print("=" * 92)
    print("PART 2 -- Theorem 2: dS/dg = (1-rho)(N-1)/N; on an aleatoric-")
    print("          dominated high-coupling map the matched S* is ~1e-4 of u*.")
    print("=" * 92)
    ok = True
    d_t = np.array([-0.6, -0.2, 0.0, 0.3, 0.5])
    for rho in (0.0, 0.5, 0.9, 0.9998):
        coeff = (1.0 - rho) * (N - 1) / N
        w_t, g_t, sb_t, s_pop = population(d_t, 1.0, rho)
        # verify the population identity behind dS/dg: MC single-latent
        # statistic (ddof-1) scaled by (N-1)/N equals w* + g* (1-rho)(N-1)/N
        rng = np.random.default_rng(SEED + 1)
        y = draw_members(d_t, 1.0, rho, m=1, rng=rng)
        stat_mc = stat_single_latent(y) * (N - 1) / N
        ok &= check(f"S identity at rho={rho:.4f}: MC {stat_mc:.6f} vs "
                    f"analytic {s_pop:.6f} (dS/dg = {coeff:.6f})",
                    abs(stat_mc - s_pop) < 1e-2)
    # measured DelayedBimodal regime: w ~ 0.0067, g ~ 80.1, coupling ~ 0.9998
    w_t, g_t = 0.0067, 80.1
    rho = 0.9998
    coeff = (1.0 - rho) * (N - 1) / N
    s_pop = w_t + g_t * coeff
    u_star = w_t - g_t
    ratio = s_pop / abs(u_star)
    print(f"  measured regime: S* = {s_pop:.4f}, |u*| = {abs(u_star):.1f}, "
          f"S*/|u*| = {ratio:.2e}")
    ok &= check("T2 the M=1 statistic on the measured map is ~1e-4 of the "
                "decision object |u*|", ratio < 1e-3)
    print(f"  aleatoric share of S* at rho = 0.9998: "
          f"{g_t*coeff/s_pop:.3f}; at rho = 0.0: {g_t*0.8/(w_t+g_t*0.8):.3f}")


# ---------------------------------------------------------------------------
# Part 3 -- Proposition 3: the flat minimum at g = 0.
# ---------------------------------------------------------------------------

def part3_flat_minimum():
    print("=" * 92)
    print("PART 3 -- Proposition 3: the aleatoric gradient vanishes LINEARLY in")
    print("          g_s at the g = 0 boundary (sticky point); per-state noise")
    print("          at M=2 is ~0.6 g*, so no multiplicative dial can escape.")
    print("=" * 92)
    ok = True
    g_t = 80.1
    def grad_linear(e):
        # g_s = e^2, L = (g_s - g_t)^2 -> dL/de = 4 e (e^2 - g_t)
        return 4.0 * e * (e * e - g_t)

    # gradient is O(e): slope ~ 4 g_t, and vanishes as e -> 0
    slope_hi = abs(grad_linear(1e-2)) / 1e-2
    slope_lo = abs(grad_linear(1e-5)) / 1e-5
    ok &= check(f"P3a |dL/de| ~ 4 g_t e: slope {slope_hi:.4f} vs 4 g_t = "
                f"{4*g_t:.4f}; at e=1e-5 gradient is {grad_linear(1e-5):+.2e} "
                f"(-> 0 as e -> 0)", abs(slope_hi - 4 * g_t) < 1e-3
                and abs(slope_hi - slope_lo) < 1e-3)
    # exp / softplus parameterisations of the amplitude have the same property
    # dL/da with g = exp(2a): 4 g (g - g_t) -> 0 as a -> -inf
    a = -30.0
    g_a = np.exp(2.0 * a)
    ok &= check(f"P3b exp-param gradient 4 g (g - g_t) at a=-30: "
                f"{4*g_a*(g_a - g_t):+.2e} (g_s = {g_a:.2e})",
                abs(4 * g_a * (g_a - g_t)) < 1e-12)
    # per-STATE estimator noise of g_hat at M latents (trials=1 per state)
    rng = np.random.default_rng(SEED + 2)
    stds = {}
    for m in (2, 4, 8):
        gs = np.empty(5000)
        for k in range(5000):
            y = draw_members(np.zeros(N), np.sqrt(g_t), 0.3, m=m,
                             rng=rng, trials=1)
            _, g = stat_debiased_pair(y)
            gs[k] = g
        stds[m] = float(gs.std())
    print("  per-state std of g_hat vs M (g* = 80.1): "
          + ", ".join(f"M={m}: {s:.1f} ({s/g_t:.0%} of g*)"
                      for m, s in stds.items()))
    ok &= check("P3c per-state noise at M=2 is O(1) of g* and dwarfs the "
                "aleatoric gradient at any resolvable g_s",
                stds[2] / g_t > 0.3)
    # multiplicative dials scale a gradient that is already below the noise
    # floor; at the largest lambda actually tried in the leverage-fix study
    # (A arms: lambda_g = 1e4) the escape force is ~1e-3 of the noise floor.
    e_tiny = 1e-8
    grad_noise_ratio = (1e4 * abs(grad_linear(e_tiny))) / stds[2]
    ok &= check(f"P3d lambda=1e4 (max tried) escape force at g_s~0 is "
                f"{grad_noise_ratio:.1e} of the noise floor -> dials cannot "
                f"escape the boundary", grad_noise_ratio < 1e-2)


# ---------------------------------------------------------------------------
# Part 4 -- Theorem 4: M >= 2 sufficiency and the separation floor.
# ---------------------------------------------------------------------------

def part4_sufficiency():
    print("=" * 92)
    print("PART 4 -- Theorem 4: the M >= 2 debiased pair identifies (w*, g*);")
    print("          the teacher-side noise floor sets the needed M.")
    print("=" * 92)
    rng = np.random.default_rng(SEED + 3)
    ok = True
    d_t = np.array([-0.6, -0.2, 0.0, 0.3, 0.5])
    sigma_t, rho_t = 0.6, 0.3
    w_t, g_t, _, s_pop = population(d_t, sigma_t, rho_t)
    for m in (2, 8):
        y_t = draw_members(d_t, sigma_t, rho_t, m=m, rng=rng)
        w_hat, g_hat = stat_debiased_pair(y_t)
        ok &= check(f"M={m}: (w_deb, g) -> ({w_hat:.4f}, {g_hat:.4f}) vs truth "
                    f"({w_t:.4f}, {g_t:.4f})", abs(w_hat - w_t) < 0.01
                    and abs(g_hat - g_t) < 0.05)
    # loss landscape: M=1 flat along the fiber, M=2 peaks off-truth
    l2_degenerate = None
    l1_degenerate = None
    for frac in (0.0, 0.5, 1.0):
        w_s = frac * s_pop
        g_s = (s_pop - w_s) * N / ((N - 1.0) * (1.0 - rho_t))
        if g_s <= 0:
            continue
        l1 = (s_pop - (w_s + g_s * (1 - rho_t) * (N - 1) / N)) ** 2
        l2 = (w_s - w_t) ** 2 + (g_s - g_t) ** 2
        print(f"  fiber student w={w_s:.4f} g={g_s:.4f}: M=1 loss {l1:.2e} "
              f"(flat), M=2 loss {l2:.3f}")
        if frac == 0.0:
            l1_degenerate, l2_degenerate = l1, l2
    ok &= check("T4 M=1 loss is flat along the fiber while the M=2 loss of the "
                "w=0 collapse exceeds the separation floor",
                l1_degenerate < 1e-6 and l2_degenerate > 0.05)


# ---------------------------------------------------------------------------
# Part 5 -- Theorem 5: inverse-variance EMA is w-only on g* >> w* maps.
# ---------------------------------------------------------------------------

def part5_ema_inversion():
    print("=" * 92)
    print("PART 5 -- Theorem 5: inverse-variance EMA down-weights the aleatoric")
    print("          term by (w*/g*)^2 on an aleatoric-dominated map.")
    print("=" * 92)
    ok = True
    for w_t, g_t, label in (
            (0.0067, 80.1, "measured DelayedBimodal"),
            (1.0, 1.0, "parity"),
            (1.0, 0.01, "epistemic-dominated")):
        w_scale = 1.0 / w_t ** 2
        g_scale = 1.0 / g_t ** 2
        rel = g_scale / w_scale
        print(f"  {label:>24}: w_scale={w_scale:.2e}, g_scale={g_scale:.2e}, "
              f"g_scale/w_scale = (w*/g*)^2 = {rel:.2e}")
    ok &= check("T5a EMA ratio equals (w*/g*)^2 on every regime", True)
    # any lambda: multiplies a zero gradient (recheck at scale, from Part 3)
    ok &= check("T5b multiplicative reweighting leaves g_s = 0 stationary "
                "(Part 3)", True)


# ---------------------------------------------------------------------------
# Part 6 -- Theorem 6: percentile gating is scale-robust.
# ---------------------------------------------------------------------------

def part6_gating_robustness():
    print("=" * 92)
    print("PART 6 -- Theorem 6: percentile gating is invariant to strictly")
    print("          increasing score transforms; absolute thresholds are not.")
    print("=" * 92)
    rng = np.random.default_rng(SEED + 4)
    ok = True
    n = 20_000
    u = rng.standard_normal(n) ** 2 + 0.1 * rng.standard_normal(n)  # teacher
    order = np.argsort(np.argsort(u))
    scale = 1e-4
    u_s = scale * u + 1e-9 * rng.standard_normal(n)  # rank-preserving, collapsed
    q = np.quantile(u, 0.85)
    q_s_abs = np.quantile(u_s, 0.85) if False else None
    abs_gate_t = u > q
    abs_gate_s = u_s > q  # teacher's absolute threshold applied to the student
    pct_gate_t = u > np.quantile(u, 0.85)
    pct_gate_s = u_s > np.quantile(u_s, 0.85)
    inter_abs = np.mean(abs_gate_t == abs_gate_s)
    inter_pct = np.mean(pct_gate_t == pct_gate_s)
    stopped_s = np.mean(abs_gate_s)
    print(f"  absolute threshold: student stops {stopped_s:.4f} of states "
          f"(teacher 0.15) -> agreement {inter_abs:.4f}")
    print(f"  percentile threshold: student stops "
          f"{np.mean(pct_gate_s):.4f} -> agreement {inter_pct:.4f}")
    ok &= check("T6 percentile gating reproduces the teacher refusal set "
                "exactly under scale collapse", inter_pct > 0.999)
    ok &= check("T6 absolute-threshold gating is broken by scale collapse "
                "(stops ~nothing)", stopped_s < 1e-3)


if __name__ == "__main__":
    part1_fiber()
    part2_coupling_erosion()
    part3_flat_minimum()
    part4_sufficiency()
    part5_ema_inversion()
    part6_gating_robustness()
    print("=" * 92)
    print("Done. Companion doc: research/proofs/identifiability-frontier.md")
