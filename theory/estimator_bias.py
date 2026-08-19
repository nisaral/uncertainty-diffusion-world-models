"""
Finite-sample BIAS of the MC-UBE local reward estimator (and how to remove it).

Motivation
----------
`udwm/uncertainty/mc_ube.py` estimates the Luis et al. (2023) local UBE rewards as

    mu_i  = mean_m  Q(s'_{i,m}, a'_{i,m})        # M samples from member i
    v_i   = var_m   Q(...)          (ddof=0)
    w_hat = var_i(mu_i)             (ddof=0)     # epistemic
    g_hat = mean_i(v_i)                          # mean aleatoric
    u_hat = w_hat - g_hat

`w_hat` is a variance of *noisy* member means, so it inherits the Monte Carlo
noise of each mean.  That makes `u_hat` biased at order 1/M -- and the bias is
proportional to the aleatoric term `g`, i.e. to exactly the quantity Luis's
`u = w - g` correction was introduced to remove.  Averaging over training steps
does NOT remove a bias.

Exact result (derived below, verified numerically here)
------------------------------------------------------
Let mu*_i = E[Q | theta_i], sigma_i^2 = Var[Q | theta_i], and

    w* = (1/N) sum_i (mu*_i - mu_bar*)^2 ,   g* = (1/N) sum_i sigma_i^2 ,
    u* = w* - g* .

With M i.i.d. samples per member and the ddof=0 conventions above:

    E[w_hat] = w* + ((N-1)/N) * g*/M
    E[g_hat] = g* * (M-1)/M            =  g* - g*/M
    E[u_hat] = u* + g* * (2N-1)/(N*M)

So the naive estimator OVERSTATES local epistemic uncertainty by ~2g*/M.
The bias is distribution-free: it depends on the next-value law only through
sigma_i^2, never on its shape.  This is verified here for a unimodal Gaussian
and for a bimodal mixture (the multimodal case diffusion models exist to fit).

Bias-corrected estimator (exactly unbiased for u*)
--------------------------------------------------
    g_ub  = (1/N) sum_i v_i * M/(M-1)              # Bessel-corrected
    w_deb = w_hat - ((N-1)/N) * g_ub / M           # remove MC noise of the means
    u_deb = w_deb - g_ub

Why it matters for this repo
----------------------------
1. Default `m_samples=8`, N=5  ->  bias = 0.225 * g*.  Since u* = w* - g* is a
   *difference*, the bias is routinely the same order as the signal (or larger).
2. `AdaptiveMCUBELocalRewards` spends *different* M on different states
   (m_probe=2 for most, m_max for high-w ones).  Because the bias scales as
   1/M, refined states get LESS inflation than unrefined ones -- so the adaptive
   scheme systematically COMPRESSES the very uncertainty ranking it is meant to
   sharpen.  Part 3 measures that.

Run (from repo root)::

    python theory/estimator_bias.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

N_TRIALS = 20000


# ---------------------------------------------------------------------------
# Ensemble spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ensemble:
    """Population next-value moments per member: mu*_i and sigma_i^2."""

    mu: np.ndarray      # [N]
    sigma2: np.ndarray  # [N]

    @property
    def n(self) -> int:
        return int(self.mu.shape[0])

    def population(self) -> Tuple[float, float, float]:
        w = float(((self.mu - self.mu.mean()) ** 2).mean())
        g = float(self.sigma2.mean())
        return w, g, w - g


def predicted_bias(n: int, g: float, m: int) -> Tuple[float, float, float]:
    """Analytic bias of (w_hat, g_hat, u_hat) at sample size m."""
    bias_w = ((n - 1) / n) * g / m
    bias_g = -g / m
    bias_u = g * (2 * n - 1) / (n * m)
    return bias_w, bias_g, bias_u


# ---------------------------------------------------------------------------
# Samplers for the next-value law Q(s',a') under member i
# ---------------------------------------------------------------------------

def draw_gaussian(ens: Ensemble, m: int, rng: np.random.Generator, trials: int) -> np.ndarray:
    """[trials, N, m] samples of Q with mean mu_i and variance sigma_i^2."""
    sd = np.sqrt(ens.sigma2)[None, :, None]
    return ens.mu[None, :, None] + sd * rng.standard_normal((trials, ens.n, m))


def draw_bimodal(ens: Ensemble, m: int, rng: np.random.Generator, trials: int) -> np.ndarray:
    """Two-component mixture with the SAME mu_i and sigma_i^2 as `draw_gaussian`.

    Stand-in for a multimodal diffusion next-value law: half the variance comes
    from mode separation, half from within-mode spread.  If the bias formula is
    distribution-free, this must reproduce the Gaussian numbers.
    """
    half = ens.sigma2 / 2.0
    sep = np.sqrt(half)[None, :, None]          # mode offset  +/- sep
    sd = np.sqrt(half)[None, :, None]           # within-mode sd
    signs = rng.integers(0, 2, size=(trials, ens.n, m)) * 2.0 - 1.0
    return ens.mu[None, :, None] + signs * sep + sd * rng.standard_normal((trials, ens.n, m))


# ---------------------------------------------------------------------------
# Estimators (naive == what mc_ube.py computes today)
# ---------------------------------------------------------------------------

def estimate(y: np.ndarray) -> dict:
    """y: [trials, N, m] next-value samples -> naive and debiased estimates."""
    m = y.shape[-1]
    n = y.shape[-2]
    mu_hat = y.mean(axis=-1)                                  # [trials, N]
    v_hat = y.var(axis=-1, ddof=0)                            # [trials, N]

    w_naive = mu_hat.var(axis=-1, ddof=0)                     # [trials]
    g_naive = v_hat.mean(axis=-1)
    u_naive = w_naive - g_naive

    if m > 1:
        g_ub = v_hat.mean(axis=-1) * m / (m - 1)
        w_deb = w_naive - ((n - 1) / n) * g_ub / m
    else:  # g is unidentifiable from a single sample per member
        g_ub = np.full_like(g_naive, np.nan)
        w_deb = np.full_like(w_naive, np.nan)
    u_deb = w_deb - g_ub

    return {
        "w_naive": w_naive, "g_naive": g_naive, "u_naive": u_naive,
        "w_deb": w_deb, "g_deb": g_ub, "u_deb": u_deb,
    }


# ---------------------------------------------------------------------------
# Part 1 — the bias formula is exact, and distribution-free
# ---------------------------------------------------------------------------

def part1_verify_bias() -> None:
    rng = np.random.default_rng(0)
    ens = Ensemble(
        mu=np.array([0.0, 0.5, 1.2, -0.3, 0.8]),
        sigma2=np.array([0.25, 0.50, 0.10, 0.80, 0.35]),
    )
    w_s, g_s, u_s = ens.population()

    print("=" * 78)
    print("PART 1 — finite-M bias of the MC-UBE local rewards")
    print("=" * 78)
    print(f"N = {ens.n}   w* = {w_s:.6f}   g* = {g_s:.6f}   u* = w*-g* = {u_s:.6f}")
    print(f"(u* < 0 here: aleatoric spread exceeds member disagreement -> clamped to u_min)")
    print()

    for law, draw in (("gaussian", draw_gaussian), ("bimodal", draw_bimodal)):
        print(f"next-value law = {law}")
        print(f"{'M':>4}  {'bias_u measured':>16}  {'bias_u predicted':>17}  "
              f"{'bias_u debiased':>16}  {'|u*|':>8}")
        for m in (2, 4, 8, 16, 32, 64):
            y = draw(ens, m, rng, N_TRIALS)
            est = estimate(y)
            meas = float(est["u_naive"].mean()) - u_s
            pred = predicted_bias(ens.n, g_s, m)[2]
            deb = float(est["u_deb"].mean()) - u_s
            print(f"{m:4d}  {meas:16.6f}  {pred:17.6f}  {deb:16.6f}  {abs(u_s):8.4f}")
        print()

    print("  -> measured == predicted to MC precision, for BOTH laws.")
    print("  -> the bias depends on the next-value law only through sigma_i^2,")
    print("     so multimodality (the reason to use diffusion) does not save you.")
    print("  -> debiased column is ~0 at every M, including M=2.")
    print()


# ---------------------------------------------------------------------------
# Part 2 — bias vs signal at the M values this repo actually uses
# ---------------------------------------------------------------------------

def part2_signal_to_bias() -> None:
    print("=" * 78)
    print("PART 2 — bias relative to signal at repo default budgets")
    print("=" * 78)
    n = 5
    print("bias_u / g*  =  (2N-1)/(N*M)   [fraction of the aleatoric term added to u]")
    print()
    print(f"{'M':>4}  {'bias_u / g*':>12}   note")
    notes = {
        1: "adaptive m_min floor (g unidentifiable, no correction possible)",
        2: "adaptive_mc m_probe default -> most of the batch",
        8: "ube.m_samples default (mc_ube.MCUBELocalRewards)",
        12: "adaptive_mc m_max default -> refined states only",
    }
    for m in (1, 2, 4, 8, 12, 32, 128):
        frac = (2 * n - 1) / (n * m)
        print(f"{m:4d}  {frac:12.4f}   {notes.get(m, '')}")
    print()
    print("  -> at M=8 the estimator adds 22.5% of g* to u*.  Because u* = w* - g*")
    print("     is a DIFFERENCE of two comparable quantities, that is routinely as")
    print("     large as u* itself: the reported 'epistemic' signal is substantially")
    print("     re-labelled aleatoric noise.")
    print()


# ---------------------------------------------------------------------------
# Part 3 — adaptive M distorts the uncertainty RANKING across states
# ---------------------------------------------------------------------------

def make_state_bank(n_states: int, n_members: int, rng: np.random.Generator):
    """States with heterogeneous epistemic disagreement and aleatoric noise."""
    mus = rng.normal(0.0, 1.0, size=(n_states, n_members))
    # spread member means by a per-state epistemic scale
    epi = rng.uniform(0.05, 1.5, size=(n_states, 1))
    mus = (mus - mus.mean(axis=1, keepdims=True)) * epi
    # per-state aleatoric level, independent of the epistemic level
    ale = rng.uniform(0.1, 1.2, size=(n_states, 1))
    sig2 = ale * rng.uniform(0.5, 1.5, size=(n_states, n_members))
    w = ((mus - mus.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    g = sig2.mean(axis=1)
    return mus, sig2, w, g, w - g


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))


def estimate_per_state(mus, sig2, m_per_state, rng):
    """Draw M_s samples for each state (heterogeneous M) -> naive + debiased u."""
    s, n = mus.shape
    u_naive = np.empty(s)
    u_deb = np.empty(s)
    w_probe = np.empty(s)
    for idx in range(s):
        m = int(m_per_state[idx])
        y = mus[idx][None, :, None] + np.sqrt(sig2[idx])[None, :, None] * rng.standard_normal(
            (1, n, m)
        )
        est = estimate(y)
        u_naive[idx] = est["u_naive"][0]
        u_deb[idx] = est["u_deb"][0]
        w_probe[idx] = est["w_naive"][0]
    return u_naive, u_deb, w_probe


def part3_adaptive_ranking() -> None:
    rng = np.random.default_rng(7)
    n_states, n_members = 400, 5
    mus, sig2, w_s, g_s, u_s = make_state_bank(n_states, n_members, rng)

    m_probe, m_max, refine_frac = 2, 12, 0.5
    k = int(refine_frac * n_states)

    # --- probe pass: every state at m_probe.  These estimates are what the
    #     implementation KEEPS for states it decides not to refine.
    u_pr_naive, u_pr_deb, w_pr = estimate_per_state(
        mus, sig2, np.full(n_states, m_probe), rng
    )

    # selection on the NOISY probe w -- what adaptive_mc.estimate actually does
    refine = np.zeros(n_states, dtype=bool)
    refine[np.argsort(w_pr)[-k:]] = True
    # selection on the TRUE w -- counterfactual that isolates selection noise
    oracle = np.zeros(n_states, dtype=bool)
    oracle[np.argsort(w_s)[-k:]] = True

    m_adapt = np.where(refine, m_max, m_probe)

    def adaptive_with(mask):
        """Refined states are redrawn at m_max; unrefined keep the probe estimate.

        Matching the implementation here is essential: _member_stats issues a
        fresh independent draw for the refined subset and the probe estimate
        survives untouched elsewhere.  Reusing an estimate that was also used to
        SELECT introduces a winner's-curse bias on top of the 1/M bias, and the
        two are separate failure modes -- debiasing removes only the second.
        """
        u_n, u_d = u_pr_naive.copy(), u_pr_deb.copy()
        idx = np.nonzero(mask)[0]
        r_n, r_d, _ = estimate_per_state(mus[idx], sig2[idx], np.full(idx.size, m_max), rng)
        u_n[idx], u_d[idx] = r_n, r_d
        return u_n, u_d

    u_ad_naive, u_ad_deb = adaptive_with(refine)
    u_or_naive, u_or_deb = adaptive_with(oracle)

    # --- uniform baselines at comparable and at max cost
    m_equal = int(round(m_adapt.mean()))
    u_eq_naive, u_eq_deb, _ = estimate_per_state(mus, sig2, np.full(n_states, m_equal), rng)
    u_mx_naive, u_mx_deb, _ = estimate_per_state(mus, sig2, np.full(n_states, m_max), rng)

    print("=" * 78)
    print("PART 3 — does adaptive M help or hurt the uncertainty ranking?")
    print("=" * 78)
    print(f"{n_states} states, N={n_members}, m_probe={m_probe}, m_max={m_max}, "
          f"refine_frac={refine_frac}")
    print(f"mean samples/state: adaptive={m_adapt.mean():.1f}  uniform={m_equal}  max={m_max}")
    print()
    print(f"{'scheme':<34} {'mean M':>7} {'spearman(u_hat,u*)':>20} {'MAE(u_hat,u*)':>15}")
    rows = [
        ("uniform M, naive", m_equal, u_eq_naive),
        ("adaptive M, naive  <-- repo", m_adapt.mean(), u_ad_naive),
        ("uniform M=m_max, naive", m_max, u_mx_naive),
        ("uniform M, debiased", m_equal, u_eq_deb),
        ("adaptive M, debiased", m_adapt.mean(), u_ad_deb),
        ("adaptive M, debiased, ORACLE sel", m_adapt.mean(), u_or_deb),
        ("uniform M=m_max, debiased", m_max, u_mx_deb),
    ]
    for name, mm, est in rows:
        rho = spearman(est, u_s)
        mae = float(np.abs(est - u_s).mean())
        print(f"{name:<34} {mm:7.1f} {rho:20.4f} {mae:15.4f}")
    print()

    # --- mechanism 1: the 1/M bias differs across the refine boundary
    b_lo = predicted_bias(n_members, float(g_s[~refine].mean()), m_probe)[2]
    b_hi = predicted_bias(n_members, float(g_s[refine].mean()), m_max)[2]
    print("  mechanism 1 -- heterogeneous 1/M bias (removable):")
    print(f"    bias added to unrefined states (M={m_probe}):  {b_lo:+.4f}")
    print(f"    bias added to refined   states (M={m_max}): {b_hi:+.4f}")
    print(f"    differential (refined shifted vs unrefined):  {b_hi - b_lo:+.4f}")
    print("    -> refined states get a SMALLER inflation than the ones skipped,")
    print("       pushing high-uncertainty states DOWN relative to low ones.")
    print("       Debiasing per-state at the M actually spent removes this exactly.")
    print()

    # --- mechanism 2: selection on a noisy statistic (NOT removable by debiasing)
    d_probe = spearman(u_ad_deb, u_s)
    d_oracle = spearman(u_or_deb, u_s)
    print("  mechanism 2 -- winner's curse from selecting on the noisy probe w:")
    print(f"    spearman, debiased + probe-w selection:  {d_probe:.4f}")
    print(f"    spearman, debiased + true-w  selection:  {d_oracle:.4f}")
    print(f"    gap attributable to selection noise:     {d_oracle - d_probe:+.4f}")
    print("    -> unrefined states keep the very estimate that selected them, so")
    print("       their u is conditioned on a low w draw.  Debiasing corrects the")
    print("       mean at a GIVEN M; it cannot undo conditioning on the draw.")
    print("       Fixing this needs a sample-splitting probe (select on one half,")
    print("       report from the other) -- not yet implemented.")
    print()


# ---------------------------------------------------------------------------
# Part 4 — downstream effect on the UBE fixed point
# ---------------------------------------------------------------------------

def part4_fixed_point() -> None:
    print("=" * 78)
    print("PART 4 — propagation to U through the UBE fixed point")
    print("=" * 78)
    gamma = 0.99
    n, g = 5, 0.5
    print(f"gamma={gamma}, N={n}, g*={g}")
    print("A uniform local-reward error eps propagates as ||U~ - U*||_inf <= "
          "gamma^2/(1-gamma^2) * eps")
    amp = gamma**2 / (1 - gamma**2)
    print(f"amplification factor gamma^2/(1-gamma^2) = {amp:.2f}")
    print()
    print(f"{'M':>4}  {'local bias':>11}  {'U inflation (upper bd)':>23}")
    for m in (2, 4, 8, 16, 32):
        eps = predicted_bias(n, g, m)[2]
        print(f"{m:4d}  {eps:11.5f}  {amp * eps:23.4f}")
    print()
    print("  -> the bias is systematic and same-signed everywhere, so it does not")
    print("     cancel in the Bellman recursion; at gamma=0.99 it is amplified ~50x.")
    print("  -> a constant offset in U is harmless for ranking but NOT for the")
    print("     Q +/- lambda*sqrt(U) objective, whose scale is then M-dependent:")
    print("     re-tuning lambda after changing M is chasing this bias.")
    print()


# ---------------------------------------------------------------------------
# Part 5 — coupling: the bias is the mean pairwise DISAGREEMENT variance
# ---------------------------------------------------------------------------

def draw_correlated(ens: Ensemble, m: int, rng, trials: int, rho: float) -> np.ndarray:
    """[trials, N, m] draws with Var(Y_i)=sigma_i^2 and Corr(Y_i,Y_j)=rho, i!=j.

    Stands in for sampling every ensemble member from a SHARED DDIM latent: each
    member's marginal law is untouched (so w*, g*, u* are unchanged) while the
    members' sampling errors become correlated.
    """
    common = rng.standard_normal((trials, 1, m))
    idio = rng.standard_normal((trials, ens.n, m))
    sd = np.sqrt(ens.sigma2)[None, :, None]
    return ens.mu[None, :, None] + sd * (
        np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    )


def coupled_estimate(y: np.ndarray) -> dict:
    """Covariance-aware estimator: valid for dependent members, any coupling."""
    m, n = y.shape[-1], y.shape[-2]
    mu_hat = y.mean(axis=-1)                                  # [trials, N]
    w_raw = mu_hat.var(axis=-1, ddof=0)                       # [trials]
    z = y - mu_hat[..., None]                                 # centred
    denom = max(m - 1, 1)
    g = (z**2).sum(axis=-1).mean(axis=-1) / denom             # mean of diag(C)
    sigma_bar = (z.mean(axis=-2) ** 2).sum(axis=-1) / denom   # mean of ALL of C
    w_deb = w_raw - (g - sigma_bar) / m if m > 1 else np.full_like(w_raw, np.nan)
    return {"w_raw": w_raw, "w_deb": w_deb, "g": g,
            "sigma_bar": sigma_bar, "u_deb": w_deb - g}


def part5_coupling() -> None:
    rng = np.random.default_rng(11)
    ens = Ensemble(
        mu=np.array([0.0, 0.5, 1.2, -0.3, 0.8]),
        sigma2=np.array([0.25, 0.50, 0.10, 0.80, 0.35]),
    )
    w_s, g_s, u_s = ens.population()
    m = 8

    print("=" * 78)
    print("PART 5 — coupling: bias = mean pairwise DISAGREEMENT variance / M")
    print("=" * 78)
    print(f"N={ens.n}, M={m}, w*={w_s:.4f}, g*={g_s:.4f}, u*={u_s:.4f}")
    print()
    print("Allowing dependence across members:")
    print("    E[w_hat] = w* + (g* - mean(Sigma))/M,   mean(Sigma) = Var(mean_i Y_i)")
    print("    g* - mean(Sigma) = (1/2N^2) sum_ij Var(Y_i - Y_j)   [identity]")
    print("rho=0 recovers the independent case (N-1)/N * g*/M; rho=1 gives zero bias.")
    print()
    header = (f"{'rho':>5}  {'naive bias':>11}  {'predicted':>10}  "
              f"{'indep-corr bias':>16}  {'coupled bias':>13}  {'sd(w_deb)':>10}")
    print(header)
    for rho in (0.0, 0.3, 0.6, 0.9, 0.99):
        y = draw_correlated(ens, m, rng, N_TRIALS, rho)
        # population mean(Sigma) for this rho
        sd = np.sqrt(ens.sigma2)
        cov = rho * np.outer(sd, sd)
        np.fill_diagonal(cov, ens.sigma2)
        sigma_bar_pop = cov.mean()
        pred = (g_s - sigma_bar_pop) / m

        naive = estimate(y)
        coup = coupled_estimate(y)
        b_naive = float(naive["w_naive"].mean()) - w_s
        b_indep = float(naive["w_deb"].mean()) - w_s   # wrong correction under coupling
        b_coup = float(coup["w_deb"].mean()) - w_s
        print(f"{rho:5.2f}  {b_naive:11.6f}  {pred:10.6f}  {b_indep:16.6f}  "
              f"{b_coup:13.6f}  {float(coup['w_deb'].std()):10.6f}")
    print()
    print("  -> column 2 vs 3: the naive bias tracks (g*-mean(Sigma))/M, not g*/M.")
    print("  -> 'indep-corr' over-subtracts as rho grows (goes negative): the")
    print("     independent-sampling correction is INVALID under a coupling.")
    print("  -> 'coupled' stays ~0 at every rho, with no knowledge of rho.")
    print("  -> sd(w_deb) falls with rho: the variance reduction is the practical")
    print("     prize, since it is what allows M to be cut.")
    print()


def main() -> None:
    part1_verify_bias()
    part2_signal_to_bias()
    part3_adaptive_ranking()
    part4_fixed_point()
    part5_coupling()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("1. u_hat as implemented is biased by +g*(2N-1)/(N*M): O(1/M), not O(1/sqrt M).")
    print("   Theorem A1's O(1/sqrt M) concentration bound is loose enough to hide it.")
    print("2. The bias is distribution-free -> multimodal diffusion dynamics do not help.")
    print("3. Adaptive per-state M turns a constant bias into a state-dependent one")
    print("   that anti-correlates with the refinement decision.")
    print("4. A closed-form correction makes u_hat exactly unbiased at any M >= 2.")
    print("5. Adaptive M has a SECOND, separate defect: it selects on a noisy probe")
    print("   w and then keeps that same estimate for unrefined states. Debiasing")
    print("   does not fix this; sample-splitting the probe would.")
    print("6. The bias is a property of INDEPENDENT Monte Carlo, not of Monte Carlo.")
    print("   It equals the mean pairwise disagreement variance / M, so coupling the")
    print("   members through a shared DDIM latent shrinks both it and the variance.")
    print("   That coupling is available for free in an implicit sampler and is")
    print("   unnecessary for a Gaussian ensemble -> the interesting asymmetry.")
    print("=" * 78)


if __name__ == "__main__":
    main()
