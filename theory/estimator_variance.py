"""
VARIANCE of the MC-UBE local reward estimator -- and why it is NOT distribution-free.

Context
-------
`research/ESTIMATOR-BIAS-FINDING.md` establishes that the finite-M BIAS of
`u_hat = w_hat - g_hat` is `g* (2N-1)/(N M)`, and that it is *distribution-free*:
it touches the next-value law only through `sigma_i^2`.  The document draws the
conclusion that "multimodality -- the reason to use diffusion at all -- provides
no protection", and that the estimator gap "does not arise for Gaussian
ensembles, because there the member means are exact".

That is correct about the bias and incomplete about the estimator.  The bias is
removable in closed form at no sampling cost, so after the correction the only
thing left is VARIANCE -- and the variance is governed by the FOURTH moment of
the next-value law.  Multimodality is exactly what moves the fourth moment.

Result (derived here, verified numerically below)
------------------------------------------------
Write d_i = mu*_i - mu_bar*, and let sigma_i^2, mu_3i, mu_4i be the central
moments of Y_i = Qbar(s', a') under member i.  For the debiased estimator

    u_deb = w_hat - ((N-1)/N) g_ub / M - g_ub ,

    Var(u_deb) = A / M + O(1/M^2),

    A  =  (4/N^2) sum_i d_i^2 sigma_i^2          <- "epistemic x aleatoric"
        + (1/N^2) sum_i (mu_4i - sigma_i^4)      <- FOURTH moment / kurtosis
        - (4/N^2) sum_i d_i mu_3i .              <- skew cross term

Writing mu_4i = kappa_i sigma_i^4,

    A  =  (4/N^2) sum_i d_i^2 sigma_i^2  +  (1/N^2) sum_i (kappa_i - 1) sigma_i^4
          - (4/N^2) sum_i d_i mu_3i .

So:
  * The BIAS depends on the law only through sigma^2  (distribution-free).
  * The VARIANCE depends on it through kappa and mu_3 (NOT distribution-free).

Consequences, in the direction opposite to the bias story
--------------------------------------------------------
1. A rare-mode law -- one mode of probability p far from the bulk, the archetypal
   thing a Gaussian ensemble cannot represent and a diffusion model can -- has
   kappa ~ 1/p.  So the sample budget needed for an eps-accurate u_hat is
       M = Omega(kappa / eps^2) = Omega(1 / (p eps^2)).
   The model class you need for rare modes is precisely the one whose MC-UBE
   estimator is worst conditioned, and the penalty is the inverse rare-mode
   probability.  This is the missing "why the model class matters" bridge: the
   bias section argues multimodality does not help, and it does not -- but
   multimodality is what sets the cost.
2. A *symmetric* bimodal law with variance split evenly between separation and
   within-mode spread has kappa = 2.5 < 3, i.e. LESS variance than a Gaussian of
   matched sigma^2.  Multimodality is therefore not uniformly bad; only
   *asymmetric / rare-mode* multimodality is.  The repo's own bimodal control in
   `theory/estimator_bias.py` is the benign case, which is why the bias check
   passes and the variance question never surfaces.
3. Per-state sample ALLOCATION follows from A, not from w.  Minimising total
   error sum_s A_s / M_s subject to sum_s M_s = B is a Lagrange problem with the
   Neyman solution
       M_s  proportional to  sqrt(A_s) .
   A_s is driven by sigma^4 and kurtosis -- the ALEATORIC scale -- so the
   optimal rule refines high-`g` states, not high-`w` states.
   `udwm/uncertainty/adaptive_mc.py` refines on top-`w`.  Part 3 of
   `theory/estimator_bias.py` measures that top-`w` refinement is WORSE than
   uniform M at matched cost; this file gives the reason and the fix.

Run (from repo root)::

    python theory/estimator_variance.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

N_TRIALS = 60000


# ---------------------------------------------------------------------------
# Laws: all share (mu_i, sigma_i^2); they differ only in higher moments.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Law:
    name: str
    draw: Callable[[np.ndarray, np.ndarray, int, np.random.Generator, int], np.ndarray]
    # central moments given sigma^2, as multiples: (kappa, mu3/sigma^3)
    moments: Callable[[], Tuple[float, float]]


def _gaussian(mu, sig2, m, rng, trials):
    return mu[None, :, None] + np.sqrt(sig2)[None, :, None] * rng.standard_normal(
        (trials, mu.size, m)
    )


def _symmetric_bimodal(mu, sig2, m, rng, trials):
    """Half the variance from +/- separation, half within-mode.  kappa = 2.5."""
    half = sig2 / 2.0
    sep = np.sqrt(half)[None, :, None]
    sd = np.sqrt(half)[None, :, None]
    signs = rng.integers(0, 2, size=(trials, mu.size, m)) * 2.0 - 1.0
    return mu[None, :, None] + signs * sep + sd * rng.standard_normal((trials, mu.size, m))


def _rare_mode(p: float):
    """One rare mode of probability p carrying ALL the variance.  kappa ~ 1/p.

    Y = mu + a (B - p) / sqrt(p(1-p)) * sigma, B ~ Bernoulli(p), a = 1.
    Centred and scaled to variance sigma^2 exactly, so w*, g*, u* match the
    Gaussian law.  This is the shape a diffusion ensemble exists to fit and a
    Gaussian ensemble cannot: a low-probability, far-away branch.
    """

    def draw(mu, sig2, m, rng, trials):
        b = (rng.random((trials, mu.size, m)) < p).astype(np.float64)
        z = (b - p) / np.sqrt(p * (1.0 - p))
        return mu[None, :, None] + np.sqrt(sig2)[None, :, None] * z

    return draw


def _rare_mode_kappa(p: float) -> float:
    """Kurtosis of a centred, standardised Bernoulli(p)."""
    q = 1.0 - p
    # E[Z^4] for Z = (B-p)/sqrt(pq):  (p q^4 + q p^4) / (pq)^2 = (q^3 + p^3)/(pq)
    return (q**3 + p**3) / (p * q)


def _rare_mode_skew(p: float) -> float:
    q = 1.0 - p
    return (q - p) / np.sqrt(p * q)


LAWS = [
    Law("gaussian", _gaussian, lambda: (3.0, 0.0)),
    Law("bimodal-symmetric", _symmetric_bimodal, lambda: (2.5, 0.0)),
    Law("rare-mode p=0.10", _rare_mode(0.10),
        lambda: (_rare_mode_kappa(0.10), _rare_mode_skew(0.10))),
    Law("rare-mode p=0.02", _rare_mode(0.02),
        lambda: (_rare_mode_kappa(0.02), _rare_mode_skew(0.02))),
]


# ---------------------------------------------------------------------------
# Estimator (matches udwm/uncertainty/mc_ube.py combine(debias=True))
# ---------------------------------------------------------------------------

def u_debiased(y: np.ndarray) -> np.ndarray:
    """y: [trials, N, M] -> debiased u_hat per trial."""
    m, n = y.shape[-1], y.shape[-2]
    mu_hat = y.mean(axis=-1)
    w_raw = mu_hat.var(axis=-1, ddof=0)
    g_ub = y.var(axis=-1, ddof=1).mean(axis=-1)
    w_deb = w_raw - ((n - 1) / n) * g_ub / m
    return w_deb - g_ub


def predicted_A(d: np.ndarray, sig2: np.ndarray, kappa: float, skew: float) -> float:
    """Leading variance coefficient A with Var(u_deb) = A/M + O(1/M^2)."""
    n = d.size
    mu4 = kappa * sig2**2
    mu3 = skew * sig2**1.5
    return float(
        (4.0 / n**2) * np.sum(d**2 * sig2)
        + (1.0 / n**2) * np.sum(mu4 - sig2**2)
        - (4.0 / n**2) * np.sum(d * mu3)
    )


# ---------------------------------------------------------------------------
# Part 1 -- the variance coefficient A, measured vs predicted, per law
# ---------------------------------------------------------------------------

def part1_variance_is_not_distribution_free() -> None:
    rng = np.random.default_rng(3)
    mu = np.array([0.0, 0.5, 1.2, -0.3, 0.8])
    sig2 = np.array([0.25, 0.50, 0.10, 0.80, 0.35])
    d = mu - mu.mean()
    n = mu.size
    w_s = float((d**2).mean())
    g_s = float(sig2.mean())

    print("=" * 82)
    print("PART 1 -- Var(u_deb) = A/M, and A depends on the FOURTH moment")
    print("=" * 82)
    print(f"N={n}  w*={w_s:.4f}  g*={g_s:.4f}  u*={w_s - g_s:.4f}")
    print("All four laws share mu_i and sigma_i^2 exactly, so w*, g*, u* and the")
    print("finite-M BIAS are identical across rows.  Only the variance moves.")
    print()
    print(f"{'law':<22} {'kappa':>7} {'A pred':>10} {'A meas (M=16)':>14} "
          f"{'A meas (M=64)':>14} {'vs gaussian':>12}")

    a_gauss = None
    for law in LAWS:
        kappa, skew = law.moments()
        a_pred = predicted_A(d, sig2, kappa, skew)
        meas = {}
        for m in (16, 64):
            u = u_debiased(law.draw(mu, sig2, m, rng, N_TRIALS))
            meas[m] = float(u.var(ddof=1)) * m
        if a_gauss is None:
            a_gauss = a_pred
        print(f"{law.name:<22} {kappa:7.2f} {a_pred:10.4f} {meas[16]:14.4f} "
              f"{meas[64]:14.4f} {a_pred / a_gauss:11.2f}x")

    print()
    print("  -> A scales ~linearly in kappa.  A rare mode of probability p has")
    print("     kappa ~ 1/p, so Var(u_hat) ~ 1/(p M): the sample budget for an")
    print("     eps-accurate u_hat is M = Omega(1/(p eps^2)).")
    print("  -> the SYMMETRIC bimodal control (kappa=2.5) is BELOW gaussian.  It is")
    print("     the benign case, which is why theory/estimator_bias.py never sees")
    print("     this: that file varies the law but only checks the bias.")
    print()


# ---------------------------------------------------------------------------
# Part 2 -- required M vs rare-mode probability
# ---------------------------------------------------------------------------

def part2_sample_complexity() -> None:
    mu = np.array([0.0, 0.5, 1.2, -0.3, 0.8])
    sig2 = np.array([0.25, 0.50, 0.10, 0.80, 0.35])
    d = mu - mu.mean()
    w_s = float((d**2).mean())
    target = 0.10 * w_s  # want sd(u_hat) <= 10% of the epistemic signal

    print("=" * 82)
    print("PART 2 -- M needed for sd(u_hat) <= 10% of w*, by rare-mode probability")
    print("=" * 82)
    print(f"target sd = 0.10 * w* = {target:.4f}")
    print()
    print(f"{'law':<22} {'kappa':>8} {'A':>10} {'M required':>12}")
    rows = [("gaussian", 3.0, 0.0)]
    for p in (0.20, 0.10, 0.05, 0.02, 0.01):
        rows.append((f"rare-mode p={p:g}", _rare_mode_kappa(p), _rare_mode_skew(p)))
    for name, kappa, skew in rows:
        a = predicted_A(d, sig2, kappa, skew)
        m_req = a / target**2
        print(f"{name:<22} {kappa:8.2f} {a:10.4f} {m_req:12.0f}")
    print()
    print("  -> at p=0.01 the budget is ~2 orders of magnitude above the repo")
    print("     default of m_samples=8.  A Gaussian ensemble cannot represent this")
    print("     law at all, so the comparison is not 'diffusion is more expensive'")
    print("     but 'the only model class that can see the mode needs M ~ 1/p to")
    print("     score it'.  That is a statement about the estimand, not the sampler.")
    print()


# ---------------------------------------------------------------------------
# Part 3 -- allocation: Neyman (sqrt A) vs top-w refinement vs uniform
# ---------------------------------------------------------------------------

def make_state_bank(n_states: int, n_members: int, rng: np.random.Generator):
    """Epistemic and aleatoric levels drawn INDEPENDENTLY across states.

    Independence is the point: if high-w states were also high-g states, a
    top-w rule would accidentally be a Neyman rule.  They are not the same
    quantity, and the repo's own Part 3 result depends on that.
    """
    mus = rng.normal(0.0, 1.0, size=(n_states, n_members))
    epi = rng.uniform(0.05, 1.5, size=(n_states, 1))
    mus = (mus - mus.mean(axis=1, keepdims=True)) * epi
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


def draw_states(mus, sig2, m_per_state, rng, kappa_law=None):
    """Debiased u_hat per state at that state's own budget M_s."""
    s, n = mus.shape
    out = np.empty(s)
    for idx in range(s):
        m = int(max(2, m_per_state[idx]))
        if kappa_law is None:
            y = mus[idx][None, :, None] + np.sqrt(sig2[idx])[None, :, None] * \
                rng.standard_normal((1, n, m))
        else:
            y = kappa_law(mus[idx], sig2[idx], m, rng, 1)
        out[idx] = u_debiased(y)[0]
    return out


def allocate(scores: np.ndarray, budget_total: float, m_min: int = 2) -> np.ndarray:
    """M_s proportional to `scores`, floored at m_min, total budget preserved."""
    s = np.maximum(scores, 1e-12)
    free = budget_total - m_min * s.size
    if free <= 0:
        return np.full(s.size, m_min, dtype=float)
    return m_min + free * s / s.sum()


def part3_allocation() -> None:
    rng = np.random.default_rng(17)
    n_states, n_members = 600, 5
    mus, sig2, w_s, g_s, u_s = make_state_bank(n_states, n_members, rng)
    d = mus - mus.mean(axis=1, keepdims=True)

    # oracle A_s for the Gaussian law (kappa=3, skew=0)
    a_s = np.array([predicted_A(d[i], sig2[i], 3.0, 0.0) for i in range(n_states)])

    budget = 7.0 * n_states  # matched total sample budget, mean M = 7

    schemes: Dict[str, np.ndarray] = {}
    schemes["uniform M=7"] = np.full(n_states, 7.0)
    # repo rule: top-50% by w gets m_max, rest gets m_probe, matched to budget
    top = np.zeros(n_states)
    top[np.argsort(w_s)[-n_states // 2:]] = 1.0
    m_probe = 2.0
    m_max = (budget - m_probe * (n_states - top.sum())) / top.sum()
    schemes[f"top-w refine (2 / {m_max:.0f})"] = np.where(top > 0, m_max, m_probe)
    schemes["Neyman  M ~ sqrt(A)   [oracle]"] = allocate(np.sqrt(a_s), budget)
    schemes["proportional to g     [proxy]"] = allocate(np.sqrt(g_s), budget)
    schemes["proportional to w     [wrong]"] = allocate(np.sqrt(w_s), budget)

    print("=" * 82)
    print("PART 3 -- per-state sample allocation at MATCHED total budget")
    print("=" * 82)
    print(f"{n_states} states, N={n_members}, total budget = {budget:.0f} "
          f"samples/member (mean M = 7)")
    print("All rows use the DEBIASED estimator, so the 1/M bias is already gone;")
    print("what remains is purely an allocation-of-variance question.")
    print()
    print(f"{'scheme':<34} {'mean M':>7} {'spearman(u,u*)':>15} {'MAE':>9} {'RMSE':>9}")
    for name, m_alloc in schemes.items():
        # average several independent replications to stabilise the comparison
        rhos, maes, rmses = [], [], []
        for rep in range(5):
            est = draw_states(mus, sig2, m_alloc, np.random.default_rng(1000 + rep))
            rhos.append(spearman(est, u_s))
            maes.append(float(np.abs(est - u_s).mean()))
            rmses.append(float(np.sqrt(((est - u_s) ** 2).mean())))
        print(f"{name:<34} {m_alloc.mean():7.1f} {np.mean(rhos):15.4f} "
              f"{np.mean(maes):9.4f} {np.mean(rmses):9.4f}")
    print()
    corr_wg = spearman(w_s, g_s)
    corr_ag = spearman(a_s, g_s)
    corr_aw = spearman(a_s, w_s)
    print(f"  rank corr(w, g)  = {corr_wg:+.3f}   (epistemic vs aleatoric level)")
    print(f"  rank corr(A, g)  = {corr_ag:+.3f}   <- the variance coefficient tracks g")
    print(f"  rank corr(A, w)  = {corr_aw:+.3f}   <- and NOT w")
    print()
    print("  -> refining the states with the largest w is refining the wrong states.")
    print("     The estimator's error is set by A ~ sigma^4 (kurtosis) + d^2 sigma^2,")
    print("     which tracks the ALEATORIC level.  Neyman allocation M_s ~ sqrt(A_s)")
    print("     is the correct rule and is available from the same probe statistics.")
    print()


def main() -> None:
    part1_variance_is_not_distribution_free()
    part2_sample_complexity()
    part3_allocation()
    print("=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print("1. BIAS of u_hat is distribution-free (already established).")
    print("   VARIANCE of u_hat is NOT: Var(u_deb) = A/M with A carrying mu_4 and mu_3.")
    print("2. A rare mode of probability p gives kappa ~ 1/p, hence M = Omega(1/(p eps^2)).")
    print("   This is the 'why the model class matters' bridge the bias result lacks:")
    print("   the laws only a diffusion ensemble can represent are the laws whose")
    print("   epistemic score is most expensive to estimate.")
    print("3. The repo's symmetric-bimodal control has kappa=2.5 < 3, BELOW gaussian.")
    print("   It is the benign multimodal case and cannot detect this effect.")
    print("4. Optimal per-state budget is Neyman: M_s ~ sqrt(A_s), driven by the")
    print("   aleatoric scale.  adaptive_mc.py refines top-w, which is close to")
    print("   orthogonal to A -- explaining why it loses to uniform M at matched cost.")
    print("=" * 82)


if __name__ == "__main__":
    main()
