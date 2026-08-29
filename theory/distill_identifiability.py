"""
Why the decision-aware distillation loss can destroy the uncertainty it matches.

The claim
---------
`udwm/models/consistency.py::decision_preserving_distill_loss` matches the
teacher's and the student's cross-member value variance:

    t_variance = t_used.var(dim=0, unbiased=True)     # var over N members
    s_variance = s_used.var(dim=0, unbiased=True)
    value_variance = F.mse_loss(s_variance, t_variance)

Both variances are computed at **one shared diffusion latent per batch element**
(a single `t_idx` / `noise` draw, reused for every member).  That single-latent
cross-member variance is not the epistemic quantity `w`.  By the repo's own
coupling identity (`MCUBELocalRewards.combine_coupled`), for member values
Y_i = mu_i + eps_i with the eps_i correlated through the shared latent,

    E[ Var_i(Y_i) ]  =  w*  +  ( g* - Sigma_bar ),
    Sigma_bar = Var( mean_i eps_i ) = (1/N^2) sum_ij Cov(eps_i, eps_j),
    g* - Sigma_bar = (1/2N^2) sum_ij Var(eps_i - eps_j)  >= 0 .

So the matched statistic is `w* + (mean pairwise disagreement variance)` -- a
SUM of an epistemic term and a latent-conditional term.  One scalar equation,
two unknowns.  The student can satisfy it by trading epistemic disagreement
against latent-conditional disagreement at **exactly zero loss**.

That is a degenerate direction, and it is the one the online-critic study fell
down: `research/RESULTS-POLICY-SCALE-2026-08-22.md` reports next-state MSE
better on 10/10 seeds while teacher-student uncertainty rank correlation is
worse on 10/10.  A nonstationary critic makes the direction easier to walk, but
it is not what creates it -- the loss is unidentified even with a perfectly
fixed value map.  A lagged target critic therefore cannot be expected to close
it, which is a falsifiable prediction about `configs/lagged_target_distill.yaml`.

This is the ensemble/epistemic analogue of the variance-shrinkage pathology
proved for value-aware model losses under sampled models by Voelcker et al.,
"Calibrated Value-Aware Model Learning with Stochastic Environment Models"
(arXiv:2505.22772), Lemmata 1-3: an uncalibrated value-matching loss prefers
low-variance models.  There the collapsing variance is the aleatoric spread of
one stochastic model; here it is the epistemic spread across ensemble members.

The fix has the same shape as theirs: draw M >= 2 latents inside the loss and
match the DEBIASED w and g separately, which turns one equation into two.

Run (from repo root)::

    python theory/distill_identifiability.py
"""

from __future__ import annotations

import numpy as np

TRIALS = 200_000
N = 5


# ---------------------------------------------------------------------------
# A member-value model with two independently controllable knobs.
# ---------------------------------------------------------------------------

def draw_members(
    d: np.ndarray,        # [N] member mean offsets (epistemic); w = mean(d^2)
    sigma: float,         # per-member conditional sd (aleatoric scale)
    rho: float,           # cross-member correlation of the sampling error
    m: int,
    rng: np.random.Generator,
    trials: int = TRIALS,
) -> np.ndarray:
    """[trials, N, m] member values with Var(Y_i)=sigma^2, Corr(Y_i,Y_j)=rho.

    `rho` stands in for the shared-latent coupling: the DDIM map is deterministic
    given x_T, so reusing x_T across members correlates their sampling errors
    without touching any member's marginal law.
    """
    common = rng.standard_normal((trials, 1, m))
    idio = rng.standard_normal((trials, N, m))
    eps = sigma * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio)
    return d[None, :, None] + eps


def population(d: np.ndarray, sigma: float, rho: float):
    """(w*, g*, Sigma_bar, single-latent expectation under the code's ddof=1).

    Note a second, independent convention mismatch in the repo.  `w` in
    `MCUBELocalRewards.combine` is the finite-population average
    ``means.var(dim=0, unbiased=False)`` (ddof=0, matching Luis's
    ``w = (1/N) sum_i (mu_i - mu_bar)^2``), whereas both the distillation loss
    and the evaluation metrics use ``.var(dim=0, unbiased=True)`` (ddof=1).
    At N=5 that is a fixed 1.25x factor between the quantity the U-network is
    regressed on and the quantity the student is trained and scored against.
    It is a constant, so it cannot flip a rank correlation, but it does inflate
    every reported `w_rmse` by 25% and it means the two objects in the paper are
    not the same estimand.  `ddof1` below tracks the loss/metric convention.
    """
    w = float((d**2).mean())
    g = sigma**2
    # Cov matrix: sigma^2 on the diagonal, rho sigma^2 off-diagonal
    sigma_bar = (g + (N - 1) * rho * g) / N
    ddof0 = w + (g - sigma_bar)
    ddof1 = ddof0 * N / (N - 1)
    return w, g, sigma_bar, ddof1


# ---------------------------------------------------------------------------
# The two loss variants
# ---------------------------------------------------------------------------

def stat_single_latent(y: np.ndarray) -> float:
    """What the current loss matches: cross-member variance at ONE latent."""
    return float(y[:, :, 0].var(axis=1, ddof=1).mean())


def stat_debiased_pair(y: np.ndarray):
    """What the fix matches: (w_deb, g) from M >= 2 latents, coupling-aware.

    Mirrors MCUBELocalRewards.combine_coupled: estimate g and Sigma_bar from the
    same samples, so the correction needs no knowledge of the coupling.
    """
    m = y.shape[-1]
    mu = y.mean(axis=-1)                                  # [trials, N]
    w_raw = mu.var(axis=-1, ddof=0)
    z = y - mu[..., None]
    denom = max(m - 1, 1)
    g = (z**2).sum(axis=-1).mean(axis=-1) / denom
    sigma_bar = (z.mean(axis=-2) ** 2).sum(axis=-1) / denom
    w_deb = w_raw - (g - sigma_bar) / m
    return float(w_deb.mean()), float(g.mean())


# ---------------------------------------------------------------------------
# Part 1 -- the zero-loss manifold of the current objective
# ---------------------------------------------------------------------------

def part1_degenerate_direction() -> None:
    rng = np.random.default_rng(5)
    # Teacher: genuine epistemic disagreement, moderate coupling.
    d_t = np.array([-0.6, -0.2, 0.0, 0.3, 0.5])
    sigma_t, rho_t = 0.6, 0.3
    w_t, g_t, sb_t, target = population(d_t, sigma_t, rho_t)

    print("=" * 84)
    print("PART 1 -- the matched statistic is w* + (g* - Sigma_bar): one equation,")
    print("          two unknowns, so a whole family of students has ZERO loss.")
    print("=" * 84)
    print(f"teacher: w*={w_t:.4f}  g*={g_t:.4f}  Sigma_bar={sb_t:.4f}  "
          f"-> matched stat = {target:.4f}")
    print()
    print("Students below all reproduce the teacher's SINGLE-LATENT cross-member")
    print("variance to within Monte Carlo error, while their epistemic w* ranges")
    print("from a faithful copy down to ZERO.")
    print()
    print(f"{'student w*':>11} {'student g*':>11} {'rho':>6} "
          f"{'single-latent stat':>19} {'loss (M=1)':>12} {'w error':>10}")

    # Solve for the g that makes w_S + g_S(1 - (1+(N-1)rho)/N) equal `target`
    # at a chosen scale factor on the teacher's d.
    rho_s = 0.0
    shrink_factor = (1.0 - (1.0 + (N - 1) * rho_s) / N)
    target_ddof0 = target * (N - 1) / N
    rows = []
    for scale in (1.0, 0.7, 0.4, 0.0):
        d_s = d_t * scale
        w_s = float((d_s**2).mean())
        g_s = (target_ddof0 - w_s) / shrink_factor
        if g_s <= 0:
            continue
        sigma_s = float(np.sqrt(g_s))
        y = draw_members(d_s, sigma_s, rho_s, m=1, rng=rng)
        stat = stat_single_latent(y)
        rows.append((w_s, g_s, rho_s, stat, (stat - target) ** 2, w_s - w_t, d_s, sigma_s))
        print(f"{w_s:11.4f} {g_s:11.4f} {rho_s:6.2f} {stat:19.4f} "
              f"{(stat - target) ** 2:12.2e} {w_s - w_t:+10.4f}")

    print()
    print("  -> the M=1 loss is flat along this family: a student with w*=0 -- no")
    print("     epistemic disagreement whatsoever -- is indistinguishable from a")
    print("     faithful one.  It compensates entirely with latent-conditional")
    print("     spread, which is exactly 'improve next-state fit, destroy the")
    print("     uncertainty ranking'.")
    print()
    return rows, (d_t, sigma_t, rho_t, target)


# ---------------------------------------------------------------------------
# Part 2 -- the M >= 2 coupling-aware loss separates them
# ---------------------------------------------------------------------------

def part2_fix(rows, teacher) -> None:
    rng = np.random.default_rng(9)
    d_t, sigma_t, rho_t, target = teacher
    w_t, g_t, sb_t, _ = population(d_t, sigma_t, rho_t)

    # The one student that actually reproduces the teacher's (w, g, rho).  It is
    # also on the M=1 zero-loss set -- so that set contains BOTH the truth and
    # the collapse, which is the whole point.
    exact = (w_t, g_t, rho_t, None, None, 0.0, d_t, sigma_t)
    y_exact = draw_members(d_t, sigma_t, rho_t, m=1, rng=rng)
    l1_exact = (stat_single_latent(y_exact) - target) ** 2

    print("=" * 84)
    print("PART 2 -- M >= 2 with the coupling-aware debias turns 1 equation into 2")
    print("=" * 84)
    print("Row 1 is the TRUE match (same w, g and coupling as the teacher).  It is")
    print("on the M=1 zero-loss set too, alongside every degenerate row below it.")
    print()
    for m in (2, 8):
        y_t = draw_members(d_t, sigma_t, rho_t, m=m, rng=rng)
        w_ref, g_ref = stat_debiased_pair(y_t)
        print(f"M = {m}:  teacher (w_deb, g) = ({w_ref:.4f}, {g_ref:.4f})   "
              f"[truth ({w_t:.4f}, {g_t:.4f})]")
        print(f"{'':<14}{'student w*':>11} {'student g*':>11} {'w_deb':>9} "
              f"{'g_hat':>9} {'loss (w,g)':>12} {'loss (M=1)':>12}")
        table = [("TRUE match  ", exact, l1_exact)] + [
            ("degenerate  ", r, r[4]) for r in rows
        ]
        for label, r, l1 in table:
            w_s, g_s, rho_s, _s, _l, _dw, d_s, sigma_s = r
            y_s = draw_members(d_s, sigma_s, rho_s, m=m, rng=rng)
            w_hat, g_hat = stat_debiased_pair(y_s)
            loss2 = (w_hat - w_ref) ** 2 + (g_hat - g_ref) ** 2
            print(f"{label:<14}{w_s:11.4f} {g_s:11.4f} {w_hat:9.4f} {g_hat:9.4f} "
                  f"{loss2:12.4f} {l1:12.2e}")
        print()
    print("  -> the M=1 column is ~1e-6 for every row: the truth and a student")
    print("     with ZERO epistemic disagreement are equally optimal.")
    print("  -> the (w, g) column separates them by orders of magnitude and puts")
    print("     the true match at the minimum.  Cost: M-1 extra latents per")
    print("     distillation minibatch; the correction needs no knowledge of the")
    print("     coupling because g and Sigma_bar come from the same samples.")
    print()


# ---------------------------------------------------------------------------
# Part 3 -- the degeneracy exists with a PERFECTLY FIXED value map
# ---------------------------------------------------------------------------

def part3_not_a_critic_problem() -> None:
    print("=" * 84)
    print("PART 3 -- what this predicts about the repo's own next experiment")
    print("=" * 84)
    print("Nothing in Parts 1-2 involves a critic, a policy, or nonstationarity.")
    print("The value map is a fixed deterministic function throughout.  So:")
    print()
    print("  * The 20-seed FIXED-value-map study (RESULTS-STRESS-LARGE) does not")
    print("    escape the degeneracy either -- it is only that with a well-behaved")
    print("    fixed map and a small student, gradient descent happens to land")
    print("    near the faithful branch.  Note the reported top-decile recall was")
    print("    already INCONCLUSIVE (9/20): that is the ranking coordinate, and it")
    print("    is precisely the coordinate the degenerate direction moves.")
    print()
    print("  * PREDICTION: lagged target-critic distillation")
    print("    (configs/lagged_target_distill.yaml) will reduce but NOT eliminate")
    print("    the uncertainty-rank collapse, because it fixes nonstationarity and")
    print("    leaves identifiability untouched.  If the 10-seed lagged table comes")
    print("    back with rank correlation still losing on most seeds, that is not a")
    print("    failure of the correction -- it is this prediction confirming.")
    print()
    print("  * PREDICTION: adding M >= 2 latents plus separate (w_deb, g) matching")
    print("    recovers rank correlation EVEN WITH THE LIVE CRITIC, because it")
    print("    removes the direction rather than slowing the drift along it.")
    print("    That is the decisive A/B: {live-Q, lagged-Q} x {M=1, M>=2 debiased}.")
    print()
    print("  * The two mechanisms are separable and the 2x2 identifies them.  If")
    print("    M>=2 fixes it under the live critic, the paper's thesis becomes an")
    print("    identifiability result, which is stronger and more portable than")
    print("    'do not use an online critic'.")
    print()


def main() -> None:
    rows, teacher = part1_degenerate_direction()
    part2_fix(rows, teacher)
    part3_not_a_critic_problem()
    print("=" * 84)
    print("SUMMARY")
    print("=" * 84)
    print("1. The distillation loss matches w* + (g* - Sigma_bar) at M=1, which is")
    print("   ONE equation in TWO unknowns -- the objective is unidentified.")
    print("2. A student with zero epistemic disagreement sits on the zero-loss set.")
    print("3. Drawing M >= 2 latents and matching (w_deb, g) separately removes it.")
    print("4. The degeneracy is independent of the critic, so the online-critic")
    print("   collapse is a symptom, not the cause, and the lagged-critic fix is")
    print("   predicted to be partial.")
    print("=" * 84)


if __name__ == "__main__":
    main()
