"""Numerical experiments.  All functions return plain dicts; no plotting and
no file I/O happens here, so results can be checked before any figure is drawn.

Random-number generators are seeded exactly as in the original scripts.  The
call order is part of the reproducibility contract: do not reorder the loops.
"""
import time
import numpy as np
from scipy.stats import ttest_ind

from .config import SEED, N, T, NTRAJ, RS, D, ETA_LR
from .simulation import simulate, simulate_fast, diagnostics, per_traj_err


def run_uniform():
    """Uniform availability: bias, asymptotic variance, and linear-recursion error."""
    p = np.ones(N) / N
    rng0 = np.random.default_rng(SEED)
    A = rng0.normal(size=(N, D))
    wstar = A.mean(axis=0); w0 = np.ones(D) * 3.0
    out = {}
    for R in RS:
        rng = np.random.default_rng(SEED + 1000 * R)
        X = simulate(N, R, p, T, NTRAJ, rng)
        bias, tvar, sgd = diagnostics(X, p, A, wstar, w0, ETA_LR, N, T, NTRAJ, D)
        pred = (N - R) * (N - R - 1) / N ** 2
        out[R] = dict(bias=bias.tolist(), tvar=tvar.tolist(), sgd=sgd.tolist(),
                      pred=pred, tvar_end=float(tvar[-1]), sgd_end=float(sgd[-1]),
                      sgd_t11=float(sgd[10]), sgd_t20=float(sgd[19]),
                      sgd_t40=float(sgd[39]))
        print("R=%2d pred=%.5f t*Var=%.5f  err t11=%.4f t20=%.4f t40=%.4f T=%.6f"
              % (R, pred, tvar[-1], sgd[10], sgd[19], sgd[39], sgd[-1]))
    return out


def run_counterexample():
    """Skewed-propensity instance: absolute variance increases from R=0 to R=1."""
    Nf, Tf, Ntr = 5, 3000, 12000
    pf = np.array([1/100, 24/25, 1/100, 1/100, 1/100]); pf /= pf.sum()
    Z = 1 - (pf ** 2).sum()
    pi0 = {0: pf[0], 1: pf[0]*(1-pf[0])/Z}
    th = {0: pf[0]*(1-pf[0]), 1: 821931/8788000}
    out = {}
    for R in (0, 1):
        rng = np.random.default_rng(SEED + 77 + 100 * R)
        X = simulate(Nf, R, pf, Tf, Ntr, rng)
        ind = (X == 0).astype(float)
        cs = np.cumsum(ind, axis=1)
        ts = np.arange(1, Tf+1, dtype=float)
        v = ((cs - ts[None, :]*pi0[R])**2 / ts[None, :]**2).mean(axis=0)
        tv = ts * v
        out[R] = dict(tvar=tv.tolist(), theory=th[R], end=float(tv[-1]))
        print("counterexample R=%d theory=%.8f t*Var(3000)=%.8f" % (R, th[R], tv[-1]))
    return out


def run_small_rho():
    """Small rho=R/N regime at N=100: closed-form ratio vs. measured steady error."""
    Ns, Ts, Ntr = 100, 1200, 2000
    Rb = [0, 1, 2, 5, 10]
    p = np.ones(Ns)/Ns
    rng0 = np.random.default_rng(SEED + 555)
    A = rng0.normal(size=(Ns, D)); ws = A.mean(0); w0 = np.ones(D)*3.0
    out = {}
    for R in Rb:
        rng = np.random.default_rng(SEED + 7000 + 100*R)
        X = simulate(Ns, R, p, Ts, Ntr, rng)
        _, tv, sg = diagnostics(X, p, A, ws, w0, ETA_LR, Ns, Ts, Ntr, D)
        pred = (Ns-R)*(Ns-R-1)/Ns**2
        out[R] = dict(rho=R/Ns, pred=pred, tvar_end=float(tv[-1]),
                      ratio_var=float(pred/((Ns-1)/Ns)),
                      sgd_end=float(sg[-1]), sgd_ratio=float(sg[-1]))
        print("small-rho N=%d R=%2d rho=%.3f  var-ratio=%.6f  steady=%.8f"
              % (Ns, R, R/Ns, out[R]["ratio_var"], sg[-1]))
    base = out[0]["sgd_end"]
    for R in Rb:
        out[R]["sgd_ratio"] = out[R]["sgd_end"]/base
    print("  steady-state ratio to R=0:", {R: round(out[R]["sgd_ratio"], 4) for R in Rb})
    return out


def errorbar_statistics():
    """Error bars at four rounds, Welch tests, 10 independent data instances,
    and the full error curves used by the Fig. 2 plot.

    Returns ``(summary, curves)``; ``summary`` is the JSON-serialisable result.
    """
    t0 = time.time()
    rng0 = np.random.default_rng(SEED)
    A = rng0.normal(size=(N, D)); w0 = np.ones(D) * 3.0
    p = np.ones(N) / N
    times = [11, 20, 40, 1200]
    main = {}
    for R in RS:
        main[R] = per_traj_err(A, w0, R, p, SEED + 1000 * R, times,
                               NTRAJ, T, simulate, ETA_LR)
    stats = {}
    for t in times:
        stats[t] = {}
        for R in RS:
            e = main[R][t]
            stats[t][R] = dict(mean=float(e.mean()), sd=float(e.std(ddof=1)),
                               sem=float(e.std(ddof=1) / np.sqrt(len(e))))
    welch = {}
    for t in times:
        e0, e16 = main[0][t], main[16][t]
        diff = e16.mean() - e0.mean()
        se = np.sqrt(stats[t][0]["sem"] ** 2 + stats[t][16]["sem"] ** 2)
        _, pv = ttest_ind(e0, e16, equal_var=False)
        welch[t] = dict(diff=float(diff), pct=float(100 * diff / e0.mean()),
                        se=float(se), zval=float(diff / se), welch_p=float(pv))
        print("  t=%4d  R0=%.4f+-%.4f  R16=%.4f+-%.4f  diff=%+.2f%%  z=%+.2f  p=%.3g"
              % (t, e0.mean(), e0.std(ddof=1), e16.mean(), e16.std(ddof=1),
                 welch[t]["pct"], welch[t]["zval"], pv))

    rep = {t: [] for t in (11, 1200)}
    for k in range(10):
        rg = np.random.default_rng(SEED + 90000 + 7919 * k)
        Ak = rg.normal(size=(N, D))
        e = {}
        for R in (0, 16):
            e[R] = per_traj_err(Ak, w0, R, p, SEED + 300000 + 7919 * k + 1000 * R,
                                (11, 1200), 1000, T, simulate, ETA_LR)
        rep[11].append(float(e[16][11].mean() - e[0][11].mean()))
        rep[1200].append(float(e[16][1200].mean() / e[0][1200].mean()))
    repstat = {}
    for t in (11, 1200):
        a = np.array(rep[t])
        repstat[t] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                          min=float(a.min()), max=float(a.max()),
                          sign_consistent=bool(np.all(a > 0) if t == 11 else np.all(a < 1)))
        print("  repeat-A t=%4d  mean=%+.5f sd=%.5f range=[%+.5f,%+.5f] sign_ok=%s"
              % (t, a.mean(), a.std(ddof=1), a.min(), a.max(), repstat[t]["sign_consistent"]))
    print("  errorbar statistics time %.1fs" % (time.time() - t0))

    # Full error curves; each R uses its own seeded RNG, so this block does not
    # affect any of the random streams above.
    curves = {}
    for R in RS:
        rng = np.random.default_rng(SEED + 1000 * R)
        X = simulate(N, R, p, T, NTRAJ, rng)
        w = np.tile(w0[None, :], (NTRAJ, 1))
        cur = np.empty(T)
        for t in range(T):
            g = w - A[X[:, t]]; w -= ETA_LR * g
            cur[t] = ((w - A.mean(0)) ** 2).sum(axis=1).mean()
        curves[R] = cur

    summary = dict(stats=stats, welch=welch, repeat_A=repstat,
                   sgd_end={R: float(curves[R][-1]) for R in RS},
                   sgd_t11={R: float(curves[R][10]) for R in RS})
    return summary, curves


def longtail_estimates():
    """Aggregate asymptotic variance for a long-tailed propensity p_i ~ i^-1.5."""
    t0 = time.time()
    Nn, Tn, M = 100, 3000, 1200
    idx = np.arange(1, Nn + 1, dtype=float)
    p = idx ** (-1.5); p /= p.sum()
    analytic0 = float(1 - (p ** 2).sum())
    Rb = [0, 1, 2, 5, 10, 20]
    checkpoints = [1500, 2000, 2500, 3000]
    out = {}
    for R in Rb:
        rng = np.random.default_rng(SEED + 50000 + 1000 * R)
        X = simulate_fast(Nn, R, p, Tn, M, rng, burn=600)
        rows = np.arange(M); counts = np.zeros((M, Nn)); vals = {}
        for t in range(1, Tn + 1):
            counts[rows, X[:, t - 1]] += 1.0
            if t in checkpoints:
                eta = counts / t
                vals[t] = float(t * ((eta - eta.mean(0)) ** 2).sum(axis=1).mean())
        est = float(np.mean([vals[t] for t in checkpoints]))
        out[R] = dict(estimate=est, checkpoints={str(t): vals[t] for t in checkpoints},
                      spread=float(max(vals.values()) - min(vals.values())))
        print("  R=%2d  sum-sigma2 est=%.6f  spread=%.6f  ratio to R=0 %.4f"
              % (R, est, out[R]["spread"], est / out[0]["estimate"]))
    cal = out[0]["estimate"] / analytic0
    print("  analytic sum-sigma2(0)=1-sum(p^2)=%.6f ; MC/analytic=%.4f" % (analytic0, cal))
    print("  longtail time %.1fs" % (time.time() - t0))
    return dict(N=Nn, T=Tn, trajectories=M, R=Rb, p="p_i proportional to i^-1.5",
                analytic_R0=analytic0, calibration_ratio=cal,
                results={str(R): out[R] for R in Rb})


def rho_scale():
    """Closed-form variance ratio for representative (N, R) magnitudes."""
    out = {}
    for Nx, Rx in [(100, 10), (1000, 10), (100000, 10), (100000, 1), (1000000, 10)]:
        rho = Rx / Nx
        ratio = (Nx - Rx) * (Nx - Rx - 1) / Nx ** 3 / ((Nx - 1) / Nx ** 2)
        out["N%d_R%d" % (Nx, Rx)] = dict(rho=rho, ratio=ratio,
                                         improve_pct=100 * (1 - ratio))
        print("  N=%7d R=%2d rho=%.2e  ratio=%.8f  improve=%.5f%%"
              % (Nx, Rx, rho, ratio, 100 * (1 - ratio)))
    return out
