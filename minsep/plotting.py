"""Figure drawing.  Every function takes an explicit output path.

The five ``plot_manuscript_*`` functions produce the figures used in the paper;
the ``plot_baseline_*`` functions reproduce the intermediate diagnostic plots.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RS, COL, T

SHADES = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]


# ------------------------------------------------------------ paper figures
def plot_manuscript_fig1(uni, path):
    """Bias of the empirical frequency and asymptotic variance (Theorem 2)."""
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.2), dpi=400)
    t = np.arange(1, T + 1)
    for c, R in zip(COL, RS):
        ax[0].plot(t, uni[R]["bias"], color=c, lw=1.4, label="$R=%d$" % R)
        ax[1].plot(t, uni[R]["tvar"], color=c, lw=1.4)
    for c, R in zip(COL, RS):
        ax[1].axhline(uni[R]["pred"], color=c, ls=":", lw=0.9, alpha=0.85)
    ax[0].set_xscale("log"); ax[0].set_xlabel("round $t$")
    ax[0].set_ylabel(r"$\|\mathbb{E}\hat\nu_t-\pi\|_{TV}$")
    ax[0].set_title("(a) Bias of the expected empirical frequency"); ax[0].set_ylim(0, 0.9)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("round $t$"); ax[1].set_ylabel(r"$t\,\mathrm{tr}\,\mathrm{Var}(\hat\nu_t)$")
    ax[1].set_title("(b) Asymptotic variance; dotted = Theorem 2"); ax[1].set_ylim(1e-2, 3)
    for a in ax:
        a.grid(alpha=0.25, which="both", lw=0.5)
    ax[0].legend(fontsize=9.5, ncol=2, framealpha=0.85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_manuscript_fig2(eb, curves, path):
    """Quadratic estimation error with one-s.d. error bars at four rounds."""
    times = [11, 20, 40, 1200]
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.2), dpi=400)
    tgrid = np.arange(1, T + 1)
    for R in RS:
        ax[0].plot(tgrid, curves[R], color=COL[RS.index(R)], lw=1.4, label="$R=%d$" % R)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("round $t$"); ax[0].set_ylabel(r"$\|w_t-a_{\mathrm{avg}}\|^2$")
    ax[0].set_title("(a) Quadratic estimation error")
    ax[0].grid(alpha=0.25, which="both", lw=0.5); ax[0].legend(fontsize=9.5, ncol=2)

    x = np.arange(len(RS)); wdt = 0.20
    for j, (t, col) in enumerate(zip(times, SHADES)):
        m = np.array([eb["stats"][t][R]["mean"] for R in RS])
        s = np.array([eb["stats"][t][R]["sd"] for R in RS])
        ax[1].bar(x + (j - 1.5) * wdt, m, wdt, yerr=s, capsize=2.5,
                  ecolor=col, error_kw=dict(lw=0.8, alpha=0.9),
                  color=col, alpha=0.92, label="$t=%d$" % t)
    ax[1].set_yscale("log"); ax[1].set_ylim(3e-3, 6e1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(R) for R in RS])
    ax[1].set_xlabel("$R$"); ax[1].set_ylabel(r"$\|w_t-a_{\mathrm{avg}}\|^2$")
    ax[1].set_title("(b) Four rounds; bars = one s.d. across 4000 trajectories")
    ax[1].grid(alpha=0.25, axis="y", lw=0.5); ax[1].legend(fontsize=9.5, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_manuscript_fig3(fce, path):
    """Time-domain check of the skewed-propensity instance."""
    t3 = np.arange(1, 3001)
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=400)
    for R, col, lab in [(0, "#2166ac", "$R=0$"), (1, "#b2182b", "$R=1$")]:
        ax.plot(t3, fce[R]["tvar"], color=col, lw=1.5, label=lab)
        ax.axhline(fce[R]["theory"], color=col, ls="--", lw=1.1,
                   label="%s, exact $\\sigma_1^2$" % lab)
    ax.set_xscale("log"); ax.set_ylim(0, 0.11)
    ax.set_xlabel("round $t$"); ax.set_ylabel(r"$t\,\mathrm{Var}(\hat\nu_{t,1})$")
    ax.set_title("Proposition 1: an instance with increasing absolute variance")
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_manuscript_fig4(nu, path):
    """Aggregate variance under long-tailed availability."""
    rr = np.array(nu["R"])
    est = np.array([nu["results"][str(R)]["estimate"] for R in rr])
    sp = np.array([nu["results"][str(R)]["spread"] for R in rr])
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=400)
    ax.errorbar(rr, est, yerr=sp, fmt="o-", color="#b2182b", lw=1.5, capsize=3,
                label=r"long-tailed $p_i\propto i^{-1.5}$, MC")
    ax.axhline(nu["analytic_R0"], color="#b2182b", ls=":", lw=1.1,
               label=r"analytic $\sum_i\sigma_i^2(0)=1-\sum_i p_i^2$")
    unif = np.array([(100 - R) * (100 - R - 1) / 100 ** 2 for R in rr])
    ax.plot(rr, unif, "s--", color="#2166ac", lw=1.4,
            label=r"uniform $p$, Theorem 2 (closed form)")
    ax.set_xlabel("$R$"); ax.set_ylabel(r"$\sum_i\sigma_i^2(R)$")
    ax.set_title(r"Non-uniform availability, $N=100$")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_manuscript_fig5(small, path):
    """Small-rho regime: Theorem 2 ratio and measured steady-state error ratio."""
    def _row(d, k):
        return d[k] if k in d else d[str(k)]
    Rb = sorted(int(k) for k in small)
    rho = np.array([_row(small, R)["rho"] for R in Rb])
    th = np.array([_row(small, R)["ratio_var"] for R in Rb])
    emp = np.array([_row(small, R)["sgd_ratio"] for R in Rb])
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=400)
    ax.plot(rho, 1 - 2 * rho, "k--", lw=1.2, label=r"$1-2\rho$ (asymptotic)")
    ax.plot(rho, th, "o-", color="#2166ac", lw=1.5,
            label=r"$\sigma^2(R)/\sigma^2(0)$, Thm. 2")
    ax.plot(rho, emp, "s-", color="#b2182b", lw=1.5,
            label=r"steady-state estimation error, ratio to $R=0$")
    ax.set_xlabel(r"$\rho=R/N$  ($N=100$)")
    ax.set_ylabel("ratio to $R=0$")
    ax.set_title("Small-$\\rho$ regime ($N=100$)")
    ax.set_ylim(0.7, 1.08); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------ baseline diagnostics
def plot_baseline_fig1(uni, path):
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.2), dpi=400)
    t = np.arange(1, T + 1)
    for c, R in zip(COL, RS):
        ax[0].plot(t, uni[R]["bias"], color=c, lw=1.4, label="$R=%d$" % R)
        ax[1].plot(t, uni[R]["tvar"], color=c, lw=1.4)
    for c, R in zip(COL, RS):
        ax[1].axhline(uni[R]["pred"], color=c, ls=":", lw=0.9, alpha=0.85)
    ax[0].set_xscale("log"); ax[0].set_xlabel("round $t$")
    ax[0].set_ylabel(r"$\|\mathbb{E}\hat\nu_t-\pi\|_{TV}$")
    ax[0].set_title("(a) Bias of the expected empirical frequency"); ax[0].set_ylim(0, 0.9)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("round $t$"); ax[1].set_ylabel(r"$t\,\mathrm{Var}(\hat\nu_t)$")
    ax[1].set_title("(b) Steady-state variance; dotted = Theorem 2"); ax[1].set_ylim(1e-2, 3)
    for a in ax:
        a.grid(alpha=0.25, which="both", lw=0.5)
    ax[0].legend(fontsize=9.5, ncol=2, framealpha=0.85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_baseline_fig2(uni, path):
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.2), dpi=400)
    t = np.arange(1, T + 1)
    for c, R in zip(COL, RS):
        ax[0].plot(t, uni[R]["sgd"], color=c, lw=1.4, label="$R=%d$" % R)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("round $t$"); ax[0].set_ylabel(r"$\|w_t-w^\star\|^2$")
    ax[0].set_title("(a) Quadratic training error")
    ax[0].grid(alpha=0.25, which="both", lw=0.5); ax[0].legend(fontsize=9.5, ncol=2)
    times = [("t11", "$t=11$"), ("t20", "$t=20$"), ("t40", "$t=40$"), ("T", "$t=1200$")]
    keys = ["sgd_t11", "sgd_t20", "sgd_t40", "sgd_end"]
    x = np.arange(len(RS)); w = 0.20
    for j, ((lab, tick), key, col) in enumerate(zip(times, keys, SHADES)):
        vals = np.array([uni[R][key] for R in RS])
        ax[1].bar(x + (j - 1.5) * w, vals, w, color=col, label=tick)
    ax[1].set_yscale("log"); ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(R) for R in RS])
    ax[1].set_xlabel("$R$"); ax[1].set_ylabel(r"$\|w_t-w^\star\|^2$")
    ax[1].set_title("(b) Ordering across four rounds")
    ax[1].grid(alpha=0.25, axis="y", lw=0.5); ax[1].legend(fontsize=9.5, ncol=2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_baseline_fig3(fce, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=400)
    t = np.arange(1, 3001)
    for R, col, lab in [(0, "#2166ac", "$R=0$"), (1, "#b2182b", "$R=1$")]:
        ax.plot(t, fce[R]["tvar"], color=col, lw=1.5, label=lab)
        ax.axhline(fce[R]["theory"], color=col, ls="--", lw=1.1,
                   label="%s, exact $\\sigma_1^2$" % lab)
    ax.set_xscale("log"); ax.set_ylim(0, 0.11)
    ax.set_xlabel("round $t$"); ax.set_ylabel(r"$t\,\mathrm{Var}(\hat\nu_{t,1})$")
    ax.set_title("Proposition 1: variance increases with $R$")
    ax.grid(alpha=0.25, which="both", lw=0.5); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_baseline_fig4(small, path):
    Rb = sorted(small.keys())
    rho = np.array([small[R]["rho"] for R in Rb])
    th = np.array([small[R]["ratio_var"] for R in Rb])
    emp = np.array([small[R]["sgd_ratio"] for R in Rb])
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=400)
    ax.plot(rho, 1 - 2 * rho, "k--", lw=1.2, label=r"$1-2\rho$ (asymptotic)")
    ax.plot(rho, th, "o-", color="#2166ac", lw=1.5, label=r"$\sigma^2(R)/\sigma^2(0)$, Thm. 2")
    ax.plot(rho, emp, "s-", color="#b2182b", lw=1.5,
            label=r"steady training error, $R/R=0$")
    ax.set_xlabel(r"$\rho=R/N$  ($N=100$)"); ax.set_ylabel("ratio to $R=0$")
    ax.set_title("Small-$\\rho$ regime ($N=100$)")
    ax.set_ylim(0.7, 1.08); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
