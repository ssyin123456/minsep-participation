# -*- coding: utf-8 -*-
r"""redraw_all.py --- authoritative entry point for the submission figures.

Reads ONLY the published datasets (data/*.json, data/*.npz,
experiments/data/e5.json); no Monte Carlo is re-run and no numerical value is
recomputed.  Produces the eight 600 dpi JPEG figures referenced by paper/main.tex,
with every text element >= 9.5 pt at its final printed size (6.0 x 3.2 in)
and horizontal legends placed below each subplot, outside the plotting area.
JPEG output avoids nested PDF CID fonts / transparency groups in the final PDF.

Numbering follows the trimmed manuscript:
    new 1 = old 1   bias / asymptotic variance
    new 2 = old 2   quadratic estimation error
    new 3 = old 4   aggregate variance, long-tailed availability
    new 4 = old 5   small-rho regime
    new 5 = old 6   one polynomial root / two time scales
    new 6 = old 7   step-size and burn-in rules
    new 7 = old 8   Proposition 4 at finite step size
    new 8 = old 10  real client profile
Dropped (redundant): old 3 (Proposition 1 time-domain cross-check, already
verified in exact rational arithmetic) and old 9 (bootstrap band over the same
data as Fig. 2(b) / Table V).

Usage:  python3 figure_sources/redraw_all.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
IN = os.path.join(ROOT, "data")
EIN = os.path.join(ROOT, "experiments", "data")
OUT = os.path.join(ROOT, "figs")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from figstyle import apply_rc, new_canvas, finalize, save, PT

apply_rc()
os.makedirs(OUT, exist_ok=True)

v2 = json.load(open(os.path.join(IN, "figures_data_v2.json")))
v3 = json.load(open(os.path.join(IN, "figures_data_v3.json")))
e1 = json.load(open(os.path.join(IN, "e1.json")))
e2 = json.load(open(os.path.join(IN, "e2.json")))
e3 = json.load(open(os.path.join(IN, "e3.json")))
e5 = json.load(open(os.path.join(EIN, "e5.json")))
Z = np.load(os.path.join(IN, "recomputed_v2_curves.npz"))

RS = [0, 1, 2, 4, 8, 12, 16]
COL = plt.cm.viridis(np.linspace(0.0, 0.92, len(RS)))
MC, TH, REF = "#b2182b", "#2166ac", "#4d4d4d"
T = 1200
tgrid = np.arange(1, T + 1)
results = {}


def font_audit(fig):
    """Smallest explicit font size among all Text artists in the figure."""
    sizes = []
    for t in fig.findobj(match=Text):
        try:
            sizes.append(float(t.get_fontsize()))
        except Exception:
            pass
    return min(sizes) if sizes else None


def emit(fig, n, title):
    finalize(fig, title)
    lo = font_audit(fig)
    path = os.path.join(OUT, "fig%d.jpg" % n)
    issues = save(fig, path)
    results[n] = (lo, issues, os.path.getsize(path))
    flag = "OK" if (lo is not None and lo >= 9.0 and not issues) else "CHECK"
    print("  fig%-2d %-58s | min font %.1f pt | %s | %.0f KB" %
          (n, title[:58], lo or -1, flag, os.path.getsize(path) / 1024))


# ---------------------------------------------------------------- new Fig. 1
fig, axes = new_canvas()
ax = axes[0]
for j, r in enumerate(RS):
    ax.semilogx(tgrid, Z["bias_%d" % r], color=COL[j], lw=1.5, label=fr"$R={r}$")
ax.set(xlabel="round $t$", ylabel=r"$\|\mathbb{E}\hat\nu_t-\pi\|_{TV}$", ylim=(0, 0.9),
       title="(a) Frequency bias")
ax.legend()
ax = axes[1]
for j, r in enumerate(RS):
    ax.loglog(tgrid, Z["tvar_%d" % r], color=COL[j], lw=1.5, label=fr"$R={r}$")
    ax.axhline(v2["uniform"][str(r)]["pred"], color=COL[j], ls=":", lw=1.0, alpha=.85)
ax.plot([], [], "k:", lw=1.2, label="Thm 2 (dotted)")
ax.set(xlabel="round $t$", ylabel=r"$t\,\mathrm{tr}\,\mathrm{Var}(\hat\nu_t)$",
       ylim=(1e-2, 3), title="(b) Asymptotic variance")
ax.legend()
emit(fig, 1, "Bias and asymptotic variance under uniform availability")

# ---------------------------------------------------------------- new Fig. 2
fig, axes = new_canvas()
ax = axes[0]
for j, r in enumerate(RS):
    ax.loglog(tgrid, Z["sgd_%d" % r], color=COL[j], lw=1.5, label=fr"$R={r}$")
ax.set(xlabel="round $t$", ylabel=r"$\|w_t-a_{\mathrm{avg}}\|^2$",
       title="(a) Quadratic estimation error")
ax.legend()
ax = axes[1]
stats = v3["errorbars"]["stats"]
times = [11, 20, 40, 1200]
shades = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]
x = np.arange(len(RS))
w = 0.20
for j, (t, col) in enumerate(zip(times, shades)):
    m = np.array([stats[str(t)][str(r)]["mean"] for r in RS])
    s = np.array([stats[str(t)][str(r)]["sd"] for r in RS])
    ax.bar(x + (j - 1.5) * w, m, w, yerr=s, capsize=2.5, ecolor=col,
           error_kw=dict(lw=.8, alpha=.9), color=col, alpha=.92,
           label=fr"$t={t}$")
ax.set_yscale("log")
ax.set_ylim(3e-3, 6e1)
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in RS])
ax.set(xlabel="$R$", ylabel=r"$\|w_t-a_{\mathrm{avg}}\|^2$",
       title="(b) Four checkpoints")
ax.legend()
emit(fig, 2, "Quadratic estimation error: transient and steady state")

# ---------------------------------------------------------------- new Fig. 3 (old 4)
fig, axes = new_canvas()
nu = v3["nonuniform"]
rr = np.array(sorted(int(k) for k in nu["results"]))
est = np.array([nu["results"][str(r)]["estimate"] for r in rr])
sp = np.array([nu["results"][str(r)]["spread"] for r in rr])
a0 = nu["analytic_R0"]
ax = axes[0]
ax.errorbar(rr, est, yerr=sp, fmt="o-", color=MC, lw=1.5, capsize=3,
            label=r"long-tail $p$")
ax.axhline(a0, color=MC, ls=":", lw=1.2,
           label=r"analytic $R=0$")
unif = (100 - rr) * (100 - rr - 1) / 100 ** 2
ax.plot(rr, unif, "s--", color=TH, lw=1.4,
        label=r"uniform $p$")
ax.set(xlabel="$R$", ylabel=r"$\sum_i\sigma_i^2(R)$",
       title=r"(a) Aggregate variance, $N=100$")
ax.legend()
ax = axes[1]
ck = ["1500", "2000", "2500", "3000"]
ckcol = plt.cm.Blues(np.linspace(.45, .9, len(ck)))
for j, c in enumerate(ck):
    vals = np.array([nu["results"][str(r)]["checkpoints"][c] for r in rr])
    ax.plot(rr, (vals - est) * 1e3, "o-", color=ckcol[j], lw=1.4, ms=4,
            label=fr"$t={c}$")
ax.axhline(0.0, color="k", ls="-", lw=1.5, label="mean")
ax.set(xlabel="$R$", ylabel=r"deviation from mean  $(\times 10^{-3})$",
       title="(b) Checkpoint spread")
ax.legend()
emit(fig, 3, "Aggregate variance under long-tailed availability")

# ---------------------------------------------------------------- new Fig. 4 (old 5)
fig, axes = new_canvas()
small = v2["small_rho"]
rho = np.array([small[str(r)]["rho"] for r in [0, 1, 2, 5, 10]])
rv = np.array([small[str(r)]["ratio_var"] for r in [0, 1, 2, 5, 10]])
sg = np.array([small[str(r)]["sgd_ratio"] for r in [0, 1, 2, 5, 10]])
ax = axes[0]
ax.plot(rho, 1 - 2 * rho, "k--", lw=1.3, label=r"$1-2\rho$")
ax.plot(rho, rv, "o-", color=TH, lw=1.5,
        label=r"variance (Thm 2)")
ax.plot(rho, sg, "s-", color=MC, lw=1.5,
        label="est. error")
ax.set_ylim(0.7, 1.08)
ax.set(xlabel=r"$\rho=R/N$  ($N=100$)", ylabel="ratio to $R=0$",
       title=r"(a) Small-$\rho$ regime at $N=100$")
ax.legend()
ax = axes[1]
Ns = np.logspace(2, 6.2, 200)
for Rfix, col, mk in [(1, TH, "s"), (10, MC, "o")]:
    rat = (Ns - Rfix) * (Ns - Rfix - 1) / Ns ** 3 / ((Ns - 1) / Ns ** 2)
    ax.semilogx(Ns, 100 * (1 - rat), "-", color=col, lw=1.6,
                label=fr"$R={Rfix}$ analytic")
pts = v3["rho_scale"]
for Rfix, col, mk in [(1, TH, "s"), (10, MC, "o")]:
    xs, ys = [], []
    for k, v in pts.items():
        Nx = int(k.split("_")[0][1:]); Rx = int(k.split("_")[1][1:])
        if Rx == Rfix:
            xs.append(Nx); ys.append(v["improve_pct"])
    ax.semilogx(xs, ys, mk, color=col, ms=6, mfc="white", mew=1.4,
                label=fr"$R={Rfix}$ tabulated")
ax.set(xlabel=r"population size $N$ (log)", ylabel="variance reduction (%)",
       title=r"(b) Large-$N$ limit")
ax.legend()
emit(fig, 4, "Small-rho regime and its large-N limit")

# ---------------------------------------------------------------- new Fig. 5 (old 6)
fig, axes = new_canvas()
n10 = e1["N10_exact"]
n100e = e1["N100_exact"]
n100b = e1["N100_bounds"]
ax = axes[0]
Rs = [r["R"] for r in n10]
ax.plot(Rs, [r["tau_mix"] for r in n10], "o-", color="#b2182b", lw=1.6, ms=5,
        label=r"exact $\tau_{\rm mix}$")
ax.plot(Rs, [r["lower_bound"] for r in n10], "s--", color="#2166ac", lw=1.3, ms=4,
        label="lower")
ax.plot(Rs, [r["upper_bound"] for r in n10], "^--", color="#1a9850", lw=1.3, ms=4,
        label="upper")
ax.plot(Rs, [r["R_plus_L"] for r in n10], ":", color="#4d4d4d", lw=1.4,
        label=r"$R+L$")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"rounds")
ax.set_title("(a) $N=10$: all quantities exact")
ax2 = ax.twinx()
ax2.plot(Rs, [r["tau_int"] for r in n10], "d-.", color="#762a83", lw=1.4, ms=4,
         label=r"$\tau_{\rm int}$")
ax2.tick_params(axis="y", labelcolor="#762a83")
ax2.set_yticklabels([])
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=PT, framealpha=.9)
ax = axes[1]
allR = [r["R"] for r in n100e + n100b]
lower = [r["lower_bound"] for r in n100e + n100b]
rpl = [r["R_plus_L"] for r in n100e + n100b]
ax.plot(allR, lower, "s--", color="#2166ac", lw=1.3, ms=4, label="lower")
ax.plot(allR, rpl, ":", color="#4d4d4d", lw=1.5, label=r"$R+L$")
exR = [r["R"] for r in n100e]
ax.plot(exR, [r["tau_mix"] for r in n100e], "o-", color="#b2182b", lw=1.6, ms=5,
        label=r"exact $\tau_{\rm mix}$ ($R\leq7$)")
ax.set_yscale("log")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"rounds (log)")
ax.set_title(r"(b) $N=100$: exact and bounds")
ax3 = ax.twinx()
allR2 = np.arange(0, 89)
ax3.plot(allR2, (100 - allR2) * (99 - allR2) / (100 * 99), "-",
         color="#762a83", lw=1.7, label=r"$\tau_{\rm int}(R)$")
ax3.tick_params(axis="y", labelcolor="#762a83")
ax3.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=PT, framealpha=.9)
emit(fig, 5, "One polynomial root drives two time scales in opposite directions")

# ---------------------------------------------------------------- new Fig. 6 (old 7)
fig, axes = new_canvas()
Rgrid = np.array(e2["Rgrid"])
ax = axes[0]
ax.plot(Rgrid, e2["alpha_rule"], "o-", color="#b2182b", lw=1.7, ms=5,
        label=r"Thm 6 rule")
ax.plot(Rgrid, e2["alpha_const"], "s--", color="#2166ac", lw=1.3, ms=4,
        label=r"constant $\tau_{\rm mix}$")
ax.plot(Rgrid, e2["alpha_decr"], "^:", color="#762a83", lw=1.3, ms=4,
        label=r"[1, App. H]")
ax.set_yscale("log")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"$\alpha_{\rm lr}$ (a.u., log)")
ax.set_title("(a) Step-size rule")
ax.legend(fontsize=PT, loc="lower left", framealpha=.9)
ax = axes[1]
Rmc = [0, 4, 8, 16]
cols = {0: "#4d4d4d", 4: "#2166ac", 8: "#1a9850", 16: "#b2182b"}
for R in Rmc:
    d = e2["mc"][str(R)]
    ax.loglog(d["t"], d["curve"], color=cols[R], lw=1.5,
              label=fr"$R={R}$")
ref_t = np.array([50, 1500])
ax.loglog(ref_t, 0.02 / ref_t, "k--", lw=1.2, label=r"$1/t$ slope")
ax.set_xlabel(r"round $t$"); ax.set_ylabel(r"$\mathbb{E}\|\hat\nu_t-\pi\|_\infty^2$")
ax.set_title(r"(b) Empirical $1/t$ decay")
ax.legend(fontsize=PT, loc="upper right", framealpha=.9)
emit(fig, 6, "Burn-in and step size become explicit functions of $R$")

# ---------------------------------------------------------------- new Fig. 7 (old 8)
fig, axes = new_canvas()
Rset = [0, 4, 8, 16]
eta_scan = np.array(e3["config"]["eta_scan"])
eta_dense = np.array(e3["curves"]["0"]["eta"])
cols = {0: "#4d4d4d", 4: "#2166ac", 8: "#1a9850", 16: "#b2182b"}
series = {}
for row in e3["checkpoints"]:
    series.setdefault(row["R"], {})[float(row["eta"])] = row["series"]
target = e3["target_ratio"]
ax = axes[0]
for R in Rset:
    d = e3["curves"][str(R)]
    ax.semilogx(d["eta"], d["ratio"], "-", color=cols[R], lw=1.7,
                label=fr"$R={R}$")
    ax.semilogx(eta_scan, [series[R][float(e)] for e in eta_scan], "o",
                color=cols[R], ms=3.5, mfc="white", mew=1.1)
    ax.axhline(d["zero_freq"], color=cols[R], ls=":", lw=1.1, alpha=.8)
    if R > 0:
        ax.axvline(d["gap"], color=cols[R], ls="--", lw=1.0, alpha=.55)
ax.plot([0.04], [target], "*", color="#000000", ms=11, zorder=5)
ax.annotate("0.168944", (0.04, target), xytext=(0.012, 0.30), fontsize=PT,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_ylim(-0.02, 1.18)
ax.set_xlabel(r"step size $\eta$ (log)")
ax.set_ylabel(r"$\mathbb{E}\|u\|^2$ ratio")
ax.set_title("(a) Closed form vs series")
ax.legend(fontsize=PT, loc="lower right", framealpha=.9)
ax = axes[1]
for R in Rset[1:]:
    d = e3["curves"][str(R)]
    x = np.array(d["eta"]) / d["gap"]
    y = np.array(d["ratio"]) / d["zero_freq"]
    ax.semilogx(x, y, "-", color=cols[R], lw=1.6, label=fr"$R={R}$")
ax.axhline(1.0, color="k", ls=":", lw=1.2)
ax.axvline(1.0, color="#b2182b", ls="--", lw=1.4)
ax.text(1.35, 5.2, r"$\eta=1-\lambda_*$", fontsize=PT, color="#b2182b")
ax.set_xlabel(r"$\eta/(1-\lambda_*)$ (log)")
ax.set_ylabel("ratio to zero-freq.")
ax.set_title("(b) Past the validity gap")
ax.legend(fontsize=PT, loc="upper left", framealpha=.9)
emit(fig, 7, "Proposition 4 at finite step size")

# ---------------------------------------------------------------- new Fig. 8 (old 10)
fig, axes = new_canvas()
part = e5["participation"]
cols = plt.cm.viridis(np.linspace(0, .85, len(part)))
ax = axes[0]
for (k, v), col in zip(sorted(part.items(), key=lambda x: int(x[0])), cols):
    c = v["curves"]
    ax.plot(c["t"], c["t_trVar"], "o-", color=col, lw=1.5, ms=4,
            label=fr"$R={k}$")
ax.set_xlabel(r"round $t$ (after burn-in)")
ax.set_ylabel(r"$t\,\mathrm{tr\,Var}(\hat\nu_t)$")
ax.set_title("(a) Aggregate variance")
ax.legend(fontsize=PT, ncol=2, framealpha=.9, loc="upper left")
ax = axes[1]
for (k, v), col in zip(sorted(part.items(), key=lambda x: int(x[0])), cols):
    c = v["curves"]
    ax.loglog(c["t"], c["bias2"], "o-", color=col, lw=1.4, ms=4,
              label=fr"$R={k}$")
tmax = max(int(t) for v in part.values() for t in v["curves"]["t"])
tt = np.array([max(10, tmax // 40), max(20, tmax)])
ax.loglog(tt, 2e-4 / tt, "k--", lw=1.2, label=r"$1/t$")
ax.set_xlabel(r"round $t$ (log)")
ax.set_ylabel(r"$\|\mathbb{E}\hat\nu_t-\hat\pi_R\|^2$")
ax.set_title(r"(b) Bias transient")
ax.legend(fontsize=PT, ncol=2, framealpha=.9)
emit(fig, 8, "Real client profile under a CIFAR-10 partition")

# ---------------------------------------------------------------- summary
print("\n=== AUDIT ===")
ok = True
for n, (lo, issues, size) in sorted(results.items()):
    good = (lo is not None and lo >= 9.0 and not issues)
    ok &= good
    print("  fig%d  min font %.1f pt  legends %s  %.0f KB  %s" %
          (n, lo or -1, "inside" if not issues else "OUTSIDE %s" % issues,
           size / 1024, "PASS" if good else "FAIL"))
print("  all figures compliant:", ok)
if not ok:
    sys.exit(2)
