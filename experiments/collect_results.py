# -*- coding: utf-8 -*-
"""collect_results.py --- turn the offline GPU outputs into publication assets.

Reads : data/e4.json, data/e5.json (produced by the run_* scripts)
Writes: data/summary.json   (numbers quoted in the manuscript)
        data/table_vi.tex   (Table V, ready to paste / auto-inserted by v3)
        The bootstrap-band figure was dropped from the manuscript as redundant
        with Table V; make_fig9() below is kept only as a diagnostic.

Usage:  python collect_results.py
        python collect_results.py --e4 data/e4_smoke.json --e5 data/e5_smoke.json
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os_fs, sys as _sys_fs
_sys_fs.path.insert(0, _os_fs.path.abspath(_os_fs.path.join(_os_fs.path.dirname(_os_fs.path.abspath(__file__)), "..", "scripts")))
from figstyle import apply_rc, new_canvas, finalize, save, audit, PT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(HERE, ".."))


def load(path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path + "  -- run the offline script first")
        return None
    with open(path) as f:
        return json.load(f)


def make_fig9(e4, out):  # diagnostic only; not a submission figure
    apply_rc()
    fig, axes = new_canvas()

    # (a) Fig.2(b) updated: quadratic error at checkpoints with 95% CI
    ax = axes[0]
    a = e4["e4a"]
    Rs = sorted(int(k) for k in a)
    cps = sorted(int(t) for t in a[str(Rs[0])]["checkpoints"])
    cols = plt.cm.plasma(np.linspace(.05, .85, len(cps)))
    for j, t in enumerate(cps):
        means, los, his, ana = [], [], [], []
        for R in Rs:
            v = a[str(R)]
            m = v["checkpoints"][str(t)]
            # CI was computed on the FINAL checkpoint; scale it to this one by
            # the ratio of means (conservative; the final CI is reported in
            # Table VI and in the text, panel (a) shows the shape only).
            s = m / max(v["mean"], 1e-12)
            means.append(m)
            los.append(v["lo"] * s if t == cps[-1] else np.nan)
            his.append(v["hi"] * s if t == cps[-1] else np.nan)
            ana.append(v["analytic_abs"])
        ax.plot(Rs, means, "o-", color=cols[j], lw=1.6, ms=4,
                label=fr"$t={t}$")
        if t == cps[-1]:
            ax.fill_between(Rs, los, his, color=cols[j], alpha=.18,
                            label="95% bootstrap CI (final $t$)")
            ax.plot(Rs, ana, "k--", lw=1.5, label="Prop. 4 closed form")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e1)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$\mathbb{E}\|w_t-a_{\rm avg}\|^2$")
    ax.set_title("(a) Fig. 2(b) with a bootstrap confidence band")
    ax.legend(fontsize=PT, framealpha=.9, loc="center left", ncol=2)

    # (b) Fig.4 updated: aggregate variance with 95% CI
    ax = axes[1]
    b = e4["e4b"]
    Rb = sorted(int(k) for k in b)
    mean = [b[str(R)]["mean"] for R in Rb]
    lo = [b[str(R)]["lo"] for R in Rb]
    hi = [b[str(R)]["hi"] for R in Rb]
    ax.fill_between(Rb, lo, hi, color="#b2182b", alpha=.2, label="95% bootstrap CI")
    ax.plot(Rb, mean, "o-", color="#b2182b", lw=2, ms=5,
            label=r"$\sum_i\sigma_i^2(R)$, long tail")
    Z0 = b[str(Rb[0])]["Z0"]
    ax.axhline(Z0, color="#2166ac", ls=":", lw=1.5,
               label=fr"analytic $R=0$: $Z={Z0:.4f}$")
    ax.set_xlabel(r"$R$"); ax.set_ylabel(r"$\sum_i\sigma_i^2(R)$")
    ax.set_title("(b) Fig. 4 with a bootstrap confidence band")
    ax.legend(fontsize=PT, framealpha=.9)

    fig.suptitle("Uncertainty quantified: $10^5$ trajectories, "
                 "$B=2000$ bootstrap resamples", fontsize=PT+1, y=0.975)
    finalize(fig)
    save(fig, out)
    print("saved", out)


def make_table_vi(e4, out):
    a = e4["e4a"]
    lines = [
        r"\begin{table}[!t]",
        r"\small", r"\centering",
        r"\caption{Bootstrap validation of the Proposition 4 closed form at "
        r"$10^5$ trajectories ($B=2000$): steady-state quadratic error of "
        r"Fig.~\ref{fig:2}(b) with 95\% confidence intervals.}",
        r"\label{table:V}",
        r"\begin{tabular}{@{}cccc@{}}",
        r"\toprule",
        r"$R$ & measured & 95\% CI & Prop.~4 closed form \\",
        r"\midrule",
    ]
    for R in sorted(int(k) for k in a):
        v = a[str(R)]
        lines.append(fr"{R} & {v['mean']:.5f} & "
                     fr"[{v['lo']:.5f},{v['hi']:.5f}] & "
                     fr"{v['analytic_abs']:.5f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4", default=os.path.join(HERE, "data/e4.json"))
    ap.add_argument("--e5", default=os.path.join(HERE, "data/e5.json"))
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    e4 = load(args.e4)
    e5 = load(args.e5, required=False)
    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
    pvi = os.path.join(HERE, "data", args.out_prefix + "table_vi.tex")
    make_table_vi(e4, pvi)

    summary = dict(
        e4a={k: dict(mean=v["mean"], lo=v["lo"], hi=v["hi"],
                     analytic_abs=v["analytic_abs"],
                     measured_ratio=v["measured_ratio"],
                     analytic_ratio=v["analytic_ratio"])
             for k, v in e4["e4a"].items()},
        e4b={k: dict(mean=v["mean"], lo=v["lo"], hi=v["hi"], Z0=v["Z0"])
             for k, v in e4["e4b"].items()},
        e4_backend=e4.get("backend"),
        e4_smoke=e4.get("smoke", False),
        e5_present=e5 is not None,
    )
    if e5:
        summary["e5"] = dict(
            profile_source=e5["profile"]["meta"].get("source"),
            Z0=e5["profile"]["Z0"],
            smoke=e5.get("smoke"),
            participation={k: dict(mean=v["mean"], lo=v["lo"], hi=v["hi"])
                           for k, v in e5["participation"].items()},
            scope_disclaimer=e5["scope_disclaimer"])
    ps = os.path.join(HERE, "data", args.out_prefix + "summary.json")
    with open(ps, "w") as f:
        json.dump(summary, f, indent=1)
    print("saved", ps)

    print("\n=== CHECKS ===")
    print("  E4-A: analytic inside 95% CI for all R :",
          all(v["lo"] <= v["analytic_abs"] <= v["hi"]
              for v in e4["e4a"].values()))
    b0 = e4["e4b"][str(min(int(k) for k in e4["e4b"]))]
    print("  E4-B: R=0 calibration inside CI :",
          b0["lo"] <= b0["Z0"] <= b0["hi"], f"({b0['mean']:.5f} vs {b0['Z0']:.5f})")
    if e4.get("smoke"):
        print("  WARNING: e4.json came from a --smoke run -- "
              "these numbers must not be used in the manuscript")
    else:
        print("  e4.json is a full run -> ready for the manuscript")


if __name__ == "__main__":
    main()
