# -*- coding: utf-8 -*-
r"""Recompute every number behind Figs. 1-5 from the pure-function package.

    python3 -m minsep.reproduce            # core datasets + curve cache
    python3 -m minsep.reproduce --full     # + error bars / long tail / baseline plots
    python3 -m minsep.reproduce --check    # compare against the published JSON

The published data/figures_data_*.json and figs/ are never overwritten.
Re-computed datasets go to data/sim_out/. The deterministic curve cache
data/recomputed_v2_curves.npz is refreshed in place; re-running produces the
same 23 arrays, so this is safe and keeps redraw_fig1_5.py self-contained.
"""
import argparse, json
import numpy as np

from .config import DATA_DIR, OUTPUT_DIR, RS, ensure_dirs
from . import experiments as X


DIFFS = []


def cmp(name, got, want, tol=1e-9):
    try:
        e = abs(float(got) - float(want))
    except (TypeError, ValueError):
        e = 0.0 if got == want else float("inf")
    DIFFS.append((name, e))
    return e


def dump(name, obj):
    with open(OUTPUT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + "/" + str(k)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def compare_block(label, got, want):
    a, b = flatten(got), flatten(want)
    for k in sorted(set(a) & set(b)):
        cmp("%s %s" % (label, k), a[k], b[k])
    return len(set(a) & set(b))


def refresh_curves(uni, fce):
    arrays = {}
    for R in RS:
        arrays["bias_%d" % R] = np.asarray(uni[R]["bias"], dtype=np.float64)
        arrays["tvar_%d" % R] = np.asarray(uni[R]["tvar"], dtype=np.float64)
        arrays["sgd_%d" % R] = np.asarray(uni[R]["sgd"], dtype=np.float64)
    for R in (0, 1):
        arrays["fce_%d" % R] = np.asarray(fce[R]["tvar"], dtype=np.float64)
    np.savez(DATA_DIR / "recomputed_v2_curves.npz", **arrays)
    print("  refreshed data/recomputed_v2_curves.npz (%d arrays)" % len(arrays))


def check(uni, fce, small, scale):
    d2 = json.load(open(DATA_DIR / "figures_data_v2.json", encoding="utf-8"))
    d3 = json.load(open(DATA_DIR / "figures_data_v3.json", encoding="utf-8"))
    for R in RS:
        e = d2["uniform"][str(R)]
        for k in ("pred", "tvar_end", "sgd_end", "sgd_t11", "sgd_t20", "sgd_t40"):
            cmp("uniform R%d %s" % (R, k), uni[R][k], e[k])
    for R in (0, 1):
        e = d2["counterexample"][str(R)]
        for k in ("theory", "end"):
            cmp("counterexample R%d %s" % (R, k), fce[R][k], e[k])
    for R in sorted(small):
        e = d2["small_rho"][str(R)]
        for k in ("rho", "pred", "tvar_end", "ratio_var", "sgd_end", "sgd_ratio"):
            cmp("small_rho R%d %s" % (R, k), small[R][k], e[k])
    for k, v in scale.items():
        e = d3["rho_scale"][k]
        for f in ("rho", "ratio", "improve_pct"):
            cmp("rho_scale %s %s" % (k, f), v[f], e[f])
    worst = max(e for _, e in DIFFS)
    bad = [n for n, e in DIFFS if e > 1e-9]
    print("  CHECK: %d values, max abs diff = %.3g, failures = %d"
          % (len(DIFFS), worst, len(bad)))
    for n in bad[:5]:
        print("    FAIL", n)
    return not bad and worst <= 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="also recompute error bars, long-tail MC and baseline plots")
    ap.add_argument("--check", action="store_true",
                    help="compare recomputed values with the published JSON")
    a = ap.parse_args()
    ensure_dirs()

    print("=== recomputing core datasets (seed 20261116) ===")
    uni = X.run_uniform()
    fce = X.run_counterexample()
    small = X.run_small_rho()
    scale = X.rho_scale()
    refresh_curves(uni, fce)
    dump("recomputed_v2.json", dict(config="see configs/ and minsep/config.py",
                                    uniform=uni, counterexample=fce,
                                    small_rho=small))
    dump("recomputed_v3.json", dict(rho_scale=scale))

    if a.full:
        print("=== error bars, Welch tests, independent-A repeats ===")
        eb, curves = X.errorbar_statistics()
        dump("recomputed_errorbars.json", eb)
        print("=== long-tailed availability (N=100) ===")
        nu = X.longtail_estimates()
        dump("recomputed_nonuniform.json", nu)
        from . import plotting as P
        out = OUTPUT_DIR / "baseline"
        out.mkdir(parents=True, exist_ok=True)
        P.plot_manuscript_fig1(uni, out / "fig1.png")
        P.plot_manuscript_fig2(eb, curves, out / "fig2.png")
        P.plot_manuscript_fig3(fce, out / "fig3.png")
        P.plot_manuscript_fig4(nu, out / "fig4.png")
        P.plot_manuscript_fig5(small, out / "fig5.png")
        print("  historical-layout diagnostics -> data/sim_out/baseline/")
        pub = json.load(open(DATA_DIR / "figures_data_v3.json", encoding="utf-8"))
        n1 = compare_block("errorbars", eb, pub["errorbars"])
        n2 = compare_block("nonuniform", nu, pub["nonuniform"])
        print("  cross-checked %d + %d values against figures_data_v3.json"
              % (n1, n2))

    if a.check:
        print("=== consistency with published data ===")
        ok = check(uni, fce, small, scale)
        if not ok:
            raise SystemExit(1)

    print("\nDone. For the eight submission figures run:")
    print("  make figures")


if __name__ == "__main__":
    main()
