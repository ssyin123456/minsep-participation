# -*- coding: utf-8 -*-
"""E3 --- Proposition 4 closed form vs independent truncated covariance
series, eta scan, and the 1-lambda_star validity boundary. Pure CPU, analytic.

Outputs: data/e3.json, figs/fig7.jpg
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os_fs, sys as _sys_fs
_sys_fs.path.insert(0, _os_fs.path.dirname(_os_fs.path.abspath(__file__)))
from figstyle import apply_rc, new_canvas, finalize, save, audit, PT

sys.path.insert(0, os.path.dirname(__file__))
import common as C

ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(f"{ROOT}/data", exist_ok=True)
os.makedirs(f"{ROOT}/figs", exist_ok=True)

N = 20
Rset = [0, 4, 8, 16]
eta_scan = np.array([0.04, 0.02, 0.01, 0.005, 0.001])
# upper end 0.5 keeps the R=4 gap (0.4638) inside the plotted range
eta_dense = np.logspace(np.log10(0.0008), np.log10(0.5), 130)

print("E3: closed form vs truncated series (independent verification)")
rows = []
maxerr = 0.0
for R in Rset:
    for eta in eta_scan:
        a = C.prop4_ratio(N, R, eta)
        b = C.prop4_ratio_by_series(N, R, eta, M=8000)
        e = abs(a - b)
        maxerr = max(maxerr, e)
        zf = C.zero_freq_ratio(N, R)
        rows.append(dict(R=R, eta=float(eta), closed=a, series=b,
                         abs_err=e, zero_freq=zf,
                         ratio_to_zerofreq=a / zf if zf > 0 else None))
    lam = C.lam_star(N, R)
    print(f"  R={R:2d} 1-lam*={1-lam if R>0 else float('nan'):.6f}  "
          f"zero-freq={C.zero_freq_ratio(N,R):.6f}  "
          f"closed(0.04)={C.prop4_ratio(N,R,0.04):.9f}")

target = C.prop4_ratio(N, 16, 0.04)
print(f"\n  target N=20,R=16,eta=0.04 : {target:.9f}  (manuscript: 0.168944)")
print(f"  max |closed - series| over grid : {maxerr:.3e}")

# dense curves
curves = {}
for R in Rset:
    curves[str(R)] = dict(eta=[float(x) for x in eta_dense],
                          ratio=[float(C.prop4_ratio(N, R, float(x))) for x in eta_dense],
                          lam=C.lam_star(N, R) if R > 0 else 0.0,
                          gap=(1 - C.lam_star(N, R)) if R > 0 else 1.0,
                          zero_freq=C.zero_freq_ratio(N, R))

# ---- Fig 8 ---------------------------------------------------------------
apply_rc()
fig, axes = new_canvas()

ax = axes[0]
cols = {0: "#4d4d4d", 4: "#2166ac", 8: "#1a9850", 16: "#b2182b"}
for R in Rset:
    d = curves[str(R)]
    ax.semilogx(d["eta"], d["ratio"], "-", color=cols[R], lw=2,
                label=fr"$R={R}$")
    # independent series checkpoints
    ax.semilogx(eta_scan, [C.prop4_ratio_by_series(N, R, float(e), M=8000)
                           for e in eta_scan], "o", color=cols[R], ms=3.5,
                mfc="white", mew=1.1)
    ax.axhline(d["zero_freq"], color=cols[R], ls=":", lw=1.1, alpha=.8)
    if R > 0:
        ax.axvline(d["gap"], color=cols[R], ls="--", lw=1.0, alpha=.55)
ax.plot([0.04], [target], "*", color="#000000", ms=12, zorder=5)
ax.annotate("0.168944", (0.04, target), xytext=(0.012, 0.30), fontsize=PT,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_ylim(-0.02, 1.18)
ax.set_xlabel(r"step size $\eta$ (log)")
ax.set_ylabel(r"$\mathbb{E}\|u\|^2(R)/\mathbb{E}\|u\|^2(0)$")
ax.set_title("(a) Closed form (line) vs independent series (circles)")
ax.legend(fontsize=PT, loc="lower right", framealpha=.9)
ax.text(0.015, 0.985, r"dotted: zero-frequency $\tau_{\rm int}$"
        "\n" r"dashed: $\eta=1-\lambda_*$",
        fontsize=PT, transform=ax.transAxes, va="top",
        bbox=dict(fc="white", ec="0.85", alpha=.9, pad=2))

ax = axes[1]
for R in Rset[1:]:
    d = curves[str(R)]
    x = np.array(d["eta"]) / d["gap"]
    y = np.array(d["ratio"]) / d["zero_freq"]
    ax.semilogx(x, y, "-", color=cols[R], lw=1.9, label=fr"$R={R}$")
ax.axhline(1.0, color="k", ls=":", lw=1.2)
ax.axvline(1.0, color="#b2182b", ls="--", lw=1.4)
ax.text(1.35, 5.2, r"$\eta=1-\lambda_*$", fontsize=PT, color="#b2182b")
ax.set_xlabel(r"$\eta/(1-\lambda_*)$ (log)")
ax.set_ylabel("finite-$\\eta$ error / zero-frequency prediction")
ax.set_title("(b) Zero-frequency approximation fails past the gap")
ax.legend(fontsize=PT, loc="upper left", framealpha=.9)

fig.suptitle("Proposition 4 predicts the steady-state error at any step size",
             fontsize=PT+1, y=0.975)
finalize(fig)
save(fig, f"{ROOT}/figs/fig7.jpg")
print("saved figs/fig7.jpg")

out = dict(config=dict(N=N, Rset=Rset, eta_scan=[float(x) for x in eta_scan],
                       M_series=8000, generated="2026-11-20"),
           checkpoints=rows, target_ratio=target, max_closed_series_error=maxerr,
           curves=curves,
           note=("Vertical dashed lines mark eta=1-lambda_star(R,N); the "
                 "zero-frequency constant is accurate only well left of them."))
with open(f"{ROOT}/data/e3.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved data/e3.json")

ok = (abs(target - 0.168944) < 2e-5) and (maxerr < 1e-8)
print(f"\nACCEPTANCE: target 0.168944 reproduced ({target:.6f}), "
      f"closed-vs-series {maxerr:.2e} -> {'PASS' if ok else 'FAIL'}")
