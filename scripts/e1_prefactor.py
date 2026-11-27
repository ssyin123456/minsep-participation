# -*- coding: utf-8 -*-
"""E1 --- TV prefactor certificate, extended to N=100, and Fig. 6.

Exact:  N=10, R=1..8  (lump for R<=7, direct P_R for R=8)
        N=100, R=1..7 (identity-tracking lump, D_R independent of N)
Bounds only: N=100, R=8..88 (state space / D_R too large for exact iteration)

Outputs: data/e1.json, figs/fig5.jpg
"""
import json, os, sys, time
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
rng = np.random.default_rng(20261120)


def cert(N, R, method):
    r = C.C_certificate(N, R, t_max=400, method=method)
    r.update(N=N, R=R, tau_int=C.tau_int(N, R),
             lower_bound=max(R, int(np.ceil(np.log(2) / abs(np.log(r["lam"]))))))
    return r


print("E1: N=10 exact (R=1..8)")
n10 = []
for R in range(1, 9):
    m = "lump" if R <= 7 else "direct"
    r = cert(10, R, m)
    n10.append(r)
    print(f"  R={R} tau={r['tau_mix']:3d} C={r['C']:.3f} "
          f"lower={r['lower_bound']:3d} upper={r['upper_bound']:7.2f} [{m}]")

print("E1: N=100 exact (R=1..7, identity-tracking lump)")
n100_exact = []
for R in range(1, 8):
    r = cert(100, R, "lump")
    n100_exact.append(r)
    print(f"  R={R} tau={r['tau_mix']:3d} C={r['C']:.3f} "
          f"R+L={r['R_plus_L']:6.2f} upper={r['upper_bound']:6.2f} "
          f"D_R={r['n_states']}")

print("E1: N=100 bounds only (R=8..88)")
n100_bounds = []
for R in [8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 88]:
    lam = C.lam_star(100, R, high_prec=(R > 40))
    L = C.L_of(100, R, lam)
    n100_bounds.append(dict(N=100, R=R, lam=lam, L=L,
                            lower_bound=max(R, int(np.ceil(np.log(2) / abs(np.log(lam))))),
                            R_plus_L=R + L, tau_int=C.tau_int(100, R),
                            tau_mix=None, C=None))
    print(f"  R={R:2d} lam={lam:.6f} L={L:8.2f} lower={n100_bounds[-1]['lower_bound']:4d} "
          f"tau_int={C.tau_int(100,R):.5f}")

# ---- Fig 6 ---------------------------------------------------------------
apply_rc()
fig, axes = new_canvas()

# (a) N=10 : fully exact
ax = axes[0]
Rs = [r["R"] for r in n10]
ax.plot(Rs, [r["tau_mix"] for r in n10], "o-", color="#b2182b", lw=1.8, ms=5,
        label=r"exact $\tau_{\rm mix}$")
ax.plot(Rs, [r["lower_bound"] for r in n10], "s--", color="#2166ac", lw=1.3, ms=4,
        label=r"lower bound (Thm 1 / 4)")
ax.plot(Rs, [r["upper_bound"] for r in n10], "^--", color="#1a9850", lw=1.3, ms=4,
        label=r"upper bound (Thm 5, with $\ln C$)")
ax.plot(Rs, [r["R_plus_L"] for r in n10], ":", color="#4d4d4d", lw=1.5,
        label=r"$R+L$ (leading term)")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"rounds")
ax.set_title("(a) $N=10$: all quantities exact")
ax.legend(fontsize=PT, loc="upper left", framealpha=.9)
ax2 = ax.twinx()
ax2.plot(Rs, [r["tau_int"] for r in n10], "d-.", color="#762a83", lw=1.5, ms=4,
         label=r"$\tau_{\rm int}$ (right)")
ax2.set_ylabel(r"$\tau_{\rm int}(R)$", color="#762a83")
ax2.tick_params(axis="y", labelcolor="#762a83")
ax2.grid(False)
ax2.legend(fontsize=PT, loc="center right", framealpha=.9)

# (b) N=100 : exact points + provable bounds
ax = axes[1]
allR = [r["R"] for r in n100_exact + n100_bounds]
lower = [r["lower_bound"] for r in n100_exact + n100_bounds]
rpl = [r["R_plus_L"] for r in n100_exact + n100_bounds]
ax.plot(allR, lower, "s--", color="#2166ac", lw=1.4, ms=4, label="lower bound (Thm 1 / 4)")
ax.plot(allR, rpl, ":", color="#4d4d4d", lw=1.7, label=r"$R+L$ (leading term)")
exR = [r["R"] for r in n100_exact]
ax.plot(exR, [r["tau_mix"] for r in n100_exact], "o-", color="#b2182b", lw=1.8, ms=5,
        label=r"exact $\tau_{\rm mix}$ ($R\leq7$)")
ax.set_yscale("log")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"rounds (log)")
allR2 = np.arange(0, 89)
ax.set_title(r"(b) $N=100$: exact for $R\leq7$, provable bounds for $R>7$")
ax.legend(fontsize=PT, loc="upper left", framealpha=.9)
ax3 = ax.twinx()
ax3.plot(allR2, [C.tau_int(100, int(x)) for x in allR2], "-", color="#762a83", lw=2,
         label=r"$\tau_{\rm int}(R)$ (right, linear)")
ax3.set_ylabel(r"$\tau_{\rm int}(R)$", color="#762a83")
ax3.tick_params(axis="y", labelcolor="#762a83")
ax3.grid(False)
ax3.legend(fontsize=PT, loc="upper right", framealpha=.9)

fig.suptitle("One polynomial root drives two time scales in opposite directions",
             fontsize=PT+1, y=0.975)
finalize(fig)
save(fig, f"{ROOT}/figs/fig5.jpg")
print("saved figs/fig5.jpg")

out = dict(config=dict(seed=20261120, accuracy=0.25, generated="2026-11-20"),
           N10_exact=n10, N100_exact=n100_exact, N100_bounds=n100_bounds,
           note=("R>7 at N=100: exact tau_mix is not computable (both (N)_R and "
                 "D_R grow too fast); only the provable lower bound and the "
                 "leading term R+L are shown. C(q) is NOT claimed for R>7."))
with open(f"{ROOT}/data/e1.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved data/e1.json")

# acceptance
slack = [r["upper_bound"] - r["tau_mix"] for r in n10 + n100_exact]
print(f"\nACCEPTANCE: min(upper - tau_mix) = {min(slack):.3f} "
      f"(>=0 required), {len(slack)} exact chains, all "
      f"{'PASS' if min(slack) >= -1e-9 else 'FAIL'}")
print("tau_int monotone decreasing:",
      all(n100_bounds[i]["tau_int"] > n100_bounds[i+1]["tau_int"]
          for i in range(len(n100_bounds)-1)))
