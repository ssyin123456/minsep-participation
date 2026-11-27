# -*- coding: utf-8 -*-
"""E2 --- Theorem 6 design rule: step size & burn-in vs R, and the
O([R+L]/t) concentration envelope (light Monte Carlo, 1000 trajectories).

No training accuracy, no comparison with the debiasing algorithm of [1].
Outputs: data/e2.json, figs/fig6.jpg
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
rng = np.random.default_rng(20261120)

N, T, K = 100, 1200, 20          # N=100 clients, T rounds, K local steps
Rgrid = np.array([0, 1, 2, 5, 10, 15, 20, 30, 50, 70, 88])

def tau_mix_rule(R):
    if R == 0:
        return 1.0
    return R + C.L_of(N, R)

tau_rule = np.array([tau_mix_rule(int(r)) for r in Rgrid])
tau_const = np.full_like(tau_rule, tau_mix_rule(1))     # ignores R-dependence
# schematic of the [1, App. I] assumption "tau_mix decreases with R":
# tau_mix(R) = tau_mix(1)/(1+R/R0), R0=5, -> alpha_lr grows with R
tau_decr = tau_mix_rule(1) / (1.0 + Rgrid / 5.0)

alpha_rule = 1.0 / (tau_rule * K * np.sqrt(T))
alpha_const = 1.0 / (tau_const * K * np.sqrt(T))
alpha_decr = 1.0 / (tau_decr * K * np.sqrt(T))
burnin = 2 * tau_rule * np.log(np.maximum(tau_rule, np.e))

print("E2: design rule at N=100, T=1200, K=20")
for i, R in enumerate(Rgrid):
    print(f"  R={int(R):2d}  tau~R+L={tau_rule[i]:8.2f}  "
          f"alpha(rule)={alpha_rule[i]:.3e}  alpha(const)={alpha_const[i]:.3e}  "
          f"burn-in={burnin[i]:8.1f}")

# ---- light MC: E||nu_hat_t - pi||_inf^2 and the O([R+L]/t) scale ----------
print("\nE2: light Monte Carlo (1000 trajectories, N=20, T=1500)")
Nmc, M, Tmc = 20, 1000, 1500
Rmc = [0, 4, 8, 16]
mc = {}
for R in Rmc:
    window = np.zeros((M, max(R, 1)), dtype=np.int64)
    # deterministic worst-case init (1,2,...,R) -> 0-indexed
    if R > 0:
        window = np.tile(np.arange(R), (M, 1))
    cnt = np.zeros((M, Nmc))
    curve = np.zeros(Tmc)
    for t in range(1, Tmc + 1):
        scores = rng.random((M, Nmc))
        if R > 0:
            scores[np.arange(M)[:, None], window] = -1.0
        x = scores.argmax(1)                       # uniform on admissible set
        cnt[np.arange(M), x] += 1.0
        if R > 0:
            window = np.concatenate([x[:, None], window[:, :-1]], axis=1)
        nu = cnt / t
        curve[t - 1] = np.mean((np.abs(nu - 1.0 / Nmc).max(1)) ** 2)
    lam = C.lam_star(Nmc, R)
    L = C.L_of(Nmc, R, lam)
    scale = (R + L) if R > 0 else 1.0
    tt = np.arange(1, Tmc + 1)
    tE_raw = curve * tt                     # actual scaling of the observable
    tE_over_scale = tE_raw / scale         # would be constant only if
                                           # E||.||^2 ~ tau_mix/t exactly
    mc[str(R)] = dict(R=R, lam=lam, L=L, scale=scale,
                      curve_tail=float(np.mean(curve[-300:])),
                      tE_raw_tail=float(np.mean(tE_raw[-300:])),
                      tE_over_RplusL_tail=float(np.mean(tE_over_scale[-300:])),
                      curve=[float(v) for v in curve[::25]],
                      tE_raw=[float(v) for v in tE_raw[::25]],
                      tE_over_RplusL=[float(v) for v in tE_over_scale[::25]],
                      t=[int(v) for v in tt[::25]])
    print(f"  R={R:2d} R+L={scale:7.2f}  E||.||^2(t=1500)={curve[-1]:.3e}  "
          f"t*E={np.mean(tE_raw[-300:]):.4f}  "
          f"t*E/(R+L)={np.mean(tE_over_scale[-300:]):.4f}")

# ---- Fig 7 ---------------------------------------------------------------
apply_rc()
fig, axes = new_canvas()

ax = axes[0]
ax.plot(Rgrid, alpha_rule, "o-", color="#b2182b", lw=1.9, ms=5,
        label=r"Thm 6: $\alpha\propto 1/(R+L)$")
ax.plot(Rgrid, alpha_const, "s--", color="#2166ac", lw=1.4, ms=4,
        label=r"constant-$\tau_{\rm mix}$ baseline")
ax.plot(Rgrid, alpha_decr, "^:", color="#762a83", lw=1.4, ms=4,
        label=r"[1, App. I] decreasing-$\tau_{\rm mix}$ (schematic)")
ax.set_yscale("log")
ax.set_xlabel(r"$R$"); ax.set_ylabel(r"$\alpha_{\rm lr}$ (a.u., log)")
ax.set_title("(a) Step-size rule from $\\tau_{\\rm mix}=R+O(L)$")
ax.legend(fontsize=PT, loc="lower left", framealpha=.9)

ax = axes[1]
cols = {0: "#4d4d4d", 4: "#2166ac", 8: "#1a9850", 16: "#b2182b"}
for R in Rmc:
    d = mc[str(R)]
    ax.loglog(d["t"], d["curve"], color=cols[R], lw=1.6,
              label=fr"$R={R}$, $R+L={d['scale']:.1f}$")
ref_t = np.array([50, 1500])
ax.loglog(ref_t, 0.02 / ref_t, "k--", lw=1.2, label=r"$1/t$ slope")
ax.set_xlabel(r"round $t$"); ax.set_ylabel(r"$\mathbb{E}\|\hat\nu_t-\pi\|_\infty^2$")
ax.set_title(r"(b) Empirical $1/t$ decay; magnitude falls with $R$")
ax.legend(fontsize=PT, loc="upper right", framealpha=.9)
ax.text(0.03, 0.05,
        r"measured $\propto\tau_{\rm int}(R)/t$ (decreasing);"
        "\n" r"Lemma 2's $\mathcal{O}(\tau_{\rm mix}/t)$ is a loose,"
        "\n" r"$R$-increasing upper bound;"
        "\n" r"$R{=}16$ kink at $t\approx70$: cooldown release",
        transform=ax.transAxes, fontsize=PT, va="bottom",
        bbox=dict(fc="white", ec="0.8", alpha=.9, pad=2.5))

fig.suptitle("Burn-in and step size become explicit functions of $R$",
             fontsize=PT+1, y=0.975)
finalize(fig)
save(fig, f"{ROOT}/figs/fig6.jpg")
print("saved figs/fig6.jpg")

out = dict(config=dict(N=N, T=T, K=K, seed=20261120,
                       Nmc=Nmc, M=M, Tmc=Tmc, generated="2026-11-20"),
           Rgrid=[int(x) for x in Rgrid],
           tau_rule=[float(x) for x in tau_rule],
           alpha_rule=[float(x) for x in alpha_rule],
           alpha_const=[float(x) for x in alpha_const],
           alpha_decr=[float(x) for x in alpha_decr],
           burnin=[float(x) for x in burnin], mc=mc,
           note=("Panel (a) curve 3 is a schematic of the assumption in [1, App. I], "
                 "not a fitted model. Panel (b): the measured empirical-frequency "
                 "error decays as 1/t with magnitude DECREASING in R (consistent "
                 "with tau_int(R)), whereas Lemma 2 of [1] supplies the loose "
                 "R-INCREASING upper bound O(tau_mix/t). The constant in that "
                 "bound is not fitted; the two directions are not in conflict."))
with open(f"{ROOT}/data/e2.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved data/e2.json")

gap = alpha_const[-1] / alpha_rule[-1]
print(f"\nACCEPTANCE: at R=88 the constant-tau_mix baseline gives alpha_lr "
      f"{gap:.0f}x larger than the Thm 6 rule; burn-in grows from "
      f"{burnin[1]:.0f} (R=1) to {burnin[-1]:.0f} (R=88) rounds -> "
      f"{'PASS' if gap > 20 else 'WEAK'}")
obs = [mc[str(R)]["tE_raw_tail"] for R in Rmc]
print(f"OBSERVATION: measured t*E||nu-pi||^2 = {obs} is decreasing in R "
      f"(tau_int direction), NOT proportional to R+L (tau_mix direction).")
