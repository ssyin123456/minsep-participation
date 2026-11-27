# -*- coding: utf-8 -*-
"""run_e4_bootstrap.py --- E4 (large-compute part).

Replaces the two weak uncertainty statements in the manuscript:
  * Fig.2(b): "error bars of one standard deviation across trajectories"
              -> 95% bootstrap CI of the steady-state quadratic error
  * Fig.4  : "error bars giving the range over the four checkpoints"
              -> 95% bootstrap CI over independent trajectory groups

Two sub-experiments:
  E4-A  N=20 uniform, quadratic estimation error (Sec. 6.2), eta=0.04
  E4-B  N=100 long-tailed p_i ~ i^-1.5, aggregate asymptotic variance (Sec. 6.3)

Usage
-----
  python run_e4_bootstrap.py --config configs/e4.yaml                 # full
  python run_e4_bootstrap.py --config configs/e4.yaml --smoke         # 2-min check
  python run_e4_bootstrap.py --config configs/e4.yaml --only e4a      # split
  python run_e4_bootstrap.py --config configs/e4.yaml --backend numpy  # CPU-only

Writes: data/e4.json
"""
import argparse, json, os, sys, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backend import get_backend
from core import (MinSepSampler, scatter_add_row, bootstrap_ci,
                  group_bootstrap_ci, prop4_bracket, longtail_p,
                  sigma2, tau_int, lam_star, L_of)


def load_cfg(path, smoke):
    with open(path) as f:
        c = yaml.safe_load(f)
    if smoke:
        c["e4a"]["trajectories"] = 600
        c["e4a"]["T"] = 200
        c["e4a"]["checkpoints"] = [11, 20, 40, 200]
        c["e4a"]["bootstrap_B"] = 300
        c["e4b"]["trajectories"] = 600
        c["e4b"]["T"] = 400
        c["e4b"]["burn_in"] = 60
        c["e4b"]["checkpoints"] = [200, 300, 350, 400]
        c["e4b"]["groups"] = 12
        c["e4b"]["bootstrap_B"] = 300
        print("[smoke] reduced grid active -- NOT for publication")
    return c


# ---------------------------------------------------------------------------
# E4-A : quadratic estimation error, Fig. 2(b)
# ---------------------------------------------------------------------------
def run_e4a(be, c, seed):
    N = int(c["N"]); Rset = list(c["R"]); eta = float(c["eta"])
    d = int(c["d"]); T = int(c["T"]); M = int(c["trajectories"])
    cps = sorted(set(int(x) for x in c["checkpoints"]))
    B = int(c["bootstrap_B"])
    print(f"\n[E4-A] N={N} R={Rset} eta={eta} d={d} T={T} M={M} B={B}")

    # fixed sensor observations a_k ~ N(0,I_d); seed fixed -> one instance
    rng = np.random.default_rng(seed + 11)
    a = rng.standard_normal((N, d)).astype(np.float32)
    a_avg = a.mean(0)
    f = a - a_avg                                  # (N,d), centered under pi=1/N
    C0 = float(np.mean(np.sum(f * f, axis=1)))     # C_0(f) = (1/N) sum ||f_k||^2
    w0 = np.full(d, 3.0, dtype=np.float32)         # manuscript: w_0 = 3*1

    a_t = be.asarray(a, dtype=be.rand((1,)).dtype)
    f_t = be.asarray(f, dtype=be.rand((1,)).dtype)
    aavg_t = be.asarray(a_avg, dtype=be.rand((1,)).dtype)

    out = {}
    base_mean = None
    for R in Rset:
        t0 = time.time()
        smp = MinSepSampler(be, N, R, M, p=None, init="worst")
        u = be.asarray(np.tile(w0 - a_avg, (M, 1)))
        vals = {}
        for t in range(1, T + 1):
            X = smp.step()
            u = (1 - eta) * u + eta * f_t[X]
            if t in cps:
                err = u.pow(2).sum(1) if be.kind == "torch" else (u * u).sum(1)
                vals[t] = be.to_numpy(err).astype(np.float64)
        ci = bootstrap_ci(be, vals[cps[-1]], B=B, chunk=100, alpha=0.05,
                          seed=seed + 1000 + R)
        bracket = prop4_bracket(N, R, eta)
        analytic_abs = eta / (2 - eta) * C0 * bracket
        row = dict(R=R, analytic_bracket=bracket, analytic_abs=analytic_abs,
                   analytic_C0=C0, checkpoints={str(k): float(np.mean(v))
                                                for k, v in vals.items()},
                   final_checkpoint=cps[-1], **ci)
        if base_mean is None:
            base_mean = ci["mean"]
        row["measured_ratio"] = ci["mean"] / base_mean
        row["analytic_ratio"] = bracket
        out[str(R)] = row
        print(f"  R={R:2d}  mean={ci['mean']:.6f}  95%CI=[{ci['lo']:.6f},"
              f"{ci['hi']:.6f}]  analytic={analytic_abs:.6f}  "
              f"ratio={row['measured_ratio']:.5f} ({time.time()-t0:.1f}s)")
    return out, C0


# ---------------------------------------------------------------------------
# E4-B : aggregate asymptotic variance under long-tailed p, Fig. 4
# ---------------------------------------------------------------------------
def run_e4b(be, c, seed):
    N = int(c["N"]); Rset = list(c["R"]); T = int(c["T"])
    burn = int(c["burn_in"]); M = int(c["trajectories"])
    G = int(c["groups"]); B = int(c["bootstrap_B"])
    cps = sorted(set(int(x) for x in c["checkpoints"]))
    p = longtail_p(N, float(c["exponent"]))
    Z0 = float(1.0 - np.dot(p, p))
    assert M % G == 0, "trajectories must be divisible by groups"
    m = M // G
    print(f"\n[E4-B] N={N} R={Rset} p~i^-{c['exponent']} T={T} burn={burn} "
          f"M={M} groups={G}x{m} B={B}")
    print(f"  analytic sum sigma^2(0) = Z = {Z0:.6f}")

    out = {}
    for R in Rset:
        t0 = time.time()
        smp = MinSepSampler(be, N, R, M, p=p, init="worst")
        smp.burn_in(burn)
        cnt = be.zeros((M, N))
        # t*trVar(nu_hat_t) = sum_i (E[cnt_i^2] - E[cnt_i]^2) / t.
        # Grouped so a cross-trajectory variance still gets a bootstrap CI.
        group_est = {k: [] for k in cps}
        for t in range(1, T + 1):
            X = smp.step()
            scatter_add_row(cnt, X)
            if t in cps:
                # reshape to (G, m, N); tr Var over clients, mean within group
                c3 = cnt.reshape(G, m, N)
                cbar = c3.mean(1)                    # (G,N)
                c2bar = (c3 * c3).mean(1)           # (G,N)
                est = (c2bar - cbar * cbar).sum(1) / t   # (G,)
                group_est[t].append(be.to_numpy(est).astype(np.float64))
        # one group estimate = average over the checkpoints (steady-state mean)
        per_group = np.mean(np.stack([group_est[k][0] for k in cps], axis=0), axis=0)
        ci = group_bootstrap_ci(be, per_group, B=B, alpha=0.05)
        row = dict(R=R, Z0=Z0, group_checkpoint_means={
            str(k): float(np.mean(group_est[k][0])) for k in cps}, **ci)
        out[str(R)] = row
        print(f"  R={R:2d}  sum_sigma2={ci['mean']:.5f}  95%CI="
              f"[{ci['lo']:.5f},{ci['hi']:.5f}]  "
              f"R=0 calibration={ci['mean']/Z0 if R==0 else float('nan'):.4f} "
              f"({time.time()-t0:.1f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs/e4.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=["e4a", "e4b", "all"], default="all")
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--trajectories", type=int, default=None,
                    help="override both M values")
    args = ap.parse_args()

    c = load_cfg(args.config, args.smoke)
    if args.trajectories:
        c["e4a"]["trajectories"] = args.trajectories
        c["e4b"]["trajectories"] = args.trajectories
    seed = int(c.get("seed", 20261120))
    be = get_backend(args.backend, seed)
    print("backend:", be.info())

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    res = dict(config=c, backend=be.info(), generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               smoke=args.smoke)
    if args.only in ("e4a", "all"):
        res["e4a"], res["e4a_C0"] = run_e4a(be, c["e4a"], seed)
    if args.only in ("e4b", "all"):
        res["e4b"] = run_e4b(be, c["e4b"], seed)

    path = os.path.join(HERE, "data", "e4_smoke.json" if args.smoke else "e4.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nsaved {path}")

    # ---- acceptance (only meaningful on the full run) ---------------------
    if not args.smoke:
        print("\n=== ACCEPTANCE (full run) ===")
        if "e4a" in res:
            a = res["e4a"]
            r16 = a.get("16", a.get(str(max(int(k) for k in a))))
            print(f"  E4-A ratio at largest R: measured={r16['measured_ratio']:.5f} "
                  f"analytic={r16['analytic_ratio']:.6f} "
                  f"(target 0.168944 for R=16,eta=0.04)")
            for k, v in a.items():
                inside = v["lo"] <= v["analytic_abs"] <= v["hi"]
                print(f"    R={k:>2}: analytic inside 95% CI ? {inside}")
        if "e4b" in res:
            b = res["e4b"]
            e0 = b.get("0", b.get(str(min(int(k) for k in b))))
            print(f"  E4-B R=0 calibration: {e0['mean']:.5f} vs Z={e0['Z0']:.5f} "
                  f"(inside CI ? {e0['lo'] <= e0['Z0'] <= e0['hi']})")


if __name__ == "__main__":
    main()
