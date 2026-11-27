# -*- coding: utf-8 -*-
"""E4a (light part of E4) --- verify the R=1 aggregate-variance criterion
against an exact Poisson solve, then measure the Dirichlet violation rate.

Criterion (O1 / Sec. 6.3), with Z=1-sum p^2, pi_w=p_w(1-p_w)/Z:
    A   = 1 - 2 sum_w pi_w p_w
    s2  = sum_w pi_w^2
    T   = sum_w pi_w^2 p_w
    sum_i sigma_i^2(1) = A(1+s2) - 2 s2 + 4 T
    sum_i sigma_i^2(0) = Z
    "aggregate decreases" iff the former <= the latter.

The exact reference solves (I-P_1+1 pi^T) h_i = f_i for every client i on the
N-state kernel P_1(w,x)=p_x/(1-p_w).

Outputs: data/e4a.json   (E4 bootstrap CIs are in experiments/data/e4.json)
"""
import json, os, sys
import numpy as np

ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(f"{ROOT}/data", exist_ok=True)
rng = np.random.default_rng(20261120)


def criterion(p):
    p = np.asarray(p, float)
    Z = 1.0 - np.dot(p, p)
    pi = p * (1 - p) / Z
    A = 1.0 - 2.0 * np.dot(pi, p)
    s2 = np.dot(pi, pi)
    T = np.dot(pi * pi, p)
    s1 = A * (1 + s2) - 2 * s2 + 4 * T
    s0 = Z
    return s1, s0


def exact_aggregate(p):
    """sum_i sigma_i^2(1) by solving the Poisson equation on the N-state
    previous-client chain, independently of the closed-form criterion."""
    p = np.asarray(p, float)
    N = len(p)
    P = np.zeros((N, N))
    for w in range(N):
        denom = 1.0 - p[w]
        for x in range(N):
            if x != w:
                P[w, x] = p[x] / denom
    Z = 1.0 - np.dot(p, p)
    pi = p * (1 - p) / Z
    # stationary check
    assert np.max(np.abs(pi @ P - pi)) < 1e-10
    M = np.eye(N) - P + np.ones((N, 1)) * pi.reshape(1, -1)
    total = 0.0
    for i in range(N):
        # centered observable is fbar_i(w) = 1{w=i} - pi_i  (CONSTANT pi_i),
        # NOT 1{w=i} - pi_w. Using pi_w here silently breaks stationarity.
        f = np.full(N, -pi[i])
        f[i] += 1.0
        assert abs(np.dot(pi, f)) < 1e-10
        h = np.linalg.solve(M, f)
        total += 2.0 * np.dot(pi, f * h) - np.dot(pi, f * f)
    return float(total)


print("E4a: verify criterion against exact Poisson solve")
maxerr = 0.0
for N in [3, 5, 8, 12, 20, 50]:
    for trial in range(6):
        p = rng.dirichlet(np.ones(N) * (0.3 + 0.4 * trial))
        p = np.clip(p, 1e-9, None)
        p /= p.sum()
        s1, s0 = criterion(p)
        e = exact_aggregate(p)
        maxerr = max(maxerr, abs(s1 - e))
print(f"  max |criterion - exact Poisson| over 36 profiles = {maxerr:.3e}")

# skewed instance of Proposition 1 (N=5) and a long tail (N=100)
p_skew = np.array([1/100, 24/25, 1/100, 1/100, 1/100])
s1, s0 = criterion(p_skew)
print(f"  Proposition-1 profile: sum sigma^2(0)={s0:.6f}  "
      f"sum sigma^2(1)={s1:.6f}  increases={s1 > s0}")
i = np.arange(1, 101)
p_tail = i ** (-1.5); p_tail /= p_tail.sum()
s1, s0 = criterion(p_tail)
print(f"  long tail N=100:      sum sigma^2(0)={s0:.6f}  "
      f"sum sigma^2(1)={s1:.6f}  decreases={s1 <= s0}")

print("\nE4a: Dirichlet violation rate (20000 profiles per alpha)")
res = {}
for N in [5, 20, 100]:
    for alpha in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        M = 20000
        pmat = rng.dirichlet(np.ones(N) * alpha, size=M)
        Z = 1.0 - np.einsum('ij,ij->i', pmat, pmat)
        pi = pmat * (1 - pmat) / Z[:, None]
        A = 1.0 - 2.0 * np.einsum('ij,ij->i', pi, pmat)
        s2 = np.einsum('ij,ij->i', pi, pi)
        T = np.einsum('ij,ij->i', pi * pi, pmat)
        s1 = A * (1 + s2) - 2 * s2 + 4 * T
        viol = float(np.mean(s1 > Z))
        res[f"N{N}_a{alpha}"] = dict(N=N, alpha=alpha, profiles=M,
                                    violation_rate=viol,
                                    mean_ratio=float(np.mean(s1 / Z)))
        print(f"  N={N:3d} alpha={alpha:3.1f}: violate={viol:.4f}  "
              f"mean s1/s0={np.mean(s1/Z):.4f}")

out = dict(config=dict(seed=20261120, profiles_per_setting=20000,
                       generated="2026-11-20"),
           poisson_check=dict(max_abs_error=maxerr, profiles_checked=36),
           prop1_profile=dict(p=[float(x) for x in p_skew],
                              s0=float(criterion(p_skew)[1]),
                              s1=float(criterion(p_skew)[0])),
           longtail_N100=dict(s0=float(criterion(p_tail)[1]),
                              s1=float(criterion(p_tail)[0])),
           dirichlet=res,
           note=("The violation rate depends strongly on the Dirichlet "
                 "concentration alpha; report the measured value, not a "
                 "universal '16%'. Only the R=0 -> R=1 transition is tested."))
with open(f"{ROOT}/data/e4a.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved data/e4a.json")
print(f"\nACCEPTANCE: criterion verified to {maxerr:.2e} -> "
      f"{'PASS' if maxerr < 1e-8 else 'FAIL'}")
