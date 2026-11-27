# -*- coding: utf-8 -*-
"""S.2, S.3: TV identity of the lump, and the j>=2 spectral-block certificate."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import lam_star, LumpChain, lump_matrix, full_state_tv, check

fails = []
def ck(c, label):
    if not check(c, label): fails.append(label)

print("== S.2: TV identity |delta_w P_R^t - zeta| = |delta_a0 q^t - pi| ==")
for N, R in [(6,2),(8,3)]:
    fs = full_state_tv(N, R, R+6)
    L = LumpChain(N, R)
    h = L.run(R+6)
    err = max(abs(fs[t] - L.tv(mu)) for t, mu in enumerate(h))
    ck(err < 1e-12, "TV identity at (N,R)=(%d,%d), max err %.2e" % (N,R,err))

print("== S.3: eigenvalues with |.| > lambda*^2 are exactly the companion block ==")
for N, R in [(6,2),(7,2),(8,2),(10,2),(8,3),(9,3),(12,3),
             (8,4),(10,4),(12,4),(10,5)]:
    Q, pi, st = lump_matrix(N, R)
    if len(st) > 2500: continue
    ls = lam_star(N, R); thr = ls**2
    ev = np.linalg.eigvals(Q)
    nontr = [e for e in ev if abs(e-1) > 1e-8]
    hi = [e for e in nontr if abs(e) > thr + 1e-8]
    lo = [e for e in nontr if abs(e) <= thr + 1e-8]
    mxlo = max((abs(e) for e in lo), default=0.0)
    ck(len(hi) == R*R and mxlo <= thr + 1e-8 and abs(mxlo/thr-1) < 1e-6,
       "(%3d,%2d) D_R=%4d: %d companion eigs (R^2=%d), rest max %.6f = lambda*^2" %
       (N,R,len(st),len(hi),R*R,mxlo))

print("\nSPECTRAL: %d checks failed" % len(fails))
sys.exit(1 if fails else 0)
