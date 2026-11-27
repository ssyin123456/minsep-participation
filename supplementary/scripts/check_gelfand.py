# -*- coding: utf-8 -*-
"""S.4: Gelfand constant. Verifies the corrected epsilon choice and the
absence of a bound on C_eps uniform in (N,R)."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import lam_star, lump_matrix, check

fails = []
def ck(c, label):
    if not check(c, label): fails.append(label)

print("== S.4: the epsilon=(1-lambda*)/2 inequality in the source note is reversed ==")
for N, R in [(6,2),(20,2),(8,3),(12,4)]:
    ls = lam_star(N, R)
    lhs = abs(np.log((1+ls)/2)); rhs = 0.5*abs(np.log(ls))
    ck(lhs <= rhs + 1e-12,
       "(%3d,%2d) |ln((1+lam)/2)|=%.6f <= 0.5|ln lam|=%.6f" % (N,R,lhs,rhs))

print("== S.4: epsilon = sqrt(lambda*) - lambda* gives rate sqrt(lambda*) exactly ==")
for N, R in [(6,2),(8,3),(12,4),(100,7)]:
    ls = lam_star(N, R); eps = np.sqrt(ls) - ls
    ck(0 < eps < 1 - ls and abs((ls+eps) - np.sqrt(ls)) < 1e-12,
       "(%3d,%2d) eps=%.6f in (0,1-lam), rate=%.6f" % (N,R,eps,ls+eps))

print("== S.4: C_eps = sup_m ||qbar^m||_{1->1}/(sqrt(lambda*))^m grows with N ==")
vals = {}
for N, R in [(6,2),(8,2),(12,2),(20,2),(40,2),(80,2),
             (9,3),(12,3),(20,3),(40,3),(12,4),(20,4),(30,4)]:
    Q, pi, st = lump_matrix(N, R)
    rate = np.sqrt(lam_star(N, R))
    qbar = Q - np.ones((len(pi),1))*pi[None,:]
    M = np.eye(len(pi)); sup = 0.0
    for mm in range(1, 120):
        M = M @ qbar
        sup = max(sup, np.abs(M).sum(axis=1).max()/rate**mm)
    vals[(N,R)] = sup
    print("     (%3d,%2d) D_R=%4d  C_eps >= %8.3f" % (N,R,len(pi),sup))
ck(vals[(80,2)] > vals[(6,2)] and vals[(40,3)] > vals[(9,3)],
   "C_eps increases with N at fixed R (no uniform bound)")

print("\nGELFAND: %d checks failed" % len(fails))
sys.exit(1 if fails else 0)
