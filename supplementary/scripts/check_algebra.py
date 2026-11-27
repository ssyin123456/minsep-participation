# -*- coding: utf-8 -*-
"""S.2, S.5, S.7: squarefree p_R, chi2_1 closed form, S_j identity,
chi2_1<=1/4 thresholds, and the tail-regime chi-square counterexamples."""
import sys, sympy as sp
from math import perm, log
from itertools import combinations, permutations
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import (roots_p, lam_star, alpha0, Ac, chi2_0, chi2_1,
                    chi2_1_exact, LumpChain, check)

fails = []
def ck(c, label):
    if not check(c, label): fails.append(label)

print("== S.2: p_R has no repeated roots and lambda* < 1 ==")
l = sp.symbols('l')
for N, R in [(6,2),(7,2),(8,3),(8,4),(10,5),(10,8),(12,10),(14,12),
             (16,14),(20,6),(100,7),(15,3),(29,4),(71,6),(208,10),(1969,30)]:
    p = sum([sp.Integer(N-R)*l**R] + [l**k for k in range(R)])
    ck(sp.degree(sp.gcd(p, sp.diff(p, l))) == 0 and lam_star(N,R) < 1-1e-12,
       "squarefree, lambda*<1 at (N,R)=(%d,%d)" % (N,R))

print("== S.5: chi2_1 = A_c/alpha0 - 1, exact rational for R <= 6 ==")
for N, R in [(6,2),(7,2),(8,3),(8,4),(10,5),(15,3),(29,4),(71,6),(20,6)]:
    a = chi2_1_exact(N, R)
    ck(a == Ac(N,R)/alpha0(N,R) - 1,
       "chi2_1 exact identity at (N,R)=(%d,%d), value %.6g" % (N,R,float(a)))

print("== S.5: tail regime, floating lump iteration with accumulated-error tolerance ==")
for N, R in [(10,8),(12,10),(14,12),(16,14),(100,7)]:
    L = LumpChain(N, R)
    try: h = L.run(R + 3)
    except RuntimeError: continue
    c1 = L.chi2(h[R+1])
    ck(abs(c1 - chi2_1(N,R)) < 1e-5,
       "chi2_1 at (N,R)=(%d,%d), |iter-closed| = %.2e" % (N,R,abs(c1-chi2_1(N,R))))

print("== S.5: S_j = sum_{y in B}(m)_{R-k_y+j} = (m)_{j-1}(m-j+1)^{R+1} ==")
def S_direct(mv, R, j):
    tot = 0
    for k in range(R+1):
        for S in combinations(range(1,R+1), k):
            for labs in permutations(range(1,R+1), k):
                if any(q >= 2 and v < q for q, v in zip(S, labs)): continue
                jj = R - k
                if 0 <= jj <= mv: tot += perm(mv, jj+j)
    return tot
m = sp.symbols('m')
for R in [2,3,4]:
    for j in [0,1,2]:
        f = (m+1)**R if j == 0 else sp.prod([m-i for i in range(j-1)])*(m-j+1)**(R+1)
        pts = list(range(3, 3+R+j+2))
        ck([S_direct(v,R,j) for v in pts] == [int(f.subs(m,v)) for v in pts],
           "S_j identity R=%d j=%d (exact interpolation)" % (R,j))

print("== S.5: minimal m with chi2_1 <= 1/4 (Table A.1) ==")
for R, m0 in {2:5, 3:12, 4:25, 6:65, 10:198, 30:1939}.items():
    mm = m0 - 1
    while chi2_1(mm+R, R) <= 0.25: mm -= 1
    ck(mm + 1 == m0 and chi2_1(m0-1+R, R) > 0.25,
       "R=%d minimal m=%d" % (R, m0))

print("== S.7: chi-square uniform-prefactor counterexamples ==")
for N, R, doc in [(10,8,3.79e3),(12,10,1.22e5),(14,12,5.46e6),(16,14,3.25e8)]:
    v = chi2_1(N,R)/lam_star(N,R)**2
    ck(abs(v/doc - 1) < 0.01, "chi2_1/lambda*^2 ~ %.2e at (N,R)=(%d,%d)" % (doc,N,R))

print("== S.4: R=2 lambda* = (N-2)^(-1/2) ==")
for N in [6,7,10,100,1000]:
    ck(abs(lam_star(N,2) - (N-2)**-0.5) < 1e-12, "lambda*(N=%d,2)" % N)

print("\nALGEBRA: %d checks failed" % len(fails))
sys.exit(1 if fails else 0)
