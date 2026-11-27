# -*- coding: utf-8 -*-
"""S.5: Theorem 7 (main regime), the maximal-coupling bound, and the R=2
generating function / residue constant 1.4492."""
import sys, sympy as sp
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import lam_star, chi2_1, LumpChain, R2_residues, check

fails = []
def ck(c, label):
    if not check(c, label): fails.append(label)

print("== S.5: main regime chi2_1 <= 1/4 implies tau_mix <= R+1 ==")
for N, R in [(7,2),(15,3),(29,4),(71,6)]:
    L = LumpChain(N, R)
    h = L.run(R+4)
    tau = L.tau_mix(h)
    ck(chi2_1(N,R) <= 0.25 and tau is not None and tau <= R+1 and tau >= R,
       "(%3d,%2d) chi2_1=%.5f, tau_mix=%s in [R, R+1]" % (N,R,chi2_1(N,R),tau))

print("== S.5: m >= 4R^2 implies tau_mix <= 2R (maximal coupling) ==")
for R in [2,3,4,5,6]:
    N = 4*R*R + R
    L = LumpChain(N, R)
    h = L.run(2*R+2)
    tau = L.tau_mix(h)
    ck(tau is not None and tau <= 2*R,
       "R=%d N=%d m=4R^2: tau_mix=%s <= 2R=%d" % (R,N,tau,2*R))

print("== S.5: R=2 generating function P/Q, residues, sup sum|a_k| = 1.4492 ==")
z = sp.symbols('z')
def PQ(N):
    m = sp.Integer(N-2); Ns = sp.Integer(N)
    Q = (1-(m**-2-2/m)*z+z**2/m**2)*(1-z/m)*(1-z/m**2)
    P = ((3*Ns-4)/m**2 + Ns*(Ns-3)*(Ns**2-3*Ns+3)/m**5*z
         + (Ns-3)*(Ns**3-5*Ns**2+7*Ns-4)/m**6*z**2
         - (Ns**2-2*Ns+2)/m**6*z**3)
    return sp.simplify(P), sp.simplify(Q)

def residues(N):
    P, Q = PQ(N)
    m = sp.Integer(N-2); Ns = sp.Integer(N)
    z1 = (-1+sp.I*sp.sqrt(4*Ns-9))/(2*m)
    rho = [sp.expand(z1**2), sp.conjugate(sp.expand(z1**2)),
           sp.Rational(1,1)/m, sp.Rational(1,1)/m**2]
    out = []
    for i, r in enumerate(rho):
        zeta = sp.simplify(1/r)
        den = sp.prod([sp.simplify(1-rj*zeta) for j, rj in enumerate(rho) if j != i])
        out.append(complex(sp.N(sp.simplify(P.subs(z,zeta)/den), 30)))
    return out

for N in [6,7,10]:
    P, Q = PQ(N)
    ser = sp.series(P/Q, z, 0, 5).removeO()
    coeff = [float(sp.N(ser.coeff(z,k),20)) for k in range(5)]
    L = LumpChain(N, 2)
    h = L.run(2+6+2)
    cs = [L.chi2(h[2+mm]) for mm in range(5)]
    ck(max(abs(coeff[k]-cs[k]) for k in range(5)) < 1e-10,
       "N=%d: P/Q series equals chi2_0..chi2_4" % N)
S = {N: sum(abs(x) for x in R2_residues(N)) for N in [4,5,6,7,8,10,20,100,1000]}
ck(abs(S[6]-1.449219) < 1e-5 and abs(S[1000]-1.0) < 2e-3
   and S[6] == max(S[N] for N in S if N >= 6),
   "sup_{N>=6} sum|a_k| = %.6f at N=6, -> 1" % S[6])
print("     small-N boundary: N=4 -> %.6f, N=5 -> %.6f (larger than 1.4492)" % (S[4],S[5]))

print("\nCHI2: %d checks failed" % len(fails))
sys.exit(1 if fails else 0)
