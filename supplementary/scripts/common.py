# -*- coding: utf-8 -*-
"""Shared routines for the supplement of
'Client Participation under Minimum Separation: Chain-Level Constants'.

All routines are exact (Fraction / integer) where a closed form is claimed;
floating point is used only for eigenvalue certificates and is labelled so.

Range convention of the main paper: uniform availability p_i = 1/N,
1 <= R <= N-2.  R = N-1 is the periodic endpoint and is excluded.
"""
from math import perm, comb, log
from fractions import Fraction as F
from collections import defaultdict
import numpy as np

# --------------------------------------------------------------------- #
# closed forms
# --------------------------------------------------------------------- #
def roots_p(N, R):
    """Roots of p_R(lambda) = (N-R) lambda^R + lambda^{R-1} + ... + 1."""
    return np.roots([float(N - R)] + [1.0] * R)

def lam_star(N, R):
    return float(np.max(np.abs(roots_p(N, R))))

def L(N, R):
    return log(4) / abs(log(lam_star(N, R)))

def alpha0(N, R):
    """R-step Doeblin constant, exact rational."""
    return F((N - R) ** R, perm(N, R))

def Ac(N, R):
    """(1 - 1/m)^{R+1} + 1/m, exact rational, m = N-R."""
    m = N - R
    return F((m - 1) ** (R + 1), m ** (R + 1)) + F(1, m)

def chi2_0(N, R):
    return float(1 / alpha0(N, R) - 1)

def chi2_1(N, R):
    """One-step chi-square divergence, exact rational closed form."""
    return float(Ac(N, R) / alpha0(N, R) - 1)

def sigma2(N, R):
    """Theorem 2: asymptotic variance of one client's empirical frequency."""
    return (N - R) * (N - R - 1) / N ** 3

def tau_int(N, R):
    return sigma2(N, R) / ((N - 1) / N ** 2)

# --------------------------------------------------------------------- #
# lumped chain q (identity-tracking classes)
# --------------------------------------------------------------------- #
class LumpChain:
    """Lumped window chain. A class c = (c_1,...,c_R) records, at each window
    coordinate, either 0 (fresh) or the identity of an initial label a_i in
    {1,...,R}; retained labels are distinct. Only reachable classes with
    R - k_c <= m are kept."""

    def __init__(self, N, R):
        self.N, self.R, self.m = N, R, N - R
        self.B = R + 1
        self.NR = float(perm(N, R))
        self.c0 = self.enc(tuple(range(1, R + 1)))
        self._pi_cache = {}

    def enc(self, c):
        return sum(int(x) * self.B ** j for j, x in enumerate(c))

    def dec(self, z):
        c, B, R = [], self.B, self.R
        for _ in range(R):
            z, r = divmod(z, B)
            c.append(r)
        return tuple(c)

    def pi(self, z):
        if z in self._pi_cache:
            return self._pi_cache[z]
        c = self.dec(z)
        k = sum(1 for x in c if x > 0)
        j = self.R - k
        v = perm(self.m, j) / self.NR if 0 <= j <= self.m else 0.0
        self._pi_cache[z] = v
        return v

    def trans(self, z):
        c = self.dec(z)
        R, m, N = self.R, self.m, self.N
        k = sum(1 for x in c if x > 0)
        ty = N - 2 * R + k
        if ty < 0:
            return []
        used = set(x for x in c if x > 0)
        I = [a for a in range(1, R + 1) if a not in used]
        tail = c[:-1]
        out = [(self.enc((0,) + tail), ty / m)]
        out += [(self.enc((a,) + tail), 1.0 / m) for a in I]
        return [(z2, p) for z2, p in out if p > 0]

    def run(self, T, max_states=4_000_000):
        mu = {self.c0: 1.0}
        hist = [mu]
        for t in range(T):
            nx = defaultdict(float)
            for z, v in mu.items():
                for z2, p in self.trans(z):
                    nx[z2] += v * p
            mu = {k: v for k, v in nx.items() if v > 1e-18}
            hist.append(mu)
            if len(mu) > max_states:
                raise RuntimeError("support too large (%d) at t=%d" % (len(mu), t + 1))
        return hist

    def tv(self, mu):
        s_pi = s_abs = 0.0
        for z, v in mu.items():
            p = self.pi(z)
            s_pi += p
            s_abs += abs(v - p)
        return 0.5 * (s_abs + 1.0 - s_pi)

    def chi2(self, mu):
        s_pi = s = 0.0
        for z, v in mu.items():
            p = self.pi(z)
            if p <= 0:
                continue
            s_pi += p
            s += (v - p) ** 2 / p
        return s + 1.0 - s_pi

    def tau_mix(self, hist):
        for t, mu in enumerate(hist):
            if self.tv(mu) <= 0.25 + 1e-12:
                return t
        return None


def lump_states(N, R):
    m = N - R
    states = []

    def rec(c, k):
        if len(c) == R:
            if R - k <= m:
                states.append(tuple(c))
            return
        q = len(c) + 1
        used = set(x for x in c if x > 0)
        for v in [0] + [i for i in range(1, R + 1) if i not in used]:
            rec(c + [v], k + (1 if v > 0 else 0))

    rec([], 0)
    return states


def lump_matrix(N, R):
    """Dense row-stochastic transition matrix Q of the lumped chain and its
    stationary vector pi. Used only for spectral / Gelfand certificates."""
    m = N - R
    states = lump_states(N, R)
    idx = {c: i for i, c in enumerate(states)}
    n = len(states)
    Q = np.zeros((n, n))
    NR = float(perm(N, R))
    for c, i in idx.items():
        k = sum(1 for x in c if x > 0)
        ty = N - 2 * R + k
        used = set(x for x in c if x > 0)
        I = [a for a in range(1, R + 1) if a not in used]
        if ty > 0:
            Q[i, idx[(0,) + c[:-1]]] += ty / m
        for a in I:
            Q[i, idx[(a,) + c[:-1]]] += 1.0 / m
    pi = np.array([perm(m, R - sum(1 for x in c if x > 0)) / NR for c in states])
    return Q, pi, states


def full_state_tv(N, R, T):
    """Direct iteration of the (N)_R-dimensional P_R, for the TV-identity
    cross-check of Lemma A. Feasible only for small N."""
    import itertools
    states = list(itertools.permutations(range(1, N + 1), R))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    for i, w in enumerate(states):
        S = set(w)
        for x in range(1, N + 1):
            if x not in S:
                P[i, idx[(x,) + w[:-1]]] = 1.0 / (N - R)
    v = np.zeros(n)
    v[idx[tuple(range(1, R + 1))]] = 1.0
    z = np.ones(n) / n
    out = [0.5 * np.abs(v - z).sum()]
    for _ in range(T):
        v = v @ P
        out.append(0.5 * np.abs(v - z).sum())
    return out


def check(condition, label):
    print("  [%s] %s" % ("PASS" if condition else "FAIL", label))
    return bool(condition)

# --------------------------------------------------------------------- #
# exact (rational) one-step chi-square, independent of lump iteration
# --------------------------------------------------------------------- #
def chi2_1_exact(N, R):
    """E_pi[rho^2]/alpha0^2 - 1 computed by exact rational enumeration of the
    one-step admissibility set B.  This is the independent check of
    chi2_1 = A_c/alpha0 - 1; no floating point is involved."""
    m = N - R
    NR = perm(N, R)
    a0 = F(m ** R, NR)
    S = F(0)

    def rec(q, used, k):
        nonlocal S
        if q > R:
            jj = R - k
            if 0 <= jj <= m:
                ty = m - R + k
                S += F(perm(m, jj) * ty ** 2, NR * m ** 2)
            return
        choices = list(range(0, R + 1)) if q == 1 else [0] + list(range(q, R + 1))
        for v in choices:
            if v > 0 and v in used:
                continue
            rec(q + 1, used | ({v} if v > 0 else set()), k + (1 if v > 0 else 0))

    rec(1, set(), 0)
    return S / a0 ** 2 - 1


def R2_residues(N):
    """Absolute residues of P(z)/Q(z) for R=2 (four simple poles)."""
    import sympy as sp
    z = sp.symbols('z')
    m = sp.Integer(N - 2); Ns = sp.Integer(N)
    Q = (1 - (m**-2 - 2/m)*z + z**2/m**2)*(1 - z/m)*(1 - z/m**2)
    P = ((3*Ns-4)/m**2 + Ns*(Ns-3)*(Ns**2-3*Ns+3)/m**5*z
         + (Ns-3)*(Ns**3-5*Ns**2+7*Ns-4)/m**6*z**2
         - (Ns**2-2*Ns+2)/m**6*z**3)
    z1 = (-1 + sp.I*sp.sqrt(4*Ns-9))/(2*m)
    rho = [sp.expand(z1**2), sp.conjugate(sp.expand(z1**2)),
           sp.Rational(1, 1)/m, sp.Rational(1, 1)/m**2]
    out = []
    for i, r in enumerate(rho):
        zeta = sp.simplify(1/r)
        den = sp.prod([sp.simplify(1 - rj*zeta) for j, rj in enumerate(rho) if j != i])
        out.append(complex(sp.N(sp.simplify(P.subs(z, zeta)/den), 30)))
    return out

# --------------------------------------------------------------------- #
# exact (rational) one-step chi-square, independent of lump iteration
# --------------------------------------------------------------------- #
def chi2_1_exact(N, R):
    """E_pi[rho^2]/alpha0^2 - 1 computed by exact rational enumeration of the
    one-step admissibility set B.  This is the independent check of
    chi2_1 = A_c/alpha0 - 1; no floating point is involved."""
    m = N - R
    NR = perm(N, R)
    a0 = F(m ** R, NR)
    S = F(0)

    def rec(q, used, k):
        nonlocal S
        if q > R:
            jj = R - k
            if 0 <= jj <= m:
                ty = m - R + k
                S += F(perm(m, jj) * ty ** 2, NR * m ** 2)
            return
        choices = list(range(0, R + 1)) if q == 1 else [0] + list(range(q, R + 1))
        for v in choices:
            if v > 0 and v in used:
                continue
            rec(q + 1, used | ({v} if v > 0 else set()), k + (1 if v > 0 else 0))

    rec(1, set(), 0)
    return S / a0 ** 2 - 1


def R2_residues(N):
    """Absolute residues of P(z)/Q(z) for R=2 (four simple poles)."""
    import sympy as sp
    z = sp.symbols('z')
    m = sp.Integer(N - 2); Ns = sp.Integer(N)
    Q = (1 - (m**-2 - 2/m)*z + z**2/m**2)*(1 - z/m)*(1 - z/m**2)
    P = ((3*Ns-4)/m**2 + Ns*(Ns-3)*(Ns**2-3*Ns+3)/m**5*z
         + (Ns-3)*(Ns**3-5*Ns**2+7*Ns-4)/m**6*z**2
         - (Ns**2-2*Ns+2)/m**6*z**3)
    z1 = (-1 + sp.I*sp.sqrt(4*Ns-9))/(2*m)
    rho = [sp.expand(z1**2), sp.conjugate(sp.expand(z1**2)),
           sp.Rational(1, 1)/m, sp.Rational(1, 1)/m**2]
    out = []
    for i, r in enumerate(rho):
        zeta = sp.simplify(1/r)
        den = sp.prod([sp.simplify(1 - rj*zeta) for j, rj in enumerate(rho) if j != i])
        out.append(complex(sp.N(sp.simplify(P.subs(z, zeta)/den), 30)))
    return out
