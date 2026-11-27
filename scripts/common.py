# -*- coding: utf-8 -*-
"""
common.py --- shared numerical core for experiments E1/E2/E3.

All results are exact / analytic; no Monte Carlo in this module.

Conventions (identical to the manuscript):
  lambda_star(N,R) : largest modulus of the R roots of
                     (N-R)*l^R + l^(R-1) + ... + l + 1 = 0
  L(N,R)           : ln 4 / |ln lambda_star|
  tau_int(N,R)     : (N-R)(N-R-1) / [N(N-1)]
  Prop4 ratio      : E||u||^2(R) / E||u||^2(0), C0(f) cancels
  lump TV          : exact d_TV(P_R^t, zeta_R) from a worst-case
                     deterministic start, via a 2^R-state bit-mask chain
"""
import numpy as np
from math import lgamma

# ----------------------------------------------------------------------------
# 1. Spectral radius of the companion block
# ----------------------------------------------------------------------------
def lam_star(N, R, high_prec=False):
    """Largest root modulus of p_R. Equivalent polynomial after removing the
    l=1 spurious root: (N-R) l^(R+1) - (N-R-1) l^R - 1 = 0."""
    if R <= 0:
        return 0.0
    if high_prec or R > 40:
        return _lam_star_mp(N, R)
    c = np.zeros(R + 2, dtype=float)
    c[0] = N - R            # l^(R+1)
    c[1] = -(N - R - 1)    # l^R
    c[-1] = -1.0           # constant
    r = np.roots(c)
    r = r[np.abs(r - 1.0) > 1e-7]
    return float(np.max(np.abs(r)))


def _lam_star_mp(N, R, dps=60):
    import mpmath as mp
    mp.mp.dps = dps
    # roots of (N-R) l^(R+1) - (N-R-1) l^R - 1, descending coefficients
    coeffs = [mp.mpf(N - R), -mp.mpf(N - R - 1)] + [mp.mpf(0)] * (R - 1) + [mp.mpf(-1)]
    try:
        rr = mp.polyroots(coeffs, maxsteps=200, error=False)
    except Exception:
        rr = mp.polyroots(coeffs, maxsteps=500)
    rr = [z for z in rr if abs(z - 1) > mp.mpf(10) ** (-dps // 2)]
    return float(max(abs(z) for z in rr))


def L_of(N, R, lam=None):
    if lam is None:
        lam = lam_star(N, R)
    if lam <= 0.0 or lam >= 1.0:
        return 0.0
    return np.log(4.0) / abs(np.log(lam))


def tau_int(N, R):
    """Normalized integrated autocorrelation time, uniform availability."""
    return (N - R) * (N - R - 1) / (N * (N - 1))


def sigma2(N, R):
    return (N - R) * (N - R - 1) / N ** 3


def cov0(N):
    return (N - 1) / N ** 2


# ----------------------------------------------------------------------------
# 2. Covariance sequence delta_k (uniform availability) and Prop 4
# ----------------------------------------------------------------------------
def delta_seq(N, R, M):
    """delta_0..delta_M. delta_0=(N-1)/N, delta_1..R=-1/N,
    delta_k = -(1/(N-R)) sum_{j=1}^R delta_{k-j} for k>R."""
    d = np.zeros(M + 1)
    d[0] = (N - 1) / N
    for k in range(1, min(R, M) + 1):
        d[k] = -1.0 / N
    for k in range(R + 1, M + 1):
        d[k] = -d[k - R:k].sum() / (N - R)
    return d


def G_of_z(N, R, z):
    if abs(1 - z) < 1e-14:
        # limit z->1 via l'Hopital handled by series; use small offset
        z = 1 - 1e-9
    A = z * (1 - z ** R) / (1 - z)
    return ((N - R) * (1 - z ** R) + A - R * z ** R) / ((N - R) + A)


def prop4_bracket(N, R, eta):
    """The bracket [1 - 2 kappa G(kappa)/((N-1)(1-kappa))] of Proposition 4.
    For R=0, G=0 and the bracket is 1. E||u||^2 = eta/(2-eta) C0(f) * bracket."""
    if R == 0:
        return 1.0
    kappa = 1.0 - eta
    G = G_of_z(N, R, kappa)
    return 1.0 - 2.0 * kappa * G / ((N - 1) * (1.0 - kappa))


def prop4_ratio(N, R, eta):
    """E||u||^2(R)/E||u||^2(0); the common factor eta/(2-eta)*C0 cancels."""
    return prop4_bracket(N, R, eta)


def prop4_ratio_by_series(N, R, eta, M=4000):
    """Independent verification of prop4_ratio by truncating the covariance
    series, WITHOUT using the generating function G(z)."""
    kappa = 1.0 - eta
    d = delta_seq(N, R, M)
    m = np.arange(M + 1)
    w = kappa ** m
    s = float(np.dot(w[1:], d[1:]))          # sum_{m>=1} kappa^m delta_m
    bracket = 1.0 + (2.0 * N / (N - 1)) * s
    return bracket


def zero_freq_ratio(N, R):
    """tau_int(R)/tau_int(0) = sigma2(R)/sigma2(0), the eta->0 prediction."""
    return tau_int(N, R)


# ----------------------------------------------------------------------------
# 3. Exact worst-case TV distance
#    (a) direct_tv        : sparse iteration of the full P_R (small N)
#    (b) correct_lump_tv  : identity-tracking lump, D_R states, N-independent
#
#    A previous "bit-mask" lump that only recorded special-vs-fresh was WRONG:
#    it merges (J,phi=a_1) with (J,phi=a_2), but from the deterministic start the
#    a_i identities are NOT exchangeable, so merging shrinks the L1 distance and
#    underestimates d(t) (e.g. (10,4) gave tau=5 instead of 7). The lump must
#    track WHICH a_j occupies each window coordinate, as in Sec. V-H.
# ----------------------------------------------------------------------------
import itertools
import scipy.sparse as sp


def direct_tv(N, R, t_max=240, tol=1e-12):
    """Exact d(t) by iterating the full window chain P_R (uniform p), from the
    deterministic worst-case start w=(0,1,...,R-1). Feasible for (N)_R <= 3e6."""
    states = list(itertools.permutations(range(N), R))
    idx = {s: k for k, s in enumerate(states)}
    n = len(states)
    rows, cols, vals = [], [], []
    for k, s in enumerate(states):
        forb = set(s)
        adm = [x for x in range(N) if x not in forb]
        for x in adm:
            rows.append(k)
            cols.append(idx[(x,) + s[:-1]])
            vals.append(1.0 / len(adm))
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    pi = np.ones(n) / n
    v = np.zeros(n)
    v[0] = 1.0
    ds = []
    for t in range(t_max + 1):
        ds.append(float(0.5 * np.abs(v - pi).sum()))
        if ds[-1] < tol and t > R:
            for _ in range(6):
                v = v @ P
                ds.append(float(0.5 * np.abs(v - pi).sum()))
            break
        v = v @ P
    return np.array(ds), n


def _pc_int(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def correct_lump_tv(N, R, t_max=240, tol=1e-12):
    """Exact d(t) via the identity-tracking lump of Sec. V-H.

    State c=(c_1,...,c_R), c_j in {0,1,...,R}: c_j=j means label a_j sits in
    window coordinate j (j=1 is newest), c_j=0 means a fresh (F) label. Legal
    states have distinct nonzero digits. Number of legal states is
        D_R = sum_k C(R,k) (R)_k,
    independent of N -- this is why it works at N=100 for small R.

    Transition (admissible set has size N-R, of which R-k are special labels
    and N-2R+k are fresh):
        c -> (0, c_1,...,c_{R-1})          w.p. (N-2R+k)/(N-R)
        c -> (j, c_1,...,c_{R-1})          w.p. 1/(N-R), for each admissible a_j
    Stationary weight of a legal state with k special labels:
        pi(c) = (N-R)_{R-k} / (N)_R .
    """
    B = R + 1
    S = B ** R
    powB = B ** np.arange(R, dtype=np.int64)
    codes = np.arange(S, dtype=np.int64)
    digs = (codes[:, None] // powB[None, :]) % B          # (S,R), digs[:,j-1]=v_j
    nz = digs > 0
    or_bits = np.bitwise_or.reduce(np.where(nz, (1 << digs), 0), axis=1)
    k = np.zeros(S, dtype=np.int16)
    uniq = np.unique(or_bits)
    for b in uniq:
        k[or_bits == b] = _pc_int(int(b))
    legal = (k == nz.sum(1))

    # stationary weights by k
    logw = np.full(R + 1, -np.inf)
    for kk in range(R + 1):
        a = N - 2 * R + kk + 1
        if a > 0:
            logw[kk] = (lgamma(N - R + 1) - lgamma(a)
                        - (lgamma(N + 1) - lgamma(N - R + 1)))
    w = np.exp(logw - np.max(logw[np.isfinite(logw)]))
    w[~np.isfinite(logw)] = 0.0
    pi = np.zeros(S)
    pi[legal] = w[np.clip(k[legal], 0, R)]
    pi[~legal] = 0.0
    pi /= pi.sum()

    p_star = np.zeros(S)
    p_star[legal] = np.clip((N - 2 * R + k[legal]) / (N - R), 0.0, 1.0)
    p_digit = 1.0 / (N - R)
    base = (codes % (B ** (R - 1))) * B                    # shift right, free slot 0

    code0 = sum(j * B ** (j - 1) for j in range(1, R + 1))  # c=(1,2,...,R)
    v = np.zeros(S)
    v[code0] = 1.0
    ds = []
    legal_idx = np.where(legal)[0]
    for t in range(t_max + 1):
        d = float(0.5 * np.abs(v - pi).sum())
        ds.append(d)
        if d < tol and t > R:
            for _ in range(6):
                v = _lump_step(v, legal_idx, base, p_star, p_digit, or_bits, R, B, S)
                ds.append(float(0.5 * np.abs(v - pi).sum()))
            break
        v = _lump_step(v, legal_idx, base, p_star, p_digit, or_bits, R, B, S)
    return np.array(ds), int(legal.sum())


def _lump_step(v, legal_idx, base, p_star, p_digit, or_bits, R, B, S):
    vn = np.bincount(base[legal_idx], weights=v[legal_idx] * p_star[legal_idx],
                     minlength=S)
    for j in range(1, R + 1):
        sel = legal_idx[(or_bits[legal_idx] & (1 << j)) == 0]
        if sel.size:
            vn += np.bincount(base[sel] + j, weights=v[sel] * p_digit, minlength=S)
    return vn


def tau_mix_from_ds(ds):
    hit = np.where(np.array(ds) <= 0.25)[0]
    return int(hit[0]) if len(hit) else None


def C_certificate(N, R, ds=None, t_max=240, method="lump"):
    lam = lam_star(N, R)
    n_states = None
    if ds is None:
        if method == "direct":
            ds, n_states = direct_tv(N, R, t_max=t_max)
        else:
            ds, n_states = correct_lump_tv(N, R, t_max=t_max)
    ratios = []
    for m in range(1, len(ds) - R):
        if ds[R + m] > 1e-11:
            ratios.append(ds[R + m] / lam ** m)
    sup = max(ratios) if ratios else 0.0
    upper = R + L_of(N, R, lam) + np.log(max(sup, 1.0)) / abs(np.log(lam))
    return dict(lam=lam, L=L_of(N, R, lam), C=sup,
                tau_mix=tau_mix_from_ds(ds), R_plus_L=R + L_of(N, R, lam),
                upper_bound=upper, n_states=n_states, method=method)
