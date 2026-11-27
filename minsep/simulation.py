"""Vectorised simulation of the R-th order minimum-separation participation chain.

The window state is the tuple of the R most recent clients.  At each round the
next client is drawn from the propensity vector p renormalised over the clients
that are not currently in the cooldown window.  Client labels are 0-based
internally; paper label j corresponds to internal label j-1.
"""
import numpy as np


def simulate(N, R, p, T, ntraj, rng):
    """Draw trajectories by renormalising p over the admissible set (O(ntraj*N))."""
    p = np.asarray(p, float); p /= p.sum()
    W = np.tile(np.arange(R, dtype=np.int64)[None, :], (ntraj, 1)) if R > 0 \
        else np.zeros((ntraj, 0), dtype=np.int64)
    X = np.empty((ntraj, T), dtype=np.int16)
    rows = np.arange(ntraj)
    for t in range(T):
        mask = np.zeros((ntraj, N), dtype=bool)
        for j in range(R):
            mask[rows, W[:, j]] = True
        wgt = (~mask) * p[None, :]
        c = np.cumsum(wgt, axis=1)
        c /= c[:, -1:]
        u = rng.random(ntraj)
        pick = (c >= u[:, None]).argmax(axis=1)
        X[:, t] = pick
        if R > 0:
            W[:, 1:] = W[:, :-1]
            W[:, 0] = pick
    return X


def simulate_fast(N, R, p, T, ntraj, rng, burn=600):
    """Rejection-sampling variant, equivalent to conditional renormalisation.

    Draws a candidate from p and rejects it if it lies in the cooldown window.
    Cost is O(ntraj*R) per step instead of O(ntraj*N); used for the long N=100
    trajectories.  ``burn`` burn-in rounds are discarded before recording X.
    """
    c = np.cumsum(p)
    W = np.tile(np.arange(R, dtype=np.int64)[None, :], (ntraj, 1)) if R > 0 \
        else np.zeros((ntraj, 0), dtype=np.int64)
    X = np.empty((ntraj, T), dtype=np.int16)
    for t in range(burn + T):
        u = rng.random(ntraj)
        pick = np.searchsorted(c, u)
        if R > 0:
            bad = (W == pick[:, None]).any(axis=1)
            while bad.any():
                k = int(bad.sum())
                pick[bad] = np.searchsorted(c, rng.random(k))
                bad = (W == pick[:, None]).any(axis=1)
        if t >= burn:
            X[:, t - burn] = pick
        if R > 0:
            W[:, 1:] = W[:, :-1]; W[:, 0] = pick
    return X


def diagnostics(X, pi, A, wstar, w0, eta_lr, N, T, ntraj, d):
    """Bias-TV, t*Var of the empirical frequency, and constant-gain linear error."""
    rows = np.arange(ntraj)
    eta = np.zeros((ntraj, N))
    w = np.tile(w0[None, :], (ntraj, 1))
    bias, tvar, sgd = np.empty(T), np.empty(T), np.empty(T)
    for t in range(T):
        eta[rows, X[:, t]] += 1.0
        nu = eta / (t + 1)
        m = nu.mean(axis=0)
        bias[t] = 0.5 * np.abs(m - pi).sum()
        tvar[t] = (t + 1) * ((nu - m) ** 2).sum(axis=1).mean()
        g = w - A[X[:, t]]
        w -= eta_lr * g
        sgd[t] = ((w - wstar) ** 2).sum(axis=1).mean()
    return bias, tvar, sgd


def per_traj_err(A, w0, R, p, seed, t_list, ntraj, T, simulate_fn, eta_lr):
    """Per-trajectory squared error at selected rounds (for error bars)."""
    rng = np.random.default_rng(seed)
    X = simulate_fn(len(p), R, p, T, ntraj, rng)
    w = np.tile(w0[None, :], (ntraj, 1))
    out = {}
    for t in range(T):
        g = w - A[X[:, t]]; w -= eta_lr * g
        if t + 1 in t_list:
            out[t + 1] = ((w - A.mean(0)) ** 2).sum(axis=1)
    return out
