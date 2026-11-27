# -*- coding: utf-8 -*-
"""core.py --- vectorised minimum-separation sampler + analytic constants.

Sampling rule (identical to Sec. 3.1 of the manuscript):
    at window w, draw X from p conditioned on [N] \\ w:
        P(X=x | w) = p_x / sum_{y not in w} p_y .
Vectorised with a Gumbel-max trick, which is EXACT (not an approximation):
    uniform p : argmax_{x not in w} U_x          ~ Uniform(admissible)
    general p : argmax_{x not in w} [log p_x+G_x] ~ p restricted to admissible
with G_x = -log(-log U'_x) i.i.d. standard Gumbel.
"""
import numpy as np

# ---------------------------------------------------------------------------
# analytic constants (closed forms of the manuscript, CPU / numpy only)
# ---------------------------------------------------------------------------
def lam_star(N, R):
    if R <= 0:
        return 0.0
    c = np.zeros(R + 2)
    c[0] = N - R
    c[1] = -(N - R - 1)
    c[-1] = -1.0
    r = np.roots(c)
    r = r[np.abs(r - 1.0) > 1e-7]
    return float(np.max(np.abs(r)))


def L_of(N, R):
    lam = lam_star(N, R)
    if lam <= 0 or lam >= 1:
        return 0.0
    return float(np.log(4.0) / abs(np.log(lam)))


def tau_int(N, R):
    return (N - R) * (N - R - 1) / (N * (N - 1))


def sigma2(N, R):
    """per-client asymptotic variance, uniform availability (Theorem 2)"""
    return (N - R) * (N - R - 1) / N ** 3


def prop4_bracket(N, R, eta):
    """Proposition 4: E||u||^2 = eta/(2-eta) * C0(f) * bracket."""
    if R == 0:
        return 1.0
    k = 1.0 - eta
    if abs(1 - k) < 1e-14:
        k = 1 - 1e-9
    A = k * (1 - k ** R) / (1 - k)
    G = ((N - R) * (1 - k ** R) + A - R * k ** R) / ((N - R) + A)
    return 1.0 - 2.0 * k * G / ((N - 1) * (1.0 - k))


def longtail_p(N, exponent=1.5):
    i = np.arange(1, N + 1, dtype=float)
    p = i ** (-exponent)
    return p / p.sum()


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------
class MinSepSampler:
    """Vectorised sampler over M independent trajectories.

    window : (M,R) int64, initialised to the deterministic worst-case window
             (0,1,...,R-1) tiled M times, as in Secs. 6.2/6.3 of the manuscript.
    """

    def __init__(self, be, N, R, M, p=None, init="worst", chunk=None):
        self.be = be
        self.N, self.R, self.M = int(N), int(R), int(M)
        self.chunk = chunk or M
        xp = be.xp
        self.logp = None
        if p is not None:
            p = np.asarray(p, dtype=np.float64)
            p = np.maximum(p, 1e-300)
            p = p / p.sum()
            self.logp = be.asarray(np.log(p).astype(np.float32))
        if init == "worst":
            w = np.tile(np.arange(self.R, dtype=np.int64), (self.M, 1))
            self.window = be.asarray(w)
        elif init == "random":
            # distinct labels per row, via a random permutation of [N]
            w = np.argsort(be.to_numpy(be.rand((self.M, self.N))), axis=1)[:, :self.R]
            self.window = be.asarray(np.asarray(w, dtype=np.int64))
        else:
            raise ValueError("init must be 'worst' or 'random'")
        self._dtype_i = xp.int64
        self._dtype_f = be.rand((1,)).dtype

    def step(self):
        """Return X (M,) int64 and advance every trajectory by one round."""
        be = self.be
        xp = be.xp
        M, N, R = self.M, self.N, self.R
        U = be.rand((M, N))
        # avoid log(0) / log(1) in the Gumbel transform
        U = be.minimum(be.maximum(U, be.asarray(np.float32(1e-7))),
                       be.asarray(np.float32(1 - 1e-7)))
        if self.logp is None:
            scores = U                      # argmax of iid uniforms is uniform
        else:
            G = -be.log(-be.log(U))        # standard Gumbel
            scores = self.logp[None, :] + G
        if R > 0:
            # mask the R currently-cooling labels: set to a very negative value
            if be.kind == "torch":
                scores = scores.clone()
                scores.scatter_(1, self.window, -1e30)
            else:
                rows = np.arange(M)[:, None]
                scores[rows, be.to_numpy(self.window)] = -1e30
            # positional axis works in BOTH numpy and torch
            X = scores.argmax(1)
            X = X.to(xp.int64) if be.kind == "torch" else X.astype(np.int64)
            self.window = be.cat([X[:, None], self.window[:, :-1]], axis=1)
        else:
            X = scores.argmax(1)
            X = X.to(xp.int64) if be.kind == "torch" else X.astype(np.int64)
        return X

    def burn_in(self, steps):
        for _ in range(int(steps)):
            self.step()

    def estimate_pi(self, steps, skip=None):
        """Advance the chain by `steps` rounds AND return an independent
        estimate of the stationary marginal pi_R from the injected labels.

        This replaces burn_in() when a bias curve is needed: using the same
        sample to estimate both pi_R and nu_hat_t would make the bias
        identically zero. Only the tail of the burn-in is pooled so the
        deterministic worst-case transient is skipped.
        """
        steps = int(steps)
        if skip is None:
            skip = min(self.R + 10, max(1, steps // 4))
        cnt = np.zeros(self.N, dtype=np.float64)
        for t in range(steps):
            X = self.step()
            if t >= skip:
                np.add.at(cnt, self.be.to_numpy(X).reshape(-1), 1.0)
        s = cnt.sum()
        if s <= 0:
            return np.ones(self.N) / self.N
        return cnt / s


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def scatter_add_row(cnt, X):
    """cnt[i, X_i] += 1 for every row i. Row indices are DISTINCT, so numpy's
    advanced-index `+=` is safe (no repeated-index loss)."""
    if hasattr(cnt, "scatter_add_"):
        import torch
        idx = torch.as_tensor(X, device=cnt.device).long()
        one = torch.ones(cnt.shape[0], 1, device=cnt.device, dtype=cnt.dtype)
        cnt.scatter_add_(1, idx[:, None], one)
        return cnt
    cnt[np.arange(cnt.shape[0]), np.asarray(X)] += 1.0
    return cnt


def bootstrap_ci(be, v, B=2000, chunk=100, alpha=0.05, seed=20261120):
    """95% bootstrap CI of the mean of a per-trajectory scalar array v (M,).

    Uses `chunk` rows at a time so the index tensor never exceeds ~80 MB.
    """
    v = be.asarray(np.asarray(v, dtype=np.float32))
    M = int(v.shape[0])
    means = []
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        idx = be.randint(0, M, (b, M))
        means.append(be.to_numpy(v[idx].mean(1)))
    m = np.concatenate(means)
    return dict(mean=float(np.mean(m)),
                lo=float(np.quantile(m, alpha / 2)),
                hi=float(np.quantile(m, 1 - alpha / 2)),
                sd=float(np.std(m)), B=B)


def group_bootstrap_ci(be, group_values, B=2000, alpha=0.05):
    """CI from G independent group estimates (used for Fig.4, where the
    estimand is a cross-trajectory variance and has no per-trajectory value)."""
    g = np.asarray(group_values, dtype=np.float64)
    rng = np.random.default_rng(20261120 + 31337)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    m = g[idx].mean(1)
    return dict(mean=float(np.mean(m)),
                lo=float(np.quantile(m, alpha / 2)),
                hi=float(np.quantile(m, 1 - alpha / 2)),
                groups=len(g), B=B)
