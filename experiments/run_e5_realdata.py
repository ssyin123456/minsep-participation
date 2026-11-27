# -*- coding: utf-8 -*-
"""run_e5_realdata.py --- E5 (optional insurance, 1 GPU-day cap).

WHAT THIS EXPERIMENT ACTUALLY DOES
---------------------------------
A real dataset supplies the *client-size heterogeneity*: CIFAR-10 is split
across N=100 clients with a Dirichlet partition and the availability profile
is set to p_i proportional to the number of samples held by client i. That
real p then drives the minimum-separation chain. We measure

    (1) t * tr Var(nu_hat_t)      -- the aggregate sampling variance
    (2) || E nu_hat_t - pi_R ||^2 -- the bias transient

and display them with the O([R+L]/t) and 1/t curves of Theorem 6 as
reference scales (see the scope limit below).

WHAT IT DOES NOT DO
-------------------
It does NOT train a model, report accuracy, or compare against the debiasing
algorithm of [1]. The closed forms are validated by exact state-space
iteration elsewhere (Tables I, III, IV); a real-data run cannot make them
truer. Its only purpose is to show that the chain constants are observable
under a realistic, non-uniform client profile, which addresses the JMLR
"audience too narrow" desk-reject risk.

IMPORTANT SCOPE LIMIT
---------------------
Theorem 5 (tau_mix = R + O(L)) is proved for UNIFORM p only. Under a real
non-uniform p, Remark 4 shows tau_mix depends on the spectrum of p and is
unbounded at fixed (N,R). Therefore the R+L and 1/t curves in Fig. 8 are
REFERENCE SCALES, not claims of validity for the non-uniform case. This is
stated in the figure caption and in e5.json.

Usage
-----
  python run_e5_realdata.py --config configs/e5.yaml
  python run_e5_realdata.py --config configs/e5.yaml --smoke
  python run_e5_realdata.py --config configs/e5.yaml --sizes-file my_sizes.npy
  python run_e5_realdata.py --config configs/e5.yaml --mode fedavg   # optional

Writes: data/e5.json, figs/fig8.jpg
"""
import argparse, glob, json, os, sys, time
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MINSEP_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from backend import get_backend
from core import (MinSepSampler, scatter_add_row, group_bootstrap_ci,
                  tau_int, L_of, lam_star)


# ---------------------------------------------------------------------------
# real client-size profiles
# ---------------------------------------------------------------------------
def cifar10_dirichlet_sizes(N, alpha, seed, data_dir, download=False, url=None):
    """Return client sample counts from a Dirichlet partition of CIFAR-10
    training labels.

    Only `ds.targets` is used -- no image tensor is ever loaded and no model
    is trained. `url` optionally overrides torchvision's default
    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz, which has been
    throttled to ~55 kB/s since mid-2026. Any mirror must have MD5
    c58f30108f718f92721af3b95e74349a (torchvision verifies it anyway).
    """
    from torchvision import datasets
    if url:
        datasets.CIFAR10.url = url
    ds = datasets.CIFAR10(root=data_dir, train=True, download=download)
    y = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    sizes = np.zeros(N, dtype=np.int64)
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        prop = rng.dirichlet(np.ones(N) * alpha)
        cuts = np.cumsum(prop) * len(idx)
        cuts = np.round(cuts).astype(np.int64)
        cuts[-1] = len(idx)
        parts = np.split(idx, cuts[:-1])
        for i, part in enumerate(parts[:N]):
            sizes[i] += len(part)
    return sizes, dict(source="CIFAR10_Dirichlet", samples=int(len(y)),
                       alpha=alpha, N=N)


def femnist_sizes(leaf_dir, N=None):
    """Read LEAF FEMNIST train json files and count samples per client."""
    files = sorted(glob.glob(os.path.join(leaf_dir, "**", "all_data_*_train.json"),
                             recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(leaf_dir, "all_data_*_train.json")))
    if not files:
        raise FileNotFoundError("no LEAF FEMNIST train json found under " + leaf_dir)
    sizes = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for u in d.get("users", []):
            sizes.append(len(d["user_data"][u]["x"]))
    sizes = np.asarray(sizes, dtype=np.int64)
    if N is not None and len(sizes) > N:
        sizes = sizes[:N]
    return sizes, dict(source="FEMNIST_LEAF", files=len(files),
                       clients=int(len(sizes)))


def synthetic_trace_sizes(N, seed):
    """Fallback when no dataset is reachable: a diurnal availability trace."""
    rng = np.random.default_rng(seed)
    base = 100 + 40 * np.sin(np.linspace(0, 2 * np.pi, N))
    sizes = np.maximum(10, np.round(base + rng.normal(0, 8, N))).astype(np.int64)
    return sizes, dict(source="synthetic_diurnal_trace", N=N,
                       note="NOT real data -- clearly mark as such")


def build_p(sizes, mode, floor=1e-4):
    """p_i proportional to client size, with a floor so every p_i > 0."""
    s = np.asarray(sizes, dtype=np.float64)
    s = np.maximum(s, s.max() * floor)
    if mode == "uniform_from_real_sizes":
        p = np.ones_like(s) / len(s)
    else:
        p = s / s.sum()
    return p


# ---------------------------------------------------------------------------
# participation-only measurement (default)
# ---------------------------------------------------------------------------
def run_participation(be, c, seed, sizes, meta):
    N = int(c["N"]); Rset = list(c["R"])
    T = int(c["T"]); burn = int(c["burn_in"])
    M = int(c["trajectories"]); G = int(c["groups"])
    B = int(c["bootstrap_B"])
    cps = sorted(set(int(x) for x in c["checkpoints"]))
    p = build_p(sizes, c.get("p_mode", "proportional"))
    assert M % G == 0, "trajectories must be divisible by groups"
    m = M // G
    Z0 = float(1.0 - np.dot(p, p))
    print(f"\n[E5 participation] N={N} R={Rset} T={T} burn={burn} "
          f"M={M} groups={G}x{m}")
    print(f"  p from {meta['source']}: min={p.min():.3e} max={p.max():.3e} "
          f"max/min={p.max()/p.min():.1f},  Z=1-sum p^2 = {Z0:.5f}")

    out = {}
    for R in Rset:
        t0 = time.time()
        smp = MinSepSampler(be, N, R, M, p=p, init="worst")
        # INDEPENDENT pi_R estimate from the burn-in tail. For R=0 the exact
        # marginal is p itself. Never reuse the checkpoint sample for pi_R:
        # that would force the bias to be identically zero.
        pi_hat = p.copy() if R == 0 else smp.estimate_pi(burn)
        cnt = be.zeros((M, N))
        curves = {"t": [], "t_trVar": [], "bias2": []}
        group_est = {}
        for t in range(1, T + 1):
            X = smp.step()
            scatter_add_row(cnt, X)
            if t in cps:
                c3 = cnt.reshape(G, m, N)
                cbar = c3.mean(1)
                c2bar = (c3 * c3).mean(1)
                # (1/m) sum (x-xbar)^2 is the biased sample variance; rescale
                # by m/(m-1) so that t*trVar -> Z0 at R=0 and the bootstrap
                # intervals are centred on the unbiased estimator.
                est = (c2bar - cbar * cbar).sum(1) * (m / (m - 1)) / t
                group_est[t] = be.to_numpy(est).astype(np.float64)
                nu_bar = be.to_numpy(cnt.mean(0)).astype(np.float64) / t
                d = nu_bar - pi_hat
                curves["t"].append(t)
                curves["t_trVar"].append(float(np.mean(group_est[t])))
                curves["bias2"].append(float(np.dot(d, d)))
        per_group = np.mean(np.stack([group_est[k] for k in cps], axis=0), axis=0)
        ci = group_bootstrap_ci(be, per_group, B=B)
        lam = lam_star(N, R); L = L_of(N, R)
        out[str(R)] = dict(R=R, **ci, tau_int_uniform_ref=tau_int(N, R),
                           lam_star=lam, L=L, R_plus_L=R + L,
                           pi_hat=[float(x) for x in pi_hat],
                           pi_source=("exact p" if R == 0 else
                                      "independent burn-in tail estimate"),
                           curves=curves)
        print(f"  R={R:2d}  t*trVar={ci['mean']:.5f}  95%CI="
              f"[{ci['lo']:.5f},{ci['hi']:.5f}]  R+L={R+L:6.2f} "
              f"({time.time()-t0:.1f}s)")
    return out, dict(p=[float(x) for x in p], Z0=Z0,
                     sizes=[int(x) for x in sizes], meta=meta)


# ---------------------------------------------------------------------------
# OPTIONAL FedAvg skeleton -- records participation ONLY, never accuracy
# ---------------------------------------------------------------------------
def run_fedavg_skeleton(be, c, seed, sizes, meta):
    """A minimal training loop whose ONLY logged quantity is the empirical
    participation frequency. No accuracy, no loss, no comparison with [1].
    Provided so the real-data check can be extended to a full training loop
    if a reviewer asks; it is NOT part of the paper's claims."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import datasets, transforms
    except ImportError as e:
        return dict(skipped=True, reason="torch/torchvision unavailable: " + str(e))

    N = int(c["N"]); R = int(c["fedavg_R"]); T = int(c["fedavg_T"])
    K = int(c["fedavg_K"]); p = build_p(sizes, c.get("p_mode", "proportional"))
    dev = be.device if be.kind == "torch" else "cpu"
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    ds = datasets.CIFAR10(root=c["data_dir"], train=True,
                          download=bool(c.get("download", False)),
                          transform=tf)
    # one flat feature per sample is enough for a participation-only log
    y = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, N, size=len(y))

    smp = MinSepSampler(be, N, R, 1, p=p, init="worst")
    cnt = np.zeros(N)
    curve = []
    for t in range(1, T + 1):
        X = be.to_numpy(smp.step())[0]
        for _ in range(K):                # schematic local steps, no gradient
            pass
        cnt[X] += 1
        if t % max(1, T // 20) == 0:
            nu = cnt / t
            curve.append(dict(t=t, nu_norm=float(np.dot(nu - 1.0/N, nu - 1.0/N))))
    return dict(skipped=False, R=R, T=T, K=K, mode="participation_log_only",
                curve=curve,
                note="No model was trained and no accuracy is reported. "
                     "This skeleton is not used in any claim of the paper.")


# ---------------------------------------------------------------------------
def make_fig8(be, res, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os as _os_fs, sys as _sys_fs
    _sys_fs.path.insert(0, _os_fs.path.dirname(_os_fs.path.abspath(__file__)))
    from figstyle import apply_rc, new_canvas, finalize, save, audit, PT
    apply_rc()
    part = res["participation"]
    fig, axes = new_canvas()
    cols = plt.cm.viridis(np.linspace(0, .85, len(part)))
    ax = axes[0]
    for (k, v), col in zip(sorted(part.items(), key=lambda x: int(x[0])), cols):
        c = v["curves"]
        ax.plot(c["t"], c["t_trVar"], "o-", color=col, lw=1.6, ms=4,
                label=fr"$R={k}$")
    ax.set_xlabel(r"round $t$ (after burn-in)")
    ax.set_ylabel(r"$t\,\mathrm{tr\,Var}(\hat\nu_t)$")
    ax.set_title("(a) Aggregate sampling variance under a real client profile")
    ax.legend(fontsize=PT, ncol=2, framealpha=.9, loc="upper left")

    ax = axes[1]
    for (k, v), col in zip(sorted(part.items(), key=lambda x: int(x[0])), cols):
        c = v["curves"]
        ax.loglog(c["t"], c["bias2"], "o-", color=col, lw=1.5, ms=4,
                  label=fr"$R={k}$")
    tmax = max(int(t) for v in part.values() for t in v["curves"]["t"])
    tt = np.array([max(10, tmax // 40), max(20, tmax)])
    ax.loglog(tt, 2e-4 / tt, "k--", lw=1.2, label=r"$1/t$ slope (reference)")
    ax.set_xlabel(r"round $t$ (log)")
    ax.set_ylabel(r"$\|\mathbb{E}\hat\nu_t-\hat\pi_R\|^2$")
    ax.set_title(r"(b) Bias transient (qualitative); $1/t$ is a reference")
    ax.legend(fontsize=PT, ncol=2, framealpha=.9)

    fig.suptitle("Real client profile under a CIFAR-10 partition",
                 fontsize=PT+1, y=0.975)
    finalize(fig)
    save(fig, out_pdf)
    print("saved", out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs/e5.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("--mode", choices=["participation", "fedavg"], default=None)
    ap.add_argument("--sizes-file", default=None,
                    help="optional .npy of client sizes, bypasses dataset I/O")
    ap.add_argument("--trajectories", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)
    if args.smoke:
        c["T"] = 300; c["burn_in"] = 50
        c["checkpoints"] = [100, 200, 250, 300]
        c["trajectories"] = 600; c["groups"] = 12
        c["bootstrap_B"] = 300
        c["fedavg_T"] = 100
        print("[smoke] reduced grid active -- NOT for publication")
    if args.trajectories:
        c["trajectories"] = args.trajectories
    if args.mode:
        c["mode"] = args.mode
    seed = int(c.get("seed", 20261120))
    be = get_backend(args.backend, seed)
    print("backend:", be.info())

    # ---- client sizes from a real dataset ---------------------------------
    if args.sizes_file:
        sizes = np.load(args.sizes_file)
        meta = dict(source="user_supplied_sizes_file", N=len(sizes))
    elif c.get("dataset", "cifar10") == "cifar10":
        try:
            sizes, meta = cifar10_dirichlet_sizes(
                int(c["N"]), float(c["dirichlet_alpha"]), seed,
                c["data_dir"], download=bool(c.get("download", False)),
                url=c.get("cifar10_url") or None)
            if c.get("cifar10_url"):
                meta["mirror_used"] = True
        except Exception as e:
            print("WARNING: CIFAR-10 unavailable (", e,
                  ") -- falling back to a SYNTHETIC trace; Fig. 8 would NOT be "
                  "real data. STOP: fix ./data/cifar-10-batches-py/ or drop E5 "
                  "(E1--E4 are sufficient).")
            sizes, meta = synthetic_trace_sizes(int(c["N"]), seed)
    elif c.get("dataset") == "femnist":
        try:
            sizes, meta = femnist_sizes(c["leaf_dir"], int(c["N"]))
        except Exception as e:
            print("WARNING: FEMNIST unavailable (", e,
                  ") -- falling back to a SYNTHETIC trace.")
            sizes, meta = synthetic_trace_sizes(int(c["N"]), seed)
    else:
        sizes, meta = synthetic_trace_sizes(int(c["N"]), seed)
    if len(sizes) != int(c["N"]):
        sizes = np.resize(sizes, int(c["N"]))
        meta["resized_to"] = int(c["N"])

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
    res = dict(config=c, backend=be.info(), smoke=args.smoke,
               generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    part, pinfo = run_participation(be, c, seed, sizes, meta)
    res["participation"] = part
    res["profile"] = pinfo
    if c.get("mode") == "fedavg":
        res["fedavg_skeleton"] = run_fedavg_skeleton(be, c, seed, sizes, meta)

    # ---- hard scope disclaimer, also written into the JSON ----------------
    res["scope_disclaimer"] = (
        "Theorem 5 (tau_mix = R + O(L)) holds for uniform p only. This run uses "
        "a real heterogeneous profile, so R+L and the 1/t curve are reference "
        "scales, NOT validated bounds. No model is trained, no accuracy is "
        "reported, and no comparison with the debiasing algorithm of [1] is "
        "made. The closed forms are validated by exact state-space iteration "
        "(Tables I, III, IV), not by this figure.")

    path = os.path.join(HERE, "data", "e5_smoke.json" if args.smoke else "e5.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("saved", path)

    png = os.path.join(ROOT, "figs",
                       "fig8_smoke.jpg" if args.smoke else "fig8.jpg")
    make_fig8(be, res, png)

    if not args.smoke:
        print("\n=== ACCEPTANCE ===")
        print("  all R: t*trVar CI finite and positive:",
              all(v["lo"] > 0 and np.isfinite(v["hi"]) for v in part.values()))
        print("  profile is REAL (not synthetic fallback):",
              "synthetic" not in meta.get("source", ""))


if __name__ == "__main__":
    main()
