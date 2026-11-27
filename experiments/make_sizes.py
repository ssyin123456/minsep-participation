# -*- coding: utf-8 -*-
"""make_sizes.py --- zero-download helper for E5.

Produces `my_sizes.npy`, a (N,) int array of REAL client sample counts, from
either
  * a CIFAR-10 / MNIST label array you already have on disk (.npy), or
  * a LEAF FEMNIST train json directory.

Then:
    python run_e5_realdata.py --config configs/e5.yaml --sizes-file my_sizes.npy

IMPORTANT: the counts must come from a real dataset. If you generate them from
a random array, E5 is a SYNTHETIC run and must be dropped, per README.md.
"""
import argparse, glob, json, os
import numpy as np


def from_labels(path, N, alpha, seed):
    """Dirichlet partition of a real label vector into N clients."""
    y = np.asarray(np.load(path)).reshape(-1)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    sizes = np.zeros(N, dtype=np.int64)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        prop = rng.dirichlet(np.ones(N) * alpha)
        cuts = np.round(np.cumsum(prop) * len(idx)).astype(np.int64)
        cuts[-1] = len(idx)
        parts = np.split(idx, cuts[:-1])
        for i, part in enumerate(parts[:N]):
            sizes[i] += len(part)
    return sizes, dict(source="labels_npy",
                       labels_file=os.path.abspath(path),
                       n_labels=int(len(y)), n_classes=int(len(classes)),
                       alpha=alpha, N=N)


def from_leaf(leaf_dir, N):
    """Count samples per real FEMNIST writer from LEAF train json files."""
    files = sorted(glob.glob(os.path.join(leaf_dir, "**",
                                          "all_data_*_train.json"),
                             recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(leaf_dir,
                                              "all_data_*_train.json")))
    if not files:
        raise FileNotFoundError("no LEAF train json under " + leaf_dir)
    sizes = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        for u in d.get("users", []):
            sizes.append(len(d["user_data"][u]["x"]))
    sizes = np.asarray(sizes, dtype=np.int64)
    if len(sizes) > N:
        sizes = sizes[:N]
    return sizes, dict(source="femnist_leaf", files=len(files),
                       clients=int(len(sizes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help=".npy array of REAL labels")
    ap.add_argument("--leaf-dir", help="LEAF FEMNIST train json directory")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", default="my_sizes.npy")
    args = ap.parse_args()
    if bool(args.labels) == bool(args.leaf_dir):
        ap.error("give exactly one of --labels or --leaf-dir")

    if args.labels:
        sizes, meta = from_labels(args.labels, args.N, args.alpha, args.seed)
    else:
        sizes, meta = from_leaf(args.leaf_dir, args.N)
    if len(sizes) != args.N:
        sizes = np.resize(sizes, args.N)
        meta["resized_to"] = args.N
    np.save(args.out, sizes)
    print("saved", args.out, "shape", sizes.shape,
          "sum", int(sizes.sum()), "min/max", int(sizes.min()), int(sizes.max()))
    print("meta:", meta)
    print("REMINDER: these counts must come from a REAL dataset. "
          "Random counts make E5 synthetic -> drop it.")
    print("next: python run_e5_realdata.py --config configs/e5.yaml "
          "--sizes-file " + args.out)


if __name__ == "__main__":
    main()
