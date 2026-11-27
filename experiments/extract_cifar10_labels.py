# -*- coding: utf-8 -*-
r"""extract_cifar10_labels.py --- LOCAL, zero-dependency helper for E5.

E5 only reads CIFAR-10's 50,000 TRAINING LABELS to build a Dirichlet client
partition; it never loads an image tensor and never trains a model. So instead
of copying the whole 170 MB archive, you can run this on your own
computer (Python standard library only -- no torch, no torchvision, no numpy
required for this step), then upload ONE small .npy file.

Usage (on your LOCAL machine, from the directory containing
cifar-10-batches-py/):

    python extract_cifar10_labels.py --dir ./cifar-10-batches-py
    python make_sizes.py --labels cifar10_labels.npy --out my_sizes.npy

Then pass ONLY `my_sizes.npy` (~800 bytes) to
run:

    python run_e5_realdata.py --config configs/e5.yaml --sizes-file my_sizes.npy

NOTE for the manuscript: with --sizes-file, the JSON records
"source: user_supplied_sizes_file". State explicitly in the caption / code
appendix that the client sizes come from the CIFAR-10 training labels via a
Dirichlet(alpha=0.5) partition, and ship this script with the code appendix.
Never fabricate the sizes: random counts would make E5 a SYNTHETIC run.
"""
import argparse, os, pickle, struct, sys

BATCHES = ["data_batch_%d" % i for i in range(1, 6)]
N_EXPECT = 50000


def dget(d, key):
    if key in d:
        return d[key]
    return d.get(key.encode("latin1"))


def read_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    labels = dget(d, "labels")
    if labels is None:
        raise ValueError("no 'labels' key in %s (keys=%s)" % (path, list(d)[:8]))
    return list(labels)


def save_npy_uint8(path, values):
    """Write a minimal valid .npy v1.0 without numpy (shape (N,), dtype |u1)."""
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (%d,), }" % len(values)
    pad = (16 - ((10 + len(header) + 1) % 16)) % 16
    header = header + " " * pad + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY" + bytes([1, 0]))
        f.write(struct.pack("<H", len(header)))
        f.write(header.encode("latin1"))
        f.write(bytes(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./cifar-10-batches-py",
                    help="extracted CIFAR-10 folder (local machine)")
    ap.add_argument("--out", default="cifar10_labels.npy")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    print("reading from:", folder)
    if not os.path.isdir(folder):
        print("FAIL: folder not found. Extract cifar-10-python.tar.gz first:")
        print("  tar -xzf cifar-10-python.tar.gz")
        sys.exit(1)

    all_labels = []
    for name in BATCHES:
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            print("FAIL: missing", p)
            print("  (if the file is one level deeper, move the contents up)")
            sys.exit(1)
        lab = read_batch(p)
        print("  %-14s %6d labels" % (name, len(lab)))
        all_labels.extend(lab)

    print("total:", len(all_labels), "labels; class counts:",
          [all_labels.count(c) for c in range(10)])
    if len(all_labels) != N_EXPECT:
        print("FAIL: expected %d training labels" % N_EXPECT)
        sys.exit(1)

    out = os.path.abspath(args.out)
    save_npy_uint8(out, all_labels)
    print("\nsaved", out, os.path.getsize(out), "bytes")
    print("next (local, needs numpy only):")
    print("  python make_sizes.py --labels %s --out my_sizes.npy" % args.out)
    print("then pass my_sizes.npy to E5 with --sizes-file.")


if __name__ == "__main__":
    main()
