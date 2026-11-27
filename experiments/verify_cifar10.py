# -*- coding: utf-8 -*-
r"""verify_cifar10.py --- verify a MANUALLY uploaded CIFAR-10 before E5.

Run this from the `experiments/` directory, AFTER placing the
extracted cifar-10-batches-py/ folder:

    cd experiments
    python verify_cifar10.py

It performs NO network access. It checks:
  1. ./data/cifar-10-batches-py/ exists at the exact level torchvision expects
  2. all 8 files are present with plausible sizes
  3. each pickle batch really contains 10000 labelled samples
  4. (optional) torchvision CIFAR10(download=False) loads 50000 targets

The most common failure is an extra directory level, e.g.
    data/cifar-10-batches-py/cifar-10-batches-py/data_batch_1
or having uploaded the .tar.gz without extracting it. The script prints the
exact fix for both.

Exit code 0 = ready; 1 = fix the layout first.
"""
import os, sys, pickle, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
# e5.yaml uses data_dir: ./data, which is relative to the run directory.
DATA_DIR = os.path.join(HERE, "data")
BASE = "cifar-10-batches-py"
FOLDER = os.path.join(DATA_DIR, BASE)

# Lower bounds only. Exact byte counts differ slightly between mirrors;
# torchvision performs its own authoritative per-file MD5 check below.
MIN_BYTES = {
    "batches.meta":      100,
    "data_batch_1":      30_000_000,
    "data_batch_2":      30_000_000,
    "data_batch_3":      30_000_000,
    "data_batch_4":      30_000_000,
    "data_batch_5":      30_000_000,
    "test_batch":        30_000_000,
    "readme.html":       50,
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def dget(d, key):
    """CIFAR-10 pickles use str keys in torchvision but bytes in some mirrors."""
    if key in d:
        return d[key]
    b = key.encode("latin1")
    return d.get(b)


def main():
    print("=== CIFAR-10 offline verification (NO network) ===")
    print("expected folder:", FOLDER)

    # ---- common layout mistakes: give an exact fix -------------------------
    tgz = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    nested = os.path.join(FOLDER, BASE)

    # Extra level check MUST run even when FOLDER itself exists (the most
    # common upload mistake): data/cifar-10-batches-py/cifar-10-batches-py/
    if os.path.isdir(nested):
        print("\nFAIL: extra directory level detected.")
        print("  found:", nested)
        print("  fix:  mv " + nested + "/* " + FOLDER + "/ && rmdir " + nested)
        sys.exit(1)
    if not os.path.isdir(FOLDER):
        if os.path.isfile(tgz):
            print("\nFAIL: you uploaded the .tar.gz but did not extract it.")
            print("  fix:  cd " + DATA_DIR + " && tar -xzf cifar-10-python.tar.gz")
            sys.exit(1)
        alt = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d))
               and "cifar" in d.lower()] if os.path.isdir(DATA_DIR) else []
        print("\nFAIL: folder not found. Contents of ./data/:", alt or "(empty)")
        print("  fix: upload the EXTRACTED cifar-10-batches-py/ directory to")
        print("       " + DATA_DIR + "/")
        sys.exit(1)

    # ---- file presence + sizes ---------------------------------------------
    ok = True
    print("\n%-16s %12s %12s  %s" % ("file", "bytes", "min", "status"))
    for name, lo in MIN_BYTES.items():
        p = os.path.join(FOLDER, name)
        if not os.path.isfile(p):
            print("%-16s %12s %12d  MISSING" % (name, "-", lo))
            ok = False
            continue
        got = os.path.getsize(p)
        good = got >= lo
        print("%-16s %12d %12d  %s" % (name, got, lo, "OK" if good else "TOO SMALL"))
        ok &= good
    if not ok:
        print("\nFAIL: incomplete or damaged upload. Re-extract and re-upload.")
        sys.exit(1)

    # ---- actual label counts (standard library only) -----------------------
    print("\nlabel counts from the pickle batches:")
    total = 0
    for i in range(1, 6):
        p = os.path.join(FOLDER, "data_batch_%d" % i)
        d = load_pickle(p)
        labels = dget(d, "labels")
        key = dget(d, "data")
        n = len(labels)
        total += n
        print("  data_batch_%d : %6d labels, data shape %s" %
              (i, n, getattr(key, "shape", "?")))
        if n != 10000:
            ok = False
    print("  train total  :", total)
    if total != 50000:
        print("FAIL: expected 50000 training labels")
        sys.exit(1)

    meta = load_pickle(os.path.join(FOLDER, "batches.meta"))
    print("  classes      :", dget(meta, "label_names"))

    # ---- optional torchvision round-trip -----------------------------------
    try:
        from torchvision import datasets
        ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=False)
        print("\ntorchvision CIFAR10(download=False): %d targets  -> OK" %
              len(ds.targets))
    except ImportError:
        print("\ntorchvision not installed here (fine on a local machine); "
              "the pickle checks above are sufficient.")
    except Exception as e:
        print("\nWARNING: torchvision could not load the folder:", e)
        print("  E5 would then fall back to a SYNTHETIC trace -- do NOT run E5 "
              "in that case. Fix the layout first.")
        sys.exit(1)

    print("\n=== READY ===")
    print("configs/e5.yaml already has `download: false`; no edit needed.")
    print("  python run_e5_realdata.py --config configs/e5.yaml --smoke")
    print("  python run_e5_realdata.py --config configs/e5.yaml")
    sys.exit(0)


if __name__ == "__main__":
    main()
