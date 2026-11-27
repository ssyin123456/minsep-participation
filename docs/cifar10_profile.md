# Preparing the E5 client profile

E5 uses **only the class labels** of the CIFAR-10 training split. No image
tensor is loaded and no model is trained. The labels are partitioned across
`N=100` clients with a symmetric Dirichlet(0.5) allocation; the client sizes
`n_i` define the availability profile `p_i = n_i / sum(n_i)` (with a small
floor so every `p_i > 0`).

`experiments/configs/e5.yaml` has `download: false`, so the dataset must be
placed locally before running. From the `experiments/` directory the expected
path is:

```
experiments/data/cifar-10-batches-py/     # eight files, one level
```

Prepare or verify it with:

```bash
python3 experiments/verify_cifar10.py     # must print === READY ===
python3 experiments/extract_cifar10_labels.py --out sizes.npy
python3 experiments/run_e5_realdata.py \
    --config experiments/configs/e5.yaml --sizes-file sizes.npy
```

`make_sizes.py` builds a sizes file from an existing local copy; passing
`--sizes-file` avoids the 170 MB archive entirely. If the run falls back to a
synthetic trace, E5 must be dropped: E1-E4 are sufficient and no claim
depends on E5.
