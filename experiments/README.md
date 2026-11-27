# Offline experiments E4 and E5

This package produced the completed `data/e4.json` (bootstrap confidence
intervals, Table V) and `data/e5.json` (CIFAR-10 client profile, Fig. 8).
Both runs are finished; the JSON files are the published results, so
re-running is optional.

## Setup

```bash
python3 -m pip install -r ../requirements.txt   # add torch for GPU
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`backend.py` selects `torch.cuda` when available and falls back to numpy.
CPU runs are for smoke tests only.

## Smoke test

```bash
python3 run_e4_bootstrap.py --config configs/e4.yaml --smoke
python3 run_e5_realdata.py  --config configs/e5.yaml --smoke
```

Expected signatures: E4-A R=16 ratio about 0.17 (analytic 0.168944 at full
T); E4-B R=0 `t*trVar` about 0.796 against analytic 0.79354; E5 bias-squared
curves are not identically zero. Stop if any of these fail.

## Full run

```bash
python3 run_e4_bootstrap.py --config configs/e4.yaml
python3 run_e5_realdata.py  --config configs/e5.yaml       # optional
python3 collect_results.py                                # Table V + summary
```

Submission figures are rebuilt by `make figures`; JSON outputs stay in `data/`.

## Design choices

- `core.py` implements the minimum-separation sampler with a Gumbel-max trick,
  exact for both uniform and general `p`.
- The group variance uses the `m/(m-1)` correction, so `t*trVar -> Z0` at
  R=0 and the bootstrap intervals are centred on the unbiased estimator.
- `pi_R` is estimated from an **independent** burn-in tail; reusing the
  checkpoint sample would force the bias curve to be identically zero.
- E5 is a sanity check on observability, not a validation of the closed
  forms. Theorem 5 assumes uniform `p`; the `R+L` and `1/t` curves in Fig. 8
  are reference scales only. No model is trained and no accuracy is reported.

`configs/e5.yaml` has `download: false`. See `../docs/cifar10_profile.md` for
preparing the CIFAR-10 label partition offline.
