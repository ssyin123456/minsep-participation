# Minimum-Separation Client Participation

Reproducibility package for the manuscript *On the Mixing Time and the
Sampling Variance of Minimum-Separation Client Participation in Federated
Learning* (Shishu Yin, 2026; JMLR submission).

Minimum separation bars a selected client from the following `R` rounds and
induces an `R`-th order Markov participation chain. The paper evaluates two
time scales of that chain: the worst-case mixing time of the full window
state, and the zero-frequency asymptotic variance of the empirical sampling
frequency. This repository contains the data, the computation source, and
every script needed to regenerate the numbers, tables and figures.

## Quick start

```bash
python3 -m pip install -r requirements.txt
make check          # published datasets: analytic + cross-file consistency
make verify         # + deterministic Monte Carlo re-run (fixed seed 20261116)
make reproduce      # recompute every published scalar, compare with the JSONs
make figures        # rebuild the 8 compliant 600 dpi JPEG figures used by paper/main.tex
make supplement     # all Appendix C derivations (S.2-S.9)
```

All commands are CPU-only. `experiments/` (E4 bootstrap, E5 CIFAR-10 profile)
can use `torch.cuda` if available and falls back to numpy; the shipped
`experiments/data/*.json` are the completed runs, so re-running them is
optional. No model is trained and no image tensor is loaded.

## Repository layout

| path | contents |
|---|---|
| `minsep/` | **authoritative computation source** (pure functions, no file I/O in `experiments.py`) |
| `data/` | published datasets for Figs. 1-5 (`figures_data_v*.json`), E1-E4a (`e*.json`), and the full-length curve cache |
| `scripts/` | `common.py` (analytic core), `figstyle.py` (shared figure style), E1-E4a generators, Figs. 6-8 |
| `figure_sources/` | `redraw_all.py` (the 8 submission JPEGs), `verify_figures.py` (raster audit), `redraw_fig1_5.py` (diagnostic), `verify_data.py` |
| `experiments/` | offline E4 bootstrap and E5 CIFAR-10 profile; configs, run scripts and completed results |
| `figs/` | the eight submission figures as 6.0 x 3.2 in, 600 dpi JPEG (3600 x 1920 px), used by `paper/main.tex` |
| `supplementary/` | Appendix C verification scripts (`check_*.py`, `run_all.py`) and their README |
| `paper/` | manuscript source: `main.tex`, `refs.bib`, `jmlr2e.sty` |
| `configs/` | `default.yaml` (shared simulation configuration) |
| `docs/` | short notes on the numerical checks, data verification and the E5 profile |

The release assets are 600 dpi JPEG figures: `pdfLaTeX` embeds them 1:1 at
`width=\textwidth`, with no raster scaling and no compression artefacts. Their canvas
is 6.0 x 3.2 in (width = `\textwidth`; the extra height holds the legends),
so matplotlib font sizes are the printed font sizes; every text element is at
least 9.5 pt, as JMLR requires for illustrations. Legends sit **below each
subplot, horizontally arranged** (one row where the label widths allow, wrapped
to two or more rows otherwise), with no frame, so they never cover a curve.
Run `make figures` to rebuild and verify all eight.

## Compiling the manuscript

`paper/main.tex` is a single document: Appendix C holds the supplementary
derivations after the conclusion and before the bibliography. It looks for
figures in `figs/` and `../figs/`, so it compiles both from the repository
root and from `paper/`:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

For Overleaf, upload `paper/` together with `figs/` at the project root;
the ready-to-upload submission zip is provided separately.

## Reproducibility contract

- Fixed RNG seed `20261116` (Figs. 1-5) and `20261120` (E1-E4a). The RNG call
  order in `minsep/simulation.py` is part of the contract and must not be
  reordered.
- `python3 -m minsep.reproduce --check` recomputes every published scalar and
  compares it against `data/figures_data_*.json`; `make verify` independently
  re-runs the Monte Carlo from `minsep.experiments`.
- Published JSONs are never overwritten by `check`/`verify`/`reproduce`;
  the eight submission JPEGs are rebuilt only by `make figures`. Recomputed
  datasets go to `data/sim_out/`.
- The E5 variance uses the `m/(m-1)` correction; `pi_R` is estimated from an
  independent burn-in tail.

## Citation

See `CITATION.cff`. License: see `LICENSE`.
