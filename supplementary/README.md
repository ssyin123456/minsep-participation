# Reproducibility README

Referenced by the *Reproducibility* paragraph of the paper. This file
covers (1) experiments E1–E5 and (2) the derivations now printed as
Appendix C of `paper/main.tex` (Sections S.2–S.9). All paths are relative
to the repository root.

## 1. Status

| Item | Status | Compute | Outputs |
|---|---|---|---|
| E1 prefactor certificate, Fig. 5, Table IV | done | CPU, exact | `data/e1.json`, `figs/fig5.jpg` |
| E2 step size / burn-in, Fig. 6 | done | CPU, 1000 trajectories | `data/e2.json`, `figs/fig6.jpg` |
| E3 finite step size, Fig. 7 | done | CPU, analytic | `data/e3.json`, `figs/fig7.jpg` |
| E4a R=1 aggregate criterion | done | CPU, exact Poisson solve | `data/e4a.json` |
| E4 bootstrap CI, Table V | done | RTX 4090 D, 10⁵ trajectories, B=2000 | `experiments/data/e4.json` (bootstrap-band figure dropped as redundant) |
| E5 CIFAR-10 client profile, Fig. 8 | done | offline labels, 2×10⁴ trajectories | `experiments/data/e5.json`, `figs/fig8.jpg` |
| Supplement derivations S.1–S.9 | done | CPU, exact / dense | `supplementary/scripts/` |

Fixed seed `20261116` for the figure data; the E4/E5 runs use the same
sampler and their own fixed seed recorded in `experiments/data/summary.json`.
The eight submission figures are 6.0 x 3.2 in, 600 dpi JPEG (3600 x 1920 px), 1x2 panels, drawn by `figstyle.py` with all text >= 9.5 pt at its final printed size and horizontal legends below each subplot.

## 2. One command per figure and table

From the repository root:

```bash
make check          # analytic + cross-file consistency checks
make verify         # + deterministic Monte Carlo re-run
make reproduce      # recompute every published scalar and compare
make figures        # rebuild + verify the eight submission JPEG figures
```

E4/E5 (offline, on the GPU machine):

```bash
cd experiments
python verify_cifar10.py                          # must print === READY ===
python run_e4_bootstrap.py --config configs/e4.yaml
python run_e5_realdata.py  --config configs/e5.yaml   # download: false
python collect_results.py
```

`configs/e5.yaml` has `download: false`; the CIFAR-10 label directory must be
`experiments/data/cifar-10-batches-py/` (eight files, one level). Only labels
are read; no image tensor is loaded and no model is trained. If the run falls
back to a synthetic trace, E5 must be dropped.

## 3. Main-text assets

The eight submission figures are `figs/fig1.jpg` … `figs/fig8.jpg`
(6.0 x 3.2 in, 600 dpi JPEG, uniform 1x2 layout, legends below the axes). They are rebuilt from the
published datasets by `figure_sources/redraw_all.py`; `verify_figures.py`
checks pixel dimensions, JPEG decoding and embedded dpi. Two figures were dropped as redundant: the
Proposition 1 time-domain cross-check (exact rational verification is in
Sec. 6.1) and the bootstrap band (its CIs are in Table V and Sec. 6.4).
The original experimental scripts (`scripts/e*.py`, `experiments/*.py`) also
use the compliant `figstyle.py` and remain the computation source.
Table III/IV give exact total-variation prefactor certificates (15 + 7
chains); Table V gives the E4 bootstrap CIs. The R=0->1 violation
fractions are reported in Section 6.3 of the text.

E1 scope: the lumped state space grows exponentially in `R`, so exact
`τ_mix` at `N=100` is reported only for `R≤7`; `C(q)` is not claimed for
`R>7`. The identity-tracking lump is required: a bit-mask lump that records
only initial-vs-fresh underestimates TV (`(10,4)` gives 5 instead of 7).

## 4. Supplement scripts

`supplementary/scripts/` is self-contained (Python 3.9+, numpy, sympy).
`common.py` holds the closed forms, the identity-tracking lumped chain, its
dense matrix, and the full-state `P_R` iterator. Run all checks with:

```bash
cd supplementary/scripts
python3 run_all.py            # prints SUPPLEMENT VERIFICATION: ALL PASS
```

| Script | Supplement sections | What it verifies |
|---|---|---|
| `check_algebra.py` | S.2, S.5, S.7 | `p_R` squarefree on 16 chains; `χ²₁` closed form (exact rational on 9 chains); `S_j` identity by interpolation; Table S.3 thresholds; Table S.5 counterexamples; `λ*=(N−2)^(−1/2)` |
| `check_spectral.py` | S.2, S.3 | TV identity against the full state space; Table S.1 (companion block = R² eigenvalues, rest ≤ λ*²) |
| `check_gelfand.py` | S.4 | corrected ε choice; `C_ε` grows with N (Table S.2) |
| `check_chi2.py` | S.5 | `τ_mix≤R+1` and `τ_mix≤2R`; `P/Q` series vs `χ²_m`; Table S.4 residues |
| `check_certificates.py` | S.7 | Table S.6 prefactor certificates; records Conjecture S.7.1 as open |

Key reproduced values: `χ²₁(71,6)=0.24768`, `τ_mix(71,6)=7`,
`min m` thresholds `5,12,25,65,198,1939` for `R=2,3,4,6,10,30`,
`Σ|a_k|` peak `1.449219` at `(N,R)=(6,2)`, `χ²₁/λ*²=3.25×10⁸` at
`(16,14)`, `max prefactor = 1.4817` at `(10,8)`.

**Certificate inventories (distinct, see Section S.7).** Main-text
Tables III–IV: 22 chains (15+7), verify equation (28) chain by chain,
`C(q)∈[0.010,1.575]`. Source note: 26 ℓ¹/TV certificates with prefactor
`≤1.58` (includes `α₀≈1.6×10⁻⁹`). Supplement Table S.6: independent
recomputation on 19 chains, maximum `1.4817`. A further 22 `(N,R)` pairs
(`m=3…98`) compare the `χ²₁` closed form against exact forward iteration
(`≤3.6×10⁻¹⁵`). Conjecture S.7.1 is a **sufficient**, not equivalent,
condition for the full-range `τ_mix=R+O(L)`; its lower-bound half is a
theorem (Sections S.2, S.6).

## 5. Scope notes carried over from the checks

- The derivations are printed as Appendix C of `paper/main.tex` (after the
  conclusion, before the bibliography). There is no separately compiled
  supplementary PDF; set `main.tex` as the Overleaf main document.
- The `R=2` bound `χ²_m ≤ 1.4492 λ*^(2m)` holds for `N≥6`. For `N=4,5` the
  residue sums are `2.241071` and `1.671156` (Table S.4).
- Conjecture S.7.1 (an absolute `ℓ¹` prefactor on all `(N,R)`) is open. The
  chi-square version is false. No numerical certificate in the supplement is
  used as a theorem in the main text.

## Note on the merged submission

The submission PDF is a single document. The supplementary derivations are
Appendix C of `main.tex` (after the conclusion, before the bibliography), not
a separately compiled file. Only `main.tex` should be set as the Overleaf main
document; the scripts in this directory are for verification only.
