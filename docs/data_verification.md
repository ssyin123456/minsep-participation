# Data verification

`figure_sources/verify_data.py` performs a consistency audit over the published datasets. `python3 -m minsep.reproduce --check` independently recomputes every scalar from the modular package and compares it with the same JSON files:

- analytic identities (Theorem 2 prediction, Proposition 1 ratio 9.447352,
  the long-tail calibration 0.7935390988, and the five small-rho scalings);
- derived quantities (28 SEMs, four Welch comparisons, ten independent-design
  repeats);
- cross-file consistency between `figures_data_v2.json`, `figures_data_v3.json`
  and `figures_data_v3_estimation.json`;
- a deterministic re-run of `run_uniform`, `run_counterexample` and
  `run_small_rho` from `minsep.experiments` (with `--full`), which reproduces
  every published summary number to within 1e-9 and refreshes
  `data/recomputed_v2_curves.npz`;
- agreement with the numbers quoted in `paper/main.tex`.

```bash
python3 figure_sources/verify_data.py           # fast, analytic checks
python3 figure_sources/verify_data.py --full    # + Monte Carlo re-run (~40 s)
```

One stale pair of numbers in the manuscript (`0.7935 / 0.6602`) was corrected
to `0.7891 / 0.6737`, matching Fig. 4 and Section VI. The visible deviation of
`t*Var(T)` from the Theorem 2 prediction (max 0.0089 in absolute terms,
4.75% relative at R=16 where the predicted value is only 0.03) is a finite-T
effect, not a data error.
