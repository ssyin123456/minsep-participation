# minsep — reproducible computation package

A small importable package. Every path is resolved from `config.py`'s own
location, so a clone works anywhere with no editing.

| module | contents |
|---|---|
| `config.py` | fixed seed (`20261116`), `N/T/NTRAJ/RS`, repository-relative paths |
| `simulation.py` | `simulate` (conditional renormalisation), `simulate_fast` (rejection sampling), `diagnostics`, `per_traj_err` |
| `experiments.py` | `run_uniform`, `run_counterexample`, `run_small_rho`, `errorbar_statistics`, `longtail_estimates`, `rho_scale` — pure functions returning dicts |
| `plotting.py` | baseline diagnostic figures (explicit output paths, written to `data/sim_out/baseline/`) |
| `reproduce.py` | CLI: recompute, refresh the curve cache, compare with published data |

```bash
python3 -m minsep.reproduce             # core datasets + data/recomputed_v2_curves.npz
python3 -m minsep.reproduce --full      # + error bars, long tail, baseline plots
python3 -m minsep.reproduce --check     # compare every published scalar
python3 figure_sources/redraw_all.py # the 8 compliant submission JPEGs
```

`experiments.py` never writes files or draws figures; the RNG call order is
part of the reproducibility contract and must not be reordered. The published
`data/figures_data_*.json` and `figs/` are never overwritten by these
commands: recomputed datasets go to `data/sim_out/`.
