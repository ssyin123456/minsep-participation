# Figure sources

| script | role |
|---|---|
| `redraw_all.py` | **authoritative**: rebuilds the eight submission figures (`figs/fig1.jpg` ... `fig8.jpg`) from the published datasets, with all text >= 9.5 pt at the final 6.0 x 3.2 in printed size |
| `verify_figures.py` | checks that each submission JPEG is 3600 x 1920 px at 600 dpi, decodes cleanly, and is present |
| `redraw_fig1_5.py` | diagnostic 1x2 redraw of the early figures; writes to `data/sim_out/diagnostic/`, never to `figs/` |
| `verify_data.py` | consistency audit of the published datasets; `--full` also re-runs the Monte Carlo |

```bash
make figures          # rebuild + verify the submission JPEGs
make figures-full     # rerun the full CPU chain first, then rebuild
python3 figure_sources/redraw_all.py
python3 figure_sources/verify_figures.py
```

Figures are 600 dpi JPEGs with physical size 6.0 x 3.2 in, embedded 1:1 by
`pdflatex` at `width=\textwidth`. Rasterising the glyphs avoids nested PDF
CID-font subsets and transparency groups, which some PDF viewers drop when a
matplotlib PDF is included as a Form XObject. PNG is deliberately not used:
pdfTeX re-encodes it as a Flate stream, which made earlier compiles time out.
The canvas is 6.0 x 3.2 in: its width equals `\textwidth` (6.0 in in
`jmlr2e.sty`), so matplotlib font sizes are the printed font sizes and no
downscaling occurs. The extra 0.6 in of height holds the horizontal legends.
Each subplot group (twin axes included) gets one frameless legend placed below
the x-axis text, laid out in as few horizontal rows as fit; `figstyle.audit`
asserts that no legend overlaps the plotting area, the axis labels or another
legend.  The authoritative computation source is the top-level
`minsep/` package.  The original experimental scripts (`scripts/e*.py`,
`experiments/*.py`) also use the compliant `scripts/figstyle.py`; their outputs
are diagnostic and are not the submitted figures.

Manuscript numbering (trimmed): 1 = bias/asymptotic variance, 2 = quadratic
error, 3 = aggregate variance, 4 = small-rho, 5 = two time scales, 6 = design
rules, 7 = finite step size, 8 = real client profile.  Dropped as redundant:
the Proposition 1 time-domain cross-check (exact rational verification is in
Sec. 6.1) and the bootstrap band figure (its confidence intervals are in
Table V and Sec. 6.4).
