PYTHON ?= python3

.PHONY: help check verify reproduce figures figures-full supplement clean

help:
	@echo "make check        - analytic + cross-file consistency checks"
	@echo "make verify       - + deterministic Monte Carlo re-run (fixed seed)"
	@echo "make reproduce    - recompute all published scalars and compare"
	@echo "make figures      - rebuild the 8 submission JPEG figures (JMLR-compliant type)"
	@echo "make figures-full - rerun the full CPU chain, then rebuild submission figures"
	@echo "make supplement   - run all Appendix C derivations (S.2-S.9)"
	@echo "make clean        - remove caches and generated raster previews"

check:
	$(PYTHON) figure_sources/verify_data.py

verify:
	$(PYTHON) figure_sources/verify_data.py --full

reproduce:
	$(PYTHON) -m minsep.reproduce --check

figures:
	$(PYTHON) figure_sources/redraw_all.py
	$(PYTHON) figure_sources/verify_figures.py

figures-full:
	$(PYTHON) -m minsep.reproduce
	$(PYTHON) scripts/e1_prefactor.py
	$(PYTHON) scripts/e2_stepsize.py
	$(PYTHON) scripts/e3_etascan.py
	$(PYTHON) scripts/e4a_dirichlet.py
	$(PYTHON) experiments/collect_results.py
	$(PYTHON) figure_sources/redraw_fig1_5.py   # diagnostics -> data/sim_out/diagnostic/
	$(MAKE) figures

supplement:
	$(PYTHON) supplementary/scripts/run_all.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf data/sim_out build figs/*.png figs/*.pdf
