"""Global constants and repository-relative paths.

Every script resolves paths from this file's location, so the repository can be
cloned anywhere and run without editing a single absolute path.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Simulation parameters (identical to the values used for the paper figures)
SEED = 20261116
N, T, NTRAJ = 20, 1200, 4000
RS = [0, 1, 2, 4, 8, 12, 16]
D, ETA_LR = 5, 0.04
COL = plt.cm.viridis(np.linspace(0.0, 0.92, len(RS)))

# ---- Repository layout
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "figs"
ASSET_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "data" / "sim_out"


def ensure_dirs():
    for d in (DATA_DIR, FIGURE_DIR, ASSET_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
