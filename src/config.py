from pathlib import Path
import numpy as np


# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


# -------------------------
# Reproducibility
# -------------------------

RANDOM_SEED = 42


# -------------------------
# Problem sizes
# -------------------------

# Use small values first while debugging.
# Later, assignment requires [50, 100, 200, 500].
N_VALUES_DEBUG = [3, 4]
N_VALUES_MAIN = [50, 100, 200, 500]


# -------------------------
# Gaussian point clouds
# -------------------------

MU_SOURCE = np.array([0.0, 0.0])
MU_TARGET = np.array([2.0, 1.0])

# Non-diagonal covariance matrices.
# Non-diagonal means the point clouds are not axis-aligned.
COV_SOURCE = np.array([
    [1.0, 0.6],
    [0.6, 1.5],
])

COV_TARGET = np.array([
    [1.2, -0.4],
    [-0.4, 0.8],
])