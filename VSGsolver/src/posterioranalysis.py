import numpy as np
import matplotlib.pyplot as plt
import corner
from pathlib import Path

# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

experiment_name = "11antbispectrum"
posterior_base = OUTPUTS_DIR / experiment_name / "samples_dir" / "posterior"
posterior_folder = max(posterior_base.iterdir(), key=lambda p: p.stat().st_mtime)
truth_file = DATA_DIR / "observed11z.npy"
output_image = OUTPUTS_DIR / experiment_name / "corner_plot.png"

# =========================
# Load posterior samples
# =========================
n_params = 4

files = sorted(p for p in posterior_folder.iterdir() if p.suffix == ".npz")

all_z = []
for f in files:
    data = np.load(f)
    all_z.append(data["z"][:n_params])

# Stack into (N_samples, n_params)
pwpts = np.vstack(all_z)

print("Posterior shape:", pwpts.shape)

# =========================
# Load true parameter values
# =========================
true_vals = np.load(truth_file)

# =========================
# Labels and ranges
# =========================
labels = [f"Param {i+1}" for i in range(n_params)]
manual_ranges = [(-1, 1)] * n_params   # change if needed

truths = true_vals[:n_params]

# =========================
# Corner plot
# =========================
fig = corner.corner(
    pwpts,
    labels=labels,
    truths=truths,
    range=manual_ranges,
    bins=40,
    smooth=1.0,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 11},
    truth_color="#ff4c4c",
    contour_kwargs={"linewidths": 1.5},
    fill_contours=True,
    levels=[0.68, 0.95],
)

fig.suptitle(f"{n_params}D Posterior Corner Plot Bispectrum", fontsize=22, y=1.02)
fig.savefig(output_image, dpi=150, bbox_inches="tight")
print(f"Saved corner plot to {output_image}")
plt.show()
