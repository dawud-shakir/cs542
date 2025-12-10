# runtime.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent # plots/memory folder
FIG_PATH = ROOT_DIR / "runtime.pdf"
save_fig = True

import matplotlib as mpl
# Set font types for better compatibility with vector graphic formats
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# set global font sizes and grid style
mpl.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    # 'grid.color': 'k',         # black grid lines
    'grid.alpha': 1.0,        # fully opaque
    'grid.linewidth': 0.2,    # line thickness
})


# Process counts
procs = np.array([1, 4, 16, 64])

# Actual times per hidden size
actual_64    = np.array([5.25, 4.17, 3.42, 4.08])
actual_2048  = np.array([65.97, 249.1275, 597.53, 12.52])
actual_4096  = np.array([236.17, 380.25, 1166.39, 37.88])

# actual_64 /= 300.0 # Normalize times by number of batches
# actual_2048 /= 300.0
# actual_4096 /= 300.0

# Compute ideal times: T(1) / p
ideal_64   = actual_64[0]   / procs
ideal_2048 = actual_2048[0] / procs
ideal_4096 = actual_4096[0] / procs


actual_time = [actual_64, actual_2048, actual_4096]
ideal_time = [ideal_64, ideal_2048, ideal_4096]
hidden_size = [64, 2048, 4096]


plt.figure(figsize=(8,8))

colors = ['blue', 'orange', 'green']
markers = ['o', 's', '^']

combined_handles = []
combined_labels = []

# Main lines
for i, (h, actual, ideal) in enumerate(zip(hidden_size, actual_time, ideal_time)):
    # actual
    line_actual, = plt.plot(procs, actual, marker=markers[i], linestyle='-', color=colors[i])

    # ideal
    line_ideal, = plt.plot(procs, ideal, marker='None', linestyle='--', color=colors[i])

    # Combine into one legend entry
    combined_handles.append((line_actual, line_ideal))
    combined_labels.append(f"h = {h}")

# Legend 1: grouped actual+ideal per hidden size
legend1 = plt.legend(
    combined_handles,
    combined_labels,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    title="Matrix Size",
    loc="lower left",
    framealpha=0.5,

)


plt.gca().add_artist(legend1)
# legend1.set_visible(False)

# Legend 2: explain which style = actual/ideal
actual_proxy, = plt.plot([], [], '-', label="Actual", color='black')
ideal_proxy,  = plt.plot([], [], '--', label="Ideal", color='black')

plt.legend(
    handles=[actual_proxy, ideal_proxy],
    # title="Line Meaning",
    loc="upper right",
    framealpha=0.5,
)

plt.xscale('log', base=2)
plt.yscale('log', base=10)


# plt.xscale('log', base=4)
plt.xticks(procs, procs)
plt.xlabel(r"Number of Processes ($p$)")
plt.ylabel("Time (sec)")
# plt.title("Ideal vs Actual Strong Scaling of Runtime (per Batch)")
plt.title("Strong Scaling of Training Runtime")
plt.grid(True)

plt.tight_layout()

if save_fig:
    plt.savefig(FIG_PATH, dpi=150) # Images should be 150dpi (no smaller or larger)
    print(f"Saved memory plot to {FIG_PATH}")

plt.show()



