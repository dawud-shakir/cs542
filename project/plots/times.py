# bar_plot.py
# Create bar plot showing epoch times for different hidden layer sizes and process counts

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent # plots folder
FIG_PATH = ROOT_DIR / "times_plot.pdf"

import matplotlib as mpl
# Set font types for better compatibility with vector graphic formats
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

# set global font sizes
mpl.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

# font_size_above_bar = 10

import matplotlib.pyplot as plt
import numpy as np

# matrix sizes on x-axis
matrix_labels = [r"64 $\times$ 64", r"2048 $\times$ 2048", r"4096 $\times$ 4096"]
# number of processes (each list index corresponds to these counts)
proc_labels = ["1", "4", "16", "64"]

# timings: each list corresponds to timings for [1,4,16,64] processes
t_64   = [5.25, 4.17, 3.42, 4.08]
t_2048 = [65.97, 249.1275, 597.53, 12.52]
t_4096 = [236.17, 380.25, 1166.39, 37.88]

# arrange data so we have one list per process containing values across matrix sizes
vals_per_proc = [
    [t_64[i], t_2048[i], t_4096[i]]
    for i in range(len(proc_labels))
]

x = np.arange(len(matrix_labels))  # base x positions for matrix sizes
n_procs = len(proc_labels)
bar_width = 0.18

# center the group of bars on each x tick
offsets = (np.arange(n_procs) - (n_procs - 1) / 2) * bar_width

colors = plt.get_cmap("tab10").colors

fig, ax = plt.subplots(figsize=(10, 6))

# use symlog so small values are shown linearly and large values on a log scale
ax.set_yscale("symlog", linthresh=10)  # linear within +/-10, log beyond

bars = []
for i, (offset, vals) in enumerate(zip(offsets, vals_per_proc)):
    b = ax.bar(x + offset, vals, width=bar_width, color=colors[i % len(colors)], label=f"{proc_labels[i]} procs")
    bars.append(b)

# annotate each bar with its value
# for b_group in bars:
#     for rect in b_group:
#         h = rect.get_height()
#         # place label slightly above for positive heights; adjust for symlog spacing
#         y = h * (1.05 if h > 0 else 0.95)
#         ax.text(rect.get_x() + rect.get_width() / 2, y, f"{h:.1f} s", ha="center", va="bottom", fontsize=font_size_above_bar, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(matrix_labels)
ax.set_xlabel("Hidden layer size in a 4-layer FFN")
ax.set_ylabel("Time (s, log scale)")
# ax.set_title("Epoch time by hidden-layer size and process count")
ax.set_title("Mean Per Process Epoch Time\nBatches Per Epoch: 60,  Batch Size: 1,000")

ax.legend(title="Processes", loc="upper left")
# ax.legend(title="Processes", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, which="both", axis="y", ls=":", lw=0.6)

plt.tight_layout()

plt.savefig(FIG_PATH, dpi=150) # Images should be 150dpi (no smaller or larger)
plt.show()
# ...existing code...