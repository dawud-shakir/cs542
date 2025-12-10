# bar_plot.py
# Create bar plot showing epoch times for different hidden layer sizes and process counts

import colorsys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent # plots/memory folder
FIG_PATH = ROOT_DIR / "memory_bar_plot.pdf"

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
    'legend.fontsize': 18,
    'grid.color': 'k',         # black grid lines
    'grid.alpha': 1.0,        # fully opaque
    'grid.linewidth': 2.8,    # line thickness
})

import matplotlib.pyplot as plt
import numpy as np

# matrix sizes on x-axis
matrix_labels = [r"64 $\times$ 64", r"2048 $\times$ 2048", r"4096 $\times$ 4096"]
# number of processes (each list index corresponds to these counts)
proc_labels = ["1", "4", "16", "64"]

# memory: each corresponds to USS memory for [1,4,16,64] processes
mem_64   = [21.82, 16.40, 5.01, 3.44]
mem_2048 = [38.5, 43.38, 13.85, 4.11]    # -35.54
mem_4096 = [590.29, 62.32, 36.02, 33.70]




# arrange data so we have one list per process containing values across matrix sizes
vals_per_proc = [
    [mem_64[i], mem_2048[i], mem_4096[i]]
    for i in range(len(proc_labels))
]

x = np.arange(len(matrix_labels))  # base x positions for matrix sizes
n_procs = len(proc_labels)
bar_width = 0.18

# center the group of bars on each x tick
offsets = (np.arange(n_procs) - (n_procs - 1) / 2) * bar_width

# colors = plt.get_cmap("tab10").colors   # blue, orange, green red

def tweak_tab10(contrast=1.2, lighten=0.0):
    base = np.array(plt.get_cmap("tab10").colors)
    out = []
    for r,g,b in base:
        h,l,s = colorsys.rgb_to_hls(r,g,b)
        l = np.clip(l * contrast + lighten, 0, 1)   # boost/dim luminance
        s = np.clip(s * contrast, 0, 1)             # boost/dim saturation
        out.append(colorsys.hls_to_rgb(h, l, s))
    return out

# use contrast>1 for stronger colors, <1 to soften; lighten positive to brighten
colors = tweak_tab10(contrast=1.15, lighten=0.01)


fig, ax = plt.subplots(figsize=(10, 6))

# use symlog so small values are shown linearly and large values on a log scale
ax.set_yscale("symlog", linthresh=10)  # linear within +/-10, log beyond

bars = []
for i, (offset, vals) in enumerate(zip(offsets, vals_per_proc)):
    b = ax.bar(x + offset, vals, width=bar_width, color=colors[i % len(colors)], label=f"{proc_labels[i]} procs")
    bars.append(b)

# # annotate each bar with its value
# for b_group in bars:
#     for rect in b_group:
#         h = rect.get_height()
#         # place label slightly above for positive heights; adjust for symlog spacing
#         y = h * (1.05 if h > 0 else 0.95)
#         ax.text(rect.get_x() + rect.get_width() / 2, y, f"{h:.1f} MB", ha="center", va="bottom", fontsize=8, rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(matrix_labels)
ax.set_xlabel("Hidden layer size")
ax.set_ylabel("Memory (MB, log scale)")

ax.set_title("Mean Per Process Memory Increase\nBatches Per Epoch: 60,  Batch Size: 1,000")

ax.set_axisbelow(True)
ax.grid(True, which="both", axis="y", ls=":", lw=mpl.rcParams["grid.linewidth"], zorder=0)


# ax.legend(title="Processes", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.legend(title="Processes", loc="best")


plt.tight_layout()


plt.savefig(FIG_PATH, dpi=150) # Images should be 150dpi (no smaller or larger)
print(f"Saved memory plot to {FIG_PATH}")

plt.show()
