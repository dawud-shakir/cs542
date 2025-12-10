# superheatmap_top3.py

import matplotlib as mpl
# Set font types for better compatibility with vector graphic formats
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "ops_superheatmap.pdf"

# Poster-ish fonts (tweak as needed)
mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
})

# Process counts and hidden sizes
ps = [1, 4, 16, 64]
hs = [64, 2048, 4096]

# ---------------------------
# OPERATION TIMES (seconds)
# from your logs
# rows: h, cols: p
# ---------------------------

# matmul
matmul = np.array([
    [0.00008, 0.00014, 0.00020, 0.00040],   # h=64
    [0.30611, 0.09429, 0.04404, 0.01714],   # h=2048
    [2.12576, 0.62162, 0.23180, 0.10950],   # h=4096
])

# log_softmax
log_softmax = np.array([
    [0.00233, 0.00200, 0.00195, 0.00249],
    [0.16553, 0.06555, 0.03930, 0.02143],
    [0.70710, 0.19407, 0.09644, 0.04645],
])

# stack_ones_on_top
stack = np.array([
    [0.00012, 0.00024, 0.00045, 0.00115],
    [0.06713, 0.06317, 0.08134, 0.07376],
    [0.24997, 0.26647, 0.28236, 0.30333],
])

# Build a 9x4 matrix by stacking (op, h) blocks
# Row order: matmul(h=64,2048,4096), log_softmax(...), stack(...)
ops_blocks = [matmul, log_softmax, stack]
data_s = np.vstack(ops_blocks)   # shape (9, 4)
data_ms = data_s * 1000.0        # convert to ms

# Row labels
row_labels = [
    r"matmul, $h=64$",
    r"matmul, $h=2048$",
    r"matmul, $h=4096$",
    r"log\_softmax, $h=64$",
    r"log\_softmax, $h=2048$",
    r"log\_softmax, $h=4096$",
    r"stack\_ones\_on\_top, $h=64$",
    r"stack\_ones\_on\_top, $h=2048$",
    r"stack\_ones\_on\_top, $h=4096$",
]

# fig, ax = plt.subplots(figsize=(6, 6))  # adjust for your poster

fig, ax = plt.subplots(figsize=(12, 9))  # enlarge heatmap
im = ax.imshow(
    data_ms,
    norm=LogNorm(vmin=data_ms.min(), vmax=data_ms.max()),
    aspect='auto'   # <-- allows cells to expand vertically/horizontally
)

# # Keep numbers readable
# for i in range(data_ms.shape[0]):
#     for j in range(data_ms.shape[1]):
#         ax.text(j, i, f"{data_ms[i, j]:.2f}",
#                 ha='center', va='center',
#                 fontsize=18)  # nice big values for poster

# Log-scaled heatmap (large dynamic range)
im = ax.imshow(
    data_ms,
    norm=LogNorm(vmin=data_ms.min(), vmax=data_ms.max()),
)

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("time (ms)")

# Axes ticks
ax.set_xticks(range(len(ps)))
ax.set_xticklabels(ps)
ax.set_xlabel("Processes $p$")

ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)
ax.set_ylabel("Operation, hidden size")

ax.set_title("Top Operations Runtime vs. $h$ and $p$")

# Annotate each cell with value (ms)
max_val = data_ms.max()
for i in range(data_ms.shape[0]):
    for j in range(data_ms.shape[1]):
        val = data_ms[i, j]
        # Simple contrast heuristic
        text_color = "white" if val > max_val / 3 else "black"
        ax.text(
            j, i,
            f"{val:.0f}" if val >= 10.0 else f"{val:.2f}",
            ha="center",
            va="center",
            color=text_color,
        )
        

fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
fig.show()
plt.close(fig)

