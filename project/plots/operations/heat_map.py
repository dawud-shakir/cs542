# heatmaps_top3.py


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
mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    # "axes.titlesize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})



# Process counts and hidden sizes
ps = [1, 4, 16, 64]
hs = [64, 2048, 4096]

# ---------------------------
# OPERATION TIMES (seconds)
# pulled from your logs
# ---------------------------

# matmul
matmul = np.array([
    [0.00008, 0.00014, 0.00020, 0.00040],
    [0.30611, 0.09429, 0.04404, 0.01714],
    [2.12576, 0.62162, 0.23180, 0.10950],
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

ops = {
    "matmul": matmul,
    "log_softmax": log_softmax,
    "stack_ones_on_top": stack,
}

# ---------------------------
# Heatmap generator
# ---------------------------

def make_heatmap(name, data_s):
    data_ms = data_s * 1000   # convert to ms (looks nicer)
    # fig, ax = plt.subplots(figsize=(5, 3))

    # im = ax.imshow(
    #     data_ms,
    #     norm=LogNorm(vmin=data_ms.min(), vmax=data_ms.max()),
    # )

    fig, ax = plt.subplots(figsize=(8, 3.5))  # enlarge heatmap
    im = ax.imshow(
    data_ms,
    norm=LogNorm(vmin=data_ms.min(), vmax=1.5e3),#data_ms.max()),
    aspect='auto'   # <-- allows cells to expand vertically/horizontally
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("time (ms)")

    ax.set_xticks(range(len(ps)))
    ax.set_yticks(range(len(hs)))
    ax.set_xticklabels(ps)
    ax.set_yticklabels(hs)

    ax.set_xlabel("$p$")
    ax.set_ylabel("$h$")
    # ax.set_title(name) # too cluttered / redundant with latex caption

    # Add annotation (too cluttered)
    maxv = data_ms.max()
    for i in range(len(hs)):
        for j in range(len(ps)):
            val = data_ms[i, j]
            text_color = "white" if val > maxv / 3 else "black"
            ax.text(j, i, 
                    f"{val:.2f}" if val < 10.0 else f"{val:.0f}", 
                    ha="center", 
                    va="center", 
                    color=text_color,
                    fontsize=18)

    fig.tight_layout()
    fig.savefig(ROOT / f"{name}_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

# Generate all three heatmaps
for name, opdata in ops.items():
    make_heatmap(name, opdata)
