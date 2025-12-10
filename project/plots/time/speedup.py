# speedup.py

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent  # plots/memory folder
FIG_PATH = ROOT_DIR / "speedup.pdf"
save_fig = True

import matplotlib as mpl
# Set font types for better compatibility with vector graphic formats
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'


plot_font = 22

# set global font sizes and grid style

mpl.rcParams.update({
    'font.size': plot_font,
    'axes.titlesize': plot_font,
    'axes.labelsize': plot_font,
    'xtick.labelsize': plot_font * 0.9,  # ~36
    'ytick.labelsize': plot_font * 0.9,
    'legend.fontsize': plot_font * 0.9,
    # 'legend.alpha': 0.8,
    # 'grid.color': 'k',       # black grid lines
    'grid.alpha': 1.0,       # fully opaque
    'grid.linewidth': 0.2,   # line thickness
})


# Process counts
procs = np.array([1, 4, 16, 64])

# Actual times per hidden size (total runtime)
t_64    = np.array([5.25, 4.17, 3.42, 4.08])
t_2048  = np.array([65.97, 249.1275, 597.53, 12.52])
t_4096  = np.array([236.17, 380.25, 1166.39, 37.88])

# Normalize times by number of batches
t_64    /= 300.0
t_2048  /= 300.0
t_4096  /= 300.0

# Speedup S(p) = T(1) / T(p)
S_64    = t_64[0]   / t_64
S_2048  = t_2048[0] / t_2048
S_4096  = t_4096[0] / t_4096

# Ideal speedup: S_ideal(p) = p
ideal_64   = procs.astype(float)   # same ideal curve for each h
ideal_2048 = procs.astype(float)
ideal_4096 = procs.astype(float)

actual_speedup = [S_64, S_2048, S_4096]
ideal_speedup  = [ideal_64, ideal_2048, ideal_4096]
hidden_size    = [64, 2048, 4096]

plt.figure(figsize=(8, 8))


colors = ['blue', 'orange', 'green']
markers = ['o', 's', '^']

plt.plot(procs, ideal_64, color='purple', marker='D', linestyle='--', label='Ideal')
plt.plot(procs, actual_speedup[0], color=colors[0], marker=markers[0], linestyle='-', label=r'Actual ($h$=64)')
plt.plot(procs, actual_speedup[1], color=colors[1], marker=markers[1], linestyle='-', label=r'Actual ($h$=2048)')
plt.plot(procs, actual_speedup[2], color=colors[2], marker=markers[2], linestyle='-', label=r'Actual ($h$=4096)')

plt.xscale('log', base=2)
plt.yscale('log', base=10)

plt.xticks(procs, procs)
# plt.yticks(procs)  
plt.xlabel(r"Number of Processes ($p$)")
plt.ylabel(r"Speedup $\mathcal{S}$(p) = T(1) / T($p$)")
plt.title("Strong Scaling Speedup")
plt.grid(True)

# anchor legend outside plot
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=False)

plt.legend(loc='upper left', framealpha=0.5)

plt.tight_layout()

if save_fig:
    plt.savefig(FIG_PATH, dpi=150)  # Images should be 150dpi (no smaller or larger)
    print(f"Saved speedup plot to {FIG_PATH}")

plt.show()


