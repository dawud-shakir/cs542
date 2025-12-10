import json
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
    'legend.loc': 'best',       # best location
    'grid.color': 'k',         # black grid lines
    'grid.alpha': 1.0,        # fully opaque
    'grid.linewidth': 0.8,    # line thickness
})

import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]     # three levels up -> repo root
LOG_DIR = BASE_DIR / "logs"
if not LOG_DIR.exists():
    raise FileNotFoundError(f"Expected logs directory at {LOG_DIR}")
PLOT_DIR    = Path(__file__).resolve().parent

data_path = LOG_DIR / "all.json"
with open(data_path, "r") as f:
    all_data = json.load(f)

# -------------------------------------
# Helpers
# -------------------------------------

def avg_epoch_time_entry(entry):
    if "per_epoch" in entry:
        per_epoch = entry["per_epoch"]
        epochs = sorted(per_epoch.keys(), key=lambda k: int(k))
        cumulative = [per_epoch[e]["wall_clock_sec"] for e in epochs]
        inc = [cumulative[0]] + [cumulative[i] - cumulative[i-1]
                                for i in range(1, len(cumulative))]
        return float(np.mean(inc))
    elif "epoch_stats" in entry:
        means = [e["mean"] for e in entry["epoch_stats"]]
        return float(np.mean(means))
    else:
        return None

results = []
for hidden_str, hdict in all_data.items():
    h = int(hidden_str)
    for run_name, run_dict in hdict.items():
        analysis = run_dict.get("analysis", {})
        meta = analysis.get("training_metadata", {})
        nproc = meta.get("number_of_processes")
        if nproc is None:
            continue
        avg_t = avg_epoch_time_entry(analysis)
        if avg_t is not None:
            results.append({"h": h, "nproc": int(nproc), "avg_epoch": avg_t})

by_h = {}
for r in results:
    h = r["h"]
    nproc = r["nproc"]
    by_h.setdefault(h, {})[nproc] = r["avg_epoch"]

markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
# -------------------------------------
# Speedup plot
# -------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)
procs_sorted = sorted({r["nproc"] for r in results})
procs_non1 = [p for p in procs_sorted if p != 1]

for i, h in enumerate(sorted(by_h.keys())):
    T1 = by_h[h].get(1)
    if T1 is None:
        continue
    S = [T1 / by_h[h][p] for p in procs_non1]
    ax.plot(procs_non1, S, marker=markers[i], linewidth=2, label=f"h={h}")

ax.set_xscale("log", base=2)
ax.set_xlabel("Number of Processes")
ax.set_xticks(procs_non1)
ax.set_xticklabels([str(p) for p in procs_non1])

ax.set_ylabel("Speedup " + r"$\mathcal{S}(p) = \mathcal{T}(1)/\mathcal{T}(p)$")
ax.set_title("Speedup vs Number of Processes")
ax.legend()
ax.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_speedup.pdf", dpi=300)

# -------------------------------------
# Average epoch time plot
# -------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111)
for i, h in enumerate(sorted(by_h.keys())):
    procs = sorted(by_h[h].keys())
    times = [by_h[h][p] for p in procs]
    ax.plot(procs, times, marker=markers[i], linewidth=2, label=f"h={h}")

ax.set_xscale("log", base=2)
ax.set_xlabel("Number of Processes")
ax.set_xticks(procs_sorted)
ax.set_xticklabels([str(p) for p in procs_sorted])

ax.set_ylabel("Epoch Time (s)")
ax.set_title("Mean Epoch Time vs Processes")
ax.legend()
ax.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.savefig(PLOT_DIR / "fig_epoch_time_vs_procs.pdf", dpi=300)


# Data for memory plot in /logs uses RSS instead of USS; commented out for now
# # -------------------------------------
# # Peak memory plot (h=64)
# # -------------------------------------

# mem_by_H = {}
# for hidden_str, hdict in all_data.items():
#     h = int(hidden_str)
#     for run_name, run_dict in hdict.items():
#         meta_m = run_dict.get("memory", {}).get("training_metadata", {})
#         nproc = meta_m.get("number_of_processes")
#         if nproc is None:
#             continue
#         overall = run_dict["memory"].get("overall", {})
#         peak = overall.get("overall_peak_mb")
#         if peak:
#             mem_by_H.setdefault(h, {})[int(nproc)] = peak

# h = 64
# if h in mem_by_H:
#     procs = sorted(mem_by_H[h].keys())
#     peaks = [mem_by_H[h][p] for p in procs]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.bar([str(p) for p in procs], peaks)
#     ax.set_xlabel("Number of Processes")
#     ax.set_ylabel("Peak memory (MB)")
#     ax.set_title("Peak memory vs processes (h=64)")
#     plt.tight_layout()
#     plt.savefig(PLOT_DIR / "fig_memory_peak_H64.pdf", dpi=300)

# -------------------------------------
# Per-epoch timing plots for each h
# -------------------------------------

def epoch_durations_for(hidden_str):
    durs = {}
    for run_name, run_dict in all_data[hidden_str].items():
        analysis = run_dict["analysis"]
        nproc = analysis["training_metadata"]["number_of_processes"]
        if "per_epoch" in analysis:
            per_epoch = analysis["per_epoch"]
            epochs = sorted(per_epoch.keys(), key=lambda k: int(k))
            cumulative = [per_epoch[e]["wall_clock_sec"] for e in epochs]
            inc = [cumulative[0]] + [
                cumulative[i]-cumulative[i-1] for i in range(1,len(cumulative))
            ]
            durs[nproc] = inc
        elif "epoch_stats" in analysis:
            stats = analysis["epoch_stats"]
            durs[nproc] = [s["mean"] for s in stats]
    return durs

for hidden_str in all_data.keys():
    h = int(hidden_str)
    durs = epoch_durations_for(hidden_str)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    epochs = None
    for nproc in sorted(durs.keys()):
        vals = durs[nproc]
        if epochs is None:
            epochs = list(range(1, len(vals)+1))
        ax.plot(epochs, vals, marker="o", linewidth=2, label=f"p={nproc}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch Time (s)")
    ax.set_title(f"Epoch Time Per Epoch for h={h}")
    ax.legend()
    ax.grid(True, linestyle="--")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"fig_epoch_time_H{h}.pdf", dpi=300)
