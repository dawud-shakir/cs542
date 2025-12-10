import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

# Ideal memory
def ideal_memory(h):
    total_elements = 784*h + 2*h*h + h*10
    total_mb = total_elements * 8 / 1e6
    procs = np.array([1, 4, 16, 64])
    return procs, total_mb / procs

# Actual measured memory
mem_64   = np.array([21.82, 16.40, 5.01, 3.44])
mem_2048 = np.array([38.5,  43.38, 13.85, 4.11])
mem_4096 = np.array([590.29, 62.32, 36.02, 33.70])

procs = np.array([1, 4, 16, 64])
hidden_sizes = [64, 2048, 4096]
actual_mem   = [mem_64, mem_2048, mem_4096]

plt.figure(figsize=(9, 6))

colors = ["blue", "orange", "green"]

combined_handles = []
combined_labels = []

# Main lines
for i, (h, mem) in enumerate(zip(hidden_sizes, actual_mem)):
    # actual
    line_actual, = plt.plot(procs, mem, marker='o', linestyle='-', color=colors[i])

    # ideal
    _, ideal = ideal_memory(h)
    line_ideal, = plt.plot(procs, ideal, marker='x', linestyle='--', color=colors[i])

    # Combine into one legend entry
    combined_handles.append((line_actual, line_ideal))
    combined_labels.append(f"h = {h}")

# Legend 1: grouped actual+ideal per hidden size
legend1 = plt.legend(
    combined_handles,
    combined_labels,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    title="Matrix Size",
    loc="upper right"
)
plt.gca().add_artist(legend1)

# Legend 2: explain which style = actual/ideal
actual_proxy, = plt.plot([], [], 'o-', label="Actual", color='black')
ideal_proxy,  = plt.plot([], [], 'x--', label="Ideal", color='black')

plt.legend(
    handles=[actual_proxy, ideal_proxy],
    # title="Line Meaning",
    loc="lower left"
)

plt.xscale('log')
plt.yscale('log')

# Explicit tick marks for number of processes
plt.xticks([1, 4, 16, 64], ["1", "4", "16", "64"])

plt.xlabel(r"Number of Processes ($p$, log scale)")
plt.ylabel("Memory per Rank (MB, log scale)")
plt.title("Actual vs. Ideal Memory Scaling (USS Memory per Rank)")
plt.grid(True, which='both')
plt.tight_layout()
plt.show()