from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
save_plots = True
show_plots = True

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" 


import json
import re
import pandas as pd
from pathlib import Path



# load the JSON back into Python structures
def load_training_log(json_file_path=LOG_DIR / 'serial_output.json'):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract metadata
    params = {
        "total_epochs": data["metadata"].get("total_epochs"),
        "batch_size": data["metadata"].get("batch_size"),
        "learning_rate_alpha": data["metadata"].get("learning_rate"),
        "hidden_layer_size": data["metadata"].get("hidden_layer_size"),
        "num_processes": data["metadata"].get("num_processes"),
        "hidden_layer_scaling": data["metadata"].get("hidden_layer_scaling"),
        "mpi_wtime_accuracy": data["metadata"].get("mpi_wtime_accuracy")
    }
    
    data_info = {
        "training_data_shape": data["metadata"].get("training_data_shape"),
        "test_data_shape": data["metadata"].get("test_data_shape"),
        "pixel_range": data["metadata"].get("pixel_range"),
        "X_train_size_mb": data["metadata"].get("X_train_size_mb"),
        "y_train_size_mb": data["metadata"].get("y_train_size_mb"),
        "X_test_size_mb": data["metadata"].get("X_test_size_mb"),
        "y_test_size_mb": data["metadata"].get("y_test_size_mb")
    }
    
    # Extract epochs and batches into a DataFrame
    all_batches = []
    epoch_tests = []
    for epoch in data["epochs"]:
        epoch_tests.append(epoch["test_accuracy"])
        for batch in epoch["batches"]:
            all_batches.append({
                "epoch": epoch["epoch"],
                "batch": batch["batch"],
                "loss": batch["loss"],
                "train_acc": batch["training_accuracy"],
                "memory_uss_mb": batch["memory_uss_mb"],
                "train_time_sec": batch["train_time_sec"],
                "test_acc": epoch["test_accuracy"]  # Add test accuracy per epoch to each batch
            })
    
    df = pd.DataFrame(all_batches)
    
    # Final accuracies
    final_acc = data["final"].get("training_accuracy")
    test_acc = data["final"].get("test_accuracy")
    
    return df, epoch_tests, final_acc, params, data_info, test_acc

# ... existing code ...

df, epoch_tests, final_acc, params, data_info, test_acc = load_training_log()

train_loss = df["loss"].tolist()
train_acc = df["train_acc"].tolist()
# Now you can use df["test_acc"].tolist() for per-batch test accuracy (repeated per epoch)
test_acc = df["test_acc"].tolist()
train_loss = df["loss"].tolist()
train_acc = df["train_acc"].tolist()


n_epochs = params["total_epochs"]
batch_size = params["batch_size"]
alpha = params["learning_rate_alpha"]




############################################################################
# Plots 
############################################################################

   
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
    'legend.loc': 'best',
    'grid.color': 'k',         # black grid lines
    'grid.alpha': 1.0,        # fully opaque
    'grid.linewidth': 0.8,    # line thickness
})

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def _add_info_box(ax, txt, loc="upper right", fontsize=mpl.rcParams['axes.labelsize'], facecolor="white", alpha=0.9):
    at = AnchoredText(txt, loc=loc, prop=dict(size=fontsize), frameon=True)
    at.patch.set_facecolor(facecolor)
    at.patch.set_alpha(alpha)
    at.patch.set_edgecolor("black")
    ax.add_artist(at)
        


# Instead of separate plots, create a single plot with two y-axes
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Plot with two y-axes: Loss on left, Accuracy on right
plt.figure(figsize=(10, 6))
ax1 = plt.gca()
# Left y-axis for Loss
ax1.plot(train_loss, color="orange", linewidth=2, label='Training Loss')
ax1.set_xlabel('Batch')
ax1.set_ylabel('Mean Squared Error (Loss)', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.grid(True)

# Right y-axis for Accuracy
ax2 = ax1.twinx()
ax2.plot(train_acc, color="red", linewidth=2, label='Training Accuracy')
ax2.plot(test_acc, linestyle='dashed', color='green', linewidth=2, label='Test Accuracy')
ax2.set_ylabel('Correct Predictions (%)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Title and info box
# plt.title('Training Loss and Accuracy')
ax = plt.gca()
info = f"Epochs: {n_epochs}\nBatch size: {batch_size}\nLR: {alpha}"
_add_info_box(ax, info, loc="center right")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# anchor legend outside plot
ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.02, 1), loc='lower right', borderaxespad=0, frameon=False)

# ax1.legend(lines1 + lines2, labels1 + labels2, loc='center')

plt.tight_layout()
if save_plots: plt.savefig(FIG_DIR / "loss_and_accuracy.pdf", dpi=150)

if show_plots: plt.show()

# Plot 1: Training loss

plt.figure(figsize=(6, 6))
# plt.suptitle(f"Epochs: {n_epochs}, Batch size: {batch_size}, Learning rate: {alpha}")

# plt.subplot(1, 3, 1)
plt.plot(train_loss, color="orange", linewidth=2)
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Mean Squared Error')

ax = plt.gca()
info = f"Epochs: {5}\nBatch size: {1000}\nLR: {0.01}"
_add_info_box(ax, info, loc="upper right")


plt.grid(True)
plt.tight_layout()
if save_plots: plt.savefig(FIG_DIR / "losses.pdf", dpi=150)

# Plot 2: Training accuracy
plt.figure(figsize=(8, 8))
# plt.suptitle(f"Epochs: {n_epochs}, Batch size: {batch_size}, Learning rate: {alpha}")

# plt.subplot(1, 3, 2)
# for _ in range(epoch + 1):
plt.plot(train_acc, color="red", linewidth=2, label='Training Accuracy')
plt.plot(test_acc, linestyle='dashed', color='green', linewidth=2, label='Test Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Batch')
plt.ylabel('Correct Predictions (%)')

ax = plt.gca()
info = f"Epochs: {n_epochs}\nBatch size: {batch_size}\nLR: {alpha}"
_add_info_box(ax, info, loc="lower right")

plt.grid(True)
plt.legend(loc="center right")
plt.tight_layout()
if save_plots: plt.savefig(FIG_DIR / "accuracy.pdf", dpi=150)

# # Plot 3: Testing accuracy
# plt.figure(figsize=(6, 6))
# plt.suptitle(f"Epochs: {n_epochs}, Batch size: {batch_size}, Learning rate: {alpha}")

# # plt.subplot(1, 3, 3)
# for _ in range(epoch + 1):
#     plt.plot(test_accuracies)
# plt.title('Testing Accuracy')
# plt.xlabel('Batch')
# plt.ylabel('Correct Predictions (%)')
# plt.grid(True)
# plt.tight_layout()
# if save_plots: plt.savefig(PLOT_DIR / "test_accuracy.pdf")

if show_plots: plt.show()