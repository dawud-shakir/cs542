
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" 
import json
import re

def parse_log_to_json(log_file_path, json_output_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    data = {
        "metadata": {},
        "epochs": [],
        "final": {}
    }
    
    current_epoch = None
    batches = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse loading data
        if line.startswith("Loaded"):
            match = re.match(r"Loaded (\w+)\.dat \((\d+\.\d+) MB\) in (\d+\.\d+) sec", line)
            if match:
                data["metadata"][f"{match.group(1)}_size_mb"] = float(match.group(2))
                data["metadata"][f"{match.group(1)}_load_time_sec"] = float(match.group(3))
        
        elif "Training data:" in line:
            match = re.search(r"Training data: \((\d+), (\d+)\), Labels: \((\d+), (\d+)\)", line)
            if match:
                data["metadata"]["training_data_shape"] = [int(match.group(1)), int(match.group(2))]
                data["metadata"]["training_labels_shape"] = [int(match.group(3)), int(match.group(4))]
        
        elif "Test data:" in line:
            match = re.search(r"Test data: \((\d+), (\d+)\), Labels: \((\d+), (\d+)\)", line)
            if match:
                data["metadata"]["test_data_shape"] = [int(match.group(1)), int(match.group(2))]
                data["metadata"]["test_labels_shape"] = [int(match.group(3)), int(match.group(4))]
        
        elif "Pixel value range:" in line:
            match = re.search(r"Pixel value range: (\d+\.\d+), (\d+\.\d+)", line)
            if match:
                data["metadata"]["pixel_range"] = [float(match.group(1)), float(match.group(2))]
        
        elif "Hidden layer size:" in line:
            match = re.search(r"Hidden layer size: \((.*)\)", line)
            if match:
                data["metadata"]["hidden_layer_size"] = [int(x) for x in match.group(1).split(', ')]
        
        elif "Batch size:" in line:
            match = re.search(r"Batch size: (\d+)", line)
            if match:
                data["metadata"]["batch_size"] = int(match.group(1))
        
        elif "Total epochs:" in line:
            match = re.search(r"Total epochs: (\d+)", line)
            if match:
                data["metadata"]["total_epochs"] = int(match.group(1))
        
        elif "Number of processes:" in line:
            match = re.search(r"Number of processes: (\d+)", line)
            if match:
                data["metadata"]["num_processes"] = int(match.group(1))
        
        elif "Learning rate (alpha):" in line:
            match = re.search(r"Learning rate \(alpha\): (\d+\.\d+)", line)
            if match:
                data["metadata"]["learning_rate"] = float(match.group(1))
        
        elif "Hidden layer scaling:" in line:
            match = re.search(r"Hidden layer scaling: (\d+\.\d+)", line)
            if match:
                data["metadata"]["hidden_layer_scaling"] = float(match.group(1))
        
        elif "Early stopping conditions:" in line:
            # This is text, perhaps store as string
            data["metadata"]["early_stopping"] = line.split(": ")[1]
        
        elif "MPI Wtime accuracy:" in line:
            match = re.search(r"MPI Wtime accuracy: (\d+\.\d+e[+-]\d+)", line)
            if match:
                data["metadata"]["mpi_wtime_accuracy"] = float(match.group(1))
        
        # Parse epoch test accuracies
        elif "#### Epoch" in line and "test accuracy:" in line:
            match = re.search(r"#### Epoch (\d+) test accuracy: (\d+\.\d+) ####", line)
            if match:
                epoch_num = int(match.group(1))
                test_acc = float(match.group(2))
                if current_epoch is not None:
                    data["epochs"].append({
                        "epoch": current_epoch,
                        "batches": batches,
                        "test_accuracy": test_acc
                    })
                current_epoch = epoch_num
                batches = []
        
        # Parse batch lines
        elif "Epoch" in line and "Batch" in line and "Loss:" in line:
            match = re.search(r"Epoch (\d+), Batch (\d+), Loss: (\d+\.\d+), Training Accuracy: (\d+\.\d+), Process: (\d+) of (\d+), Per Process Memory \(USS\): (\d+\.\d+) MB, Train Time: (\d+\.\d+) sec", line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss = float(match.group(3))
                train_acc = float(match.group(4))
                process = int(match.group(5))
                total_processes = int(match.group(6))
                memory_uss = float(match.group(7))
                train_time = float(match.group(8))
                batches.append({
                    "batch": batch,
                    "loss": loss,
                    "training_accuracy": train_acc,
                    "process": process,
                    "total_processes": total_processes,
                    "memory_uss_mb": memory_uss,
                    "train_time_sec": train_time
                })
        
        # Final accuracies
        elif "Final training accuracy:" in line:
            match = re.search(r"Final training accuracy: (\d+\.\d+)", line)
            if match:
                data["final"]["training_accuracy"] = float(match.group(1))
        
        elif "Final test accuracy:" in line:
            match = re.search(r"Final test accuracy: (\d+\.\d+)", line)
            if match:
                data["final"]["test_accuracy"] = float(match.group(1))
        
        elif "Train Time:" in line and "sec" in line:
            match = re.search(r"Train Time: (\d+\.\d+) sec", line)
            if match:
                data["metadata"]["total_train_time_sec"] = float(match.group(1))
    
    # Append the last epoch
    if current_epoch is not None:
        data["epochs"].append({
            "epoch": current_epoch,
            "batches": batches,
            "test_accuracy": None  # If not present, but in this log it is
        })
    
    # Write to JSON
    with open(json_output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Usage
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" 
parse_log_to_json(LOG_DIR / 'serial_output.log', LOG_DIR / 'serial_output.json')
