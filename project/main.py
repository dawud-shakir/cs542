"""
main.py
"""

################################################################################
# Imports and Setup
################################################################################


import numpy as np
np.random.seed(0)  # Reproducibility
np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

import package.layer as nn

import os               # for file paths
import psutil         # for memory usage
proc = psutil.Process(os.getpid())
import sys

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"

from mpi4py import MPI
from package import pmat, print_ordered_by_rank
from package.mnist import read_mnist_data
from package.utilities import get_memory_usage

################################################################################
# Data Loading and Saving
################################################################################

def write_data_files():
##### Do this once: Write data as files #####
    X_train, y_train, X_test, y_test = read_mnist_data()

    p_X_train = pmat.from_numpy(X_train)
    p_y_train = pmat.from_numpy(y_train.reshape(-1,1))  # make 2d
    p_X_test = pmat.from_numpy(X_test)
    p_y_test = pmat.from_numpy(y_test.reshape(-1,1))    # make

    p_X_train.to_file(DATA_DIR / "X_train.dat")
    p_y_train.to_file(DATA_DIR / "y_train.dat")
    p_X_test.to_file(DATA_DIR / "X_test.dat")
    p_y_test.to_file(DATA_DIR / "y_test.dat")

    return p_X_train, p_y_train, p_X_test, p_y_test

def load_data_files():
    rank = MPI.COMM_WORLD.Get_rank()
    
    if not (DATA_DIR / "X_train.dat").is_file() or not (DATA_DIR / "y_train.dat").is_file():
        if rank == 0: print("Data files not found. Writing data files...")
        t0 = MPI.Wtime()

        p_X_train, p_y_train, p_X_test, p_y_test = write_data_files()

        t1 = MPI.Wtime()
        ttot = t1 - t0
        time = MPI.COMM_WORLD.reduce(ttot, op=MPI.MAX, root=0)
        if rank == 0: print(f"Loaded and wrote data in {time:.2f} sec")

        return p_X_train, p_y_train, p_X_test, p_y_test
        

#### Load from files #####

    if rank == 0: print("Loading data files...")
    
    t0 = MPI.Wtime()
    p_X_train, nbytes = pmat.from_file(DATA_DIR / "X_train.dat")
    t1 = MPI.Wtime()
    ttot = t1 - t0
    time = MPI.COMM_WORLD.reduce(ttot, op=MPI.MAX, root=0)

    if rank == 0: print(f"Loaded X_train.dat ({nbytes / 1024**2:.2f} MB) in {time:.2f} sec")

    t0 = MPI.Wtime()
    p_y_train, nbytes = pmat.from_file(DATA_DIR / "y_train.dat")
    t1 = MPI.Wtime()
    ttot = t1 - t0
    time = MPI.COMM_WORLD.reduce(ttot, op=MPI.MAX, root=0)
    if rank == 0: print(f"Loaded y_train.dat ({nbytes / 1024**2:.2f} MB) in {time:.2f} sec")
    
    t0 = MPI.Wtime()
    p_X_test, nbytes = pmat.from_file(DATA_DIR / "X_test.dat")
    t1 = MPI.Wtime()
    ttot = t1 - t0
    time = MPI.COMM_WORLD.reduce(ttot, op=MPI.MAX, root=0)
    if rank == 0: print(f"Loaded X_test.dat ({nbytes / 1024**2:.2f} MB) in {time:.2f} sec")

    t0 = MPI.Wtime()
    p_y_test, nbytes = pmat.from_file(DATA_DIR / "y_test.dat")
    t1 = MPI.Wtime()
    ttot = t1 - t0
    time = MPI.COMM_WORLD.reduce(ttot, op=MPI.MAX, root=0)
    if rank == 0: print(f"Loaded y_test.dat ({nbytes / 1024**2:.2f} MB) in {time:.2f} sec")

    return p_X_train, p_y_train, p_X_test, p_y_test


# X_train, y_train, X_test, y_test = read_mnist_data()

# Uncomment to write data files once
# write_data_files()
# if MPI.COMM_WORLD.Get_rank() == 0:
#     print("Wrote data files."); 


p_X_train, p_y_train, p_X_test, p_y_test = load_data_files()


################################################################################
# Hyperparameters and Network 
################################################################################
show_plot = False

alpha = 1e-3
n_epochs = 5
batch_size = 1000
# batch_size = X_train.shape[0]

# Early stopping threshold (set to None to disable)
stop_at_accuracy = None     #0.99 #0.5      # 0.99
stop_at_loss = None          #0.01 # 1.0          # 0.01
stop_at_epoch = 1        # 5
stop_at_batch = None        # 272

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0 # scale factor for hidden layer size

hidden_size = int(64*scale)

# 28*28 = 784 input features
fc1 = nn.Parallel_Layer(input_size=28*28,  output_size=hidden_size); fc1.phi, fc1.phi_prime = nn.ReLU, nn.ReLU_derivative
fc2 = nn.Parallel_Layer(input_size=hidden_size, output_size=hidden_size); fc2.phi, fc2.phi_prime = nn.ReLU, nn.ReLU_derivative
fc3 = nn.Parallel_Layer(input_size=hidden_size, output_size=hidden_size); fc3.phi, fc3.phi_prime = nn.ReLU, nn.ReLU_derivative
fc4 = nn.Parallel_Layer(input_size=hidden_size, output_size=10); fc4.phi= nn.log_softmax 

################################################################################
# Testing 
################################################################################
def evaluate():
    """ Test - no gradients, just forward pass."""
    h1 = fc1.forward(X_test)   # (64, batch)
    h2 = fc2.forward(h1)  # (10, 100)
    h3 = fc3.forward(h2)  # (10, 100)
    logits = fc4.forward(h3)  # Raw logits, no activation yet

    ######### patmat version ##########
    # Apply log_softmax manually
    # p_log_probs = nn.log_softmax(logits.T)  # Shape: (batch_size, 10)
    # y_hat = np.argmax(p_log_probs.get_full(), axis=1)  # Predicted class labels

    ####### Original version ##########
    # Apply log_softmax manually
    log_probs = nn.log_softmax(logits.T)  # Shape: (batch_size, 10)
    y_hat = np.argmax(log_probs, axis=1)  # Predicted class labels

    return np.sum(y_hat == y_test) / y_test.shape[0]

################################################################################
# Main
################################################################################

def main():

    pixel_value_min, pixel_value_max = p_X_train.pmin(), p_X_train.pmax()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Training data: {p_X_train.shape}, Labels: {p_y_train.shape}")
        print(f"Test data: {p_X_test.shape}, Labels: {p_y_test.shape}")
        print(f"Pixel value range: {pixel_value_min:.3f}, {pixel_value_max:.3f}]")
        # print(f"Training labels: np.unique(y_train)")
        # print(f"Test labels: np.unique(y_test)")
        print(f"Hidden layer size: ({hidden_size}, {hidden_size})")


    ############################################################################
    # Training 
    ############################################################################

    

    n_batches = p_X_train.shape[0] // batch_size
    losses, accuracies = [], []

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("*"*100)
        print("Starting training...")
        
        print(f"Hidden layer scaling: {scale:.1f}")
        print(f"Batch size: {batch_size}, Total batches per epoch: {n_batches}")
        print("Early stopping conditions:"
              f" accuracy >= {stop_at_accuracy},"
              f" loss <= {stop_at_loss},"
              f" epoch >= {stop_at_epoch},"
              f" batch >= {stop_at_batch}"
            )
        
        print("Learning rate (alpha):", alpha)
        print("Total epochs:", n_epochs)
        print("Number of processes:", MPI.COMM_WORLD.Get_size())
        print("MPI Wtime accuracy: ", MPI.Wtick())

    
    stop_early = False
    batch_count = 0
    start_mem_bytes = get_memory_usage()[0]

    # Start MPI training timer
    start_time = MPI.Wtime()

    for epoch in range(n_epochs):
        
        # Check early stopping conditions 
        if stop_early:
            break

        if (stop_at_epoch is not None and epoch >= stop_at_epoch):
            if MPI.COMM_WORLD.Get_rank() == 0:            
                print(f"Stopping early at epoch {epoch+1} due to reaching epoch threshold: {epoch+1} >= {stop_at_epoch}")
            stop_early = True
            break
   

        # Shuffle data into batches
        indicies = np.random.permutation(np.arange(p_X_train.shape[0]))  

        # Make sure indicies is the same on all ranks (in case their random seed is different)
        if MPI.COMM_WORLD.rank == 0:
            indicies = np.random.permutation(np.arange(p_X_train.shape[0]))
        else:
            indicies = np.empty(p_X_train.shape[0], dtype=int)
        MPI.COMM_WORLD.Bcast(indicies, root=0)  # Broadcast to all ranks

        # Split into batches
        batches_idx = np.array_split(indicies, n_batches)  

        # Training loop over batches
        for (batch_num, batch_idx) in enumerate(batches_idx):
            if stop_early:
                break

            ############ Start of batch ############

            # Get batch data
            p_X = p_X_train[batch_idx]  # (batch_size, 784)
            p_Y = p_y_train[batch_idx]  # (batch_size,)

            ############ Forward pass ############
            h1 = fc1.forward(p_X)                       # (hidden_size, batch_size)
            h2 = fc2.forward(h1)                        # (hidden_size, batch_size)
            h3 = fc3.forward(h2)                        # (hidden_size, batch_size)

            # Raw logits, so no activation yet
            p_logits = fc4.forward(h3)                  # (10, batch_size)


            # Apply log_softmax and NLL loss
            p_log_probs = nn.log_softmax(p_logits.T)    # (batch_size, 10)
            loss = nn.nll_loss(p_log_probs, p_Y)        

            pmat_argmax = np.argmax(p_log_probs, axis=1)
            pmat_cmp = (pmat_argmax == p_Y)
            pmat_sum = np.sum(pmat_cmp)
            pmat_acc = pmat_sum / p_Y.shape[0]

            acc = pmat_acc
            
            ############ Backward pass ############
            
            # Start with combined log_softmax + NLL derivative
            p_dL_dlogits = nn.nll_loss_derivative(p_log_probs, p_Y)  # (batch_size, 10)


            fc4.phi = nn.linear  
            fc4.phi_prime = nn.linear_derivative

            # Note: p_dL_dlogits.T is (10, batch_size) 
            p_dL_dh3 = fc4.backward(p_dL_dlogits.T)  # (hidden_size, batch_size) 
            p_dL_dh2 = fc3.backward(p_dL_dh3)   # (hidden_size, batch_size)
            p_dL_dh1 = fc2.backward(p_dL_dh2)   # (hidden_size, batch_size)
            p_dL_dh0 = fc1.backward(p_dL_dh1)   # (features, batch_size)

            ############ Update weights ############

            fc1.update_weights(alpha)
            fc2.update_weights(alpha)
            fc3.update_weights(alpha)
            fc4.update_weights(alpha)
            
            ############ End of batch ############
            batch_count += 1

            elapsed_time = MPI.Wtime() - start_time

            MPI.COMM_WORLD.Barrier()  # Synchronize before checking stopping conditions



            # Metrics
            losses.append(loss)
            accuracies.append(acc)
            mem_bytes, mem_label = get_memory_usage()
            mem_bytes -= start_mem_bytes  # Report memory used during training
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(  
                    f"Epoch {epoch+1}, "
                    f"Batch {n_batches*epoch+batch_num+1}, "
                    f"Loss: {loss:.4f}, Training Accuracy: {acc:.4f}, "
                    f"Process: {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}, "
                    f"Per Process Memory ({mem_label}): {mem_bytes / 1024**2:.2f} MB, "
                    f"Total Time: {elapsed_time:.2f} sec"
                    )

        
            # Check early stopping conditions
            if (stop_at_accuracy is not None and accuracies[-1] >= stop_at_accuracy):
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print(f"Stopping early at epoch {epoch+1} due to reaching accuracy threshold: {accuracies[-1]:.4f} >= {stop_at_accuracy}")
                stop_early = True
                break
            
            if (stop_at_loss is not None and losses[-1] <= stop_at_loss):
                if MPI.COMM_WORLD.Get_rank() == 0:                
                    print(f"Stopping early at epoch {epoch+1} due to reaching loss threshold: {losses[-1]:.4f} <= {stop_at_loss}")
                stop_early = True
                break

            if (stop_at_batch is not None and batch_count >= stop_at_batch):
                if MPI.COMM_WORLD.Get_rank() == 0:                
                    print(f"Stopping early at batch {len(losses)} due to reaching batch threshold: {len(losses)} >= {stop_at_batch}")
                stop_early = True
                break
            


        # print(f"#### Epoch {epoch+1} test accuracy: {evaluate():.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()} ####")

    total_time = MPI.Wtime() - start_time

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Total time: {total_time:.5f} sec")
        print(f"Final test accuracy: {acc:.4f}")


    ############################################################################
    # Plots 
    ############################################################################

    if show_plot and MPI.COMM_WORLD.Get_rank() == 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Epochs: {n_epochs}, Batch size: {batch_size}, Learning rate: {alpha}", fontsize=16)

        # Plot 1: Training loss
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Mean Squared Error')
        plt.grid(True)

        # Plot 2: Training accuracy
        plt.subplot(1, 2, 2)
        for i in range(epoch + 1):
            plt.plot(accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Correct Predictions (%)')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()