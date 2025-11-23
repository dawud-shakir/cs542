"""
main.py
"""

import numpy as np

np.random.seed(0)  # Reproducibility
np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

import layer as nn

import os               # for file paths
import psutil         # for memory usage
proc = psutil.Process(os.getpid())


import numpy as np
import struct           # for unpacking binary files

""" MPI """
from mpi4py import MPI
from pmat import pmat, print_ordered_by_rank


""" Data """
def load_images(filename):
    with open(filename, 'rb') as f:
        # First 16 bytes: magic number, num images, rows, cols
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number in image file!")
        # Read the rest: unsigned bytes → uint8
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)  # shape: (num_images, 28, 28)
        return data

def load_labels(filename):
    with open(filename, 'rb') as f:
        # First 8 bytes: magic number, num labels
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in label file!")
        # Read the rest: unsigned bytes → uint8
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def read_mnist_data():
    """ Load MNIST data from original ubyte files """
    # (60000, 28, 28)
    X_train = load_images(os.path.join(os.path.dirname(__file__), "train-images.idx3-ubyte"))
    # (60000,)
    y_train = load_labels(os.path.join(os.path.dirname(__file__), "train-labels.idx1-ubyte"))
    # (10000, 28, 28)
    X_test  = load_images(os.path.join(os.path.dirname(__file__), "t10k-images.idx3-ubyte"))
    # (10000,)
    y_test  = load_labels(os.path.join(os.path.dirname(__file__), "t10k-labels.idx1-ubyte"))

    """ Preprocessing """
    # Flatten to 2D: (60000, 28, 28) → (60000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Flatten to 2D: (10000, 28, 28) → (10000, 784)
    X_test = X_test.reshape(X_test.shape[0], -1)  

    # Normalize pixel by grayscale max value
    # X_train = (X_train.astype(np.float32) / 255.0)  
    # X_test = (X_test.astype(np.float32) / 255.0)

    # Convert test labels to int64 for compatibility
    y_train = y_train.astype(np.int64) 
    y_test = y_test.astype(np.int64)    
    return X_train, y_train, X_test, y_test

def write_data_files():
##### Do this once: Write data as files #####
    X_train, y_train, X_test, y_test = read_mnist_data()

    p_X_train = pmat.from_numpy(X_train)
    p_y_train = pmat.from_numpy(y_train.reshape(-1,1))  # make 2d
    p_X_test = pmat.from_numpy(X_test)
    p_y_test = pmat.from_numpy(y_test.reshape(-1,1))    # make

    p_X_train.to_file("X_train.dat")
    p_y_train.to_file("y_train.dat")
    p_X_test.to_file("X_test.dat")
    p_y_test.to_file("y_test.dat")

def load_data_files():
#### Load from files #####
    p_X_train = pmat.from_file("X_train.dat")
    p_y_train = pmat.from_file("y_train.dat")
    p_X_test = pmat.from_file("X_test.dat")
    p_y_test = pmat.from_file("y_test.dat")

    return p_X_train, p_y_train, p_X_test, p_y_test


X_train, y_train, X_test, y_test = read_mnist_data()

# Uncomment to write data files once
# write_data_files()
# if MPI.COMM_WORLD.Get_rank() == 0:
#     print("Wrote data files."); 

# p_X_train, p_y_train, p_X_test, p_y_test = load_data_files()
# exit()


if MPI.COMM_WORLD.Get_rank() == 0:
    print(f"Training data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
    print(f"Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Training labels: {np.unique(y_train)}")
    print(f"Test labels: {np.unique(y_test)}")

""" Hyperparameters """
alpha = 1e-3
n_epochs = 5
batch_size = 1000
# batch_size = X_train.shape[0]

# Early stopping threshold (set to None to disable)
stop_at_accuracy = None #0.99 #0.5      # 0.99
stop_at_loss = None #0.01 # 1.0          # 0.01
stop_at_epoch = None        # 5
stop_at_batch = None        # 272

""" Network"""
scale = 1 # scale factor for hidden layer size (64 shows good results while comparing serial to parallel)

# 28*28 = 784 input features
fc1 = nn.Parallel_Layer(input_size=28*28,  output_size=int(64*scale)); fc1.phi, fc1.phi_prime = nn.ReLU, nn.ReLU_derivative
fc2 = nn.Parallel_Layer(input_size=int(64*scale), output_size=int(64*scale)); fc2.phi, fc2.phi_prime = nn.ReLU, nn.ReLU_derivative
fc3 = nn.Parallel_Layer(input_size=int(64*scale), output_size=int(64*scale)); fc3.phi, fc3.phi_prime = nn.ReLU, nn.ReLU_derivative
fc4 = nn.Parallel_Layer(input_size=int(64*scale), output_size=10); fc4.phi= nn.log_softmax 

""" Testing """

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

def main():

    # logits_numpy = np.random.uniform(-1.0, 1.0, (5, 5))
    # logits_pmat = pmat.from_numpy(logits_numpy)

    # softmax_numpy = nn.log_softmax(logits_numpy)
    # softmax_pmat = nn.log_softmax(logits_pmat)
    
    # if not np.allclose(softmax_pmat.get_full(), softmax_numpy):
    #     softmax_pmat_str = softmax_pmat.pretty_string("f_pmat", remove_padding=False, as_type="f")
    
    #     if MPI.COMM_WORLD.Get_rank() == 0:
    #         print("softmax_numpy", softmax_numpy.shape, ":\n", softmax_numpy)
    #         print("softmax_pmat", softmax_pmat.shape, ":\n", softmax_pmat_str)

    #         print("\033[32msoftmax_pmat and softmax_numpy failed allclose!\033[0m")
    # else:
    #     if MPI.COMM_WORLD.Get_rank() == 0:
    #         print("softmax(A) ...\033[31mpassed\033[0m allclose")
    # exit()

    ####### Original version ##########    
    # print(f"### Initial test accuracy: {evaluate():.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()} ####")

    """ Training """
    n_batches = X_train.shape[0] // batch_size

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




    start_time = MPI.Wtime()
    
    total_batches = 0

    for epoch in range(n_epochs):
        
        # Check early stopping conditions 

        last_acc = accuracies[-1] if accuracies else None
        last_loss = losses[-1] if losses else None
        
        
        if (stop_at_accuracy is not None and last_acc is not None and last_acc >= stop_at_accuracy):
            print(f"Stopping early at epoch {epoch+1} due to reaching accuracy threshold: {last_acc:.4f} >= {stop_at_accuracy}")
            break
        if (stop_at_loss is not None and last_loss is not None and last_loss <= stop_at_loss):
            print(f"Stopping early at epoch {epoch+1} due to reaching loss threshold: {last_loss:.4f} <= {stop_at_loss}")
            break
        if (stop_at_epoch is not None and epoch >= stop_at_epoch):
            print(f"Stopping early at epoch {epoch+1} due to reaching epoch threshold: {epoch+1} >= {stop_at_epoch}")
            break
        if (stop_at_batch is not None and total_batches >= stop_at_batch):
            print(f"Stopping early at batch {len(losses)} due to reaching batch threshold: {len(losses)} >= {stop_at_batch}")
            break

        # Shuffle data
        indicies = np.random.permutation(np.arange(X_train.shape[0]))  

        # Split into batches
        batches_idx = np.array_split(indicies, n_batches)  

        for (batch_num, batch_idx) in enumerate(batches_idx):
            # Get batch data
            X = X_train[batch_idx]  # (batch_size, 784)
            
            
            ###### Original version ##########
            Y = y_train[batch_idx]  # (batch_size,)
            
            ###### pmat verson ####
            # Y = Y.reshape(1, -1) # make 2d

            # Forward pass
            h1 = fc1.forward(X)   # (64, batch)
            h2 = fc2.forward(h1)  # (10, 100)
            h3 = fc3.forward(h2)  # (10, 100)
            # h4 = fc4.forward(h3)  # (10, 100)

            # Raw logits, so no activation yet
            p_logits = fc4.forward(h3) 

            ####################################################################
            # Apply log_softmax and NLL loss
            ####################################################################

            ######### patmat version ##########
            p_log_probs = nn.log_softmax(p_logits.T)  # Shape: (batch_size, 10)
            loss = nn.nll_loss(p_log_probs, Y)
            


            # numpy_argmax = np.argmax(p_log_probs.get_full(), axis=1, keepdims=True)
            # numpy_cmp = np.argmax(p_log_probs.get_full(), axis=1) == Y
            # numpy_sum = np.sum(numpy_cmp)
            # numpy_acc = numpy_sum / Y.shape[0]

            pmat_argmax = np.argmax(p_log_probs, axis=1)
            p_Y = pmat.from_numpy(Y.reshape(-1,1))
            pmat_cmp = (pmat_argmax == p_Y)
            pmat_sum = np.sum(pmat_cmp)
            pmat_acc = pmat_sum / p_Y.shape[0]


            acc = pmat_acc

            # ####### Original version ##########
            # acc = np.sum(np.argmax(p_log_probs.get_full(), axis=1) == Y) / Y.shape[0]
            # loss = nn.nll_loss(p_log_probs.get_full(), Y)



            ####### Original version ##########
            # logits = p_logits.get_full()
            # log_probs = nn.log_softmax(logits.T)      # (batch_size, 10)

            # acc = np.sum(np.argmax(log_probs, axis=1) == Y) / Y.shape[0]
            # loss = nn.nll_loss(log_probs, Y)
            
            ####################################################################
            # Backward pass - start with combined log_softmax + NLL derivative
            ####################################################################
            ######### patmat version ##########
            p_dL_dlogits = nn.nll_loss_derivative(p_log_probs, Y)  # (batch_size, 10)
            
            ####### Original version ##########
            # dL_dlogits = nn.nll_loss_derivative(log_probs, Y)  # (batch_size, 10)
            # p_dL_dlogits = pmat.from_numpy(dL_dlogits) 

            fc4.phi = nn.linear  
            fc4.phi_prime = nn.linear_derivative
            p_dL_dh3 = fc4.backward(p_dL_dlogits.T)  # Note: transpose for (features, batch) format
            p_dL_dh2 = fc3.backward(p_dL_dh3)
            p_dL_dh1 = fc2.backward(p_dL_dh2)
            fc1.backward(p_dL_dh1)

            fc1.update_weights(alpha)
            fc2.update_weights(alpha)
            fc3.update_weights(alpha)
            fc4.update_weights(alpha)

            ############### Original version ########
            # # Make fc4 linear since we applied log_softmax here
            # fc4.phi = nn.linear  
            # fc4.phi_prime = nn.linear_derivative
            # dL_dh3 = fc4.backward(dL_dlogits.T)  # Note: transpose for (features, batch) format
            # dL_dh2 = fc3.backward(dL_dh3)
            # dL_dh1 = fc2.backward(dL_dh2)
            # fc1.backward(dL_dh1)

            # fc1.update_weights(alpha)
            # fc2.update_weights(alpha)
            # fc3.update_weights(alpha)
            # fc4.update_weights(alpha)

            # Metrics
            losses.append(loss)
            accuracies.append(acc)
            
            total_batches += n_batches
            elapsed_time = MPI.Wtime() - start_time

            if MPI.COMM_WORLD.Get_rank() == 0:
                #  RSS = Resident Set Size, a measure of the physical memory (RAM) currently used by a process. It includes the code, data, and stack segments that are resident in memory, excluding swapped-out pages.

                print(f"Epoch {epoch+1}, Batch {n_batches*epoch+batch_num+1}, Loss: {loss:.4f}, Training Accuracy: {acc:.4f}, Process: {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}, Per Process Memory (RSS): {proc.memory_info().rss / 1024**2:.2f} MB, Total Time: {elapsed_time:.5f} sec")

        ### Original version ##########
        # print(f"#### Epoch {epoch+1} test accuracy: {evaluate():.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()} ####")

    total_time = MPI.Wtime() - start_time

    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Total time: {total_time:.5f} sec")
        print(f"Final test accuracy: {acc:.4f}")


    """ Plot """

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

    """ Prediction and residual plot not appropriate for classification """
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()