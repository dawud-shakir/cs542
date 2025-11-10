"""
main.py
"""

import numpy as np
np.random.seed(0)  # Reproducibility
import layer as nn

import os               # for file paths

import numpy as np
import struct           # for unpacking binary files

""" MPI """
from mpi4py import MPI
from pmat import pmat


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



# (60000, 28, 28)
X_train = load_images(os.path.join(os.path.dirname(__file__), "train-images.idx3-ubyte"))
# (60000,)
y_train = load_labels(os.path.join(os.path.dirname(__file__), "train-labels.idx1-ubyte"))
# (10000, 28, 28)
X_test  = load_images(os.path.join(os.path.dirname(__file__), "t10k-images.idx3-ubyte"))
# (10000,)
y_test  = load_labels(os.path.join(os.path.dirname(__file__), "t10k-labels.idx1-ubyte"))

""" Preprocessing """
# Flatten images (60000, 28, 28) → (60000, 784)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)  

# Normalize pixel by grayscale max value
# X_train = (X_train.astype(np.float32) / 255.0)  
# X_test = (X_test.astype(np.float32) / 255.0)

# Convert test labels to int64 for compatibility
y_train = y_train.astype(np.int64) 
y_test = y_test.astype(np.int64)    

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

""" Network"""
# 28*28 = 784 input features
fc1 = nn.Parallel_Layer(input_size=28*28,  output_size=64); fc1.phi, fc1.phi_prime = nn.ReLU, nn.ReLU_derivative
fc2 = nn.Parallel_Layer(input_size=64, output_size=64); fc2.phi, fc2.phi_prime = nn.ReLU, nn.ReLU_derivative
fc3 = nn.Parallel_Layer(input_size=64, output_size=64); fc3.phi, fc3.phi_prime = nn.ReLU, nn.ReLU_derivative
fc4 = nn.Parallel_Layer(input_size=64, output_size=10); fc4.phi= nn.log_softmax 

""" Testing """

def evaluate():
    """ Test - no gradients, just forward pass."""
    h1 = fc1.forward(X_test)   # (64, batch)
    h2 = fc2.forward(h1)  # (10, 100)
    h3 = fc3.forward(h2)  # (10, 100)
    logits = fc4.forward(h3)  # Raw logits, no activation yet

    # Apply log_softmax manually
    log_probs = nn.log_softmax(logits.T)  # Shape: (batch_size, 10)
    y_hat = np.argmax(log_probs, axis=1)  # Predicted class labels

    return np.sum(y_hat == y_test) / y_test.shape[0]

def main():
    print(f"### Initial test accuracy: {evaluate():.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()} ####")

    """ Training """
    n_batches = X_train.shape[0] // batch_size

    losses, accuracies = [], []

    for epoch in range(n_epochs):


        # Shuffle data
        indicies = np.random.permutation(np.arange(X_train.shape[0]))  

        # Split into batches
        batches_idx = np.array_split(indicies, n_batches)  

        for (batch_num, batch_idx) in enumerate(batches_idx):
            # Get batch data
            X = X_train[batch_idx]  # (batch_size, 784)
            Y = y_train[batch_idx]  # (batch_size,)

            # Forward pass
            h1 = fc1.forward(X)   # (64, batch)
            h2 = fc2.forward(h1)  # (10, 100)
            h3 = fc3.forward(h2)  # (10, 100)
            # h4 = fc4.forward(h3)  # (10, 100)

            # Raw logits, so no activation yet
            logits = fc4.forward(h3) 




            # Apply log_softmax
            log_probs = nn.log_softmax(logits.T)  # Shape: (batch_size, 10)

            # Accuracy    
            acc = np.sum(np.argmax(log_probs, axis=1) == Y) / Y.shape[0]

            ############## Pmat version ########
            # p_logits = pmat.from_numpy(logits) 
            # p_log_probs = nn.log_softmax(p_logits.T)  # Shape: (batch_size, 10)
            # p_loss = nn.nll_loss(p_log_probs, Y)
            # exit()

            # Loss
            loss = nn.nll_loss(log_probs, Y)
   
            
            # Backward pass - start with combined log_softmax + NLL derivative
            dL_dlogits = nn.nll_loss_derivative(log_probs, Y)  # Shape: (batch_size, 10)
            



            # Now backprop through fc4 after linear transformation
            
            
            # Make fc4 linear since we applied log_softmax here
            p_dL_dlogits = pmat.from_numpy(dL_dlogits) 

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
            
            print(f"Epoch {epoch+1}, Batch {n_batches*epoch+batch_num+1}, Loss: {loss:.4f}, Training Accuracy: {acc:.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()}")


        print(f"#### Epoch {epoch+1} test accuracy: {evaluate():.4f}, Process: {MPI.COMM_WORLD.Get_rank()+1} of {MPI.COMM_WORLD.Get_size()} ####")


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