# mnist_to_dat.py

import os               # for file paths
import numpy as np
import struct           # for unpacking binary files

from mpi4py import MPI
from pmat import pmat

def load_images(filename):
    ### Load MNIST data from original ubyte files ###
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
    ### Load MNIST labels from original ubyte files ###
    with open(filename, 'rb') as f:
        # First 8 bytes: magic number, num labels
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in label file!")
        # Read the rest: unsigned bytes → uint8
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def read_mnist_data():
    ### Load MNIST data into numpy arrays ###

    # (60000, 28, 28)
    X_train = load_images(os.path.join(os.path.dirname(__file__), "train-images.idx3-ubyte"))
    # (60000,)
    y_train = load_labels(os.path.join(os.path.dirname(__file__), "train-labels.idx1-ubyte"))
    # (10000, 28, 28)
    X_test  = load_images(os.path.join(os.path.dirname(__file__), "t10k-images.idx3-ubyte"))
    # (10000,)
    y_test  = load_labels(os.path.join(os.path.dirname(__file__), "t10k-labels.idx1-ubyte"))

    ################## Preprocessing ####################
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

def write_dat_files():
    #### Load MNIST data and labels and write them to .dat files #####
    X_train, y_train, X_test, y_test = read_mnist_data()

    p_X_train = pmat.from_numpy(X_train)
    p_y_train = pmat.from_numpy(y_train.reshape(-1,1))  # make 2d
    p_X_test = pmat.from_numpy(X_test)
    p_y_test = pmat.from_numpy(y_test.reshape(-1,1))    # make

    p_X_train.to_file("X_train.dat")
    p_y_train.to_file("y_train.dat")
    p_X_test.to_file("X_test.dat")
    p_y_test.to_file("y_test.dat")

def load_dat_files():
    #### Load data and labels from .dat files #####
    p_X_train = pmat.from_file("X_train.dat")
    p_y_train = pmat.from_file("y_train.dat")
    p_X_test = pmat.from_file("X_test.dat")
    p_y_test = pmat.from_file("y_test.dat")

    return p_X_train, p_y_train, p_X_test, p_y_test


# Uncomment to write data files once
write_dat_files()
if MPI.COMM_WORLD.Get_rank() == 0:
    print("Wrote MNIST data and labels to .dat files."); 

p_X_train, p_y_train, p_X_test, p_y_test = load_dat_files()
if MPI.COMM_WORLD.Get_rank() == 0:
    print("Loaded dat and labels from .dat files.")

# Use p_X_train, p_y_train, p_X_test, p_y_test ...