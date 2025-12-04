# mnist.py

import numpy as np
import struct           # for unpacking binary files

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

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
    # Load MNIST data from original ubyte files
    
    X_train = load_images(DATA_DIR / "train-images.idx3-ubyte") # (60000, 28, 28)
    
    y_train = load_labels(DATA_DIR / "train-labels.idx1-ubyte") # (60000,)
    
    X_test  = load_images(DATA_DIR / "t10k-images.idx3-ubyte") # (10000, 28, 28)
    
    y_test  = load_labels(DATA_DIR / "t10k-labels.idx1-ubyte") # (10000,)
    
    
    ### Preprocessing ###

    # Flatten to 2D: (60000, 28, 28) → (60000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Flatten to 2D: (10000, 28, 28) → (10000, 784)
    X_test = X_test.reshape(X_test.shape[0], -1)  

    # Normalize pixel by grayscale max value (optional)
    # X_train = (X_train.astype(np.float32) / 255.0)  
    # X_test = (X_test.astype(np.float32) / 255.0)

    # Convert test labels to int64 for compatibility
    y_train = y_train.astype(np.int64) 
    y_test = y_test.astype(np.int64)    
    return X_train, y_train, X_test, y_test