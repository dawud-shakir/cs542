import numpy as np
from mpi4py import MPI

import os

# import struct

# """ Data """
# def load_images(filename):
#     with open(filename, 'rb') as f:
#         # First 16 bytes: magic number, num images, rows, cols
#         magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
#         if magic != 2051:
#             raise ValueError("Invalid magic number in image file!")
#         # Read the rest: unsigned bytes → uint8
#         data = np.frombuffer(f.read(), dtype=np.uint8)
#         data = data.reshape(num, rows, cols)  # shape: (num_images, 28, 28)
#         return data

# def load_labels(filename):
#     with open(filename, 'rb') as f:
#         # First 8 bytes: magic number, num labels
#         magic, num = struct.unpack(">II", f.read(8))
#         if magic != 2049:
#             raise ValueError("Invalid magic number in label file!")
#         # Read the rest: unsigned bytes → uint8
#         labels = np.frombuffer(f.read(), dtype=np.uint8)
#         return labels


from pmat import pmat, print_ordered_by_rank


import psutil         # for memory usage
proc = psutil.Process(os.getpid())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Starting memory usage report...")



t0 = MPI.Wtime()


rank_start_memory = proc.memory_info().rss

p_X_train, size_X_train = pmat.from_file('X_train.dat')
p_y_train, size_y_train = pmat.from_file('y_train.dat')
p_X_test, size_X_test = pmat.from_file('X_test.dat')
p_y_test, size_y_test = pmat.from_file('y_test.dat')

rank_end_memory = proc.memory_info().rss

t1 = MPI.Wtime()
total = t1 - t0
comm.reduce(total, op=MPI.MAX, root=0)


total_size = size_X_train + size_y_train + size_X_test + size_y_test

# comm.Barrier()

if rank == 0:
    print("After loading data memory usage report...")

print_ordered_by_rank(f"rank {rank}, memory (RSS): +{(rank_end_memory - rank_start_memory) / 1024**2:.2f} MB")

# comm.Barrier()

if rank == 0:
    print(f"X_train shape: {p_X_train.shape}, y_train shape: {p_y_train.shape}")
    print(f"X_test shape: {p_X_test.shape}, y_test shape: {p_y_test.shape}")




if rank == 0:
    print(f"Loaded p_X_train of size {size_X_train / 1024**2:.2f} MB")
    print(f"Loaded p_y_train of size {size_y_train / 1024**2:.2f} MB")
    print(f"Loaded p_X_test of size {size_X_test / 1024**2:.2f} MB")
    print(f"Loaded p_y_test of size {size_y_test / 1024**2:.2f} MB")
    print(f"Total loaded data size: {total_size / 1024**2:.2f} MB")
    print(f"Time taken to load data: {total:.4f} seconds")