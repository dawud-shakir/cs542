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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def write_pmat(pmat_matricies: list[pmat], filename: str):
    comm = MPI.COMM_WORLD

    # Open the file
    amode = MPI.MODE_CREATE | MPI.MODE_WRONLY
    fh = MPI.File.Open(comm, filename, amode)

    file_offset = 0
    
    header = np.array([len(pmat_matricies)], dtype=np.int64).tobytes()
    fh.Write_at(0, header)

    file_offset += 8  # 8 bytes for number of matrices

    for (mat, pmat_matrix) in enumerate(pmat_matricies):
        # Write header
        if rank == 0:
            dtype_str = np.dtype(pmat_matrix.dtype).name  # e.g., 'float64'
            dtype_bytes = dtype_str.encode('utf-8')
            dtype_len = np.int32(len(dtype_bytes))  # store length as 4 bytes

            header = np.array([pmat_matrix.n, pmat_matrix.m], dtype=np.int64).tobytes()
            fh.Write_at(file_offset, header)
            fh.Write_at(file_offset + 16, dtype_len.tobytes())
            fh.Write_at(file_offset + 20, dtype_bytes)
            
            # n (8 bytes), m (8 bytes), dtype_len (4 bytes), dtype_str (dtype_len)
            
            file_offset += 4 + 16 + dtype_len

        offset = np.array(file_offset, dtype=np.int64)
        MPI.COMM_WORLD.Bcast(offset, root=0)
        file_offset = np.int64(offset)  # Convert back to int
        


        # Wait for header to be written before writing any data
        comm.Barrier()  
        
        coords = pmat_matrix.coords
        extent = pmat_matrix.extents[coords[0]][coords[1]]
        offset = pmat_matrix.offsets[coords[0]][coords[1]]

        local_rows, local_cols = extent[0], extent[1]
        row_offset, col_offset = offset[0], offset[1]

        for i in range(local_rows):
            global_row = row_offset + i
            file_offset = file_offset + (global_row * pmat_matrix.m + col_offset) * np.dtype(pmat_matrix.dtype).itemsize
            fh.Write_at(file_offset, pmat_matrix.local[i, :local_cols].tobytes())

        file_offset += pmat_matrix.n * pmat_matrix.m * np.dtype(pmat_matrix.dtype).itemsize

        print_ordered_by_rank("Rank", rank, "wrote matrix",  mat, "and is at offset", file_offset)

    fh.Close()

def read_pmat(filename: str) -> list[pmat]:
    comm = MPI.COMM_WORLD


    # Open the file
    amode = MPI.MODE_RDONLY
    fh = MPI.File.Open(comm, filename, amode)

    matrices = []
    file_offset = 0

    # file_size = fh.Get_size()
    num_matrices = np.empty(1, dtype=np.int64)
    fh.Read_at(0, num_matrices)
    file_offset += 8  # 8 bytes for number of matrices

    for mat in range(num_matrices[0]):

        header = np.empty(2, dtype=np.int64)

        fh.Read_at(file_offset, header)
        dtype_len = np.empty(1, dtype=np.int32)
        fh.Read_at(file_offset + 16, dtype_len)
        dtype_bytes = bytearray(dtype_len[0])
        fh.Read_at(file_offset + 20, dtype_bytes)
        dtype_str = dtype_bytes.decode('utf-8')
   
        dtype = np.dtype(dtype_str)
        nrows, ncols = header

        file_offset += 4 + 16 + dtype_len[0]
        
        # data_offset = 20 + dtype_len[0]
        
        # Empty pmat
        pmat_matrix = pmat(nrows, ncols)

        coords = pmat_matrix.coords
        extent = pmat_matrix.extents[coords[0]][coords[1]]
        offset = pmat_matrix.offsets[coords[0]][coords[1]]

        local_rows, local_cols = extent[0], extent[1]
        row_offset, col_offset = offset[0], offset[1]

        for i in range(local_rows):
            global_row = row_offset + i
            file_offset = file_offset + (global_row * ncols + col_offset) * np.dtype(dtype).itemsize
            buffer = bytearray(local_cols * np.dtype(dtype).itemsize)
            fh.Read_at(file_offset, buffer)
            pmat_matrix.local[i, :local_cols] = np.frombuffer(buffer, dtype=dtype)

        matrices.append(pmat_matrix)

        file_offset += nrows * ncols * np.dtype(dtype).itemsize

        print_ordered_by_rank("Rank", rank, "read matrix", mat, "and is at offset", file_offset)

    fh.Close()
    return matrices


################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)  # Create folder if it does not exist


filename = os.path.join(data_dir, "pmat.dat")

# base, ext = os.path.splitext(filename)
# i = 1
# while os.path.exists(filename):
#     filename = os.path.join(data_dir, f"pmat_{i}{ext}")
#     i += 1

# filename_only = os.path.basename(filename)  # 'pmat.py'

################################################################################



## Create pmat ##############

n, m = 1000,1000
numpy_matrix1 = np.arange(1, n * m + 1).reshape((n, m))

n, m = 1200,800
numpy_matrix2 = np.arange(1, n * m + 1).reshape((n, m))

numpy_matricies = [numpy_matrix1, numpy_matrix2]

### Write pmat to file ##############

pmat_write_matrix1 = pmat.from_numpy(numpy_matrix1)
pmat_write_matrix2 = pmat.from_numpy(numpy_matrix2)
# pmat_write_matrix.print_pretty("write_matrix", as_type="i")

matricies_to_write = [pmat_write_matrix1, pmat_write_matrix2]

write_pmat(matricies_to_write, filename)


if MPI.COMM_WORLD.rank == 0:
    print("Wrote matricies to", os.path.basename(filename))

## Read pmat back in ##############

matricies_read = read_pmat(filename)
# pmat_read_matrix.print_pretty("read_matrix", as_type="i")

for i, pmat_read_matrix in enumerate(matricies_read):
    numpy_matrix = numpy_matricies[i]
    if not np.allclose(pmat_read_matrix.get_full(), numpy_matrix):
        avg_diff = np.mean(np.abs(pmat_read_matrix.get_full() - numpy_matrix))

        if MPI.COMM_WORLD.rank == 0:
            print(f"\033[92mMatrices are not equal! Average difference: {avg_diff}\033[0m")
    else:
        if MPI.COMM_WORLD.rank == 0:
            print("pmat file read and write ...\033[91mpassed\033[0m")
