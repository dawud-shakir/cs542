# test_pmat.py

import numpy as np

np.set_printoptions(precision=1, suppress=True, floatmode='fixed')

from mpi4py import MPI
from pmat import pmat, create_grid_comm, print_matrix
grid = create_grid_comm()
rank = grid.Get_rank()
size = grid.Get_size()
coords = grid.coords

sizes = [
    # (10, 10),
    # (100, 50),
    # (50, 100),
    # (200, 200),
    # (1000, 2000, 10000),
    # (28*28, 64, 64),
    # (0, 0, 0),        ### doesn't work... yet
    (9, 5, 16),
    (8, 8, 16),
    (16, 12, 16),
    (64, 10, 16),
    (65, 10, 16),
    (28*28, 1000, 16),
    (64, 100, 16),
    (10, 100, 16)
]



################################################################
# Functions to test
################################################################

# def all_procs_available():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()
#     if rank != 0:
#         comm.send(1, dest=0, tag=123)  # Signal that this rank called it
#     else:
#         count = 1  # Rank 0 counts itself
#         for i in range(1, size):
#             if comm.iprobe(source=i, tag=123):  # Check if message is available (non-blocking)
#                 comm.recv(source=i, tag=123)  # Receive it
#                 count += 1
#         if count == size:
#             print("All ranks called the function.")
#         else:
#             print(f"Only {count}/{size} ranks called the function.")



def from_shared_buffer(win: MPI.Win, shared_array: np.ndarray, dtype=np.float64) -> pmat:
    ########################################################################
    # Create a new pmat and its extent and position before reading 
    # numpy array from shared memory
    ########################################################################
    coords = pmat.grid_comm.coords
    n, m = shared_array.shape        
    
    # Set up the extented matrix
    newmat = pmat(n, m, dtype=dtype)
    newmat_extent = newmat.extents[coords[0]][coords[1]]
    newmat_position = newmat.offsets[coords[0]][coords[1]]

    newmat_row_start, newmat_row_end = newmat_position[0], newmat_position[0] + newmat_extent[0]
    newmat_col_start, newmat_col_end = newmat_position[1], newmat_position[1] + newmat_extent[1]

    newmat_local = newmat.local
    
    ############################################################################
    # Synchronize processes before reading
    ############################################################################
    win.Fence()
    
    # Each rank reads its submatrix block from the shared array
    newmat_local[:newmat_extent[0], :newmat_extent[1]] = shared_array[newmat_row_start:newmat_row_end, newmat_col_start:newmat_col_end]
    newmat.local = newmat_local
    
    ############################################################################
    # End the epoch and synchronize processes again
    ############################################################################
    win.Fence()

    return newmat


def fancy_indexing_rows(A: pmat, idx):
    # idx can be a scalar, a slice, a list, or an array of integers
    # idx can have duplicates and be out of order
    
    if np.isscalar(idx):
        idx = np.array([idx])
    elif isinstance(idx, list):
        idx = np.array(idx)
    elif isinstance(idx, slice):
        # Convert slice to array of indices    
        idx = np.arange(idx.start, idx.stop, idx.step or 1)
        # Alternative: idx = np.r_[idx]

    idx = np.atleast_2d(idx)

    if idx.shape[0] != 1:
        raise NotImplementedError("Only 1D row vector indexes are supported")

    
    # idx is a row vector: (1, n_elements)
    n_elements = idx.shape[1]  # number of rows to select

    ########################################################################
    # Set up the shared array
    ########################################################################
    type = A.dtype

    type_bytes = np.dtype(type).itemsize
    shape = (n_elements, A.m)
    size = int(np.prod(shape)) * type_bytes

    win = MPI.Win.Allocate_shared(size, disp_unit=type_bytes, comm=pmat.grid_comm)

    assert win is not None, "win is None"
        
    # Buffer is allocated by rank 0, but all processes can access it
    buf, _ = win.Shared_query(0)
    shared_array = np.ndarray(buffer=buf, dtype=type, shape=shape)

    ############################################################################
    # 
    for (B_row, A_row) in enumerate(idx[0]):
        # if rank == 0: 
        #     print("A_row:", A_row, "B_row:", B_row)

        # Compute grid row rank in A and local row index
        A_block, A_block_row  = divmod(A_row, A.n_loc)

        if A_block == A.coords[0]:
            # Ranks that made it here have a nonempty block if they are in A's row comm
            row_comm = A.row_comm
            
            # The row allgather copies into needs to be the full nonempty row size (with padding) because of the way allgather works

            row = np.empty((A.m_loc * row_comm.Get_size()), dtype=A.dtype)
            A.row_comm.Allgather(A.local[A_block_row], row)
            row = row[:A.m]  # Trim any padding from the last nonempty block
            
            shared_array[B_row] = row

            # if A.coords[1] == 0:
                
                # print("A_block=",A_block,"A_block_row=",A_block_row )
                # print(row[0])
                # B_numpy[B_row] = row[0] ### Won't work ... different B_numpy for each rank
        
        # if rank == 0:
        #     print()
        # grid.Barrier()

    # Synchronize writes across all ranks (this is done in from_shared_buffer as well, but just to be safe)
    win.Fence()

    # Convert shared array to a pmat 
    B = from_shared_buffer(win, shared_array, dtype=A.dtype)

    # Clean up
    win.Free()

    return B


# A  and idx as a 1D array or list of integers (e.g., [0, 3, 9, ...]), you can write A[idx] to perform fancy indexing on the rows. This selects the rows at the specified indices.


################################################################
# End functions to test
################################################################




def check(pmat_matrix: pmat, numpy_matrix: np.ndarray, str=""):
    pmat_as_numpy = pmat_matrix.get_full()
    if grid.rank == 0:
        assert isinstance(pmat_matrix, pmat) and isinstance(numpy_matrix, np.ndarray), f"{str} failed instance check\ntype(M1)={type(pmat_matrix)} and type(M2)={type(numpy_matrix)}"
        
        assert np.allclose(pmat_as_numpy, numpy_matrix, equal_nan=True), f"{str} failed allclose\npmat:\n{pmat_as_numpy}\nnumpy:\n{numpy_matrix}"

        # assert M1_pmat.dtype == M2.dtype, f"{str} failed type check\ndtype(M1)={M1_pmat.dtype} and dtype(M2)={M2.dtype}"
        
        print(f"\t{str:<20}...\033[31mpassed\033[0m allclose")

    grid.Barrier()

def check_scalar(pmat_value, numpy_value, str=""):
    if not np.isclose(pmat_value, numpy_value, equal_nan=True): # allclose does not consider NaN == NaN unless told to
        if grid.rank == 0:
            print(f"{str} failed isclose\npmat_value:\n{pmat_value}\nnumpy_value:\n{numpy_value}")
        exit()
    else:
        if grid.rank == 0:
             print(f"\t{str:<20}...\033[31mpassed\033[0m isclose")

    grid.Barrier()

def test(n, k, m, dtype=np.double):

    ################################################################
    # Start tests
    ################################################################

    if grid.rank == 0:           
        print(f"Testing n={n}, k={k}, m={m}...")
    
    A_numpy = (np.arange(1, n * k + 1) / (n * k)).reshape(n, k).astype(dtype)    
    A_pmat = pmat.from_numpy(A_numpy)


    # idx = np.array([3, 1, 2, 3, 0])

    # idx = slice(0, 4)  # slice object
    # idx = [6, 8, 3, 0, 1, 2, 5, 7, 4] 
    # idx = np.random.choice(n, size=min(1,100), replace=True)
    # idx = np.random.permutation(n)[:min(10, n)] # random indices 

    # idx = idx.tolist()  # convert to list for testing
    # if rank == 0:
    #     print("idx:", idx)
    #     if isinstance(idx, np.ndarray):
    #         print("idx.shape:", idx.shape, "idx.dtype:", idx.dtype)
    #     else:
    #         print("idx.len:", len(idx))
        
    #     print()
    
    np.random.seed(0)
    indicies = np.random.permutation(np.arange(n))  

    # if grid.rank == 0:
    #     indicies = np.random.permutation(np.arange(n))
    # else:
    #     indicies = np.empty(n, dtype=int)
    # grid.Bcast(indicies, root=0)  # Broadcast to all ranks


    # Split into batches
    n_batches = 5
    batches_idx = np.array_split(indicies, n_batches)  

    for (batch_num, idx) in enumerate(batches_idx):
            # Get batch data
        if rank == 0:
            print(f"  Batch {batch_num + 1}/{n_batches}, idx: {idx}")
        
        B_pmat = fancy_indexing_rows(A_pmat, idx)
        B_numpy = A_numpy[idx]

        check(B_pmat, B_numpy, "indexing rows")

    ################################################################
    # End tests
    ################################################################

if __name__ == "__main__":
    
    # A = pmat.from_numpy(np.arange(20).reshape(5,4))
    # idx = [3, 1, 2, 3, 0]
    
    
    # A.print_pretty("A")
    
    # if rank == 0:
    #     print("idx:", idx)
    #     print()

    # B = fancy_indexing_rows(A, idx)
    
    # import traceback
    # try:
    for (n, k, m) in sizes:
            test(n, k, m)
    # except Exception as e:
    #     tb_str = traceback.format_exc()

       #### What if exceptions are different on different ranks? #### 
    
    #     if grid.rank == 0:
    #         print("Exception caught:\n", tb_str)