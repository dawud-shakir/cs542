# shared_memory.py
# Shared memory utilities for parallel processing

from mpi4py import MPI
import numpy as np
from sympy import per


class SharedMemoryArray:
    def __init__(self, shape, dtype='d'):
        self.shape = shape
        self.dtype = dtype
        self.size = np.prod(shape)
        self.comm = MPI.COMM_WORLD

        # Allocate a shared memory window that is size * bytes(dtype) large with a unit size of bytes(dtype) for all processes in the grid
        self.win = MPI.Win.Allocate_shared(int(self.size) * np.dtype(dtype).itemsize, np.dtype(dtype).itemsize, comm=self.comm)
        
        # Buffer is allocated by rank 0, but all processes can access it
        self.buf, _ = self.win.Shared_query(0)


        self.array = np.ndarray(buffer=self.buf, dtype=dtype, shape=shape)

    def get_array(self):
        return self.array

    def free(self):
        self.win.Free()
        self.win = None
        
    
    def __del__(self):
        if self.win is not None:
            self.free()

def create_shared_memory_array(shape, dtype='d'):
    shared_array = SharedMemoryArray(shape, dtype)
    return shared_array.get_array(), shared_array

def free_shared_memory_array(shared_array):
    shared_array.free() 



from pmat import pmat, create_grid_comm, print_matrix


def make_text_green(text):
    return f"\033[38;5;22m{text}\033[0m"

def from_numpy(M_numpy: np.ndarray) -> pmat:
    
    n, m = M_numpy.shape       

    coords = pmat.grid_comm.coords

    ########################################################################
    # Create a new pmat and its extent and position before reading 
    # numpy array from shared memory
    ########################################################################


    M_pmat = pmat(n, m)
    extent = M_pmat.extent[coords[0]][coords[1]]
    position = M_pmat.position[coords[0]][coords[1]]

    row_start, row_end = position[0], position[0] + extent[0]
    col_start, col_end = position[1], position[1] + extent[1]

    local = M_pmat.local

    ########################################################################
    # Set up the shared array
    ########################################################################
    type = M_pmat.local.dtype
    type_bytes = np.dtype(type).itemsize
    size = int(np.prod(M_pmat.shape)) * type_bytes

    win = MPI.Win.Allocate_shared(size, disp_unit=type_bytes, comm=pmat.grid_comm)

    assert win is not None, "win is None"
        
    # Buffer is allocated by rank 0, but all processes can access it
    buf, _ = win.Shared_query(0)
    shared_array = np.ndarray(buffer=buf, dtype=type, shape=(M_pmat.n, M_pmat.m))

    shared_array[:] = M_numpy[:]

    ############################################################################
    # Start an epoch and synchronize processes before reading
    ############################################################################
    win.Fence()

    # Each rank reads its submatrix block from the shared array
    local[:extent[0], :extent[1]] = shared_array[row_start:row_end, col_start:col_end]
    M_pmat.local = local.copy()  # Added copy ... didn't fix the issue
    
    ############################################################################
    # End the epoch and synchronize processes
    ############################################################################
    win.Fence()

    # Free the shared memory
    win.free()

    return M_pmat


def test_pmat_shared_memory_from_numpy(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = from_numpy(A_numpy)

    if not np.allclose(A_pmat.get_full(), A_numpy):
        raise ValueError(f"\033[38;5;22m n={n}, m={m}: submat does not match! \033[0m")
    else:
        if rank == 0:
            test_name = f"test_pmat_shared_memory_from_numpy n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")

    # if rank == 0:
    #     print("Shared Array after fencing:")
    #     print(shared_array)

    # Ensure all ranks have finished before starting the next test
    MPI.COMM_WORLD.Barrier()

##############################################################################

def remove_first_column(M: pmat) -> pmat:


    col_offset = 1  # remove first column

    assert M.m > 1, "matrix only has one column"
            
    n, m = M.shape        
    local = M.get_local()

    coords = pmat.grid_comm.coords
    extent = M.extent[coords[0]][coords[1]]
    position = M.position[coords[0]][coords[1]]

    row_start, row_end = position[0], position[0] + extent[0]
    col_start, col_end = position[1], position[1] + extent[1]

    # Set up the submatrix
    submat = pmat(n, m - 1)
    submat_extent = submat.extent[coords[0]][coords[1]]
    submat_position = submat.position[coords[0]][coords[1]]

    submat_row_start, submat_row_end = submat_position[0], submat_position[0] + submat_extent[0]
    submat_col_start, submat_col_end = col_offset + submat_position[1], col_offset + submat_position[1] + submat_extent[1]

    submat_local = np.zeros_like(submat.local)

    ########################################################################
    # Write each rank's local block to the shared array
    ########################################################################
    type = M.local.dtype
    type_bytes = np.dtype(type).itemsize
    size = int(np.prod(M.shape)) * type_bytes

    win = MPI.Win.Allocate_shared(size, disp_unit=type_bytes, comm=pmat.grid_comm)

    assert win is not None, "win is None"
        
    # Buffer is allocated by rank 0, but all processes can access it
    buf, _ = win.Shared_query(0)
    shared_array = np.ndarray(buffer=buf, dtype=type, shape=(M.n, M.m))

    # Start an epoch and synchronize processes before writing
    win.Fence()

    # Each rank writes its original matrix block to the shared array
    shared_array[row_start:row_end, col_start:col_end] = local

    ############################################################################
    # End the epoch and synchronize processes (again) before reading
    ############################################################################
    
    win.Fence()
    
    # Each rank reads its submatrix block from the shared array
    submat_local[:submat_extent[0], :submat_extent[1]] = shared_array[submat_row_start:submat_row_end, submat_col_start:submat_col_end]
    submat.local = submat_local
    
    ############################################################################
    # End the epoch and synchronize processes
    ############################################################################
    win.Fence()

    # Free the shared memory
    win.free()

    return submat

##############################################################################

def test_pmat_shared_memory_remove_column(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = pmat.from_numpy(A_numpy)
    A_pmat_submat = remove_first_column(A_pmat)

    A_numpy_submat = A_numpy[:, 1:]

    if not np.allclose(A_pmat_submat.get_full(), A_numpy_submat):
        raise ValueError(f"\033[38;5;22m n={n}, m={m}: submat does not match! \033[0m")
    else:
        if rank == 0:
            test_name = f"test_pmat_shared_memory_remove_column n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")


    # if rank == 0:
    #     print("Shared Array after fencing:")
    #     print(shared_array)

    # Ensure all ranks have finished before starting the next test
    MPI.COMM_WORLD.Barrier()

def write_pmat_to_shared_memory(M: pmat):

    # shared_array, shared_mem = create_shared_memory_array((4, 4), dtype='d')
    shared_array, shared_mem = create_shared_memory_array((M.n, M.m), dtype='d')


    # Start an epoch for RMA operations
    shared_mem.win.Fence()


    ########################################################################
    # Each rank writes its local block to the shared array
    ########################################################################

    # for p in range(num_procs):
    #     comm.Barrier()
    #     if rank == p:
    #         print("M.shape=", M.shape)
            
    local = M.get_local()

    coords = pmat.grid_comm.coords
    extent = M.extent[coords[0]][coords[1]]
    position = M.position[coords[0]][coords[1]]

    row_start, row_end = position[0], position[0] + extent[0]
    col_start, col_end = position[1], position[1] + extent[1]
    
    # shared_shape = (row_end - row_start, col_end - col_start)
    
    
    # text = f"shared_shape={shared_shape}, M.local_shape={local.shape}"
    # if local.shape != shared_shape:
    #     text = make_text_green(text + "  <-- MISMATCH!")
    # print(text)
    # print("*" * 50, '\n')

    shared_array[row_start:row_end, col_start:col_end] = local
    ############################################################################
    # End the epoch and synchronize
    shared_mem.win.Fence()

    return shared_array, shared_mem

def test_pmat_shared_memory_with_fencing(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = pmat.from_numpy(A_numpy)
    shared_array, shared_mem = write_pmat_to_shared_memory(A_pmat)


    if not np.allclose(shared_array, A_numpy):
        print("Shared Array:")
        print(shared_array)
        print("Original Matrix:")
        print(A_numpy)

        raise ValueError(f"\033[38;5;22m n={n}, m={m}: shared memory array does not match original matrix! \033[0m")
    else:
        if rank == 0:
            test_name = f"test_pmat_shared_memory_with_fencing n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")

    # if rank == 0:
    #     print("Shared Array after fencing:")
    #     print(shared_array)

    free_shared_memory_array(shared_mem)

# Example usage:
if __name__ == "__main__":
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    # # Create a shared memory array of shape (4, 4)
    # shared_array, shared_mem = create_shared_memory_array((4, 4), dtype='d')

    # # Each process writes its rank to the shared array
    # shared_array[rank, :] = rank

    # # Synchronize processes
    # comm.Barrier()

    # if rank == 0:
    #     print("Shared Array:")
    #     print(shared_array)

    # # Free the shared memory
    # free_shared_memory_array(shared_mem)




    # Want:
    # 1. Create a shared memory array for PMAT1 ( N1 x M1 )
    # 2. Each PMAT1 rank in grid writes its local block to the shared memory array
    # 3. A submatrix, PMAT2, comes along that is ( N2 x M2 )
    # 4. Calculate the local sizes for PMAT2
    # 5. Each PMAT2 rank in grid reads its local block in from PMAT1's shared memory array



    # Example: Using fencing with MPI shared memory


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    grid = create_grid_comm()

    if rank == 0:
        print(f"grid size: {grid.dims[0]} x {grid.dims[1]}")


    ##############################################################################
    # Test creating pmat from numpy using shared memory
    ##############################################################################
    n, m = 9, 5
    test_pmat_shared_memory_from_numpy(n, m)

    n, m = 8, 8
    test_pmat_shared_memory_from_numpy(n, m)

    n, m = 16, 12
    test_pmat_shared_memory_from_numpy(n, m)

    n, m = 64, 10
    test_pmat_shared_memory_from_numpy(n, m)

    n, m = 65, 10
    test_pmat_shared_memory_from_numpy(n, m)


    n, m = 28*28, 1000
    test_pmat_shared_memory_from_numpy(n, m)
    exit()

    ##############################################################################
    # Test removing first column from pmat using shared memory
    ##############################################################################
    n, m = 9, 5
    test_pmat_shared_memory_remove_column(n, m)

    n, m = 8, 8
    test_pmat_shared_memory_remove_column(n, m)

    n, m = 16, 12
    test_pmat_shared_memory_remove_column(n, m)

    n, m = 64, 10
    test_pmat_shared_memory_remove_column(n, m)

    n, m = 65, 10
    test_pmat_shared_memory_remove_column(n, m)


    ##############################################################################
    # Test writing pmat to shared memory with fencing
    # try:
    n, m = 9, 5
    test_pmat_shared_memory_with_fencing(n, m)

    n, m = 8, 8
    test_pmat_shared_memory_with_fencing(n, m)

    n, m = 16, 12
    test_pmat_shared_memory_with_fencing(n, m)

    n, m = 64, 10
    test_pmat_shared_memory_with_fencing(n, m)

    n, m = 65, 10
    test_pmat_shared_memory_with_fencing(n, m)
    # except ValueError as ve:
        
    #     if rank == 0:
    #         print(ve)
    #     MPI.Finalize
    #     exit()
