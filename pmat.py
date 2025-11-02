import numpy as np
np.set_printoptions(precision=0, suppress=True, floatmode='fixed')

from mpi4py import MPI



class pmat:
    @staticmethod
    def from_numpy(M: np.ndarray, grid_comm: MPI.Cartcomm):
        rank = grid_comm.Get_rank()
        row, col = grid_comm.Get_coords(rank)
        
        n, m = M.shape

        n_loc = np.ceil(n / grid_comm.dims[0]).astype(int)
        m_loc = np.ceil(m / grid_comm.dims[1]).astype(int)
       
        row_start = row * n_loc
        row_end = min((row + 1) * n_loc, n)
        col_start = col * m_loc
        col_end = min((col + 1) * m_loc, m)

        block = M[row_start:row_end, col_start:col_end]
        
        # Pad with zeros
        local = np.zeros((n_loc, m_loc), dtype=M.dtype)
        
        local[:block.shape[0], :block.shape[1]] = block

        # print(f"Process {rank} at coords {row},{col} has local block shape {local.shape}:\n{local}\n")

        return pmat(n, m, grid_comm, local)

    def __init__(self, n, m, grid_comm: MPI.Cartcomm, local=None):
        self.grid_comm = grid_comm
        Pr = self.grid_comm.dims[0]
        Pc = self.grid_comm.dims[1]

        self.n = n
        self.m = m
        
        self.rank = grid_comm.Get_rank()
        self.coords = self.grid_comm.Get_coords(self.rank)

        self.n_loc = np.ceil(self.n / Pr).astype(int)
        self.m_loc = np.ceil(self.m / Pc).astype(int)

        # Padding if n or m are not evenly divisible by Pr or Pc
        _, self.n_pad = divmod(self.n, Pr)
        _, self.m_pad = divmod(self.m, Pc)

        # self.print_members()
        
        self.local = np.zeros((self.n_loc, self.m_loc), dtype=np.double)
        self.local = np.ascontiguousarray(self.local)   # or self.local = self.local.copy(order='C')

        if local is not None:
            self.local[:local.shape[0], :local.shape[1]] = local    # deep copy
            
            
            # print(f"local (passed) shape: {local.shape}")
            # print(f"local (variable) shape: {self.local.shape}")
            # print(f"After copy, local: {self.local}")
        
    def print_members(self):
        if self.rank == 0:
            print(f"Pr = {Pr}, Pc = {Pc}")
            print(f"n = {n}, m = {m}")
            print(f"n_loc = {self.n_loc}, m_loc = {self.m_loc}")
            print(f"Is local contiguous? C_CONTIGUOUS={self.local.flags['C_CONTIGUOUS']}")  
        # True False

            print()
    


    def set_full(self, M):
        # Scatter full matrix M to all processes
        
        blocks = []
        for i in range(self.grid_comm.dims[0]):
            for j in range(self.grid_comm.dims[1]):
                row_start = i * self.n_loc
                row_end = min((i + 1) * self.n_loc, self.n)
                col_start = j * self.m_loc
                col_end = min((j + 1) * self.m_loc, self.m)
                
                block = M[row_start:row_end, col_start:col_end]
                blocks.append(block)

        # Scatter blocks to all processes
        local_block = self.grid_comm.scatter(blocks, root=0)
        
        # Set local block
        self.local[:local_block.shape[0], :local_block.shape[1]] = local_block

        # Do I need to do this padding if already zeroed out?

        _, n_rem = divmod(n, Pr)
        if n_rem > 0 and self.coords[0] == self.grid_comm.dims[0]-1:
                print(f"Setting extra {n_rem} rows of {self.coords} to 0")
                start = self.n_loc - n_rem
                print(f"start={start}, n_loc={self.n_loc}")
                self.local[self.n_loc - n_rem:self.n_loc+1] = 0

        _, m_rem = divmod(self.m, Pc)
        print(f"m: {self.m}, Pc: {Pc}, m//Pc: {self.m//Pc}, m_rem: {m_rem}")
        print(f"mloc: {self.m_loc}")
        if m_rem > 0 and self.coords[1] == self.grid_comm.dims[1]-1:
            print(f"Setting extra {m_rem} cols of {self.coords} to 0")
            self.local[:, self.m_loc - m_rem:self.m_loc+1] = 0

    def get_full(self):
        # Gather all blocks at root

        Pr = self.grid_comm.dims[0]
        Pc = self.grid_comm.dims[1]

        M = np.zeros((self.n_loc * Pr, self.m_loc * Pc))

        # # Only root gathers a copy of all blocks
        # all_blocks = self.grid_comm.gather(self.local, root=0)

        # All processes gather a copy of all blocks
        all_blocks = self.grid_comm.allgather(self.local)

        if all_blocks is not None:
      
            for i, block in enumerate(all_blocks):
            
                # row = i // cols
                # col = i % cols
                row, col = divmod(i, self.grid_comm.dims[1]) 

                # Add padding if needed

                if self.n_loc - block.shape[0] > 0:
                    block = np.pad(block, ((0, self.n_loc - block.shape[0]), (0,0)))
                                   
                if self.m_loc - block.shape[1] > 0:
                    block = np.pad(block, ((0,0), (0, self.m_loc - block.shape[1])))

                # Copy block into grid
                M[row*self.n_loc:(row+1)*self.n_loc, col*self.m_loc:(col+1)*self.m_loc] = block #.reshape(self.n_loc, self.m_loc)

            # Truncate to original size (n x m) in case of padding
            M = M[:self.n, :self.m]


    

        return M

    def __repr__(self):
        return f"Parallel_Matrix({self.coords[0]}, {self.coords[1]})"
    
    def __str__(self):
        return f"{self.get_full()}"
    
    def __add__(self, other):
        assert self.n == other.n and self.m == other.m, f'Add: A and B are not the same shape'
        
        return pmat(self.n, self.m, self.grid_comm, self.local + other.local)
    
    def __mul__(self, other):
        return pmat(self.n, self.m, self.grid_comm, self.local * other.local)
    
    def __neg__(self):
        return pmat(self.n, self.m, self.grid_comm, -self.local)

    def __matmul__(self, other):
        # Cannon's Algorithm

        assert self.m == other.n, f"A @ B: A.m  = {self.m} and B.n = {other.n}"

        C = np.zeros((self.n_loc, other.m_loc))

        # Total steps over self.m (or other.n) dimension
        num_steps = np.ceil(self.m / self.m_loc).astype(int) # ceil for padding

        # Deep copies for Sendrecv_replace
        A = self.local.copy()
        B = other.local.copy()

        # Skew (initial alignment)
        for _ in range(self.coords[0]):
            src, dst = self.grid_comm.Shift(1, -1)
            self.grid_comm.Sendrecv_replace(A, dest=dst, source=src)

        for _ in range(self.coords[1]):
            src, dst = self.grid_comm.Shift(0, -1)
            self.grid_comm.Sendrecv_replace(B, dest=dst, source=src)

        # # Skew (initial alignment)
        # i,j = self.coords

        # # sending A to [i-j][j]
        # # recving A from [i+j][j]

        # src, dst = self.grid_comm.Get_cart_rank([i-j,j]), self.grid_comm.Get_cart_rank([i+j,j])
        # self.grid_comm.Sendrecv_replace(A, dest=dst, source=src)

        # # sending B to [i][j-i]
        # # recving B from [i][j+i]

        # src, dst = self.grid_comm.Get_cart_rank([i,j-i]), self.grid_comm.Get_cart_rank([i,j+i])
        # self.grid_comm.Sendrecv_replace(B, dest=dst, source=src)

        for _ in range(num_steps):
            # Multiply and accumulate
            C += A @ B

            # Shift A left
            src, dst = self.grid_comm.Shift(1, -1)
            self.grid_comm.Sendrecv_replace(A, dest=dst, source=src)

            # Shift B up
            src, dst = self.grid_comm.Shift(0, -1)
            self.grid_comm.Sendrecv_replace(B, dest=dst, source=src)

        return pmat(self.n, other.m, grid_comm=self.grid_comm, local=C)
    

    @property
    def T(self):
        # For a non-square matrix, we need a separate destination array.
        
        # 1. Each process computes the local transpose of its block:
        #                       A[i][j].T
        # 2. Each process exchanges its local transpose with its transpose 
        # partner:
        #                   A[i][j].T <-> A[j][i].T 

        # Receiving type
        row_type = MPI.DOUBLE.Create_vector(self.n_loc, self.m_loc, self.m_loc)
        row_type.Commit()

        # Sending type
        col_type = MPI.DOUBLE.Create_vector(self.n_loc, 1, self.m_loc)        
        col_type.Commit()

        # Sending type
        # Repeat the column type by the number of columns (m_loc times) with a stride in bytes (unlike vector)
        mult_col_type = col_type.Create_hvector(self.m_loc, 1, self.local.dtype.itemsize)
        mult_col_type.Commit()

        # Transpose partner process in grid: A[i][j] <-> A[j][i]
        other = self.grid_comm.Get_cart_rank([self.coords[1], self.coords[0]])

        local_transpose = np.zeros((self.m_loc, self.n_loc))
        # local_transposed = np.ascontiguousarray(local_transposed)

        # if rank == 0:
        #     # Bytes described by datatype
        #     print("dtype bytes:", mult_col_type.Get_size())
        #     print("send nbytes:", self.local.nbytes)
        #     print("recv nbytes:", local_transpose.nbytes)

        self.grid_comm.Sendrecv([self.local, 1, mult_col_type], dest=other, source=other, recvbuf=[local_transpose, 1, row_type])

        # Free datatypes
        mult_col_type.Free()
        col_type.Free()
        
        row_type.Free()
        
        # Transpose dimensions
        return pmat(self.m, self.n, self.grid_comm, local_transpose) 

dtype = np.double
def maximum(scalar: dtype, M: pmat, *args, **kwargs):
    return pmat(M.n, M.m, M.grid_comm, np.maximum(scalar, M.local, *args, **kwargs))

def mean(M: pmat, *args, **kwargs):
    return pmat.from_numpy(np.mean(M.get_full(), *args, **kwargs), M.grid_comm)

# def _as_base_seq(seq):
#     return [parallel_matrix_2._to_base(x) for x in seq]

# def _hstack(tup, *args, **kwargs):
#     res = np.hstack(_as_base_seq(tup), *args, **kwargs)
#     return _wrap(res)

# def _concatenate(tup, *args, **kwargs):
#     res = np.concatenate(_as_base_seq(tup), *args, **kwargs)
#     return _wrap(res)

# def _stack(tup, *args, **kwargs):
#     res = np.stack(_as_base_seq(tup), *args, **kwargs)
#     return _wrap(res)

# def _mean(a, *args, **kwargs):
#     base = parallel_matrix_2._to_base(a)
#     res = np.mean(base, *args, **kwargs)
#     # mean over all axes can return a scalar; only wrap ndarrays
#     return res if np.isscalar(res) else _wrap(res)


def test(n, k, m, dtype=np.double):
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()

    # Process grid and panel width
    Pr = int(np.sqrt(num_procs))
    Pc = int(np.sqrt(num_procs))

    dims = [Pr, Pc]
    periods = [True, True]
    grid_comm = comm.Create_cart(dims, periods, reorder=True)


    def check(M1: pmat, M2: np.ndarray, str=""):
        M1_pmat = M1.get_full()
        if rank == 0:
            assert np.allclose(M1_pmat, M2), f"{str} failed allclose\nM1:\n{M1_pmat}\nM2:\n{M2}"
            
            print(f"\t{str}\t\t\t\t... passed allclose")

    ################################################################
    # Start tests
    ################################################################

    if rank == 0:           
        print(f"Testing n={n}, k={k}, m={m}...")
    
    A_mat = np.arange(1, n * k + 1).reshape(n, k).astype(dtype)
    B_mat = np.arange(1, k * m + 1).reshape(k, m).astype(dtype)

    A = pmat.from_numpy(A_mat, grid_comm)

    ################################################################
    # Non-Element-wise operations

    # Initial matrix
    check(A, A_mat, "A")

    # Transpose
    check(A.T, A_mat.T, "A.T")

    # Negation
    check(-A, -A_mat, "-A")

    
    # Matrix Multiply
    B = pmat.from_numpy(B_mat, grid_comm)
    check(A @ B, A_mat @ B_mat, "A @ B")

    ################################################################
    # Element-wise operations

    # Addition
    check(A + A, A_mat + A_mat, "A + A")

    # Multiplication
    check(A * A, A_mat * A_mat, "A * A")

    ################################################################
    # Function operations

    # Maximum
    check(maximum(32, A), np.maximum(32, A_mat), "maxmium(32, A)")


    ################################################################
    # End tests
    ################################################################

    if rank == 0:
        print()

if __name__ == "__main__":
    # Test input

    # Square matrices
    test(n=8, k=8, m=16)

    # Non-square matrices
    test(n=9, k=5, m=16)



