from networkx import kosaraju_strongly_connected_components
import numpy as np
np.set_printoptions(precision=0, suppress=True, floatmode='fixed')

from mpi4py import MPI



class pmat:
    @staticmethod
    def from_numpy(M_numpy: np.ndarray, grid_comm: MPI.Cartcomm):
        rank = grid_comm.Get_rank()
        row, col = grid_comm.Get_coords(rank)
        
        n, m = M_numpy.shape

        n_loc = np.ceil(n / grid_comm.dims[0]).astype(int)
        m_loc = np.ceil(m / grid_comm.dims[1]).astype(int)
       
        row_start = row * n_loc
        row_end = min((row + 1) * n_loc, n)
        col_start = col * m_loc
        col_end = min((col + 1) * m_loc, m)

        block = M_numpy[row_start:row_end, col_start:col_end]
        
        # Pad with zeros
        local = np.zeros((n_loc, m_loc), dtype=np.double)
        
        local[:block.shape[0], :block.shape[1]] = block

        # print(f"Process {rank} at coords {row},{col} has local block shape {local.shape}:\n{local}\n")

        return pmat(n, m, grid_comm, local)

    def __init__(self, n, m, grid_comm: MPI.Cartcomm, local=None):
        self.grid_comm = grid_comm
        Pr = self.grid_comm.dims[0]
        Pc = self.grid_comm.dims[1]

        self.n = n
        self.m = m
        self.shape = (n, m)
        self.ndim = 2
        
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
        return f"pmat({self.n}x{self.m}) at coords {self.coords} with local shape {self.local.shape})"
    
    def __str__(self):
        return f"{self.get_full()}"
    
    def astype(self, dtype):
        return pmat(self.n, self.m, self.grid_comm, self.local.astype(dtype))

    def __gt__(self, other):
        return pmat(self.n, self.m, self.grid_comm, self.local > other)
    
    def __add__(self, other):
        assert self.n == other.n and self.m == other.m, f'Add: A and B are not the same shape'
        
        return pmat(self.n, self.m, self.grid_comm, self.local + other.local)
    
    def __sub__(self, other):        
        return pmat(self.n, self.m, self.grid_comm, self.local - other.local)

    def __mul__(self, other):
        return pmat(self.n, self.m, self.grid_comm, self.local * other.local)
    
    def __neg__(self):
        return pmat(self.n, self.m, self.grid_comm, -self.local)

    def __matmul__(self, other: 'pmat'):
        # Cannon's Algorithm

        assert self.m == other.n, f"A @ B: A.m  = {self.m} and B.n = {other.n}"

        C = np.zeros((self.n_loc, other.m_loc))

        # Total steps over self.m (or other.n) dimension
        # num_steps = np.ceil(self.m / self.m_loc).astype(int) # ceil for padding

        num_steps = min(self.grid_comm.dims)


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
    
    def __len__(self):
        return len(self.local)

    def __getitem__(self, idx):
        # val = self.data[idx]
        # return MyArray(val) if isinstance(val, np.ndarray) else val

        full = self.get_full()
        val = full[idx]
        return pmat.from_numpy(val, self.grid_comm) if isinstance(val, np.ndarray) else val

    # def get_start_end(slc, dim_size):
    #     if isinstance(slc, slice):
    #         start = slc.start if slc.start is not None else 0
    #         end = slc.stop if slc.stop is not None else dim_size
    #         return start, end
    #     elif isinstance(slc, int):
    #         return slc, slc+1



    # def __setitem__(self, idx, value):
    #     if isinstance(idx, int):
    #         # handle single integer index    
            
        # self.data[idx] = value


    @property
    def T(self):        
        # 1. Each process computes the local transpose of its block:
        #                       A[i][j].T


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

        # 2. Each process exchanges its local transpose with its transpose 
        # partner:
        #                   A[i][j].T <-> A[j][i].T 

        # Transpose partner process in grid
        other = self.grid_comm.Get_cart_rank([self.coords[1], self.coords[0]])

        # For a non-square matrix, we need a separate destination array.
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
    

   
    # # Removes pylance warning 
    # def __array__(self, dtype=None):
    #     # how to convert to a bare numpy array
    #     return np.asarray(self.local, dtype=dtype)

    # Universal functions (np.exp, np.add, etc.)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):        # method = "__call__" or "reduce", etc.

        # Convert inputs to numpy arrays
        arrays = [np.asarray(x.local) if isinstance(x, pmat) else x
                  for x in inputs]

        if ufunc is np.log and self.n_pad > 0 and self.coords[0]==self.grid_comm.dims[0]-1:
            arrays = [np.asarray(x.local[:self.n_loc-self.n_pad, :self.m_loc]) if isinstance(x, pmat) else x
                  for x in inputs]


        
        if ufunc is np.log and self.m_pad > 0 and self.coords[1]==self.grid_comm.dims[1]-1: 
            arrays = [np.asarray(x.local[:self.n_loc, :self.m_loc-self.m_pad]) if isinstance(x, pmat) else x
                  for x in inputs]


        # # Replace zeros with nan to avoid -inf
        # #### For log only
        # if ufunc is np.log:
        # arrays = [np.where(x == 0, np.nan, x) if isinstance(x, np.ndarray) else x
        #         for x in arrays]
        
        result = getattr(ufunc, method)(*arrays, **kwargs)

        # Wrap back into your class if it returns an array
        if isinstance(result, np.ndarray):
            return pmat(self.n, self.m, self.grid_comm, result)
        else:
            # e.g. scalar results from reduction
            return result


    #     # ------------------------------------------------------
    # # Handle non-ufunc NumPy functions (np.mean, np.concatenate, etc.)
    # # ------------------------------------------------------
    def __array_function__(self, func, types, args, kwargs):
        """
        Called for non-ufunc NumPy functions that support the array function protocol.
        """
        # Only handle functions that explicitly support MyArray
        # if func is np.concatenate:
        #     arrays = [self.local if isinstance(a, pmat) else a for a in args[0]]
        #     return pmat(np.concatenate(arrays, **kwargs))
        # elif func is np.mean:
        #     a = args[0]
        #     if isinstance(a, pmat):
        #         a = self.local
        #     return pmat(np.mean(a, **kwargs))

        if func is np.ones_like:
            other = args[0]
            if isinstance(other, pmat):
                return pmat.from_numpy(np.ones((other.n, other.m)), other.grid_comm)
        elif func is np.max:
            other = args[0]
            if isinstance(other, pmat):
                return pmax(other, **kwargs)
        elif func is np.sum:
            other = args[0]
            if isinstance(other, pmat):
                return psum(other, **kwargs)


        # Unknown function — defer to NumPy’s default
        return NotImplemented
    

def psum(M: pmat, *args, **kwargs):
    axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

    assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

    if axis == 1:
        horz_group = M.grid_comm.Sub([False, True])  # rows
        row_sum = []

        for row in range(M.n_loc):
            # Reduce to the root of each group
            row_sum.append(horz_group.reduce(np.sum(M.local[row, :]), op=MPI.SUM, root=0))

        if M.grid_comm.coords[1] == 0:
            row_sum = np.array([x for x in row_sum if x is not np.nan]).flatten().reshape(-1, 1) # maxs is now a column vector
        else:
            row_sum = None

        return pmat(M.n, 1, M.grid_comm, row_sum)

def pmax(M: pmat, *args, **kwargs):
    axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

    assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

    if axis == 1:
        horz_group = M.grid_comm.Sub([False, True])  # rows
        row_max = []

        for row in range(M.n_loc):
            # Reduce to the root of each group
            row_max.append(horz_group.reduce(np.max(M.local[row, :]), op=MPI.MAX, root=0))

        if M.grid_comm.coords[1] == 0:
            row_max = np.array([x for x in row_max if x is not np.nan]).flatten().reshape(-1, 1) # maxs is now a column vector
        else:
            row_max = None

        return pmat(M.n, 1, M.grid_comm, row_max)

def pmean(M: pmat, *args, **kwargs):
    axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

    assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

    if axis == 1:
        horz_group = M.grid_comm.Sub([False, True])  # rows
        row_reduction = []

        for row in range(M.n_loc):
            # Reduce to the root of each group
            row_reduction.append(horz_group.reduce(np.sum(M.local[row, :]), op=MPI.SUM, root=0))

        if M.grid_comm.coords[1] == 0:
            # Remove Nones and make a column vector
            row_reduction = np.array([x for x in row_reduction if x is not np.nan]).flatten().reshape(-1, 1)
            
            # Divide by total number of columns
            row_reduction = row_reduction / M.m
        else:
            # Process is not a root
            row_reduction = None

        return pmat(M.n, 1, M.grid_comm, row_reduction)
    
dtype = np.double
def maximum(scalar: dtype, M: pmat, *args, **kwargs):
    return pmat(M.n, M.m, M.grid_comm, np.maximum(scalar, M.local, *args, **kwargs))



def print_pmat_on_rank0(M: pmat, msg=""):
    comm = M.grid_comm
    rank = comm.Get_rank()

    s = str(M)
    if rank == 0:
        # msg = msg + ":" if len(msg) > 0 else msg
        print(f"{msg}\n{s}\n")

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
            assert isinstance(M1, pmat) and isinstance(M2, np.ndarray), f"{str} failed instance check\ntype(M1)={type(M1)} and type(M2)={type(M2)}"
            
            assert np.allclose(M1_pmat, M2), f"{str} failed allclose\nM1:\n{M1_pmat}\nM2:\n{M2}"

            # assert M1_pmat.dtype == M2.dtype, f"{str} failed type check\ndtype(M1)={M1_pmat.dtype} and dtype(M2)={M2.dtype}"
            
            print(f"\t{str}\t\t\t\t... passed allclose")

    ################################################################
    # Start tests
    ################################################################

    if rank == 0:           
        print(f"Testing n={n}, k={k}, m={m}...")
    
    A_mat = np.arange(1, n * k + 1).reshape(n, k).astype(dtype)
    B_mat = np.arange(1, k * m + 1).reshape(k, m).astype(dtype)
    D_mat = np.arange(1, m * n + 1).reshape(m, n).astype(dtype)

    A = pmat.from_numpy(A_mat, grid_comm)

    
    check(np.ones_like(A), np.ones_like(A_mat), "ones_like")
    check(np.maximum(32, A), np.maximum(32, A_mat), "maximum(32, A)")

    ###### Too large values cause exp to overflow/go to inf (normally, -1 to 1 range)
    #######################check(np.exp(A), np.exp(A_mat), "exp(A)")
    #######################check(np.exp(A).astype(float), np.exp(A_mat).astype(float), "astype")
    check(np.log(A), np.log(A_mat), "log(A)")

    check(A > 5, A_mat > 5, "A > 5")
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

    D = pmat.from_numpy(D_mat, grid_comm)
    check(A @ B @ D, A_mat @ B_mat @ D_mat, "A @ B @ D")

    ################################################################
    # Element-wise operations

    # Addition
    check(A + A, A_mat + A_mat, "A + A")

    # Subtraction
    check(A - A, A_mat - A_mat, "A - A")

    # Multiplication
    check(A * A, A_mat * A_mat, "A * A")

    ################################################################
    # Function operations

    # Maximum
    check(maximum(32, A), np.maximum(32, A_mat), "maxmium(32, A)")

    check(pmean(A, axis=1), np.mean(A_mat, axis=1, keepdims=True), "mean(A, axis=1)")

    check(pmax(A, axis=1), np.max(A_mat, axis=1, keepdims=True), "mean(A, axis=1)")

    ################################################################
    # Vecmat and Matvec operation 
    
    # Row vector = n...1
    row_vec = np.array(np.arange(n+1, 1, -1), ndmin=2).reshape(1, -1)
    row_pvec = pmat.from_numpy(row_vec, grid_comm)
    check(row_pvec @ A, row_vec @ A_mat, "vecmat")

    # Column vector = k...1
    col_vec = np.array(np.arange(k+1, 1, -1), ndmin=2).reshape(-1, 1) 
    col_pvec = pmat.from_numpy(col_vec, grid_comm)
    check(A @ col_pvec, A_mat @ col_vec, "matvec")

    ################################################################
    # End tests
    ################################################################

    if rank == 0:
        print()

# if __name__ == "__main__":
#     # Test input

        # Large matrices
#     test(n=1000, k=2000, m=10000)

#     test(n=28*28, k=64, m=64)

#     # Non-square matrices
#     # test(n=9, k=5, m=16)

#     # Square matrices
#     # test(n=8, k=8, m=16)





