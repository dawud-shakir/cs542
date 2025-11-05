from hmac import new
import warnings
import numpy as np
np.set_printoptions(precision=0, suppress=True, floatmode='fixed')

from mpi4py import MPI

def create_grid_comm():
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()

    # Grid rows and columns
    Pr = int(np.sqrt(num_procs))
    Pc = int(np.sqrt(num_procs))

    dims = [Pr, Pc]
    periods = [True, True]
    return comm.Create_cart(dims, periods, reorder=True)


class pmat:
    ############################################################################
    # Static members and methods
    ############################################################################

    # Static grid communicator shared by all pmats
    grid_comm = create_grid_comm()

    @staticmethod
    def from_numpy(M_numpy: np.ndarray):
        """
        Convert a full numpy array to a distributed pmat
        """
        rank = pmat.grid_comm.Get_rank()
        row, col = pmat.grid_comm.Get_coords(rank)
        
        n, m = M_numpy.shape

        n_loc = np.ceil(n / pmat.grid_comm.dims[0]).astype(int)
        m_loc = np.ceil(m / pmat.grid_comm.dims[1]).astype(int)
       
        row_start = row * n_loc
        row_end = min((row + 1) * n_loc, n)
        col_start = col * m_loc
        col_end = min((col + 1) * m_loc, m)

        block = M_numpy[row_start:row_end, col_start:col_end]
        
        # Pad with zeros
        local = np.zeros((n_loc, m_loc), dtype=np.double)
        # Copy from numpy block up to padding        
        local[:block.shape[0], :block.shape[1]] = block

        return pmat(n, m, local)
    
    @staticmethod
    def resize(top: int, bottom: int, left: int, right: int, M: 'pmat'):
        # grid_comm = create_grid_comm()
        # new_M = pmat(bottom-top, right-left, grid_comm)   # start with all zeros

        full = M.get_full()[top:bottom, left:right]
        new_M = pmat.from_numpy(full)

        return new_M


    ############################################################################
    # Constructor
    ############################################################################

    def __init__(self, n, m, local=None):

        Pr = pmat.grid_comm.dims[0]
        Pc = pmat.grid_comm.dims[1]

        self.n = n
        self.m = m
        self.shape = (n, m)
        self.ndim = 2
        
        self.rank = pmat.grid_comm.Get_rank()
        self.coords = pmat.grid_comm.Get_coords(self.rank)

        self.n_loc = np.ceil(self.n / Pr).astype(int)
        self.m_loc = np.ceil(self.m / Pc).astype(int)
        self._local_shape = (self.n_loc, self.m_loc)

        # Padding if n or m are not evenly divisible by Pr or Pc
        _, self.n_pad = divmod(self.n, Pr)
        _, self.m_pad = divmod(self.m, Pc)
        self.is_padded = (self.n_pad > 0) or (self.m_pad > 0)

        self._set_local(local)

 

    ############################################################################
    # Accessors and string representations
    ############################################################################

    def __repr__(self):
        return f"pmat({self.n}x{self.m}) at coords {self.coords} with local shape {self.local.shape})"
    
    def __str__(self):
        return f"{self.get_full()}"
    
    def astype(self, dtype):
        return pmat(self.n, self.m, self.local.astype(dtype))
    
    def __getitem__(self, idx):
        # val = self.data[idx]
        # return MyArray(val) if isinstance(val, np.ndarray) else val

        full = self.get_full()
        val = full[idx]
        return pmat.from_numpy(val) if isinstance(val, np.ndarray) else val

    def remove_first_column(self):
        new_n = self.n
        new_m = self.m - 1

        other = pmat(new_n, new_m)

        horz_group = pmat.grid_comm.Sub([False, True])  # rows
        horz_group.periods = ([False, False])

        # for col in range(pmat.grid_comm.dims[1] - 1):

        if horz_group.rank < horz_group.dims[1] - 1:
            horz_group.Recv(other.local[-1,:], horz_group.rank + 1)

        if horz_group.rank > 0:
            horz_group.Send(self.local[0,:], horz_group.rank - 1)
                                
        #     pmat.grid_comm.Sendrecv(self.local[0,:], 
        
        # other.local[-1,:]

        # grid_comm = create_grid_comm()
        # new_M = pmat(bottom-top, right-left, grid_comm)   # start with all zeros


        # axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        # assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

        # row_sum = []

        #     for row in range(M.n_loc):
        #         # Reduce to the root of each group
        #         row_sum.append(horz_group.reduce(np.sum(M.local[row, :]), op=MPI.SUM, root=0))

        #     if M.grid_comm.coords[1] == 0:
        #         row_sum = np.array([x for x in row_sum if x is not np.nan]).flatten().reshape(-1, 1) # maxs is now a column vector
        #     else:
        #         row_sum = None

        #     return pmat(M.n, 1, row_sum)







        full = M.get_full()[top:bottom, left:right]
        new_M = pmat.from_numpy(full)



        return new_M
    
    def _set_local(self, local):
        # Local block will be (n_pad, m_pad) larger than local if there is zero padding.
        self.local = np.zeros((self.n_loc, self.m_loc), dtype=np.double)
        self.local = np.ascontiguousarray(self.local)
        
        if local is not None:
            self.local[:local.shape[0], :local.shape[1]] = local    # deep copy

    def set_full(self, M):
        blocks = []
        for i in range(pmat.grid_comm.dims[0]):
            for j in range(pmat.grid_comm.dims[1]):
                row_start = i * self.n_loc
                row_end = min((i + 1) * self.n_loc, self.n)
                col_start = j * self.m_loc
                col_end = min((j + 1) * self.m_loc, self.m)
                
                block = M[row_start:row_end, col_start:col_end]
                blocks.append(block)

        # Scatter blocks to all processes
        local_block = pmat.grid_comm.scatter(blocks, root=0)
        
        # Set local block
        self.local[:local_block.shape[0], :local_block.shape[1]] = local_block

    def get_full(self):
        # Gather all blocks at root
        Pr = pmat.grid_comm.dims[0]
        Pc = pmat.grid_comm.dims[1]

        M = np.zeros((self.n_loc * Pr, self.m_loc * Pc))

        # All processes gather a copy of all blocks
        all_blocks = pmat.grid_comm.allgather(self.local)

        if all_blocks is not None:
      
            for i, block in enumerate(all_blocks):
                row, col = divmod(i, pmat.grid_comm.dims[1]) 

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

    ############################################################################
    # Arithmetic operators
    ############################################################################

    def __gt__(self, other):
        return pmat(self.n, self.m, self.local > other)
    
    def __add__(self, other):
        if np.isscalar(other):
            return pmat(self.n, self.m, self.local + other)
        else:
            return pmat(self.n, self.m, self.local + other.local)
    
    def __sub__(self, other):        
        return pmat(self.n, self.m, self.local - other.local)

    def __mul__(self, other):
        if np.isscalar(other):
            return pmat(self.n, self.m, self.local * other)
        else:
            return pmat(self.n, self.m, self.local * other.local)
    
    def __rmul__(self, other):
        # Handle scalar * pmat (right multiplication)
        if np.isscalar(other):
            return pmat(self.n, self.m, other * self.local)
        else:
            return NotImplemented
        
    def __truediv__(self, other):
        if np.isscalar(other):
            return pmat(self.n, self.m, self.local / other)
        else:
            return pmat(self.n, self.m, self.local / other.local)
    
    def __rdiv__(self, other):
        # Handle scalar / pmat (right division)
        if np.isscalar(other):
            return pmat(self.n, self.m, other / self.local)
        else:
            return NotImplemented
    
    def __rtruediv__(self, other):
        # Handle scalar / pmat (right true division for Python 3)
        if np.isscalar(other):
            return pmat(self.n, self.m, other / self.local)
        else:
            return NotImplemented

    def __radd__(self, other):
        # Handle scalar + pmat (right addition)
        if np.isscalar(other):
            return pmat(self.n, self.m, other + self.local)
        else:
            return NotImplemented
    
    def __neg__(self):
        return pmat(self.n, self.m, -self.local)

    def __matmul__(self, other: 'pmat'):
        # Cannon's Algorithm

        assert self.m == other.n, f"A @ B: A.m  = {self.m} and B.n = {other.n}"

        C = np.zeros((self.n_loc, other.m_loc))

        # Total steps over grid 
        num_steps = min(pmat.grid_comm.dims)

        # Deep copies for Sendrecv_replace
        A = self.local.copy()
        B = other.local.copy()

        # Skew (initial alignment)
        for _ in range(self.coords[0]):
            src, dst = pmat.grid_comm.Shift(1, -1)
            pmat.grid_comm.Sendrecv_replace(A, dest=dst, source=src)

        for _ in range(self.coords[1]):
            src, dst = pmat.grid_comm.Shift(0, -1)
            pmat.grid_comm.Sendrecv_replace(B, dest=dst, source=src)

        for _ in range(num_steps):
            # Multiply and accumulate
            C += A @ B

            # Shift A left
            src, dst = pmat.grid_comm.Shift(1, -1)
            pmat.grid_comm.Sendrecv_replace(A, dest=dst, source=src)

            # Shift B up
            src, dst = pmat.grid_comm.Shift(0, -1)
            pmat.grid_comm.Sendrecv_replace(B, dest=dst, source=src)

        return pmat(self.n, other.m, local=C)
    
    @property
    def T(self):        
        ########################################################################
        # 1. Each process computes the local transpose of its block:
        #                       A[i][j].T
        ########################################################################

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

        ########################################################################
        # 2. Each process exchanges its local transpose with its transpose 
        # partner:
        #                   A[i][j].T <-> A[j][i].T 
        ########################################################################

        # Transpose partner process in grid
        other = pmat.grid_comm.Get_cart_rank([self.coords[1], self.coords[0]])

        # For a non-square matrix, we need a separate destination array.
        local_transpose = np.zeros((self.m_loc, self.n_loc))
        # local_transposed = np.ascontiguousarray(local_transposed)

        # if rank == 0:
        #     # Bytes described by datatype
        #     print("dtype bytes:", mult_col_type.Get_size())
        #     print("send nbytes:", self.local.nbytes)
        #     print("recv nbytes:", local_transpose.nbytes)

        pmat.grid_comm.Sendrecv([self.local, 1, mult_col_type], dest=other, source=other, recvbuf=[local_transpose, 1, row_type])

        # Free datatypes
        mult_col_type.Free()
        col_type.Free()
        
        row_type.Free()
        
        # Transpose dimensions
        return pmat(self.m, self.n, local_transpose) 


    ############################################################################
    # Universal functions (np.exp, np.add, etc.)
    ############################################################################
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        local_safe = self.local.copy()

        if ufunc is np.log:
            eps = 1e-12
            local_safe = np.where(self.local == 0, eps, self.local)
        
        # Convert inputs to numpy arrays
        arrays = [np.asarray(local_safe) if isinstance(x, pmat) else x
                  for x in inputs]

        with warnings.catch_warnings(record=True) as w:
            result = getattr(ufunc, method)(*arrays, **kwargs)

        for warning in w:
            if issubclass(warning.category, RuntimeWarning):

                print(f"\033[91m{pmat.grid_comm.coords}: Caught a RuntimeWarning during training: {warning.message}\033[0m")
                # pmat.grid_comm.Abort(-1)

        
        # Return result in a pmat
        if isinstance(result, np.ndarray):
            return pmat(self.n, self.m, result)
        else:
            return result


    ############################################################################
    # Handle non-ufunc NumPy functions (np.mean, np.concatenate, etc.)
    ############################################################################

    def __array_function__(self, func, types, args, kwargs):
        """
        Called for non-ufunc NumPy functions that support the array function protocol.
        """

        if func is np.ones_like:
            other = args[0]
            if isinstance(other, pmat):
                return pmat.from_numpy(np.ones((other.n, other.m)))
        elif func is np.max:
            other = args[0]
            if isinstance(other, pmat):
                return pmax(other, **kwargs)
        elif func is np.maximum:
            a1, a2 = args[0], args[1]
            if isinstance(a1, pmat) and not isinstance(a2, pmat):
                return pmaximum(a2, a1, **kwargs)
            elif not isinstance(a1, pmat) and isinstance(a2, pmat):
                return pmaximum(a1, a2, **kwargs)
            elif isinstance(a1, pmat) and isinstance(a2, pmat):
                return NotImplemented
        elif func is np.sum:
            other = args[0]
            if isinstance(other, pmat):
                return psum(other, **kwargs)
        elif func is np.mean:
            other = args[0]
            if isinstance(other, pmat):
                return pmean(other, **kwargs)

        return NotImplemented
    

############################################################################
# Non-ufunc pmat functions (sum, mean, max, maximum, etc.)
############################################################################

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

        return pmat(M.n, 1, row_sum)

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

        return pmat(M.n, 1, row_max)

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

        return pmat(M.n, 1, row_reduction)
    
def pmaximum(scalar, M: pmat, *args, **kwargs):
    return pmat(M.n, M.m, np.maximum(scalar, M.local, *args, **kwargs))

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

    A = pmat.from_numpy(A_mat)

    
    check(pmat.from_numpy(np.ones_like(A)), np.ones_like(A_mat), "ones_like")
    check(pmaximum(32, A), np.maximum(32, A_mat), "maximum(32, A)")

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
    B = pmat.from_numpy(B_mat)
    check(A @ B, A_mat @ B_mat, "A @ B")

    D = pmat.from_numpy(D_mat)
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
    check(pmaximum(32, A), np.maximum(32, A_mat), "maxmium(32, A)")

    pmean_result = pmean(A, axis=1)
    pmax_result = pmax(A, axis=1)
    if pmean_result is not None:
        check(pmean_result, np.mean(A_mat, axis=1, keepdims=True), "mean(A, axis=1)")
    if pmax_result is not None:
        check(pmax_result, np.max(A_mat, axis=1, keepdims=True), "max(A, axis=1)")

    ################################################################
    # Vecmat and Matvec operation 
    
    # Row vector = n...1
    row_vec = np.array(np.arange(n+1, 1, -1), ndmin=2).reshape(1, -1)
    row_pvec = pmat.from_numpy(row_vec)
    check(row_pvec @ A, row_vec @ A_mat, "vecmat")

    # Column vector = k...1
    col_vec = np.array(np.arange(k+1, 1, -1), ndmin=2).reshape(-1, 1) 
    col_pvec = pmat.from_numpy(col_vec)
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





