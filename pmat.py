from hmac import new
import warnings
import numpy as np
np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

from mpi4py import MPI

def dtype_to_mpi(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.int32:
        return MPI.INT
    elif dtype == np.int64:
        return MPI.LONG
    elif dtype == np.float32:
        return MPI.FLOAT
    elif dtype == np.float64:
        return MPI.DOUBLE
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def create_grid_comm():
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()

    # Grid rows and columns
    Pr = int(np.sqrt(num_procs))
    Pc = int(np.sqrt(num_procs))

    dims = [Pr, Pc]
    periods = [True, True]
    return comm.Create_cart(dims, periods, reorder=True)

def print_matrix(my_list, name=""):
    rows = len(my_list)
    cols = len(my_list[0])
    if name != "":
        print(f"{name}:")
    for i in range(rows):
        for j in range(cols):
            if pmat.grid_comm.rank == 0:
                print(f"{my_list[i][j]}", end=" ")
        print()
    print()

def print_ordered_by_rank(x, *args, **kwargs):
    grid = kwargs.get('comm', create_grid_comm())

    with_ranks = kwargs.get('with_ranks', np.arange(0, grid.Get_size()).tolist())

    for p in range(grid.Get_size()):
        grid.Barrier()
        if p == grid.rank and grid.rank in with_ranks:
            print(x, *args, **kwargs)

def print_pmat_on_rank0(M: 'pmat', msg=""):
    comm = M.grid_comm
    rank = comm.Get_rank()

    s = str(M)
    if rank == 0:
        # msg = msg + ":" if len(msg) > 0 else msg
        print(f"{msg}\n{s}\n")

# ANSI color helpers
def set_text_color(code):
    # code is an integer (e.g. 31 for red, 0 to reset)
    print(f"\033[{code}m", end="", flush=True)

def reset_text_color():
    print("\033[0m", end="", flush=True)

################################################################################
# Class pmat (will probably be called p_matrix later)
################################################################################

class pmat:
    ############################################################################
    # Static members and methods
    ############################################################################

    # Static grid communicator shared by all pmats
    grid_comm = create_grid_comm()

    @staticmethod
    def from_numpy(M_numpy: np.ndarray, dtype=np.float64) -> 'pmat':
        M_numpy = np.atleast_2d(M_numpy) if M_numpy is not None else None

        n, m = M_numpy.shape       

        coords = pmat.grid_comm.coords

        ########################################################################
        # Create a new pmat and its extent and position before reading 
        # numpy array from shared memory
        ########################################################################

        M_pmat = pmat(n, m, dtype=dtype)
        extent = M_pmat.extent[coords[0]][coords[1]]
        position = M_pmat.position[coords[0]][coords[1]]

        row_start, row_end = position[0], position[0] + extent[0]
        col_start, col_end = position[1], position[1] + extent[1]

        local = M_pmat.local    # Using full local array here (not just extent)

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

    # @staticmethod
    # def from_numpy(M_numpy: np.ndarray):
    #     """
    #     Convert a full numpy array to a distributed pmat
    #     """
    #     rank = pmat.grid_comm.Get_rank()
    #     row, col = pmat.grid_comm.Get_coords(rank)
        
    #     n, m = M_numpy.shape

    #     n_loc = np.ceil(n / pmat.grid_comm.dims[0]).astype(int)
    #     m_loc = np.ceil(m / pmat.grid_comm.dims[1]).astype(int)
       
    #     row_start = row * n_loc
    #     row_end = min((row + 1) * n_loc, n)
    #     col_start = col * m_loc
    #     col_end = min((col + 1) * m_loc, m)

    #     block = M_numpy[row_start:row_end, col_start:col_end]
        
    #     # Pad with zeros
    #     local = np.zeros((n_loc, m_loc), dtype=np.double)
    #     # Copy from numpy block up to padding        
    #     local[:block.shape[0], :block.shape[1]] = block

    #     return pmat(n, m, local)
    
    @staticmethod
    def resize(top: int, bottom: int, left: int, right: int, M: 'pmat'):
        # grid_comm = create_grid_comm()
        # new_M = pmat(bottom-top, right-left, grid_comm)   # start with all zeros

        full = M.get_full()[top:bottom, left:right]
        new_M = pmat.from_numpy(full)

        return new_M

    def pretty_string(self, name="", remove_padding=True, as_type=None):
        # print("p_rows, p_cols:", p_rows, p_cols)
        # print("n, m:", n, m)
        # print("n_local, n_rem:", n_local, n_rem)
        # print("m_local, m_rem:", m_local, m_rem)

        if name != "":
            if self.grid_comm.rank == 0:
                print(f"{name}:")

        if as_type is None:
            if self.local.dtype == np.int32 or self.local.dtype == np.int64:
                as_type = "i"
            elif self.local.dtype == np.float32:
                as_type = "f"
            elif self.local.dtype == np.float64:
                as_type = "d"
            elif self.local.dtype == np.bool_:
                as_type = "b"
            else:
                raise ValueError(f"Unsupported dtype for pretty print: {self.local.dtype}")

        full_matrix = self.get_full(remove_padding)
        matrix_str = ""            
        
        for row in range(self.n if remove_padding else self.n + self.n_pad):
            col = 0

            if row % self.n_loc == 0:
                row_color = (row * self.m) // self.n_loc
            while col < self.m:
                color_code = 31 + ((row_color + col))   # 7 possible colors
                # print(f"row {row} col {j} color {color_code}", flush=True)
                
                if as_type == "i":
                    block_str = " ".join(f"{int(val):3d}" for val in full_matrix[row][col : col + self.m_loc])
                elif as_type == "f" or as_type == "d":
                    block_str = "\b" + "".join(f"{float(val):10.1f}" for val in full_matrix[row][col : col + self.m_loc])
                elif as_type == "b":
                    block_str = "\b" + "".join(f"{val}" for val in full_matrix[row][col : col + self.m_loc])                

                if self.grid_comm.rank == 0:
                    Pr = row // self.n_loc
                    Pc = col // self.m_loc
                    # color_code = 31 + (Pr + Pc) % 7  # 7 possible colors
                    # set_text_color(color_code)
                    palette = [196, 46, 220, 21, 208, 93, 226, 201, 202, 51, 82, 129, 214, 200, 198, 199]
                    color_code = palette[(Pr + Pc * self.grid_comm.dims[1]) % len(palette)]

                    matrix_str += f"\033[38;5;{color_code}m{block_str}\033[0m"
                    matrix_str += " "
                    
                    # color_code = (Pr * pmatrix.grid_comm.dims[1] + Pc) * 13 % 256
                    # print(f"\033[38;5;{color_code}m{block_str}\033[0m", end=" ", flush=True)

        
                col += self.m_loc

            # New line after each global row
            if self.grid_comm.rank == 0:
                matrix_str += "\n" 
        
        pmat.grid_comm.Barrier()
        return matrix_str

    def print_pretty(self, name="", remove_padding=True, as_type=None):
        matrix_str = self.pretty_string(name, remove_padding, as_type)
        if self.grid_comm.rank == 0:
            print(matrix_str, flush=True)

    ############################################################################
    # Constructor
    ############################################################################

    def __init__(self, n, m, local=None, dtype=np.float64):
    # def __init__(self, shape: tuple[int, int], local=None):

        local = np.atleast_2d(local) if local is not None else None

        Pr = pmat.grid_comm.dims[0]
        Pc = pmat.grid_comm.dims[1]

        self.n = n
        self.m = m
        self.shape = (n, m)
        self.ndim = 2           # used in layer._as_2d
        self.dtype = dtype
        
        self.rank = pmat.grid_comm.Get_rank()
        self.coords = pmat.grid_comm.Get_coords(self.rank)

        self.n_loc = np.ceil(self.n / Pr).astype(int)
        self.m_loc = np.ceil(self.m / Pc).astype(int)
        self._local_shape = (self.n_loc, self.m_loc)

        # Padding if n or m are not evenly divisible by Pr or Pc
        _, self.n_pad = divmod(self.n, Pr)
        _, self.m_pad = divmod(self.m, Pc)
        self.is_padded = (self.n_pad > 0) or (self.m_pad > 0)

        ########### Also in method self._set_local(local) ##################

        # Local block will be (n_pad, m_pad) larger than local if there is zero padding.
        self.local = np.zeros((self.n_loc, self.m_loc), dtype=dtype)
        self.local = np.ascontiguousarray(self.local)
        
        if local is not None:
            self.local[:local.shape[0], :local.shape[1]] = local    # deep copy

        
        ####################### New Way (with extents) ########################
        
        # Block size
        n_loc = int(np.ceil(n / Pr))
        m_loc = int(np.ceil(m / Pc))

        # Row extents
        x, x_rem = divmod(n, n_loc)
        n_loc_extent = [self.n_loc] * x
        if x < Pr:
            n_loc_extent += [x_rem]
        n_loc_extent += [0] * (Pr - len(n_loc_extent))  # pad with zeros if needed

        # Column extents
        y, y_rem = divmod(m, m_loc)
        m_loc_extent = [self.m_loc] * y
        if y < Pc:
            m_loc_extent += [y_rem]
        m_loc_extent += [0] * (Pc - len(m_loc_extent))  # pad with zeros if needed

        extent = [[(0, 0) if a == 0 or b == 0 else (a, b) for b in m_loc_extent] for a in n_loc_extent]

        # extent = [[(a, b) for b in m_loc_extent] for a in n_loc_extent]


        n_loc_pos = [0]
        for v in n_loc_extent[:-1]:
            n_loc_pos.append(n_loc_pos[-1] + v)

        m_loc_pos = [0]
        for v in m_loc_extent[:-1]:
            m_loc_pos.append(m_loc_pos[-1] + v)

        position = [[(a,b) for b in m_loc_pos] for a in n_loc_pos]
        self.extent = extent
        self.position = position

        self.block_size = extent[self.coords[0]][self.coords[1]]
        self.block_loc = position[self.coords[0]][self.coords[1]]

 
    def copy(self):
        # Deep copy
        new_pmat = pmat(self.n, self.m, self.local.copy(), dtype=self.local.dtype)
        return new_pmat
    
    def __copy__(self):
        return self.copy()
    
    def __deepcopy__(self, memo):
        # p_matrix only contains numpy arrays and primitive types, so a deep copy is the same as a shallow copy
        return self.copy()
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
        # if pmat.grid_comm.rank == 0:
        #     print(f"index0 ({len(idx[0])})=\n{idx[0]}\n{"*"*50}\nindex1 ({len(idx[0])})=\n{idx[1]}")
        # exit()
        full = self.get_full()
        val = np.atleast_2d(full[idx])
        # if pmat.grid_comm.rank == 0:
        #     print(f"val.shape={val.shape}")
        # exit()

        #####
        # Todo: Index -> pmat without going through full numpy array (shared memory?)
        ####
        return pmat.from_numpy(val) if isinstance(val, np.ndarray) else val
    
    def __setitem__(self, idx, value):
        # i, i_rem = divmod(self.n, idx[0]) 
        # j, j_rem = divmod(self.m, idx[1]) 
        
        # if self.coords[0] == i and self.coords[1] == j:
        #     pass
        #     # print(f"Setting item on rank {self.rank} at coords {self.coords} for index {idx} with value {value}")

        full = self.get_full()
        full[idx] = value
        self.set_full(full)

    def remove_first_column(self):
    
        col_offset = 1  # remove first column

        assert self.m > 1, "matrix only has one column"
                
        n, m = self.shape        
        
        coords = pmat.grid_comm.coords
        extent = self.extent[coords[0]][coords[1]]
        position = self.position[coords[0]][coords[1]]

        row_start, row_end = position[0], position[0] + extent[0]
        col_start, col_end = position[1], position[1] + extent[1]

        local = self.local[:extent[0], :extent[1]]

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
        type = self.local.dtype
        type_bytes = np.dtype(type).itemsize
        size = int(np.prod(self.shape)) * type_bytes

        win = MPI.Win.Allocate_shared(size, disp_unit=type_bytes, comm=pmat.grid_comm)

        assert win is not None, "win is None"
            
        # Buffer is allocated by rank 0, but all processes can access it
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=type, shape=(self.n, self.m))

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
        submat.local = submat_local.copy()  # Added copy ... didn't fix the issue
        
        ############################################################################
        # End the epoch and synchronize processes
        ############################################################################
        win.Fence()

        # Free the shared memory
        win.free()

        return submat    

    def get_local(self):
        n_loc, m_loc = self.extent[self.coords[0]][self.coords[1]]
        return self.local[:n_loc, :m_loc]

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

    def get_full(self, remove_padding=True):
        # Gather all blocks at root
        Pr = pmat.grid_comm.dims[0]
        Pc = pmat.grid_comm.dims[1]

        M = np.zeros((self.n_loc * Pr, self.m_loc * Pc), dtype=self.dtype)

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



        return M[:self.n, :self.m] if remove_padding else M

    ############################################################################
    # Arithmetic operators
    ############################################################################
    def check_for_broadcast(A, B):
        # Python broadcasting expands a smaller array (a vector) to match a larger array (a matrix)

        # Scalar broadcasting
        if np.isscalar(B):
            A_extent = A.extent[A.coords[0]][A.coords[1]]
            A_local = A.local[:A_extent[0], :A_extent[1]]

            return A_local, B, (A.n, A.m)
        elif np.isscalar(A):
            B_extent = B.extent[B.coords[0]][B.coords[1]]
            B_local = B.local[:B_extent[0], :B_extent[1]]

            return A, B_extent, (B.n, B.m)

        
        # Operands are without padding
        A_extent = A.extent[A.coords[0]][A.coords[1]]
        A_local = A.local[:A_extent[0], :A_extent[1]]
        
        B_extent = B.extent[B.coords[0]][B.coords[1]]
        B_local = B.local[:B_extent[0], :B_extent[1]]

        # Output shape is a matrix
        output_shape = (max(A.n, B.n), max(A.m, B.m))

        # Column vector broadcasting. Avoid broadcast if A and B are both column vectors
        if B.shape == (A.n, 1) and B.m != A.m:  
            # Handle column vector as right operand
            horz_comm = B.grid_comm.Sub([False, True])            
            B_local_col = horz_comm.bcast(B_local, root=0)
            return A_local, B_local_col, output_shape
        elif A.shape == (B.n, 1) and A.m != B.m:           
            # Handle column vector as left operand
            horz_comm = A.grid_comm.Sub([False, True])            
            A_local_col = horz_comm.bcast(A_local, root=0)
            return A_local_col, B_local, output_shape
        
        # Row vector broadcasting. Avoid broadcast if A and B are both row vectors
        elif B.shape == (1, A.m) and B.n != A.n:  
            # Handle row vector as right operand
            vert_comm = B.grid_comm.Sub([True, False])            
            B_local_row = vert_comm.bcast(B_local, root=0)
            return A_local, B_local_row, output_shape
        elif A.shape == (1, B.m) and A.n != B.n:
            # Handle row vector as left operand
            vert_comm = A.grid_comm.Sub([True, False])            
            A_local_row = vert_comm.bcast(A_local, root=0)
            return A_local_row, B_local, output_shape
        
        # Nevermind: Both are matrices
        else:
            return A_local, B_local, output_shape


    def __gt__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.greater(left_operand,  right_operand))

        # return pmat(self.n, self.m, self.local > other)

    def __eq__(self, other):

        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.equal(left_operand,  right_operand), dtype=np.bool_)

        # return pmat(self.n, self.m, self.local > other)
    
    def __add__(self, other):
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, self.local + other)
        # else:
            
        #     
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.add(left_operand,  right_operand))
        # return pmat(self.n, self.m, self.local + other.local)


    def __sub__(self, other):     
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.subtract(left_operand, right_operand))


    def __mul__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.multiply(left_operand, right_operand))
        
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, self.local * other)
        # else:
        #     return pmat(self.n, self.m, self.local * other.local)
    
    def __rmul__(self, other):
        # Handle scalar * pmat (right multiplication)
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.multiply( right_operand, left_operand))
        
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, other * self.local)
        # else:
        #     return NotImplemented
        
    def __truediv__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.true_divide(left_operand, right_operand))

        # if np.isscalar(other):
        #     return pmat(self.n, self.m, self.local / other)
        # else:
        #     return pmat(self.n, self.m, self.local / other.local)
    
    def __rdiv__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.true_divide(left_operand, right_operand))

        # Handle scalar / pmat (right division)
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, other / self.local)
        # else:
        #     return NotImplemented
    
    def __rtruediv__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.true_divide(right_operand, left_operand))

        # Handle scalar / pmat (right true division for Python 3)
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, other / self.local)
        # else:
        #     return NotImplemented

    def __radd__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.add(right_operand, left_operand))

        # # Handle scalar + pmat (right addition)
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, other + self.local)
        # else:
        #     return NotImplemented
    

    ##### failed in test.py..... ########
    def __rsub__(self, other):
        left_operand, right_operand, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.subtract( right_operand, left_operand))

        # # Handle scalar + pmat (right addition)
        # if np.isscalar(other):
        #     return pmat(self.n, self.m, other - self.local)
        # else:
        #     return NotImplemented

    def __neg__(self):
        return pmat(self.n, self.m, -self.local)

    def __matmul__(self, other: 'pmat'):
        


        return matmul(self, other)

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
        return pmat(self.m, self.n, local_transpose, dtype=self.dtype) 


    ############################################################################
    # Universal functions (np.exp, np.add, etc.)
    ############################################################################
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        extent = self.extent[self.coords[0]][self.coords[1]]
        local = self.local[:extent[0], :extent[1]]

        # Convert inputs to arrays
        arrays = [np.asarray(local) if isinstance(x, pmat) else x
                  for x in inputs]

        with warnings.catch_warnings(record=True) as w:
            result = getattr(ufunc, method)(*arrays, **kwargs)

        for warning in w:
            if issubclass(warning.category, RuntimeWarning):
                
                local_str = f"\nOriginal local:\n{self.local}\n\nResult matrix:\n{result}"
                print(f"\033[91m{pmat.grid_comm.coords}: Caught a RuntimeWarning: {warning.message}\033[0m")
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
        if func is np.zeros_like:
            other = args[0]
            if isinstance(other, pmat):
                return pzeros_like(other, **kwargs)
        elif func is np.ones_like:
            other = args[0]
            if isinstance(other, pmat):
                return p_ones_like(other, **kwargs)
        elif func is np.max:
            other = args[0]
            if isinstance(other, pmat):
                return self.pmax(other, **kwargs)
        elif func is np.argmax:
            other = args[0]
            if isinstance(other, pmat):
                return self.pargmax(other, **kwargs)    
        elif func is np.sum:
            other = args[0]
            if isinstance(other, pmat):
                return self.psum(other, **kwargs)
        elif func is np.mean:
            other = args[0]
            if isinstance(other, pmat):
                return self.pmean(other, **kwargs)
        elif func is np.maximum:
            other = args[0]
            if isinstance(other, self.local.dtype):     # scalar
                return self.pmaximum(other, **kwargs)

        raise NotImplementedError(f"{func} not implemented for pmat")
    
    ############################################################################
    # Non-ufunc pmat functions (sum, mean, max, maximum, etc.)
    ############################################################################



    def psum(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        # assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

        if axis == 1:
            row_group = self.grid_comm.Sub([False, True])  # rows

            dtype = self.dtype
            coords = self.coords
            extent = self.extent[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            row_sum = []

            for row in range(extent[0]):
                # Reduce to the root of each group
                local_sum = np.sum(local[row, :])
                row_sum.append(row_group.reduce(local_sum, op=MPI.SUM, root=0))

            if self.grid_comm.coords[1] == 0:
                row_sum = np.array([x for x in row_sum if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1) # row_sum is now a column vector
            else:
                row_sum = None

            return pmat(self.n, 1, row_sum)
        elif axis == None:
            # Global sum
            coords = self.coords
            extent = self.extent[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            local_sum = np.sum(local)
            global_sum = self.grid_comm.allreduce(local_sum, op=MPI.SUM)
            return global_sum
        else:
            raise NotImplementedError("psum axis=0 not implemented yet")

    def pmax(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

        if axis == 1:
            horz_group = self.grid_comm.Sub([False, True])  # rows

            dtype = self.dtype
            coords = self.coords
            extent = self.extent[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            row_max = []

            for row in range(extent[0]):
                # Reduce to the root of each group
                local_max = np.max(local[row, :])
                row_max.append(horz_group.reduce(local_max, op=MPI.MAX, root=0))

            if self.grid_comm.coords[1] == 0:
                row_max = np.array([x for x in row_max if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1) # maxs is now a column vector
            else:
                row_max = None

            return pmat(self.n, 1, row_max)
    
    
    def pargmax(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

        if axis == 1:
            # horz_comm = self.grid_comm.Sub([False, True])  # rows

            # dtype = self.dtype
            # coords = self.coords
            # position = self.position[coords[0]][coords[1]]
            # extent = self.extent[coords[0]][coords[1]]
            # local = self.local[:extent[0], :extent[1]]

            # result = np.zeros((extent[0], 2), dtype=(np.int64, self.dtype))
            # # self.print_pretty()
            # import time

            # # start = time.perf_counter()
            # for row in range(extent[0]):
            #     local_max = dtype(np.max(local[row, :]))
            #     local_idx = np.int64(np.argmax(local[row, :]))
            #     global_idx = np.int64(local_idx + position[1])


            #     row_max_combined = np.zeros((1, horz_comm.size), dtype=dtype)
            #     row_idx_combined = np.zeros((1, horz_comm.size), dtype=np.int64)

            #     horz_comm.Allgather(local_max, row_max_combined)
            #     horz_comm.Allgather(global_idx, row_idx_combined)

            #     idx = np.argmax(row_max_combined)
            #     result[row, 0] = row_idx_combined.flatten()[idx]
            #     result[row, 1] = row_max_combined.flatten()[idx]

            # local_argmax = result[:,0].reshape(extent[0], 1)
            
            # end = time.perf_counter()

            # # if self.rank == 0:
            # #     print("allgather time: {:.6f} {}".format(end-start, self.shape))
            # return pmat(self.n, 1, local=local_argmax, dtype=np.int64)



            #### Slightly faster to use allreduce #########################


            
            horz_comm = self.grid_comm.Sub([False, True])  # rows

            dtype = self.dtype
            coords = self.coords
            position = self.position[coords[0]][coords[1]]
            extent = self.extent[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            result = np.zeros((extent[0], 2), dtype=(np.int64, self.dtype))
            
            # start = time.perf_counter()
            for row in range(extent[0]):
                local_max = dtype(np.max(local[row, :]))
                local_idx = np.int64(np.argmax(local[row, :]))
                global_idx = np.int64(local_idx + position[1])
                mpi_dtype = np.dtype([('value', np.float64), ('index', np.int32)], align=True)
                sendbuf = np.array([(local_max, global_idx)], dtype=mpi_dtype)  
                recvbuf = np.array([(0.0, 0)], dtype=mpi_dtype)

                # Allreduce with MAXLOC
                horz_comm.Allreduce([sendbuf, MPI.DOUBLE_INT], [recvbuf, MPI.DOUBLE_INT], op=MPI.MAXLOC)

                result[row,:] = recvbuf['index'][0]

            row_argmax = result[:,0].reshape(extent[0], 1) if coords[1] == 0 else None

            if self.grid_comm.coords[1] == 0:    
                row_argmax = np.array([x for x in row_argmax if x is not np.nan]).flatten().reshape(-1, 1) # row_argmax is now a column vector
            else:
                row_argmax = None

            # end = time.perf_counter()
            # if self.rank == 0:
            #     print(f"allreduce time: {(end-start):.6f}, {self.shape}")
            return pmat(self.n, 1, local=row_argmax, dtype=np.int64)
            

    def pmean(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        if axis == 0:
            raise NotImplementedError("pmean axis=0 not implemented yet")
        elif axis == 1:
            # horz_group = self.grid_comm.Sub([False, True])  # rows

            # dtype = self.dtype
            # coords = self.coords
            # extent = self.extent[coords[0]][coords[1]]
            # local = self.local[:extent[0], :extent[1]]

            # row_mean = []

            # for row in range(extent[0]):
            #     # Reduce to the root of each group
            #     local_row_sum = np.sum(self.local[row, :])
            #     row_mean.append(horz_group.reduce(local_row_sum, op=MPI.SUM, root=0))

            # if self.grid_comm.coords[1] == 0:
            #     # Remove Nones and make a column vector
            #     row_mean = np.array([x for x in row_mean if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1)
                
            #     # Divide by total number of columns
            #     row_mean = row_mean / self.m
            # else:
            #     # Process is not a root
            #     row_mean = None
            # return pmat(self.n, 1, row_mean)

            return self.psum(axis=1) / self.m

        elif axis is None:
            # Global mean
            total_sum = self.grid_comm.allreduce(np.sum(self.local), op=MPI.SUM)
            global_mean = total_sum / (self.n * self.m)
            return global_mean      # return scalar value (not pmat)
        else:
            raise ValueError(f"Invalid axis {axis} for pmean")
        
    def pmaximum(self, scalar, *args, **kwargs):
        return pmat(self.n, self.m, np.maximum(scalar, self.local, *args, **kwargs))
    

    def stack_ones_on_top(self: 'pmat') -> 'pmat':

        # row_offset = 1  # add to first row
        coords = pmat.grid_comm.coords
        n, m = self.shape        
        
        # Set up the extented matrix
        newmat = pmat(n + 1, m )
        newmat_extent = newmat.extent[coords[0]][coords[1]]
        newmat_position = newmat.position[coords[0]][coords[1]]

        newmat_row_start, newmat_row_end = newmat_position[0], newmat_position[0] + newmat_extent[0]
        newmat_col_start, newmat_col_end = newmat_position[1], newmat_position[1] + newmat_extent[1]

        newmat_local = newmat.local

        ########################################################################
        # Write each rank's local block to the shared array offset by one row
        ########################################################################
        extent = self.extent[coords[0]][coords[1]]
        position = self.position[coords[0]][coords[1]]

        row_start, row_end = position[0] + 1, position[0] + extent[0] + 1
        col_start, col_end = position[1], position[1] + extent[1]

        local = self.local[:extent[0], :extent[1]]
        
        type = self.local.dtype
        type_bytes = np.dtype(type).itemsize
        shape = (n + 1, m)
        size = int(np.prod(shape)) * type_bytes

        win = MPI.Win.Allocate_shared(size, disp_unit=type_bytes, comm=pmat.grid_comm)

        assert win is not None, "win is None"
            
        # Buffer is allocated by rank 0, but all processes can access it
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=type, shape=shape)

        shared_array[0,:] = 1  # First row of ones
        
        ############################################################################
        # Start an epoch and synchronize processes before writing
        ########################################################################### 
        win.Fence()
        
        # Each rank writes its original matrix block to the shared array
        shared_array[row_start:row_end, col_start:col_end] = local

        ############################################################################
        # End the epoch and synchronize processes (again) before reading
        ############################################################################
        win.Fence()
        
        # Each rank reads its submatrix block from the shared array
        newmat_local[:newmat_extent[0], :newmat_extent[1]] = shared_array[newmat_row_start:newmat_row_end, newmat_col_start:newmat_col_end]
        newmat.local = newmat_local
        
        ############################################################################
        # End the epoch and synchronize processes
        ############################################################################
        win.Fence()

        # Free the shared memory
        win.free()

        return newmat




############################################################################
# Utility functions
############################################################################

def pzeros_like(M_pmat: pmat) -> pmat:
    newmat = pmat(M_pmat.n, M_pmat.m)

    coords = pmat.grid_comm.coords
    extent = newmat.extent[coords[0]][coords[1]]

    newmat.local[:extent[0], :extent[1]] = np.zeros((extent[0], extent[1]))

    return newmat

def p_ones_like(M_pmat: pmat) -> pmat:
    newmat = pmat(M_pmat.n, M_pmat.m)

    coords = pmat.grid_comm.coords
    extent = newmat.extent[coords[0]][coords[1]]

    newmat.local[:extent[0], :extent[1]] = np.ones((extent[0], extent[1]))

    return newmat



############################################################################
# Matrix-matrix multiplication function
############################################################################

def matmul(A: pmat, B: pmat):
    assert A.m == B.n, f"A @ B: A.m:{A.m} != B.n:{B.n}"



    # Cannon's Algorithm
    
    C = np.zeros((A.n_loc, B.m_loc))

    # Total steps over grid 
    num_steps = min(pmat.grid_comm.dims)

    # Deep copies for Sendrecv_replace
    A_block = A.local.copy()
    B_block = B.local.copy()

    # Skew (initial alignment)
    for _ in range(A.coords[0]):
        src, dst = pmat.grid_comm.Shift(1, -1)
        pmat.grid_comm.Sendrecv_replace(A_block, dest=dst, source=src)

    for _ in range(A.coords[1]):
        src, dst = pmat.grid_comm.Shift(0, -1)
        pmat.grid_comm.Sendrecv_replace(B_block, dest=dst, source=src)

    for _ in range(num_steps):
        # Multiply and accumulate
        C += A_block @ B_block

        # Shift A left
        src, dst = pmat.grid_comm.Shift(1, -1)
        pmat.grid_comm.Sendrecv_replace(A_block, dest=dst, source=src)

        # Shift B up
        src, dst = pmat.grid_comm.Shift(0, -1)
        pmat.grid_comm.Sendrecv_replace(B_block, dest=dst, source=src)

    return pmat(A.n, B.m, local=C)






if __name__ == "__main__":
#     # Test input

        # Large matrices
#     test(n=1000, k=2000, m=10000)

#     test(n=28*28, k=64, m=64)

    # Non-square matrices
    test(n=9, k=5, m=16)

    # Square matrices
    test(n=8, k=8, m=16)





