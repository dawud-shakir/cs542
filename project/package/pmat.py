from math import ceil
import warnings
import os # for file paths
import numpy as np
np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

from mpi4py import MPI
from package.utilities import create_grid_comm
import package.utilities as utils

def normalize_index(idx):
    # arrays/lists to int arrays. slices, scalars, and tuples stay the same but normalize their content

    if isinstance(idx, tuple):
        # For tuples (e.g., (row_idx, col_idx)), normalize each element
        normalized = []
        for i in idx:
            if isinstance(i, (list, np.ndarray)):
                normalized.append(np.asarray(i, dtype=int))
            elif isinstance(i, slice):
                normalized.append(i)  # Slices are already valid
            elif np.isscalar(i):
                normalized.append(int(i))  # Convert scalars to Python int
            # elif isinstance(i, pmat):
            ### Use "duck-typing" to avoid refering to pmat before it is declared below
            elif hasattr(i, "to_numpy") and callable(i.to_numpy):            
                normalized.append(np.asarray(i.to_numpy(), dtype=int))

            else:
                raise TypeError(f"Unsupported index element type: {type(i)}")
        return tuple(normalized)
    elif isinstance(idx, (list, np.ndarray)):
        # For 1D arrays/lists, convert to NumPy int array
        return np.asarray(idx, dtype=int)
    elif isinstance(idx, slice):
        # Slices are already valid
        return idx
    elif np.isscalar(idx):
        # Convert scalars to Python int
        return int(idx)
    # elif isinstance(idx, pmat):
    ### Use "duck-typing" to avoid refering to pmat here
    elif hasattr(idx, "get_full") and callable(idx.get_full):
        return idx.get_full()
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")

################################################################################
class pmat:
################################################################################
    def apply(self, fn, approach='auto'):
        """
        Apply a user-supplied function 'fn' to the full matrix represented by this pmat.

        approach:
          - 'auto'   : pick 'numpy' if pmat.use_scatter True (safe), else prefer 'shared'
          - 'numpy'  : gather full matrix on root, apply fn there, scatter back (uses to_numpy/from_numpy)
          - 'shared' : use MPI.Win.Allocate_shared if available to create a shared one-buffer,
                       rank 0 applies the function, then it's turned into a pmat via from_shared_buffer
          - 'rma'    : use MPI one-sided Put/Get to write to a root buffer, apply fn on root,
                       then have other ranks Get their block back from root

        The provided function 'fn' should either:
          - modify the array in-place and return None, or
          - return a new numpy array of the same shape.

        Returns a new pmat with the operation result (self is unchanged).
        """
        if not callable(fn):
            raise TypeError("fn must be callable")

        if approach == 'auto':
            # Safe choice: if pmat.use_scatter, do a scatter/gather approach (to_numpy / from_numpy).
            # Otherwise prefer shared memory if available.
            approach = 'numpy' if pmat.use_scatter else 'shared'

        if approach == 'numpy':
            return self._apply_via_numpy(fn)
        elif approach == 'shared':
            return self._apply_via_shared(fn)
        elif approach == 'rma':
            return self._apply_via_rma(fn)
        else:
            raise ValueError("approach must be one of 'auto', 'numpy', 'shared', or 'rma'")

    def _apply_via_numpy(self, fn):
        # Gather to root synchronously, apply function on root, and scatter result.
        # All ranks call to_numpy(all_to_root=True), root gets full matrix, else None
        full = self.to_numpy(all_to_root=False)  # root will have full matrix, others None
        
        print(f"{pmat.grid_comm.coords}: here in apply_via_numpy")
        root = pmat.grid_comm.rank == 0
  
        if root:
            out = fn(full)
            
            if out is None:
                # assume in-place modification
                out = full
            # Validate
            out = np.atleast_2d(out)
            assert out.shape == (self.n, self.m), "fn returned array of wrong shape"
        else:
            

            out = None

        # from_numpy will broadcast shape/dtype and scatter blocks
        return pmat.from_numpy(out, dtype=self.dtype)

    def _apply_via_shared(self, fn):
        # Allocate a shared window (one buffer accessible to all ranks).
        # Each rank writes its local block to the shared array; root applies fn;
        # then we create a new pmat via from_shared_buffer which copies out appropriate local blocks.
        grid = pmat.grid_comm
        dtype = self.dtype
        coords = self.coords
        n, m = self.shape
        itemsize = np.dtype(dtype).itemsize

        size = int(np.prod((n, m))) * itemsize

        # Allocate shared memory across ranks in grid_comm
        win = MPI.Win.Allocate_shared(size, disp_unit=itemsize, comm=grid)
        # Buffer belongs to rank 0, but all processes can Shared_query(0)
        buf, _ = win.Shared_query(0)
        shared_array = np.ndarray(buffer=buf, dtype=dtype, shape=(n, m))

        # Start epoch - synchronize before writes
        win.Fence()

        # Compute local write region
        extent = self.extents[coords[0]][coords[1]]
        pos = self.offsets[coords[0]][coords[1]]
        row_start, row_end = pos[0], pos[0] + extent[0]
        col_start, col_end = pos[1], pos[1] + extent[1]
        local = self.local[:extent[0], :extent[1]]

        if extent[0] > 0 and extent[1] > 0:
            shared_array[row_start:row_end, col_start:col_end] = local

        # Ensure all writes complete
        win.Fence()

        # Only root applies the lambda (avoid duplicating side effects)
        if grid.rank == 0:
            res = fn(shared_array)
            if res is not None:
                res = np.atleast_2d(res)
                assert res.shape == (n, m), "fn returned array of wrong shape"
                shared_array[:] = res

        # Wait for function to complete on root and changes to be visible on other ranks
        win.Fence()

        # Use reusable factory to create a pmat reading from shared memory
        new_pmat = pmat.from_shared_buffer(win, shared_array, dtype=dtype)

        # Free the shared memory window
        try:
            win.Free()
        except Exception:
            # Some platforms use win.free()
            try:
                win.free()
            except Exception:
                pass

        return new_pmat

    def _apply_via_rma(self, fn):
        # RMA approach: gather the whole matrix to a root buffer using Put,
        # apply fn on root, then Get local blocks back from root.
        grid = pmat.grid_comm
        dtype = self.dtype
        coords = self.coords
        n, m = self.shape
        itemsize = np.dtype(dtype).itemsize
        root = 0

        # root will hold the full array as a 1D contiguous buffer; others have no backing memory
        if grid.rank == root:
            root_buf = np.zeros((n * m,), dtype=dtype)
        else:
            root_buf = None

        # Create a window exposing root_buf on the root, None elsewhere
        win = MPI.Win.Create(root_buf, disp_unit=itemsize, comm=grid)

        # compute local region offsets (flat offset in elements)
        extent = self.extents[coords[0]][coords[1]]
        pos = self.offsets[coords[0]][coords[1]]
        row_start, row_end = pos[0], pos[0] + extent[0]
        col_start, col_end = pos[1], pos[1] + extent[1]
        local = self.local[:extent[0], :extent[1]]
        # flattened length / target disp
        local_flat_len = int(extent[0] * extent[1])
        if local_flat_len > 0:
            # flat starting position in root_buf elements
            target_disp = int(row_start * m + col_start)

            # Put local block into root window
            win.Lock(root)
            # Put expects data formats and uses MPI datatype; default numeric dtype=double
            mpi_type = MPI.DOUBLE if dtype == np.float64 else (MPI.FLOAT if dtype == np.float32 else MPI.BYTE)
            win.Put([local.ravel(), mpi_type], target_rank=root, target_disp=target_disp)
            win.Unlock(root)

        # Make sure all puts are finished
        grid.Barrier()

        # Only root applies the function to the full buffer
        if grid.rank == root:
            # View as 2D
            full = root_buf.reshape((n, m))
            res = fn(full)
            if res is not None:
                res = np.atleast_2d(res)
                assert res.shape == (n, m), "fn returned array of wrong shape"
                full[:] = res

        # Barrier to ensure root modification done
        grid.Barrier()

        # Each rank now retrieves its local block from the root buffer using Get
        new_local = np.zeros_like(self.local)  # allocate storage for the new local block
        if local_flat_len > 0:
            recv_flat = np.empty((local_flat_len,), dtype=dtype)
            win.Lock(root)
            mpi_type = MPI.DOUBLE if dtype == np.float64 else (MPI.FLOAT if dtype == np.float32 else MPI.BYTE)
            win.Get([recv_flat, mpi_type], target_rank=root, target_disp=target_disp)
            win.Unlock(root)

            # reshape into local block shape and copy to new_local
            new_local[:extent[0], :extent[1]] = recv_flat.reshape(extent[0], extent[1])

        # Ensure all gets finished
        grid.Barrier()

        # Free window
        try:
            win.Free()
        except Exception:
            try:
                win.free()
            except Exception:
                pass

        # Create a new pmat with the same global shape and this process's new local block
        new_pmat = pmat(self.n, self.m, local=new_local, dtype=dtype)
        return new_pmat



     # Static grid communicator shared by all pmats
    grid_comm = create_grid_comm()
    use_scatter = True

    ############################################################################
    #                           File I/O
    ############################################################################
    def to_file(self, filename: str, prefix_current_directory=True) -> int:
        comm = MPI.COMM_WORLD
    
        filepath = os.path.join(os.path.dirname(__file__), filename) if prefix_current_directory else filename

        # Open the file
        amode = MPI.MODE_CREATE | MPI.MODE_WRONLY
        fh = MPI.File.Open(comm, filepath, amode)

        data_offset = 0

        # Write header
        if self.rank == 0:
            dtype_str = np.dtype(self.dtype).name  # e.g., 'float64'
            dtype_bytes = dtype_str.encode('utf-8')
            dtype_len = np.int32(len(dtype_bytes))  # store length as 4 bytes

            header = np.array([self.n, self.m], dtype=np.int64).tobytes()
            fh.Write_at(0, header)
            fh.Write_at(16, dtype_len.tobytes())
            fh.Write_at(20, dtype_bytes)
            
            # n (8 bytes), m (8 bytes), dtype_len (4 bytes), dtype_str (dtype_len)
            
            data_offset += 20 + dtype_len

        data_offset = np.array(data_offset, dtype=np.int64)
        MPI.COMM_WORLD.Bcast(data_offset, root=0)
        data_offset = int(data_offset)  # Convert back to int
        
        # Wait for header to be written before writing any data
        comm.Barrier()  
        
        coords = self.coords
        extent = self.extents[coords[0]][coords[1]]
        offset = self.offsets[coords[0]][coords[1]]

        local_rows, local_cols = extent[0], extent[1]
        row_offset, col_offset = offset[0], offset[1]

        for i in range(local_rows):
            global_row = row_offset + i
            file_offset = data_offset + (global_row * self.m + col_offset) * np.dtype(self.dtype).itemsize
            fh.Write_at(file_offset, self.local[i, :local_cols].tobytes())

       

        file_size = fh.Get_size()
        fh.Close()
        return file_size

    @staticmethod
    def from_shared_buffer(win: MPI.Win, shared_array: np.ndarray, dtype=np.float64) -> 'pmat':
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

    @staticmethod
    def index_rows(A: 'pmat', idx):
        # idx can be a scalar, a slice, a list, or an array of integers
        # idx can have duplicates and be out of order
    
        # Make sure idx is an array
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

        if pmat.use_scatter:
            full_matrix = A.to_numpy()
            return pmat.from_numpy(full_matrix[idx[0]], dtype=A.dtype)

        else:        
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
                # Compute grid row rank in A and local row index
                A_block, A_block_row  = divmod(A_row, A.n_loc)

                if A_block == A.coords[0]:
                    # Ranks that made it here have a nonempty block if they are in A's row comm
                    row_comm = A.row_comm
                    if row_comm == MPI.COMM_NULL:
                        continue

                    # The row allgather copies into needs to be the full nonempty row size (with padding) because of the way allgather works

                    row = np.empty((A.m_loc * row_comm.Get_size()), dtype=A.dtype)
                    A.row_comm.Allgather(A.local[A_block_row], row)
                    row = row[:A.m]  # Trim any padding from the last nonempty block
                    
                    shared_array[B_row] = row

            # Synchronize writes across all ranks (this is done in from_shared_buffer as well, but just to be safe)
            win.Fence()

            if pmat.use_scatter:
                # Scatter the shared array to all processes
                B = pmat.from_numpy(shared_array, dtype=A.dtype)
            else:
                # Convert shared array to a pmat 
                B = pmat.from_shared_buffer(win, shared_array, dtype=A.dtype)

            # Clean up
            win.Free()

            return B



    @staticmethod
    def from_file(filename: str, prefix_current_directory=True) -> tuple['pmat', int]:
        comm = MPI.COMM_WORLD

        filepath = os.path.join(os.path.dirname(__file__), filename) if prefix_current_directory else filename

        # Open the file
        amode = MPI.MODE_RDONLY
        fh = MPI.File.Open(comm, filepath, amode)

        file_size = fh.Get_size()

        header = np.empty(2, dtype=np.int64)

        fh.Read_at(0, header)
        dtype_len = np.empty(1, dtype=np.int32)
        fh.Read_at(16, dtype_len)
        dtype_bytes = bytearray(dtype_len[0])
        fh.Read_at(20, dtype_bytes)
        dtype_str = dtype_bytes.decode('utf-8')
        
        dtype = np.dtype(dtype_str)
        nrows, ncols = header

        data_offset = 20 + dtype_len[0]
        
        # Empty pmat
        pmat_matrix = pmat(nrows, ncols)

        coords = pmat_matrix.coords
        extent = pmat_matrix.extents[coords[0]][coords[1]]
        offset = pmat_matrix.offsets[coords[0]][coords[1]]

        local_rows, local_cols = extent[0], extent[1]
        row_offset, col_offset = offset[0], offset[1]

        for i in range(local_rows):
            global_row = row_offset + i
            file_offset = data_offset + (global_row * ncols + col_offset) * np.dtype(dtype).itemsize
            buffer = bytearray(local_cols * np.dtype(dtype).itemsize)
            fh.Read_at(file_offset, buffer)
            pmat_matrix.local[i, :local_cols] = np.frombuffer(buffer, dtype=dtype)

        fh.Close()
        return (pmat_matrix, file_size)

    def to_numpy(self, remove_padding=True, all_to_root=False) -> np.ndarray | None:
        # Gather all blocks at root
        Pr = pmat.grid_comm.dims[0]
        Pc = pmat.grid_comm.dims[1]

        # Destination matrix with padding (if any)
        dst_matrix = np.zeros((self.n_loc * Pr, self.m_loc * Pc), dtype=self.dtype)

        all_blocks = None

        if all_to_root:
            # Root gathers a copy of every blocks
            all_blocks = pmat.grid_comm.gather(self.local, root=0)
        else:
            # All processes gather a copy of every blocks
            all_blocks = pmat.grid_comm.allgather(self.local)
        if all_blocks is not None:
            
            for i, block in enumerate(all_blocks):
                row, col = divmod(i, pmat.grid_comm.dims[1]) 

                # Add padding if needed to the block
                if self.n_loc - block.shape[0] > 0:
                    block = np.pad(block, ((0, self.n_loc - block.shape[0]), (0,0)))
                                   
                if self.m_loc - block.shape[1] > 0:
                    block = np.pad(block, ((0,0), (0, self.m_loc - block.shape[1])))

                # Copy block into grid
                dst_matrix[row*self.n_loc:(row+1)*self.n_loc, col*self.m_loc:(col+1)*self.m_loc] = block #.reshape(self.n_loc, self.m_loc)


        if all_to_root:
            # Only root has the full matrix
            if pmat.grid_comm.rank == 0:
                return dst_matrix[:self.n, :self.m] if remove_padding else dst_matrix
            else:
                return None
        else:
            return dst_matrix[:self.n, :self.m] if remove_padding else dst_matrix

    @staticmethod
    def from_numpy(src_matrix: np.ndarray, dtype=np.float64) -> 'pmat':
        if src_matrix is not None:
            src_matrix = np.atleast_2d(src_matrix)

        # Make sure all processes know the shape and dtype, even if only rank 0 has the numpy array
        shape = pmat.grid_comm.bcast(src_matrix.shape if src_matrix is not None else None, root=0)
        n, m = shape[0], shape[1]

        dtype=pmat.grid_comm.bcast(src_matrix.dtype if src_matrix is not None else None, root=0)
        
        distributed_matrix = pmat(n, m, dtype=dtype)

        blocks = []

        # Only ranks with the stored blocks will be scattered
        if src_matrix is not None:
            for i in range(pmat.grid_comm.dims[0]):
                for j in range(pmat.grid_comm.dims[1]):
                    row_start = i * distributed_matrix.n_loc
                    row_end = min((i + 1) * distributed_matrix.n_loc, distributed_matrix.n)
                    col_start = j * distributed_matrix.m_loc
                    col_end = min((j + 1) * distributed_matrix.m_loc, distributed_matrix.m)
                    
                    block = src_matrix[row_start:row_end, col_start:col_end]
                    blocks.append(block)

        # Scatter blocks to all processes
        local_block = pmat.grid_comm.scatter(blocks, root=0)
        
        # Set local block
        distributed_matrix.local[:local_block.shape[0], :local_block.shape[1]] = local_block
        
        return distributed_matrix
    
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

        self.n_loc = ceil(self.n / Pr)
        self.m_loc = ceil(self.m / Pc) 
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

        # If the extent's row size -- or -- column size is zero, the block is empty
        extent = [[(0, 0) if a == 0 or b == 0 else (a, b) for b in m_loc_extent] for a in n_loc_extent]

        # extent = [[(a, b) for b in m_loc_extent] for a in n_loc_extent]


        n_loc_pos = [0]
        for v in n_loc_extent[:-1]:
            n_loc_pos.append(n_loc_pos[-1] + v)

        m_loc_pos = [0]
        for v in m_loc_extent[:-1]:
            m_loc_pos.append(m_loc_pos[-1] + v)

        offsets = [[(a,b) for b in m_loc_pos] for a in n_loc_pos]
        self.extents = extent
        self.offsets = offsets

        self.block_extent = extent[self.coords[0]][self.coords[1]]
        self.block_offset = offsets[self.coords[0]][self.coords[1]]

        # Count number of non-empty blocks in each process dimension
        nonempty_in_row = sum(1 for i in range(Pr) if extent[i][0][0] > 0)
        nonempty_in_col = sum(1 for j in range(Pc) if extent[0][j][1] > 0)


        self.nonempty_processes = (nonempty_in_row, nonempty_in_col)

        ### Create subcommunicator for non-empty blocks only
        coords = self.coords
        grid = self.grid_comm

        # Decide if THIS rank’s block is non-empty
        extent   = self.extents[coords[0]][coords[1]]
        has_work = int(extent[0] > 0 and extent[1] > 0)  # 1 if non-empty

        row_all = grid.Sub([False, True])   # all ranks in my row
        col_all = grid.Sub([True,  False])  # all ranks in my column

        # Use MPI.UNDEFINED to drop ranks with no work
        row_color = 0 if has_work else MPI.UNDEFINED
        col_color = 0 if has_work else MPI.UNDEFINED

        self.row_comm = row_all.Split(color=row_color, key=coords[1])
        self.col_comm = col_all.Split(color=col_color, key=coords[0])

        row_all.Free()
        col_all.Free()

    def __del__(self):
     # Destructor: free communicators safely
        for comm in (getattr(self, 'row_comm', None), getattr(self, 'col_comm', None)):
            if comm is not None and comm != MPI.COMM_NULL:
                try:
                    comm.Free()
                except MPI.Exception:
                    # It might already be freed or invalid during MPI_Finalize
                    pass
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
        return f"{self.to_numpy()}"
    
    def astype(self, dtype):
        return pmat(self.n, self.m, self.local.astype(dtype))
    
    def __getitem__(self, idx):

        idx = normalize_index(idx)

        if isinstance(idx, np.ndarray) and idx.ndim == 1:
            # 1D array indexing is treated as row indexing
            idx = np.atleast_2d(idx)
            return pmat.index_rows(self, idx)

        # idx0 = [idx[0]] if isinstance(idx[0], int) else idx[0]
        # idx1 = [idx[1]] if isinstance(idx[1], int) else idx[1]
        # # assert(len(idx0) == len(idx1))

        local_values = []
        for (global_i, global_j) in zip(idx[0], idx[1]):
                rank_i, block_i = divmod(global_i, self.n_loc)
                rank_j, block_j = divmod(global_j, self.m_loc)
                
                if (self.coords[0] == rank_i) and (self.coords[1] == rank_j):
                    local_values.append(self.local[block_i, block_j])

        local_values = np.array(local_values, dtype=self.local.dtype)

        
        global_values = self.grid_comm.allgather(local_values)
        # Filter out empty blocks
        non_empty_lists = [lst for lst in global_values if len(lst)!= 0] 

        global_values = np.concatenate([np.array(lst, dtype=self.dtype) for lst in non_empty_lists])
        return global_values

    def __setitem__(self, idx, value):
        idx = normalize_index(idx)

        idx0 = [idx[0]] if isinstance(idx[0], int) else idx[0]
        idx1 = [idx[1]] if isinstance(idx[1], int) else idx[1]

        for (global_i, global_j) in zip(idx0, idx1):

            rank_i, block_i = divmod(global_i, self.n_loc)
            rank_j, block_j = divmod(global_j, self.m_loc)
                
            if (self.coords[0] == rank_i) and (self.coords[1] == rank_j):
                self.local[block_i, block_j] = value

        return

    def remove_first_column(self):
        assert self.m > 1, "matrix only has one column"
    
        if pmat.use_scatter:
            all_blocks = self.to_numpy(all_to_root=True)
            

            # Remove first column
            
            all_blocks_no_bias = all_blocks[:, 1:] if all_blocks is not None else None # 
            
            no_bias = pmat.from_numpy(all_blocks_no_bias)   

            return no_bias
        
        ### Shared memory approach to remove first column #########

        col_offset = 1  # remove first column

                
        n, m = self.shape        
        
        coords = pmat.grid_comm.coords
        extent = self.extents[coords[0]][coords[1]]
        position = self.offsets[coords[0]][coords[1]]

        row_start, row_end = position[0], position[0] + extent[0]
        col_start, col_end = position[1], position[1] + extent[1]

        local = self.local[:extent[0], :extent[1]]

        # Set up the submatrix
        submat = pmat(n, m - 1, dtype=self.dtype)
        submat_extent = submat.extents[coords[0]][coords[1]]
        submat_position = submat.offsets[coords[0]][coords[1]]

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



    ############################################################################
    # Arithmetic operators
    ############################################################################
    @staticmethod
    def check_for_broadcast(A: 'pmat', B: 'pmat'):
        # Python broadcasting expands a smaller array (a vector) to match a larger array (a matrix)

        ########################################################################
        # Scalar broadcasting
        
        if np.isscalar(B):
            A_extent = A.extents[A.coords[0]][A.coords[1]]
            A_local = A.local[:A_extent[0], :A_extent[1]]

            return A_local, B, (A.n, A.m)
        elif np.isscalar(A):
            B_extent = B.extents[B.coords[0]][B.coords[1]]
            B_local = B.local[:B_extent[0], :B_extent[1]]

            return A, B_local, (B.n, B.m)

        
        # Operands are without padding
        A_extent = A.extents[A.coords[0]][A.coords[1]]
        A_local = A.local[:A_extent[0], :A_extent[1]]
        
        B_extent = B.extents[B.coords[0]][B.coords[1]]
        B_local = B.local[:B_extent[0], :B_extent[1]]

        # Output shape is a matrix
        output_shape = (max(A.n, B.n), max(A.m, B.m))

        ########################################################################
        # Row vector and column vector broadcasting. 
        # In the process grid, a vector has empty blocks where the matrix 
        # has non-empty blocks, so we need to broadcast its nonempty blocks
        # downward (columns) or rightward (rows) from the root of the matrix's
        # column and row subcommunicators.
        
        if B.shape == (A.n, 1) and B.m != A.m:  
            if A.row_comm == MPI.COMM_NULL:
                return A_local, A_local, output_shape
            else:
                B_local = A.row_comm.bcast(B_local, root=0)
                return A_local, B_local, output_shape
        elif A.shape == (B.n, 1) and A.m != B.m:           
            if B.row_comm == MPI.COMM_NULL:
                return B_local, B_local, output_shape
            else:
                A_local = B.row_comm.bcast(A_local, root=0)
                return A_local, B_local, output_shape
        elif B.shape == (1, A.m) and B.n != A.n:  
            if A.col_comm == MPI.COMM_NULL:
                return A_local, A_local, output_shape
            else:         
                B_local = A.col_comm.bcast(B_local, root=0)
                return A_local, B_local, output_shape
        elif A.shape == (1, B.m) and A.n != B.n:
            if B.col_comm == MPI.COMM_NULL:
                return B_local, B_local, output_shape
            else:
                A_local = B.col_comm.bcast(A_local, root=0)
                return A_local, B_local, output_shape

        #######################################################################
        # Nevermind: Both are matrices or both are vectors of the same shape
        else:
            return A_local, B_local, output_shape


    def __gt__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.greater(A,  B))

        # return pmat(self.n, self.m, self.local > other)

    def __eq__(self, other):

        A, B, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.equal(A,  B), dtype=np.bool_)

        # return pmat(self.n, self.m, self.local > other)
    
    def __add__(self, other): 
        A, B, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.add(A,  B))

    def __sub__(self, other):     
        A, B, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.subtract(A, B))


    def __mul__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.multiply(A, B))
    
    def __rmul__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.multiply( B, A))
        
    def __truediv__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.true_divide(A, B))
    
    def __rdiv__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.true_divide(A, B))
    
    def __rtruediv__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)

        return pmat(output_shape[0], output_shape[1], np.true_divide(B, A))

    def __radd__(self, other):
        A, B, output_shape = pmat.check_for_broadcast(self, other)
        
        return pmat(output_shape[0], output_shape[1], np.add(B, A))
    

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

    def __pow__(self, exponent):
        new_local = self.local ** exponent
        return pmat(self.n, self.m, new_local)

    # def __rpow__(self, base):
    #     # handles base ** pmat -> elementwise
    #     new_local = base ** self.local
    #     return pmat(self.n, self.m, new_local)

    # def __ipow__(self, exponent):
    #     # in-place power
    #     self.local **= exponent
    #     return self

    def __matmul__(self, other: 'pmat'):
        # Using Cannon's Algorithm


        assert self.m == other.n, f"A @ B: A.m:{self.m} != B.n:{other.n}"

        C = np.zeros((self.n_loc, other.m_loc))

        # Total steps over grid 
        num_steps = min(pmat.grid_comm.dims)

        # Deep copies for Sendrecv_replace
        A_block = self.local.copy()
        B_block = other.local.copy()

        # Skew (initial alignment)
        for _ in range(self.coords[0]):
            src, dst = pmat.grid_comm.Shift(1, -1)
            pmat.grid_comm.Sendrecv_replace(A_block, dest=dst, source=src)

        for _ in range(self.coords[1]):
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
        extent = self.extents[self.coords[0]][self.coords[1]]
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

        # assert axis == 1, f'Only axis=1 (row-wise) is supported for psum'


        if axis == 1:
            dtype = self.dtype
            coords = self.coords
            extent = self.extents[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            row_sum = []
            if self.row_comm != MPI.COMM_NULL:
                for row in range(extent[0]):
                    # Reduce to the root of each group
                    local_sum = np.sum(local[row, :])
                    row_sum.append(self.row_comm.reduce(local_sum, op=MPI.SUM, root=0))

                if self.grid_comm.coords[1] == 0:
                    row_sum = np.array([x for x in row_sum if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1) # row_sum is now a column vector
                else:
                    row_sum = None
            else:
                row_sum = None

            return pmat(self.n, 1, row_sum)
        elif axis == None:
            grid = self.grid_comm

            # Global sum
            coords = self.coords
            extent = self.extents[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]

            local_sum = np.sum(local)
                        
            global_sum = grid.allreduce(local_sum, op=MPI.SUM)
            return global_sum
        else:
            raise NotImplementedError("psum axis=0 not implemented yet")

    def pmin(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        if axis == 1:
            
            if self.row_comm != MPI.COMM_NULL:
                dtype = self.dtype
                coords = self.coords
                extent = self.extents[coords[0]][coords[1]]
                local = self.local[:extent[0], :extent[1]]

                row_min = []

                for row in range(extent[0]):
                    # Reduce to the root of each group
                    local_min = np.min(local[row, :])
                    row_min.append(self.row_comm.reduce(local_min, op=MPI.MIN, root=0))

                if self.grid_comm.coords[1] == 0:
                    row_min = np.array([x for x in row_min if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1) # maxs is now a column vector
                else:
                    row_min = None
            else:
                row_min = None

            return pmat(self.n, 1, row_min)
    
        elif axis is None:
            # Global min
            global_min = self.grid_comm.allreduce(np.min(self.local), op=MPI.MIN)
            return global_min      # return scalar value (not pmat)
        else:
            raise ValueError(f"Invalid axis {axis} for pmin")
        
    def pmax(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        if axis == 1:
            
            if self.row_comm != MPI.COMM_NULL:
                dtype = self.dtype
                coords = self.coords
                extent = self.extents[coords[0]][coords[1]]
                local = self.local[:extent[0], :extent[1]]

                row_max = []

                for row in range(extent[0]):
                    # Reduce to the root of each group
                    local_max = np.max(local[row, :])
                    row_max.append(self.row_comm.reduce(local_max, op=MPI.MAX, root=0))

                if self.grid_comm.coords[1] == 0:
                    row_max = np.array([x for x in row_max if x is not np.nan], dtype=dtype).flatten().reshape(-1, 1) # maxs is now a column vector
                else:
                    row_max = None
            else:
                row_max = None

            return pmat(self.n, 1, row_max)
    
        elif axis is None:
            # Global max
            global_max = self.grid_comm.allreduce(np.max(self.local), op=MPI.MAX)
            return global_max      # return scalar value (not pmat)
        else:
            raise ValueError(f"Invalid axis {axis} for pmax")
    
    def pargmax(self, *args, **kwargs):
        axis = kwargs.get('axis', None) # axis=0 for cols, axis=1 for rows

        assert axis == 1, f'Only axis=1 (row-wise) is supported for pmax'

        if axis == 1:
            ############## Slower using allgather than below ###################
            # subgrid = self.subgrid

            # dtype = self.dtype
            # coords = self.coords
            # position = self.position[coords[0]][coords[1]]
            # extent = self.extent[coords[0]][coords[1]]
            # local = self.local[:extent[0], :extent[1]]
            # num_rows = self.block_size[0]

            # result = np.zeros((extent[0], 2), dtype=(np.int64, self.dtype))
            
            # if subgrid == MPI.COMM_NULL:
            #     # Return type is np.int64 for argmax
            #     return pmat(self.n, 1, local=None, dtype=np.int64)

            # else:
            #     horz_comm = subgrid.Sub([False, True])  # rows

            #     for row in range(num_rows): 
                    
            #         local_max = dtype(np.max(local[row, :]))
            #         local_idx = np.int64(np.argmax(local[row, :]))
            #         global_idx = np.int64(local_idx + position[1])


            #         row_max_combined = np.zeros((1, horz_comm.size), dtype=dtype)
            #         row_idx_combined = np.zeros((1, horz_comm.size), dtype=np.int64)

            #         horz_comm.Allgather(local_max, row_max_combined)
            #         horz_comm.Allgather(global_idx, row_idx_combined)

            #         idx = np.argmax(row_max_combined)
            #         result[row, 0] = row_idx_combined.flatten()[idx]
            #         result[row, 1] = row_max_combined.flatten()[idx]

            #     argmax = result[:,0].reshape(num_rows, 1)

            #     # Return type is np.int64 for argmax
            #     return pmat(self.n, 1, local=argmax, dtype=np.int64)

            #### Slightly faster to use allreduce #########################
            dtype = self.dtype
            coords = self.coords
            position = self.offsets[coords[0]][coords[1]]
            extent = self.extents[coords[0]][coords[1]]
            local = self.local[:extent[0], :extent[1]]
            num_rows = self.block_extent[0]

            result = np.zeros((extent[0], 2), dtype=(np.int64, self.dtype))
            
            if self.row_comm == MPI.COMM_NULL:
                # Return type is np.int64 for argmax
                return pmat(self.n, 1, local=None, dtype=np.int64)

            else:
                for row in range(num_rows): 
                    
                    local_max = np.max(local[row, :]).astype(dtype)
                    local_idx = np.int64(np.argmax(local[row, :]))
                    global_idx = np.int64(local_idx + position[1])

                    mpi_dtype = np.dtype([('value', np.float64), ('index', np.int32)], align=True)
                    sendbuf = np.array([(local_max, global_idx)], dtype=mpi_dtype)  
                    recvbuf = np.array([(0.0, 0)], dtype=mpi_dtype)

                    # Allreduce with MAXLOC
                    self.row_comm.Allreduce([sendbuf, MPI.DOUBLE_INT], [recvbuf, MPI.DOUBLE_INT], op=MPI.MAXLOC)

                    result[row,:] = recvbuf['index'][0]

                argmax = result[:,0].reshape(num_rows, 1)

                # Return type is np.int64 for argmax
                return pmat(self.n, 1, local=argmax, dtype=np.int64)

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
    
        if pmat.use_scatter:
            all_blocks = self.to_numpy(all_to_root=True)
            

            # Add a row of ones on top
            all_blocks_with_bias = np.vstack([np.ones((1, self.m), dtype=self.dtype), all_blocks]) if all_blocks is not None else None
            
            with_bias  = pmat.from_numpy(all_blocks_with_bias)   

            return with_bias
        
        ### Shared memory approach to stacking a row of ones #########
        # row_offset = 1  # add to first row
        coords = pmat.grid_comm.coords
        n, m = self.shape        
        
        # Set up the extented matrix
        newmat = pmat(n + 1, m, dtype=self.dtype)
        newmat_extent = newmat.extents[coords[0]][coords[1]]
        newmat_position = newmat.offsets[coords[0]][coords[1]]

        newmat_row_start, newmat_row_end = newmat_position[0], newmat_position[0] + newmat_extent[0]
        newmat_col_start, newmat_col_end = newmat_position[1], newmat_position[1] + newmat_extent[1]

        newmat_local = newmat.local

        ########################################################################
        # Write each rank's local block to the shared array offset by one row
        ########################################################################
        extent = self.extents[coords[0]][coords[1]]
        position = self.offsets[coords[0]][coords[1]]

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
    extent = newmat.extents[coords[0]][coords[1]]

    newmat.local[:extent[0], :extent[1]] = np.zeros((extent[0], extent[1]))

    return newmat

def p_ones_like(M_pmat: pmat) -> pmat:
    newmat = pmat(M_pmat.n, M_pmat.m)

    coords = pmat.grid_comm.coords
    extent = newmat.extents[coords[0]][coords[1]]

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



