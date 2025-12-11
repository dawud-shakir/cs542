# utilities.py - utility functions

import numpy as np
np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

from mpi4py import MPI
import os
import psutil         # for memory usage

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pmat import pmat  # for type checkers only


def create_grid_comm():
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()

    # Grid rows and columns
    Pr = int(np.sqrt(num_procs))
    Pc = int(np.sqrt(num_procs))

    dims = [Pr, Pc]
    periods = [True, True]
    return comm.Create_cart(dims, periods, reorder=True)

def dtype_to_mpi(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return MPI.BOOL # Or MPI.C_BOOL
    elif dtype == np.uint8:
        return MPI.UNSIGNED_CHAR
    elif dtype == np.int32:
        return MPI.INT
    elif dtype == np.int64:
        return MPI.LONG
    elif dtype == np.float32:
        return MPI.FLOAT
    elif dtype == np.float64:
        return MPI.DOUBLE
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# This should be using COMM_WORLD or a passed communicator
def print_matrix(title: str, my_list: list):
    grid = create_grid_comm()

    rows = len(my_list)
    cols = len(my_list[0])
   
    print(f"{title}:")
    for i in range(rows):
        for j in range(cols):
            if grid.rank == 0:
                print(f"{my_list[i][j]}", end=" ")
        print()
    print()

def print_ordered_by_rank(x, *args, **kwargs):
    grid = kwargs.get('comm', create_grid_comm())
    for p in range(grid.Get_size()):
        grid.Barrier()
        if p == grid.rank:
            print(x, *args, **kwargs)

def print_pmat_on_rank0(M: 'pmat', msg=""):
    comm = M.grid_comm
    rank = comm.Get_rank()

    s = str(M)
    if rank == 0:
        # msg = msg + ":" if len(msg) > 0 else msg
        print(f"{msg}\n{s}\n")

def pretty_string(distributed_matrix, name="", remove_padding=True, as_type=None):

    if name != "":
        if distributed_matrix.grid_comm.rank == 0:
            print(f"{name}:")

    if as_type is None:
        if distributed_matrix.local.dtype == np.int32 or distributed_matrix.local.dtype == np.int64:
            as_type = "i"
        elif distributed_matrix.local.dtype == np.float32:
            as_type = "f"
        elif distributed_matrix.local.dtype == np.float64:
            as_type = "d"
        elif distributed_matrix.local.dtype == np.bool_:
            as_type = "b"
        else:
            raise ValueError(f"Unsupported dtype for pretty print: {distributed_matrix.local.dtype}")

    full_matrix = distributed_matrix.to_numpy(remove_padding)
    matrix_str = ""            
    
    for row in range(distributed_matrix.n if remove_padding else distributed_matrix.n + distributed_matrix.n_pad):
        col = 0

        if row % distributed_matrix.n_loc == 0:
            row_color = (row * distributed_matrix.m) // distributed_matrix.n_loc
        while col < distributed_matrix.m:
            color_code = 31 + ((row_color + col))   # 7 possible colors
            # print(f"row {row} col {j} color {color_code}", flush=True)
            
            if as_type == "i":
                block_str = " ".join(f"{int(val):3d}" for val in full_matrix[row][col : col + distributed_matrix.m_loc])
            elif as_type == "f" or as_type == "d":
                block_str = "\b" + "".join(f"{float(val):10.1f}" for val in full_matrix[row][col : col + distributed_matrix.m_loc])
            elif as_type == "b":
                block_str = "\b" + "".join(f"{val}" for val in full_matrix[row][col : col + distributed_matrix.m_loc])                

            if distributed_matrix.grid_comm.rank == 0:
                Pr = row // distributed_matrix.n_loc
                Pc = col // distributed_matrix.m_loc
                # color_code = 31 + (Pr + Pc) % 7  # 7 possible colors
                # set_text_color(color_code)
                palette = [196, 46, 220, 21, 208, 93, 226, 201, 202, 51, 82, 129, 214, 200, 198, 199]
                color_code = palette[(Pr + Pc * distributed_matrix.grid_comm.dims[1]) % len(palette)]

                matrix_str += f"\033[38;5;{color_code}m{block_str}\033[0m"
                matrix_str += " "
                
                # color_code = (Pr * pmatrix.grid_comm.dims[1] + Pc) * 13 % 256
                # print(f"\033[38;5;{color_code}m{block_str}\033[0m", end=" ", flush=True)

    
            col += distributed_matrix.m_loc

        # New line after each global row
        if distributed_matrix.grid_comm.rank == 0:
            matrix_str += "\n" 
    
    pmat.grid_comm.Barrier()
    return matrix_str

def print_pretty(distributed_matrix, name="", remove_padding=True, as_type=None):
    matrix_str = distributed_matrix.pretty_string(name, remove_padding, as_type)
    if distributed_matrix.grid_comm.rank == 0:
        print(matrix_str, flush=True)



def mpi_get_variable_statistics(x):
    local_min = np.min(x)
    local_max = np.max(x)
    local_sum = np.sum(x)
    local_count = x.size

    global_min = MPI.COMM_WORLD.allreduce(local_min, op=MPI.MIN)
    global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)
    global_sum = MPI.COMM_WORLD.allreduce(local_sum, op=MPI.SUM)
    global_count = MPI.COMM_WORLD.allreduce(local_count, op=MPI.SUM)

    global_mean = global_sum / global_count

    # Get stddev
    all_values = np.empty((MPI.COMM_WORLD.Get_size(), x.size), dtype=x.dtype)
    all_values[MPI.COMM_WORLD.Get_rank(), :] = x
    all_values = MPI.COMM_WORLD.allgather(all_values)
    all_values = np.concatenate(all_values, axis=0)
    global_stddev = np.std(all_values)

    return global_min, global_max, global_mean, global_stddev


def mpi_print_variable_statistics(var_name, x):
    n_procs = MPI.COMM_WORLD.Get_size()
    v_min, v_max, v_mean, v_stddev = mpi_get_variable_statistics(x)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"{var_name}")
        print(f"  Procs: {n_procs}, Min: {v_min:.3f}, Max: {v_max:.3f}, Mean: {v_mean:.3f}, Stddev: {v_stddev:.3f}\n")

def get_memory_usage():
    
    # Prefer USS (unique set size) or PSS (proportional set size) if available; if not, fall back to rss-shared or rss.
    try:
        proc = psutil.Process(os.getpid())
        import gc
        gc.collect()   
        
        mem_full = proc.memory_full_info()
        # Choose uss (unique) first, else pss (proportional)
        mem_bytes = getattr(mem_full, "uss", None) or getattr(mem_full, "pss", None) or proc.memory_info().rss
        mem_label = "USS" if getattr(mem_full, "uss", None) else ("PSS" if getattr(mem_full, "pss", None) else "RSS")
    except Exception:
        mi = proc.memory_info()
        # If shared attribute exists, subtract it to approximate private memory
        shared = getattr(mi, "shared", None)
        if shared is not None:
            mem_bytes = mi.rss - shared
            mem_label = "RSS-shared"
        else:
            mem_bytes = mi.rss
            mem_label = "RSS"
    return mem_bytes, mem_label

def time_it(fn, *args, repeat=5, warmups=0, timer_fn=MPI.Wtime):
    # optional warmups for steady-state CPU, caches, etc.
    for _ in range(warmups):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = timer_fn()
        fn(*args)
        times.append(timer_fn() - t0)
    return min(times), sum(times)/len(times)