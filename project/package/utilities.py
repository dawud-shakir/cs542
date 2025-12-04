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

# ANSI color helpers
def set_text_color(code):
    # code is an integer (e.g. 31 for red, 0 to reset)
    print(f"\033[{code}m", end="", flush=True)

def reset_text_color():
    print("\033[0m", end="", flush=True)

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