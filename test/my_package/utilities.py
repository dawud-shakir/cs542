import numpy as np
import time # perf_counter
import sys  # exit
from tqdm import tqdm # tqdm is not compatible with njit because njit compiles to machine code

def print_matrix(name, rows, cols, elements):
    print(f"{name}:")
    for i in range(rows):
        for j in range(cols):
            print(f"{elements[i*cols+j]:.1f} ", end="")
        print()

# True: error
def check_matmat(n, A, B, C):
    A2 = np.reshape(A,(n,n))
    B2 = np.reshape(B,(n,n))
    return not np.allclose(C, (A2 @ B2).flatten())

def time_it(fn, *args, repeat=5, warmups=0, timer_fn=time.perf_counter):
    # optional warmups (for steady-state CPU caches etc.)
    for _ in range(warmups):
        fn(*args)
    times = []
    for _ in range(repeat):
        t0 = timer_fn()
        fn(*args)
        times.append(timer_fn() - t0)
    return min(times), sum(times)/len(times)