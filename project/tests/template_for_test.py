# test_pmat.py

import numpy as np

np.set_printoptions(precision=1, suppress=True, floatmode='fixed')

from pmat import pmat, create_grid_comm, print_matrix
grid = create_grid_comm()
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
    
    A_mat = (np.arange(1, n * k + 1) / (n * k)).reshape(n, k).astype(dtype)    
    A = pmat.from_numpy(A_mat)

    check(A, A_mat, "from_numpy")

    ################################################################
    # End tests
    ################################################################

if __name__ == "__main__":
    
    
    # import traceback
    # try:
        for (n, k, m) in sizes:
            test(n, k, m)
    # except Exception as e:
    #     tb_str = traceback.format_exc()

       #### What if exceptions are different on different ranks? #### 
    
    #     if grid.rank == 0:
    #         print("Exception caught:\n", tb_str)