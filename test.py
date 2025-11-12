# test.py

import numpy as np

from matplotlib.pyplot import sca
from pmat import pmat, create_grid_comm, print_matrix
axis = 1
grid = create_grid_comm()

sizes = [
    # (10, 10),
    # (100, 50),
    # (50, 100),
    # (200, 200),
    # (1000, 2000, 10000),
    # (28*28, 64, 64),
    (9, 5, 16),
    (8, 8, 16),
    (16, 12, 16),
    (64, 10, 16),
    (65, 10, 16),
    (28*28, 1000, 16),
    (64, 100, 16),
    (10, 100, 16)
]

def check(M1: pmat, M2: np.ndarray, str=""):
    pmat_as_numpy = M1.get_full()
    if grid.rank == 0:
        assert isinstance(M1, pmat) and isinstance(M2, np.ndarray), f"{str} failed instance check\ntype(M1)={type(M1)} and type(M2)={type(M2)}"
        
        assert np.allclose(pmat_as_numpy, M2), f"{str} failed allclose\nM1:\n{pmat_as_numpy}\nM2:\n{M2}"

        # assert M1_pmat.dtype == M2.dtype, f"{str} failed type check\ndtype(M1)={M1_pmat.dtype} and dtype(M2)={M2.dtype}"
        
        print(f"\t{str:<20}...\033[31mpassed\033[0m allclose")

    grid.Barrier()

def check_scalar(val1, val2, str=""):
    if grid.rank == 0:


        assert np.isclose(val1, val2), f"{str} failed isclose\nval1:\n{val1}\nval2:\n{val2}"
        
        print(f"\t{str:<20}...\033[31mpassed\033[0m isclose")

    grid.Barrier()


def test(n, k, m, dtype=np.double):

    ################################################################
    # Start tests
    ################################################################

    if grid.rank == 0:           
        print(f"Testing n={n}, k={k}, m={m}...")
    
    A_mat = (np.arange(1, n * k + 1) / (n * k)).reshape(n, k).astype(dtype)
    B_mat = (np.arange(1, k * m + 1) / (k * m)).reshape(k, m).astype(dtype)
    D_mat = (np.arange(1, m * n + 1) / (n * m)).reshape(m, n).astype(dtype)

    A = pmat.from_numpy(A_mat)

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

    # Logarithm
    check(np.log(A), np.log(A_mat), "log(A)")

    # Exponentiation
    check(np.exp(A), np.exp(A_mat), "exp(A)")

    # Greater than
    check(A > 5, A_mat > 5, "A > 5")

    ################################################################
    # Function operations
    
    # Global mean (scalar)
    check_scalar(A.pmean(), np.mean(A_mat), "mean(A)")

    check(A.pmean(axis=1), np.mean(A_mat, axis=1, keepdims=True), "mean(A, axis=1)")

    check(A.pmax(axis=1), np.max(A_mat, axis=1, keepdims=True), "max(A, axis=1)")

    check(A.psum(axis=1), np.sum(A_mat, axis=1, keepdims=True), "sum(A, axis=1)")

    # Maximum takes scalar
    scalar = 0.5
    check(A.pmaximum(scalar), np.maximum(scalar, A_mat), f"maxmium({scalar:0.2}, A)")



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

    # if grid.rank == 0:
    #     A_extent = A.extent
    #     B_extent = B.extent


    #     print_matrix(A.extent, name="A extent")
    #     print_matrix(B.extent, name="B extent")
    #     print()
        
    # A.print_pretty("A", remove_padding=False)
    # B.print_pretty("B", remove_padding=False)

    #     # for i in range(Pr):
    #     #     for j in range(Pc):
    #     #         print(f"{A_extent[i][j][1]} {B_extent[j][i][0]}")
   


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