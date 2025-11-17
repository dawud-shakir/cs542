# test_pmat.py

import numpy as np

np.set_printoptions(precision=1, suppress=True, floatmode='fixed')

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

# def check_scalar(pmat_value, numpy_value, str=""):
#     if grid.rank == 0:
#         assert np.isclose(pmat_value, numpy_value), f"{str} failed isclose\npmat_value:\n{pmat_value}\nnumpy_value:\n{numpy_value}"
        
#         print(f"\t{str:<20}...\033[31mpassed\033[0m isclose")

#     grid.Barrier()


def test(n, k, m, dtype=np.double):

    ################################################################
    # Start tests
    ################################################################

    if grid.rank == 0:           
        print(f"Testing n={n}, k={k}, m={m}...")
    
    # only positive values
    A_mat = (np.arange(1, n * k + 1) / (n * k)).reshape(n, k).astype(dtype)    
    B_mat = (np.arange(1, k * m + 1) / (k * m)).reshape(k, m).astype(dtype)
    D_mat = (np.arange(1, m * n + 1) / (n * m)).reshape(m, n).astype(dtype)

    # with negative values
    A_mat = (np.arange(-(n * k)/2 + 1, (n * k)/2 + 1) / (n * k)).reshape(n, k).astype(dtype)    
    B_mat = (np.arange(-(k * m)/2 + 1, (k * m)/2 + 1)).reshape(k, m).astype(dtype)
    D_mat = (np.arange(-(m * n)/2 + 1, (m * n)/2 + 1)).reshape(m, n).astype(dtype)


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

    # Scalar operations
    check(A + 5, A_mat + 5, "A + 5")
    check(5 + A, 5 + A_mat, "5 + A")
    check(A - 5, A_mat - 5, "A - 5")
    check(5 - A, 5 - A_mat, "5 - A")
    check(A * 5, A_mat * 5, "A * 5")
    check(5 * A, 5 * A_mat, "5 * A")
    check(A / 5, A_mat / 5, "A / 5")
    check(5 / A, 5 / A_mat, "5 / A")    
    check(A > 5, A_mat > 5, "A > 5")
    check(A == A, A_mat == A_mat, "A == A")

    ################################################################
    # Function operations
    
    # Argmax

 



    # log_props_partial = np.array([
    # [-0.7, -10.9, -6.4, -41.0, -82.9],
    # [-124.1, -82.3, -22.5, -0.7, -13.4],
    # [-181.9, -168.3, -44.6, -0.7, -31.8],
    # [-146.5, -135.8, -40.0, -4.9, -0.7],
    # [-92.3, -136.3, -12.8, -14.2, -0.7],
    # [-135.4, -128.9, -29.3, -0.7, -26.4],
    # [-53.3, -35.7, -37.0, -20.0, -0.7],
    # [-111.9, -89.3, -0.7, -18.7, -20.8],
    # [-0.7, -38.8, -50.1, -42.6, -21.7],
    # [-104.2, -123.6, -33.6, -4.9, -0.7]
    # ])#.astype(dtype)

    # p_log_props_partial = pmat.from_numpy(log_props_partial)

    # p_log_props_partial.print_pretty("log_props_partial", remove_padding=False)


    # check(p_log_props_partial.pargmax(axis=1), np.argmax(log_props_partial, axis=1, keepdims=True), "argmax(log_props_partial, axis=1)")

    check(A.pargmax(axis=1), np.argmax(A_mat, axis=1, keepdims=True), "argmax(A, axis=1)")


    # Global mean (returns a scalar)
    check_scalar(A.pmean(), np.mean(A_mat), "mean(A)")

    # Row mean
    check(A.pmean(axis=1), np.mean(A_mat, axis=1, keepdims=True), "mean(A, axis=1)")

    # Global max (returns a scalar)
    check(A.pmax(axis=1), np.max(A_mat, axis=1, keepdims=True), "max(A, axis=1)")

    # Global sum (returns a scalar)
    check_scalar(A.psum(), np.sum(A_mat), "global sum(A)")
    
    # Row sum
    check(A.psum(axis=1), np.sum(A_mat, axis=1, keepdims=True), "sum(A, axis=1)")

    # Maximum sets each element to max(scalar, element)
    scalar = 0.5
    check(A.pmaximum(scalar), np.maximum(scalar, A_mat), f"maxmium({scalar:0.2}, A)")



    ################################################################
    # Vecmat and Matvec operation 
    
    # Row vector = n...1
    row_vec = np.array(np.arange(n+1, 1, -1), ndmin=2).reshape(1, -1)
    row_pvec = pmat.from_numpy(row_vec)
    check(row_pvec @ A, row_vec @ A_mat, "vecmat")



    # Column vector = k...1
    col_vec = np.array(np.arange(k+1, 1, -1) / k, ndmin=2).reshape(-1, 1) 
    col_pvec = pmat.from_numpy(col_vec)
    check(A @ col_pvec, A_mat @ col_vec, "matvec")

    ###############
    # Test Python's broadcast betweeen a matrix and a vector
    
    # Column vector operations
    col_vec = np.array(np.arange(1, n + 1) / n, ndmin=2).reshape(n, 1) 
    col_pmat = pmat.from_numpy(col_vec)

    # check(A + A_rand_mat, A_mat + A_rand, "A + rand(A)")


    check(A - col_pmat, A_mat - col_vec, "A - col_vec")

    check(col_pmat - A, col_vec - A_mat, "col_vec - A")

    check(col_pmat - col_pmat, col_vec - col_vec, "col_vec - col_vec")
    # check(A * A_rand_mat, A_mat * A_rand, "A * rand(A)")

    # Row vector operations
    row_vec = np.array(np.arange(1, k + 1), ndmin=2).reshape(1, k) 
    row_pmat = pmat.from_numpy(row_vec)

    # check(A + A_rand_mat, A_mat + A_rand, "A + rand(A)")
    check(A - row_pmat, A_mat - row_vec, "A - row_vec")
    
    check(row_pmat - A, row_vec - A_mat, "row_vec - A")
    
    check(row_pmat - row_pmat, row_vec - row_vec, "row_vec - row_vec")
    # check(A * A_rand_mat, A_mat * A_rand, "A * rand(A)")




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