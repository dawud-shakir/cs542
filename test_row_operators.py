# test_row_operators.py

import numpy as np

from pmat import pmat, create_grid_comm
axis = 1
grid = create_grid_comm()

sizes = [
    # (10, 10),
    # (100, 50),
    # (50, 100),
    # (200, 200),
    (9, 5),
    (8, 8),
    (16, 12),
    (64, 10),
    (65, 10),
    (28*28, 1000),
    (64, 100),
    (10, 100)
]

def test_max(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = pmat.from_numpy(A_numpy)

    A_pmat_result = A_pmat.pmax(axis=axis)
    A_numpy_result = np.max(A_numpy, axis=axis, keepdims=True)  # keepdims used in layer.py

    if not np.allclose(A_pmat_result.get_full(), A_numpy_result):
        results = f"\npmat result:\n{A_pmat_result.get_full()}\nnumpy result:\n{A_numpy_result}"
        raise ValueError(f"\033[38;5;22m n={n}, m={m}: results do not match!\nOriginal:\n{A_numpy}\n{results}\033[0m")
    else:
        if grid.rank == 0:
            test_name = f"{test_max.__name__} n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")


    grid.Barrier()


def test_sum(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = pmat.from_numpy(A_numpy)

    A_pmat_result = A_pmat.psum(axis=axis)
    A_numpy_result = np.sum(A_numpy, axis=axis, keepdims=True)  # keepdims used in layer.py

    if not np.allclose(A_pmat_result.get_full(), A_numpy_result):
        results = f"\npmat result:\n{A_pmat_result.get_full()}\nnumpy result:\n{A_numpy_result}"
        raise ValueError(f"\033[38;5;22m n={n}, m={m}: results do not match!\nOriginal:\n{A_numpy}\n{results}\033[0m")
    else:
        if grid.rank == 0:
            test_name = f"{test_sum.__name__} n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")


    grid.Barrier()


def test_mean(n, m):
    A_numpy = np.arange(1, n * m + 1).reshape(n, m).astype(np.double)
    A_pmat = pmat.from_numpy(A_numpy)

    A_pmat_result = A_pmat.pmean(axis=axis)
    A_numpy_result = np.mean(A_numpy, axis=axis)

    if not np.allclose(A_pmat_result.get_full(), A_numpy_result):
        results = f"\npmat result:\n{A_pmat_result.get_full()}\nnumpy result:\n{A_numpy_result}"
        raise ValueError(f"\033[38;5;22m n={n}, m={m}: results do not match!\nOriginal:\n{A_numpy}\n{results}\033[0m")
    else:
        if grid.rank == 0:
            test_name = f"{test_mean.__name__} n={n}, m={m}"
            print(f"{test_name:<30} ...\033[31m" + "passed" + "\033[0m")


    grid.Barrier()


if __name__ == "__main__":

    if grid.rank == 0:
        print("*" * 30, "test_max", "*" * 30, "\n")
    
    for (n, m) in sizes:
        test_max(n, m)

    if grid.rank == 0:
        print('\n', "*" * 30, "test_sum", "*" * 30, "\n")
    
    for (n, m) in sizes:
        test_sum(n, m)

    if grid.rank == 0:
        print('\n', "*" * 30, "test_mean", "*" * 30, "\n")

    for (n, m) in sizes:
        test_mean(n, m)