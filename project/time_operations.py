# test_pmat.py

import numpy as np

np.set_printoptions(precision=1, suppress=True, floatmode='fixed')

from mpi4py import MPI

from package.pmat import pmat, create_grid_comm
import package.layer as nn
from package.utilities import time_it

grid = create_grid_comm()
rank = grid.Get_rank()
size = grid.Get_size()
coords = grid.coords



sizes = [
    (64, 64, 64),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # (8192, 8192, 8192),
]


repeat = 5
warmups = 2
use_min_time = False

# For layer timing
hidden_size = 64
batch_size = 1000
alpha = 1e-3


pmat.use_scatter = True


if __name__ == "__main__":
    if grid.rank == 0:
        print("=" * 40)
        print(f"Number of processes: {size} in a {grid.dims} grid")
        print(f"warmups={warmups}, repeat={repeat}, use_min_time={use_min_time}")
        print("=" * 40)


    output = (str)

    dtypes = [np.float64]   
    for (n, k, m) in sizes:
        for dtype in dtypes:
            
            if grid.rank == 0:
                print("-" * 40)
                print(f"n={n}, k={k}, m={m}, dtype={dtype.__name__}")
                print("-" * 40)

            A_np = (np.arange(1, n * k + 1) / (n * k)).reshape(n, k).astype(dtype)    
            B_np = (np.arange(1, k * m + 1) / (k * m)).reshape(k, m).astype(dtype)

            A_pmat = pmat.from_numpy(A_np)
            B_pmat = pmat.from_numpy(B_np)

            items = [np.arange(0, n).astype(int), np.random.randint(0, m, size=n).astype(int)]


            X_pmat = pmat.from_numpy(np.random.rand(hidden_size, batch_size).astype(dtype))

            fc2 = nn.Parallel_Layer(input_size=hidden_size, output_size=hidden_size); 
            fc2.phi = nn.ReLU
            fc2.phi_prime = nn.ReLU_derivative


            operations = {
                "from_numpy":           lambda: pmat.from_numpy(A_np),
                "set_full (scatter)":   lambda: A_pmat.set_full(B_np),
                "get_full (gather)":    lambda: A_pmat.get_full(),
                
                "matmul":               lambda: A_pmat @ B_pmat,
                "transpose":            lambda: A_pmat.T,
                "add":                  lambda: A_pmat + B_pmat,
                "log":                  lambda: np.log(A_pmat),  
                
                "getitem (n items)":    lambda: A_pmat.__getitem__(items),
                "setitem (n items)":    lambda: A_pmat.__setitem__(items, 0),
                "remove_first_column":  lambda: A_pmat.remove_first_column(),
                "greater_than(0.5)":    lambda: A_pmat.__gt__(0.5),
                "global sum":           lambda: A_pmat.psum(),
                "global mean":          lambda: A_pmat.pmean(),
                "global max":           lambda: A_pmat.pmax(),

                "row sum":              lambda: A_pmat.psum(axis=1),
                "row mean":             lambda: A_pmat.pmean(axis=1),
                "row max":              lambda: A_pmat.pmax(axis=1),
                "argmax":               lambda: A_pmat.pargmax(axis=1),

                "stack_ones_on_top":    lambda: A_pmat.stack_ones_on_top(),

                "ReLU":                     lambda: nn.ReLU(A_pmat),
                "ReLU_derivative":          lambda: nn.ReLU_derivative(A_pmat),
                "log_softmax":              lambda: nn.log_softmax(A_pmat.T),
                "nll_loss":                 lambda: nn.nll_loss(A_pmat, items[1]),  
                "nll_loss_derivative":      lambda: nn.nll_loss_derivative(A_pmat, items[1]),

                # f"layer forward(ReLU, {hidden_size,batch_size})":           lambda: fc2.forward(X_pmat),
                # f"layer backward(ReLU-deriv, {hidden_size,batch_size})":    lambda: fc2.backward(X_pmat),
                # f"layer update(Adam optimization)":                         lambda: (fc2.backward(X_pmat), fc2.update_weights(alpha=0.001)),
            
            }

            results = []
            for op_name, op_func in operations.items():
                grid.Barrier()
                min_t, mean_t = time_it(op_func, repeat=repeat, warmups=warmups, timer_fn=MPI.Wtime)

                if use_min_time:
                    time = grid.reduce(min_t, op=MPI.MIN, root=0)
                else:
                    time = grid.reduce(mean_t, op=MPI.MAX, root=0)
                

                # For sorting later
                if grid.rank == 0:
                    results.append((op_name, time))

            # For update_weights, subtract the time taken by backward pass
            for i, (op_name, _) in enumerate(results):
                if op_name == "layer update(Adam optimization)":
                    backward_time = next(t for name, t in results if name == f"layer backward(ReLU-deriv, {hidden_size,batch_size})")
                    adjusted_time = results[i][1] - backward_time
                    results[i] = (op_name, adjusted_time)
                    break

            if grid.rank == 0:

                results.sort(key=lambda x: x[1], reverse=True)  # sort by time (descending)
                max_chars = max(len(op_name) for op_name, _ in results) 
                for op_name, time in results:


                    print(f"{(op_name + ' =>'):>{max_chars + 5}} {time:.5f} seconds")
                