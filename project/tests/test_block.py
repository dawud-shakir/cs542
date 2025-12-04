from math import ceil
from venv import create
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD

def create_grid_comm():
    # Grid rows and columns
    Pr = int(np.sqrt(comm.Get_size()))
    Pc = int(np.sqrt(comm.Get_size()))

    dims = [Pr, Pc]
    periods = [True, True]
    return comm.Create_cart(dims, periods, reorder=True)

grid = create_grid_comm()
rank = grid.Get_rank()
size = grid.Get_size()
coords = grid.coords

class Block(np.ndarray):
    def __new__(cls, data, custom_attr=None):
        # Convert data to a numpy array and view it as Block
        data = np.asarray(data)
        obj = data.view(cls)
        # Add custom attributes here
        obj.custom_attr = custom_attr or "default_value"
        return obj

    def __array_finalize__(self, obj):
        # Propagate custom attributes to new arrays (e.g., from slicing or operations)
        if obj is None:
            return
        self.custom_attr = getattr(obj, 'custom_attr', "default_value")

    def func(self):
        print(self)
        print("custom_attr=", self.custom_attr)
        return "done"

def print_ordered_by_rank(x, *args, **kwargs):
    grid = kwargs.get('comm', create_grid_comm())
    for p in range(grid.Get_size()):
        grid.Barrier()
        if p == grid.rank:
            print(x, *args, **kwargs)

def from_numpy(arr: np.ndarray):
    n, m = arr.shape
    if rank == 0: 
        print("arr=", arr.shape)
    block_size = ceil(n / grid.dims[0]), ceil(m / grid.dims[1])

    print_ordered_by_rank(coords, "block_size=", block_size)



# b1 = Block(np.array([4, 5, 6]), custom_attr=2)
# b2 = Block(np.array([1, 2, 3]) + b1)    

# print(b2.func())

n, m = 1, 5

arr = np.arange(1, n*m+1).reshape((n, m))
from_numpy(arr)