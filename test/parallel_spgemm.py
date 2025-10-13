# parallel_spgemm.py (from Charlie)

# Sparse General Matrix-Matrix Multiplication (SpGEMM)

from mpi4py import MPI
from scipy.sparse import csr_matrix
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --------------------------------------------
# 1. Assume 2D process grid
# --------------------------------------------
p = int(np.sqrt(size))
if p * p != size:
    raise ValueError("Number of processes must form a square grid")

row = rank // p
col = rank % p

# --------------------------------------------
# 2. Example sparse matrices (local blocks)
# --------------------------------------------
np.random.seed(rank)
n_local = 4  # local block dimension

# Random sparse blocks (~30% fill)
A_on = csr_matrix(np.random.rand(n_local, n_local) * (np.random.rand(n_local, n_local) > 0.7))
B_on = csr_matrix(np.random.rand(n_local, n_local) * (np.random.rand(n_local, n_local) > 0.7))

# Make "off" copies (for demonstration)
A_off = A_on.copy()
B_off = B_on.copy()

# --------------------------------------------
# 3. Neighbor ranks (Cannon-style grid)
# --------------------------------------------
left  = row * p + (col - 1) % p
right = row * p + (col + 1) % p
up    = ((row - 1) % p) * p + col
down  = ((row + 1) % p) * p + col

# --------------------------------------------
# 4. Non-blocking communication helpers
# --------------------------------------------
def isend_csr(mat, dest, tag_base):
    """Send full CSR matrix asynchronously."""
    reqs = []
    for tag, arr in enumerate([mat.data, mat.indices, mat.indptr]):
        reqs.append(comm.Isend(arr, dest=dest, tag=tag_base + tag))
    return reqs

def irecv_csr(source, tag_base):
    """Receive full CSR matrix asynchronously."""
    # Step 1: probe sizes
    sizes = []
    for tag in range(3):
        status = MPI.Status()
        comm.Probe(source=source, tag=tag_base + tag, status=status)
        count = status.Get_elements(MPI.DOUBLE if tag == 0 else MPI.INT)
        sizes.append(count)

    data = np.empty(sizes[0], dtype=np.float64)
    indices = np.empty(sizes[1], dtype=np.int32)
    indptr = np.empty(sizes[2], dtype=np.int32)

    reqs = [
        comm.Irecv(data, source=source, tag=tag_base),
        comm.Irecv(indices, source=source, tag=tag_base + 1),
        comm.Irecv(indptr, source=source, tag=tag_base + 2),
    ]
    return data, indices, indptr, reqs

# --------------------------------------------
# 5. Start non-blocking sends/receives
# --------------------------------------------
send_reqs = []
recv_reqs = []

# Send A_off left, receive A_recv from right
send_reqs += isend_csr(A_off, left, tag_base=100)
A_recv_data, A_recv_idx, A_recv_ptr, tmp_reqs = irecv_csr(right, tag_base=100)
recv_reqs += tmp_reqs

# Send B_off up, receive B_recv from down
send_reqs += isend_csr(B_off, up, tag_base=200)
B_recv_data, B_recv_idx, B_recv_ptr, tmp_reqs = irecv_csr(down, tag_base=200)
recv_reqs += tmp_reqs

# --------------------------------------------
# 6. Local SpGEMM (overlap with comm)
# --------------------------------------------
C_on  = A_on.dot(B_on)
C_off = A_on.dot(B_off)

# --------------------------------------------
# 7. Wait for non-blocking communication
# --------------------------------------------
MPI.Request.Waitall(recv_reqs)

# Rebuild CSR matrices from received buffers
A_recv = csr_matrix((A_recv_data, A_recv_idx, A_recv_ptr), shape=(n_local, n_local))
B_recv = csr_matrix((B_recv_data, B_recv_idx, B_recv_ptr), shape=(n_local, n_local))

# --------------------------------------------
# 8. Perform non-local SpGEMM
# --------------------------------------------
C_on  += A_off.dot(B_recv)
C_off += A_recv.dot(B_off)

MPI.Request.Waitall(send_reqs)  # clean up sends

# --------------------------------------------
# 9. Print partial results
# --------------------------------------------
print(f"\nRank {rank} at grid ({row},{col}):")
print("A_on:\n", A_on.toarray())
print("B_on:\n", B_on.toarray())
print("C_on:\n", C_on.toarray())
print("C_off:\n", C_off.toarray())
