from my_package import print_matrix, time_it, check_matmat

####################################################################################

from mpi4py import MPI
import numpy as np
from numba import njit      # nopython mode (if possible)



## Helper Method to get process from row, col
def get_proc(row, col, sq_procs):
    return row*sq_procs + col

## Helper Method for local matrix-matrix multiply
@njit  # compile into machine code (no tqdm)
def matmat(n, A, B, C):
    for i in range(n):    
        for j in range(n):
            val = A[i*n+j]
            for k in range(n):
                C[i*n+k] += val * B[j*n+k]




# n = 1024
# size = n*n

# A = np.ndarray(size)
# B = np.ndarray(size)
# C = np.ndarray(size)


# for i in range(n):
#     for j in range(n):
#         A[i*n+j] = (i + j)
#         B[i*n+j] = (i + j)


# # start = time.perf_counter()
# # matmat(n, A, B, C, no_bar=False)
# # time = time.perf_counter() - start
# # print(f"Time (Pure Python) n={n}: {time:.6f}")

# t_min, t_avg = time_it(matmat, n, A, B, C, True, warmups=10)
# print(f"Loop, Size: {n} x {n}, Minimum Time: {t_min:.4f}, Average Time: {t_avg:.4f}")

# # print_matrix("A", n, n, A)
# # print_matrix("B", n, n, B)
# # print_matrix("C", n, n, C)


# A2 = np.reshape(A,(n,n))
# B2 = np.reshape(B,(n,n))
# t_min, t_avg = time_it(np.matmul, A2, B2, warmups=10)
# print(f"Numpy, Size: {n} x {n}, Minimum Time: {t_min:.4f}, Average Time: {t_avg:.4f}")

# C2 = A2 @ B2

# # print_matrix("A2 @ B2", n, n, C2.flatten())

# if not np.allclose(C, (A2 @ B2).flatten()):
#     print("Error in matmat")
# else:
#     print("matmat gave the correct answer")

# sys.exit()



# A : Local portion of A (n*n)
# B : Local portion of B (n*n)
# C : Local portion (n*n), where you will hold C after matrix-matrix mult 
# Steps : 
# 1. Shift A 'rank_row' columns
# 2. Shift B 'rank_col' rows
# 3. All pairs of A and B on a single process should be multiplied
# 4. Then, send submatrix of A to neighboring process (rowwise)
# 5. and submatrix of B to neighboring process (columnwise)
#
# Return 0 at end (python interface, checks that return in not null)
def matmat_cannon(A, B, C, n, sq_num_procs, rank_row, rank_col):
    
    Comm = MPI.COMM_WORLD

    num_procs = Comm.Get_size()
    rank = Comm.Get_rank()


    size = n*n
    tag_a = 1234
    tag_b = 4321



    for i in range(size):
        C[i] = 0

    # # Set all elements of C to 0
    # C.fill(0)    

    # send_A = np.zeros(size, dtype=np.float64)
    # recv_A = np.zeros(size, dtype=np.float64)

    # send_B = np.zeros(size, dtype=np.float64)
    # recv_B = np.zeros(size, dtype=np.float64)


    # send_A = np.empty_like(A)
    # recv_A = np.empty_like(A)

    # send_B = np.empty_like(B)
    # recv_B = np.empty_like(B)

    send_A = np.ndarray(size)
    recv_A = np.ndarray(size)

    send_B = np.ndarray(size)
    recv_B = np.ndarray(size)

    # Determine Send and Recv Processes for Inital Shift

    # Shift A[i][j] left by rank row: A[i][j-i]

    ############### Test whether mod (-k) works as expected ###############
    # x = -3 % 4
    # print(f"Rank {rank} : -3 % 4 = {x}")
    # mod = ((-3 % 4) + 4) % 4
    # print(f"Rank {rank} : ((-3 % 4) + 4) % 4 = {mod}")
    ######################################################################

    send_proc_A = get_proc(rank_row, (rank_col - rank_row) % sq_num_procs, sq_num_procs)
    recv_proc_A = get_proc(rank_row, (rank_col + rank_row) % sq_num_procs, sq_num_procs)
    
    # Each up by rank column: each B[i][j] sent to B[i-j][j]
    send_proc_B = get_proc((rank_row - rank_col) % sq_num_procs, rank_col, sq_num_procs)
    recv_proc_B = get_proc((rank_row + rank_col) % sq_num_procs, rank_col, sq_num_procs)

    # 1. Perform Initial Shift : 
    # Goal : A[rank_row, rank_row+rank_col]*B[rank_row+rank_col, rank_col]
    for i in range(size):
        send_A[i] = A[i]
        send_B[i] = B[i]

    # print(f"Rank {rank} ({rank_row}, {rank_col}):\n" \
    #       f"send_proc_A: {send_proc_A}, send_proc_A2: {send_proc_A_2}\n" \
    #       f"recv_proc_A: {recv_proc_A}, recv_proc_A2: {recv_proc_A_2}\n" \
    #       f"send_proc_B: {send_proc_B}, send_proc_B2: {send_proc_B_2}\n" \
    #       f"recv_proc_B: {recv_proc_B}, recv_proc_B2: {recv_proc_B_2}\n")
         
    Comm.Sendrecv(sendbuf=send_A, dest=send_proc_A, sendtag=tag_a,
                            recvbuf=recv_A, source=recv_proc_A, recvtag=tag_a)

    Comm.Sendrecv(sendbuf=send_B, dest=send_proc_B, sendtag=tag_b,
                            recvbuf=recv_B, source=recv_proc_B, recvtag=tag_b)

    
    # 2. Perform local matrix-multiplication
    # on submatrices received in initial shift

    matmat(n, recv_A, recv_B, C)

    # 3. Determine new values for send_proc_A/B, recv_proc_A/B
    # Send A to [rank_row, rank_col+1]
    # Send B to [rank_row+1, rank_col]
    # Recv A from [rank_row, rank_col-1]
    # Recv B from [rank_row-1, rank_col]
    # Make sure to check bounds (wrap around if >= sq_num_procs or < 0)
    send_proc_A = get_proc(rank_row, (rank_col + 1) % sq_num_procs, sq_num_procs)
    recv_proc_A = get_proc(rank_row, (rank_col - 1) % sq_num_procs, sq_num_procs)

    send_proc_B = get_proc((rank_row + 1) % sq_num_procs, rank_col, sq_num_procs)
    recv_proc_B = get_proc((rank_row - 1) % sq_num_procs, rank_col, sq_num_procs)




    # 4. For each iteration, send and recv A, B, and perform multiplication
    for i in range(1, sq_num_procs):
        # Or swap in python with: 
        # send_A, recv_A = recv_A, send_A
        # send_B, recv_B = recv_B, send_B

        tmp = send_A
        send_A = recv_A
        recv_A = tmp

        tmp = send_B
        send_B = recv_B
        recv_B = tmp

        # 4a. Send A to send_proc_A
        # 4b. Recv new A from recv_proc_A

        Comm.Sendrecv(send_A, send_proc_A, tag_a, recv_A, recv_proc_A, tag_a)

        # 4c. Send B to send_proc_B
        # 4c. Recv new B from recv_proc_B

        Comm.Sendrecv(send_B, send_proc_B, tag_b, recv_B, recv_proc_B, tag_b)

        # 4e. Local matrix multiplication C += recv_A * recv_B

        # if rank == 0:
        #     if np.all(C == 0):
        #         print("Error: C is all zeros")

        matmat(n, recv_A, recv_B, C)

        # if rank == 0:
        #     print(f"matmat {i} sumC: {sum(C)}")



    return 0

Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
num_procs = Comm.Get_size()

N = 1024
sq_num_procs = int(np.sqrt(num_procs))

if sq_num_procs*sq_num_procs != num_procs:
    if rank == 0:
        print("Number of processes needs to be square")
        Comm.Abort(-1)

rank_row = rank // sq_num_procs
rank_col = rank % sq_num_procs
n = N // sq_num_procs
size = n*n

if n*n*num_procs != N*N:
    if rank == 0:
        print(f"Cannot evenly split {N} rows and cols over {size} processes")
        Comm.Abort(-1)



if rank == 0:
    print(f"num_procs: {num_procs}, sq_num_procs: {sq_num_procs}, n: {n}, n*n (size): {size}")

# A = np.empty(size, dtype=np.float64)
# B = np.empty(size, dtype=np.float64)
# C = np.empty(size, dtype=np.float64)

A = np.ndarray(size)
B = np.ndarray(size)
C = np.ndarray(size)

for i in range(n):
    for j in range(n):
        A[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1
        B[i*n+j] = ((rank_row*n)+i)*N + (rank_col*n)+j+1


# start = MPI.Wtime()
# matmat_cannon(A, B, C, n, sq_num_procs, rank_row, rank_col)
# time = MPI.Wtime() - start

time, _ = time_it(matmat_cannon, A, B, C, n, sq_num_procs, rank_row, rank_col, repeat=5, warmups=2, timer_fn=MPI.Wtime)
sum_C = np.sum(C)

A2 = np.resize(A, (n,n))
B2 = np.resize(B, (n,n))
time2, _ = time_it(np.matmul, A2, B2, repeat=5, warmups=2, timer_fn=MPI.Wtime)
sum_C2 = np.sum(np.matmul(A2, B2).flatten())

print(f"rank {rank} sumC: {sum_C}, sumC2: {sum_C2}")

total_sum_C = np.zeros_like(sum_C)
Comm.Reduce(sum_C, total_sum_C, op=MPI.SUM, root=0)
max_time = np.zeros_like(time)
Comm.Reduce(np.array([time]), max_time, op=MPI.MAX, root=0)

# total_sum_C2 = np.zeros_like(sum_C2)
# Comm.Reduce(sum_C2, total_sum_C2, op=MPI.SUM, root=0)
# max_time2 = np.zeros_like(time2)
# Comm.Reduce(np.array([time2]), max_time2, op=MPI.MAX, root=0)

total_sum_C2 = np.array([sum_C2])
Comm.Reduce(np.array([sum_C2]), total_sum_C2, op=MPI.SUM, root=0)
max_time2 = np.zeros_like(time2)
Comm.Reduce(np.array([time2]), max_time2, op=MPI.MAX, root=0)


if rank == 0:
    print(f"Cannon's Method : sumC {total_sum_C:25f} (Mine), Elapsed Time {max_time:e}")
    print(f"Cannon's Method : sumC 295244544829852483584.000000 (C/C++), Elapsed Time 2.477775e-02")
    print(f"Cannon's Method : sumC {total_sum_C2[0]:25f} (Python), Elapsed Time {max_time2:e}")

