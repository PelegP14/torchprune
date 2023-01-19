import numpy as np
from .testEM2 import computeCost, computeDistanceToSubspace, EMLikeAlg, EMLikeAlgWithJOpt, computeSuboptimalSubspace


def getProjectiveClustering(
        P, j, k, verbose=True, steps=300, NUM_INIT_FOR_EM=10
):
    n = P.shape[0]
    w = np.ones(n)  # unit weights
    # steps = 15 # number of EM steps

    if not verbose:
        import os
        import sys

        sys.stdout = open(os.devnull, "w")  # disable printing
    flats, runtime = EMLikeAlg(
        P, w, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
    )
    if not verbose:
        sys.stdout = sys.__stdout__  # re-enable printing

    return flats


def getProjectiveClusteringWithJOPT(
        P, j, k, verbose=True, steps=300, NUM_INIT_FOR_EM=10
):
    n = P.shape[0]
    w = np.ones(n)  # unit weights
    # steps = 15 # number of EM steps

    if not verbose:
        import os
        import sys

        sys.stdout = open(os.devnull, "w")  # disable printing
    flats, runtime = EMLikeAlg(
        P, w, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
    )
    d = P.shape[1]
    max_rank = min(d, n // k)
    new_flats = np.zeros((k, max_rank, d))
    partition = list(_partitionToClosestFlat_new(P, flats))
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [P[idx, :] for idx in idx_list]
    size_per_cluster = np.array([len(idx_list[i]) for i in range(k)])
    j_list = np.zeros((k,), dtype=np.int)
    for size in set(size_per_cluster):
        if size < j:
            continue
        A_list_size = []
        for idx in np.where(size_per_cluster == size)[0]:
            A_list_size.append(A_list[idx])
        k_size = len(A_list_size)
        j_list_size = calculate_j_s(A_list_size, k_size, j, can_improve_max)
        j_list[np.where(size_per_cluster == size)[0]] = j_list_size
    for i in range(k):
        new_flats[i, :, :], _ = computeSuboptimalSubspace(
            A_list[i], w[idx_list[i]], j_list[i], padding=new_flats.shape[1]
        )
    if not verbose:
        sys.stdout = sys.__stdout__  # re-enable printing

    return new_flats


def getJOpt(
        P, j, k, verbose=True, steps=300, NUM_INIT_FOR_EM=10
):
    n = P.shape[0]
    w = np.ones(n)  # unit weights
    # steps = 15 # number of EM steps

    if not verbose:
        import os
        import sys

        sys.stdout = open(os.devnull, "w")  # disable printing
    flats, j_s, partition, runtime = EMLikeAlgWithJOpt(
        P, w, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM
    )
    if not verbose:
        sys.stdout = sys.__stdout__  # re-enable printing

    return flats, j_s, partition


def getCost(P, flats):
    n = P.shape[0]
    w = np.ones(n)  # unit weights

    cost = computeCost(P, w, flats)[0]
    return cost


def _partitionToClosestFlat_new(A, flats):
    dists = np.empty((A.shape[0], flats.shape[0]))
    for l in range(flats.shape[0]):
        _, dists[:, l] = computeCost(A, np.ones(A.shape[0]), flats[l, :, :])

    cluster_indices = np.argmin(
        dists, 1
    )  # determine for each point, the closest flat to it
    return cluster_indices


def calculate_j_s(A_list, k, j, can_improve):
    singular_array = get_singular_values(A_list)
    j_list = [j] * k
    idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array, j_list)
    while idx_to_increase != idx_to_decrease and improvement_is_possible:
        print("improving j")
        j_list[idx_to_increase] += 1
        j_list[idx_to_decrease] -= 1
        idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array, j_list)
    return j_list


def get_singular_values(A_list):
    k = len(A_list)
    max_j = min(A_list[0].shape[0], A_list[0].shape[1])
    singular_array = np.zeros((k, max_j + 1))
    for i, A_i in enumerate(A_list):
        singular_array[i, :max_j] = np.linalg.svd(A_i, compute_uv=False)
    return singular_array


def can_improve_max(singular_array, j_list):
    current_sv = np.array([singular_array[i, j_list[i]] for i in range(len(j_list))])
    lower_sv = np.array([singular_array[i, j_list[i] - 1] if j_list[i] > 0 else np.inf for i in range(len(j_list))])
    idx_to_increase = np.argmax(current_sv)
    idx_to_decrease = np.argmin(lower_sv)
    return idx_to_increase, idx_to_decrease, current_sv[idx_to_increase] > lower_sv[idx_to_decrease]


def can_improve_mean(singular_array, j_list):
    increase_diff = np.array(
        [singular_array[i, j_list[i]] - singular_array[i, j_list[i] + 1] if singular_array[i, j_list[i]] != 0 else 0 for
         i in range(len(j_list))])
    decrease_diff = np.array(
        [singular_array[i, j_list[i] - 1] - singular_array[i, j_list[i]] if j_list[i] > 0 else np.inf for
         i in range(len(j_list))])
    idx_to_increase = np.argmax(increase_diff)
    idx_to_decrease = np.argmin(decrease_diff)
    return idx_to_increase, idx_to_decrease, increase_diff[idx_to_increase] > decrease_diff[idx_to_decrease]


def can_improve_frobenius(singular_array, j_list):
    increase_diff = np.array(
        [singular_array[i, j_list[i]] ** 2 - singular_array[i, j_list[i] + 1] ** 2 if singular_array[
                                                                                          i, j_list[i]] != 0 else 0 for
         i in range(len(j_list))])
    decrease_diff = np.array(
        [singular_array[i, j_list[i] - 1] ** 2 - singular_array[i, j_list[i]] ** 2 if j_list[i] > 0 else np.inf for
         i in range(len(j_list))])
    idx_to_increase = np.argmax(increase_diff)
    idx_to_decrease = np.argmin(decrease_diff)
    return idx_to_increase, idx_to_decrease, increase_diff[idx_to_increase] > decrease_diff[idx_to_decrease]


def raw_alds_base(A, j, k, can_improve_func):
    listU = []
    listV = []
    n = A.shape[0]
    partition = [i // int(n / k) for i in range(n)]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [A[idx, :] for idx in idx_list]
    j_s = calculate_j_s(A_list, k, j, can_improve_func)
    for z in range(k):
        A_z = A_list[z]
        U_z, V_z = lowRank(A_z, j_s[z])
        listU.append(U_z)
        listV.append(V_z)

    return partition, listU, listV


def raw_messi_base(A, j, k, can_improve_func, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    # returns (k, j, d) tensor to represent the k flats
    # by j basis vectors in R^d
    flats = getProjectiveClustering(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    # partition[i] == z means row i of A belongs to flat z
    # where 0 <= i < n and 0 <= z < k
    partition = list(_partitionToClosestFlat_new(A, flats))

    n = A.shape[0]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [A[idx, :] for idx in idx_list]
    size_per_cluster = np.array([len(idx_list[i]) for i in range(k)])
    j_list = np.zeros((k,), dtype=np.int)
    for size in set(size_per_cluster):
        if size < j:
            continue
        A_list_size = []
        for idx in np.where(size_per_cluster == size)[0]:
            A_list_size.append(A_list[idx])
        k_size = len(A_list_size)
        j_list_size = calculate_j_s(A_list_size, k_size, j, can_improve_func)
        j_list[np.where(size_per_cluster == size)[0]] = j_list_size
    listU = []
    listV = []

    for z in range(k):
        A_z = A_list[z]
        U_z, V_z = lowRank(A_z, j_list[z])
        listU.append(U_z)
        listV.append(V_z)
    return partition, listU, listV


def raw_j_opt(A, j, k, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    flats, j_s, partition = getJOpt(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    n = A.shape[0]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [A[idx, :] for idx in idx_list]

    listU = []
    listV = []

    for z in range(k):
        A_z = A_list[z]
        U_z, V_z = lowRank(A_z, j_s[z])
        listU.append(U_z)
        listV.append(V_z)
    return partition, listU, listV

def raw_j_opt_for_clustering(A, j, k, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    flats, j_s, partition = getJOpt(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    n = A.shape[0]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [A[idx, :] for idx in idx_list]

    listU = []
    listV = []

    for z in range(k):
        A_z = A_list[z]
        U_z, V_z = lowRank(A_z, j_s[z])
        listU.append(U_z)
        listV.append(V_z)
    return partition, listU, listV, idx_list

def calc_j_opt_error(A, j, k, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    flats, j_s, partition = getJOpt(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    n = A.shape[0]
    idx_list = [[row for row in range(n) if partition[row] == z] for z in range(k)]
    A_list = [A[idx, :] for idx in idx_list]

    max_sing_value = 0
    for z in range(k):
        A_z = A_list[z]
        sing_values = np.linalg.svd(A_z,compute_uv=False)
        max_sing_value = max(max_sing_value,sing_values[j_s[z]] if sing_values.size > j_s[z] else 0)

    return max_sing_value

def base_messi_error(A, j, k, partition, order):
    listU = []
    listV = []
    n = A.shape[0]
    for z in range(k):
        indices_z = [row for row in range(n) if partition[row] == z]
        if len(indices_z) == 0:
            continue
        A_z = A[indices_z, :]
        U_z, V_z = lowRank(A_z, j)
        listU.append(U_z)
        listV.append(V_z)

    u_stitched, v_stitched = stitch(partition, listU, listV)

    error = np.linalg.norm(A.T - (v_stitched.T @ u_stitched.T), ord=order) \
            / np.linalg.norm(A.T, ord=order)
    return error


def raw_messi(A, j, k, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    # returns (k, j, d) tensor to represent the k flats
    # by j basis vectors in R^d
    flats = getProjectiveClustering(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    # partition[i] == z means row i of A belongs to flat z
    # where 0 <= i < n and 0 <= z < k
    partition = list(_partitionToClosestFlat_new(A, flats))

    listU = []
    listV = []
    n = A.shape[0]
    for z in range(k):
        indices_z = [row for row in range(n) if partition[row] == z]
        A_z = A[indices_z, :]
        U_z, V_z = lowRank(A_z, j)
        listU.append(U_z)
        listV.append(V_z)

    return partition, listU, listV


def raw_best_pick(A, j, k, steps=300, NUM_INIT_FOR_EM=10, verbose=True):
    # First run JOPT
    partition_j, listU_j, listV_j = raw_j_opt(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    partition_pc, listU_pc, listV_pc = raw_messi(
        A, j, k, steps=steps, NUM_INIT_FOR_EM=NUM_INIT_FOR_EM, verbose=verbose
    )

    u_j_stitched, v_j_stitched = stitch(partition_j, listU_j, listV_j)

    u_pc_stitched, v_pc_stitched = stitch(partition_pc, listU_pc, listV_pc)

    A_jopt = u_j_stitched @ v_j_stitched
    A_pc = u_pc_stitched @ v_pc_stitched
    print("costs calculated {} for j, {} for pc".format(cost_calc(A,listU_j,listV_j,partition_j),cost_calc(A,listU_pc,listV_pc,partition_pc)))
    print("errors calculated {} for j, {} for pc".format(np.linalg.norm(A - A_jopt, ord='fro'), np.linalg.norm(A - A_pc, ord='fro')))

    if np.linalg.norm(A - A_jopt, ord='fro') > np.linalg.norm(A - A_pc, ord='fro'):
        print("picked pc")
        partition = partition_pc
        listU = listU_pc
        listV = listV_pc
    else:
        print("picked j opt")
        partition = partition_j
        listU = listU_j
        listV = listV_j

    return partition, listU, listV

def cost_calc(A,listU,listV,partition):
    n, d = A.shape
    k = len(listU)
    max_rank = max((u.shape[1] for u in listU))
    Vs = np.zeros((k,max_rank,d))
    for i,(u,v) in enumerate(zip(listU,listV)):
        subspace, _ = computeSuboptimalSubspace(u @ v, np.ones(u.shape[0]),u.shape[1])
        Vs[i,:subspace.shape[0],:subspace.shape[1]] = subspace

    return computeCost(A,np.ones(n),Vs)[0]
def alds_bound(weight, k_split, j):
    def _compute_sv_for_weight(weight, k_split):
        """Compute SVD for one layer."""

        # compute rank
        rank_k = min(weight.shape[0], weight.shape[1] // k_split)
        # pre-allocate singular values ...
        singular_values = np.zeros((k_split, rank_k + 1))
        # compute singular values for each part of the decomposed weight
        for idx_k, w_per_g_k in enumerate(np.split(weight, k_split, axis=1)):
            singular_values[idx_k, :rank_k] = np.linalg.svd(w_per_g_k, compute_uv=False)

        # note has shape [k_split x rank_per_k]
        return singular_values

    # grab all singular values for decomposition
    # has shape [k_split x rank_per_k]
    singular_values = _compute_sv_for_weight(weight, k_split)

    # compute operator norm for current operator
    # has shape []
    op_norm = np.linalg.norm(weight, ord=2)

    # compute operator norm of "residual operator" W - What
    # --> corresponds to biggest singular values not included.
    # What consists of multiple k_splits.
    # see corresponding paper for op_norm derivation
    # We take max over k-splits here.
    op_norm_residual = singular_values.max(axis=0)

    # resulting relative error for each layer and each possible rank_j!
    # we take the j-th rel error that defines our bound
    rel_error = op_norm_residual / op_norm

    return rel_error[j]


"""
Computes the low-rank approximation of a matrix.

This computes a (possibly non-unique) minimizer M' of
the Frobenius norm of (M - M') for input M where
M' has rank at most r.  Uniqueness is determined by uniqueness
of the greatest r singular values of M.  The computation
is by truncating the SVD of M.

@param M
    n x d matrix, the matrix to be approximated
@param r
    integer >= 1, the rank of the approximation
@return
    U,V where U.shape==(n,r) and V.shape==(r,d)
    defining an r-rank approximation of M
"""


def lowRank(M, r):
    if M.shape[0] < r:
        full_M = np.zeros((r, M.shape[1]))
        full_M[:M.shape[0], :] = M
        M = full_M
    U, D, Vt = np.linalg.svd(M)

    # truncate to:
    #   left-most r columns of U
    #   first r values of D
    #   top-most r rows of Vt
    U_trunc = U[:, :r]
    D_trunc = np.diag(D[:r])  # also convert from vector to matrix
    Vt_trunc = Vt[:r, :]

    # arbitrary choice to combine D with either side
    return U_trunc.dot(D_trunc), Vt_trunc


"""
Combines k-matrices into one large block matrix (not a true block matrix unless
you permute the rows correctly).
"""


def stitch(partition, listU, listV):
    U = _stitchU(partition, listU)
    V = _stitchV(listV)
    return U, V


"""
Stitches together U_1, ..., U_k into a single
n x r matrix U.  The column space, of dimension r,
is the direct-sum space of the column spaces of
U_1, ..., U_k.

@param partition
    list of length n containing values in [0, ..., k-1].
    partition[i] is the component of the partition that
    row i of the input matrix belongs to.
@param listU
    list of the left-hand-side matrices of the low-rank
    approximation of the submatrices built by partitioning
    the rows of the input matrix
@return U
    a global reconstruction that can be used to as the
    left-hand-side matrix of the decomposition of the input
    matrix
"""


def _stitchU(partition, listU):
    # n: rows of original matrix == size of list partitions
    n = len(partition)

    # r: middle space between R^n and R^d, of dimension r where
    # r is the sum of all column-spaces of each U_z in listU
    r = 0
    for U_z in listU:
        r += U_z.shape[1]

    U = np.zeros((n, r))  # final U is mostly zeros

    # counter[z] stores current row of listU[z]
    counters = [0] * len(listU)#length k

    for row in range(n):
        index_component = partition[row]
        index_row = counters[index_component]  # row of U_z to use
        counters[index_component] += 1

        # insert row of U_z in column-space of R^r starting at col_start
        col_start = 0
        for u in listU[:index_component]:
            col_start += u.shape[1]
        component = listU[index_component]
        col_end = col_start + component.shape[1]
        U[row, col_start:col_end] = component[index_row]
    return U


"""
Takes the k matrices V_1, ..., V_k each of dimension j x d
from the k low-rank approximations, and stitches them together
into one large jk x d matrix.

@param listV
    [V_1, ..., V_k]
@return:
    jk x d matrix V that can be used as the right-hand-side
    matrix of the decomposition of the input matrix A ~ U.dot(V)
"""


def _stitchV(listV):
    return np.concatenate(listV)


def main():
    arr = np.random.normal(np.arange(50), 1, (50, 50)).T
    k = 5
    j = 6
    raw_alds_base(arr, j, k, can_improve_max)


if __name__ == "__main__":
    main()
