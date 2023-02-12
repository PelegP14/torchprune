import numpy as np
from scipy.linalg import null_space
from sklearn.utils.extmath import cartesian
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
import copy
import time
import random

# import getdim # file in directory to load BERT
LAMBDA = 1
Z = 2
# NUM_INIT_FOR_EM = 1
STEPS = 20
M_ESTIMATOR_FUNCS = {
    "lp": (lambda x: np.abs(x) ** Z / Z),
    "huber": (
        lambda x: x ** 2 / 2
        if np.abs(x) <= LAMBDA
        else LAMBDA * (np.abs(x) - LAMBDA / 2)
    ),
    "cauchy": (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
    "geman_McClure": (lambda x: x ** 2 / (2 * (1 + x ** 2))),
    "welsch": (
        lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-(x ** 2) / LAMBDA ** 2))
    ),
    "tukey": (
        lambda x: LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3)
        if np.abs(x) <= LAMBDA
        else LAMBDA ** 2 / 6
    ),
}
global OBJECTIVE_LOSS
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS["lp"]


def computeDistanceToSubspace(point, X):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    if point.ndim > 1:
        return np.linalg.norm(np.dot(point, null_space(X)), ord=2, axis=1)
    return np.linalg.norm(np.dot(point, null_space(X)))


def computeDistanceToSubspaceviaNullSpace(point, null_space):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    if point.ndim > 1:
        return np.linalg.norm(np.dot(point, null_space), ord=2, axis=1)
    return np.linalg.norm(np.dot(point, null_space))


def computeCost(P, w, X, show_indices=False):
    """
    This function represents our cost function which is a generalization of k-means where the means are now J-flats.

    :param P: A weighed set, namely, a PointSet object.
    :param X: A numpy matrix of J x d which defines the basis of the subspace which we would like to compute the
              distance to.
    :param v: A numpy array of d entries which defines the translation of the J-dimensional subspace spanned by the
              rows of X.
    :return: The sum of weighted distances of each point to the affine J dimensional flat which is denoted by (X,v)
    """
    global OBJECTIVE_LOSS
    if X.ndim == 2:
        dist_per_point = OBJECTIVE_LOSS(
            computeDistanceToSubspaceviaNullSpace(P, null_space(X))
        )
        cost_per_point = np.multiply(w, dist_per_point)
    else:
        temp_cost_per_point = np.empty((P.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            temp_cost_per_point[:, i] = np.multiply(
                w,
                OBJECTIVE_LOSS(
                    computeDistanceToSubspaceviaNullSpace(
                        P, null_space(X[i, :, :])
                    )
                ),
            )

        cost_per_point = np.min(temp_cost_per_point, 1)
        indices = np.argmin(temp_cost_per_point, 1)
    if not show_indices:
        return np.sum(cost_per_point), cost_per_point
    else:
        return np.sum(cost_per_point), cost_per_point, indices


def computeSuboptimalSubspace(P, w, J, padding=0):
    """
    This function computes a suboptimal subspace in case of having the generalized K-means objective function.

    :param P: A weighted set, namely, an object of PointSet.
    :return: A tuple of a basis of J dimensional spanning subspace, namely, X and a translation vector denoted by v.
    """

    start_time = time.time()

    _, _, V = np.linalg.svd(
        P, full_matrices=True
    )  # computing the spanning subspace
    if padding == 0:
        return V[:J, :], time.time() - start_time
    ret_mat = np.zeros((padding, P.shape[1]))
    if J!=0:
        ret_mat[:J, :] = V[:J, :]
    return ret_mat, time.time() - start_time


def EMLikeAlg(P, w, j, k, steps, NUM_INIT_FOR_EM=10):
    """
    The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with respect to a user chosen

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of each flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An integer denoting the max number of EM steps
    :return: A list of k j-flats which locally optimize the cost function
    """

    start_time = time.time()
    #######################################
    #### TODO: REMEMBER TO CHANGE #########
    ########################################
    np.random.seed(random.seed())
    # np.random.seed(42)
    n, d = P.shape
    min_Vs = None
    optimal_cost = np.inf
    # print ("started")
    for iter in range(-1, 2 * NUM_INIT_FOR_EM):  # run EM for 10 random initializations
        Vs = np.empty((k, max(j,1), d))
        idxs = np.arange(n)
        if iter < NUM_INIT_FOR_EM:
            if iter > -1:
                np.random.shuffle(idxs)
            idxs = np.array_split(idxs, k)  # ;print(idxs)
        else:
            split_idxs = [0]
            for kidx in range(k - 1):
                split = np.random.randint(1, n)
                while split in split_idxs:
                    split = np.random.randint(1, n)
                split_idxs.append(split)
            split_idxs.sort()
            split_idxs.append(n)
            new_idx = [idxs[split_idxs[i]:split_idxs[i + 1]] for i in range(k)]
            idxs = new_idx
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], _ = computeSuboptimalSubspace(
                P[idxs[i], :], w[idxs[i]], j, padding=Vs.shape[1]
            )

        last_error = computeCost(P,w,Vs)[0]
        improved = True
        count = 0
        while improved and count<steps:
            # find best k j-flats which can attain local optimum
            count += 1
            dists = np.empty(
                (n, k)
            )  # distance of point to each one of the k j-flats
            for l in range(k):
                _, dists[:, l] = computeCost(P, w, Vs[l, :, :])

            cluster_indices = np.argmin(
                dists, 1
            )  # determine for each point, the closest flat to it
            unique_idxs = np.unique(
                cluster_indices
            )  # attain the number of clusters

            for (
                    idx
            ) in (
                    unique_idxs
            ):  # recompute better flats with respect to the updated cluster matching
                Vs[idx, :, :], _ = computeSuboptimalSubspace(
                    P[np.where(cluster_indices == idx)[0], :],
                    w[np.where(cluster_indices == idx)[0]],
                    j,
                    padding=Vs.shape[1]
                )

            current_cost = computeCost(P, w, Vs)[0]
            if current_cost < last_error:
                last_error = current_cost
            else:
                improved = False

        print("took {} iterations to converge".format(count))
        current_cost = computeCost(P, w, Vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            optimal_cost = current_cost
        print(
            "finished iteration number {} with cost {}".format(
                iter, optimal_cost
            )
        )
    return min_Vs, time.time() - start_time

def EMLikeAlgGivenInit(P, w, j, k, partition, steps):
    """
    The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with respect to a user chosen

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of each flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An integer denoting the max number of EM steps
    :return: A list of k j-flats which locally optimize the cost function
    """

    start_time = time.time()
    #######################################
    #### TODO: REMEMBER TO CHANGE #########
    ########################################
    np.random.seed(random.seed())
    # np.random.seed(42)
    n, d = P.shape
    min_Vs = None
    optimal_cost = np.inf
    # print ("started")
    for iter in range(-1, 1):  # run EM for 10 random initializations
        Vs = np.empty((k, max(j,1), d))
        idxs = np.arange(n)
        if iter == -1:
            idxs = np.array_split(idxs, k)  # ;print(idxs)
        else:
            idxs = [np.where(np.array(partition) == i)[0] for i in range(k)]
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], _ = computeSuboptimalSubspace(
                P[idxs[i], :], w[idxs[i]], j, padding=Vs.shape[1]
            )

        last_error = computeCost(P,w,Vs)[0]
        improved = True
        count = 0
        while improved and count<steps:
            # find best k j-flats which can attain local optimum
            count += 1
            dists = np.empty(
                (n, k)
            )  # distance of point to each one of the k j-flats
            for l in range(k):
                _, dists[:, l] = computeCost(P, w, Vs[l, :, :])

            cluster_indices = np.argmin(
                dists, 1
            )  # determine for each point, the closest flat to it
            unique_idxs = np.unique(
                cluster_indices
            )  # attain the number of clusters

            for (
                    idx
            ) in (
                    unique_idxs
            ):  # recompute better flats with respect to the updated cluster matching
                Vs[idx, :, :], _ = computeSuboptimalSubspace(
                    P[np.where(cluster_indices == idx)[0], :],
                    w[np.where(cluster_indices == idx)[0]],
                    j,
                    padding=Vs.shape[1]
                )

            current_cost = computeCost(P, w, Vs)[0]
            if current_cost < last_error:
                last_error = current_cost
            else:
                improved = False

        print("took {} iterations to converge".format(count))
        current_cost = computeCost(P, w, Vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            optimal_cost = current_cost
        print(
            "finished iteration number {} with cost {}".format(
                iter, optimal_cost
            )
        )
    return min_Vs, time.time() - start_time


def EMLikeAlgWithJOpt(P, w, j, k, steps, NUM_INIT_FOR_EM=10):
    """
    The function at hand, is an EM-like algorithm which is heuristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with J optimization

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of each flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An integer denoting the max number of EM steps
    :return: A list of k j-flats which locally optimize the cost function
    """

    start_time = time.time()
    #######################################
    #### TODO: REMEMBER TO CHANGE #########
    ########################################
    np.random.seed(random.seed())
    # np.random.seed(42)
    n, d = P.shape
    max_rank = min(d, n // k)
    min_Vs = None
    min_j_s = None
    best_cluster_indices = None
    optimal_cost = np.inf
    # print ("started")
    for iter in range(-1, NUM_INIT_FOR_EM):  # run EM for 10 random initializations
        Vs = np.zeros((k, max_rank, d))
        idxs = np.arange(n)
        if iter > -1:
            np.random.shuffle(idxs)
        idxs = np.array_split(idxs, k)  # ;print(idxs)
        last_error = 0
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], _ = computeSuboptimalSubspace(
                P[idxs[i], :], w[idxs[i]], j, padding=max_rank
            )
            last_error += computeCost(
                P[idxs[i], :],
                w[idxs[i]],
                Vs[i]
            )[0]
        cluster_indices = np.repeat(range(k),n//k)
        j_s = np.ones((k,),dtype=int) * j
        # for i in range(
        #         steps
        # ):
        improved = True
        count = 0
        while improved and count<steps:
            count += 1
            # find the best clusters with the same size and varying j
            cluster_indices = find_constrained_clusters(P, Vs, j_s, w)

            idx_list = [[row for row in range(n) if cluster_indices[row] == z] for z in range(k)]
            P_list = [P[idx, :] for idx in idx_list]
            j_s = update_j_s(j_s, P_list, can_improve_max)

            unique_idxs = np.unique(
                cluster_indices
            )  # attain the number of clusters
            current_cost = 0
            for (
                    idx
            ) in (
                    unique_idxs
            ):  # recompute better flats with respect to the updated cluster matching
                Vs[idx, :, :], _ = computeSuboptimalSubspace(
                    P[np.where(cluster_indices == idx)[0], :],
                    w[np.where(cluster_indices == idx)[0]],
                    j_s[idx],
                    padding=max_rank
                )
                current_cost += computeCost(
                    P[np.where(cluster_indices == idx)[0], :],
                    w[np.where(cluster_indices == idx)[0]],
                    Vs[idx]
                )[0]
            if current_cost < last_error:
                last_error = current_cost
            else:
                improved = False
        print("took {} iterations to converge".format(count))
        current_cost = computeCost(P, w, Vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            min_j_s = copy.deepcopy(j_s)
            best_cluster_indices = copy.deepcopy(cluster_indices)
            optimal_cost = current_cost
        print(
            "finished iteration number {} with cost {}".format(
                iter, optimal_cost
            )
        )
    return min_Vs, min_j_s, best_cluster_indices, time.time() - start_time


def find_constrained_clusters(P, Vs, j_s, w):
    n = P.shape[0]
    k = j_s.shape[0]
    min_cluster_size = n // k
    dists = np.empty(
        (n, k)
    )  # distance of point to each one of the k j-flats
    for l in range(k):
        _, dists[:, l] = computeCost(P, w, Vs[l, :, :])

    edges, costs, capacities, supplies = create_graph(P, Vs, dists, min_cluster_size)
    cluster_indicies = solve_min_cost_flow(edges, costs, capacities, supplies, n, k)

    return cluster_indicies


def create_graph(P, Vs, dists, min_cluster_size):
    n = P.shape[0]
    k = Vs.shape[0]

    p_idx = np.arange(n)
    f_idx = np.arange(n, n + k)
    a_idx = n + k

    p_f_edges = cartesian([p_idx, f_idx])
    f_a_edges = cartesian([f_idx, [a_idx]])

    edges = np.concatenate([p_f_edges, f_a_edges])

    p_f_costs = dists.flatten()
    costs = np.concatenate([p_f_costs, np.zeros((edges.shape[0] - p_f_costs.shape[0],))])

    capacities = np.concatenate([np.ones(p_f_costs.shape[0]), n * np.ones(k)])

    supplies_p = np.ones(n)
    supplies_f = -1 * min_cluster_size * np.ones(k)
    supplies_a = -1 * (n - min_cluster_size * k)
    supplies = np.concatenate([supplies_p, supplies_f, [supplies_a]])

    edges = edges.astype('int32')
    costs = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give extra precision
    capacities = capacities.astype('int32')
    supplies = supplies.astype('int32')

    return edges, costs, capacities, supplies


def solve_min_cost_flow(edges, costs, capacities, supplies, n, k):
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    for count, supply in enumerate(supplies):
        min_cost_flow.set_node_supply(count, supply)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_M = np.array([min_cost_flow.flow(i) for i in range(n * k)]).reshape(n, k).astype('int32')

    labels = labels_M.argmax(axis=1)
    return labels


def update_j_s(j_s, A_list, can_improve):
    singular_array = get_singular_values(A_list)
    idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array, j_s.tolist())
    while idx_to_increase != idx_to_decrease and improvement_is_possible:
        j_s[idx_to_increase] += 1
        j_s[idx_to_decrease] -= 1
        idx_to_increase, idx_to_decrease, improvement_is_possible = can_improve(singular_array, j_s.tolist())
    return j_s


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

"""

"""


def main():
    pass


if __name__ == "__main__":
    main()
