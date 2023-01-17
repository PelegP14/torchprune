import Utils
from helper_functions import Fast_Caratheodory
import numpy as np
from scipy.optimize import linprog
from numpy import linalg as la
from scipy.linalg import null_space
from numpy.linalg import matrix_rank
from sklearn.decomposition import TruncatedSVD
import time
import copy
import sys


######################################################## Caratheodory ##################################################
def computeInitialWeightVector(P, p):
    """
    This function given a point, solves the linear program dot(self.P.P^T, x) = p where x \in [0, \infty)^n,
    and n denotes the number of rows of self.P.P.

    :param p: A numpy array representing a point.
    :return: A numpy array of n non-negative weights with respect to each row of self.P.P
    """
    N = P.shape[0] # number of rows of P

    # # Solve the linear program using scipy
    # ts = time.time()
    Q = P.T
    Q = np.vstack((Q, np.ones((1, N))))
    b = np.hstack((p, 1))
    res = linprog(np.ones((N,)), A_eq=Q, b_eq=b, options={'maxiter': int(1e7), 'tol': 1e-10})
    w = res.x
    assert (np.linalg.norm(np.dot(P.T, w) - p) <= 1e-9, np.linalg.norm(np.dot(P.T, w) - p))
    return w


def attainCaratheodorySet(P, p):
    """
    The function at hand returns a set of at most d+1 indices of rows of P where d denotes the dimension of
    rows of P. It calls the algorithms implemented by Alaa Maalouf, Ibrahim Jubran and Dan Feldman at
    "Fast and Accurate Least-Mean-Squares Solvers".

    :param p: A numpy array denoting a point.
    :return: The indices of points from self.P.P which p is a convex combination of.
    """
    d = P.shape[1]
    u = computeInitialWeightVector(P, p)  # compute initial weight vector
    # print('Sum of weights {}'.format(np.sum(u)))

    if np.count_nonzero(u) > (d + 1):  # if the number of positive weights exceeds d+1
        u = Fast_Caratheodory(P, u.flatten(), False)

    assert(np.linalg.norm(p - np.dot(P.T, u)) <= 1e-9, np.linalg.norm(p - np.dot(P.T, u)))
    return np.where(u != 0)[0]


############################################################ AMVEE #####################################################
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def computeAxesPoints(E, C):
    """
    This function finds the vertices of the self.E (the MVEE of P or the inscribed version of it)

    :return: A numpy matrix containing the vertices of the ellipsoid.
    """
    if not isPD(E):
        E = nearestPD(E)

    # L = np.linalg.cholesky(self.E)  # compute the cholesky decomposition of self.E
    # U, D, V = np.linalg.svd(L, full_matrices=True)  # attain the length of each axis of the ellipsoid and the
    #                                                 # rotation of the ellipsoid
    _, D, V = np.linalg.svd(E, full_matrices=True)
    ellips_points = np.multiply(1.0 / np.sqrt(D[:, np.newaxis]), V.T)  # attain the vertices of the ellipsoid assuming it was
                                                                       # centered at the origin
    return np.vstack((ellips_points + C.flatten(), - ellips_points + C.flatten()))


def volumeApproximation(P):
    """
    This is our implementation of Algorithm 4.1 at the paper "On Khachiyan’s Algorithm for te Computation of Minimum
    Volume Enclosing Ellipsoids" by Michael J. Todd and E. Alper Yıldırım. It serves to compute a set of at most
    2*self.P.d points which will be used for computing an initial ellipsoid.

    :return: A numpy array of 2 * self.P.d indices of points from self.P.P
    """
    basis = None
    basis_points = []
    n, d = P
    if n <= 2 * d:
        # if number of points is less than 2*self.P.d, then return their indices in self.P.P
        return [i for i in range(n)]

    v = np.random.randn(d)  # start with a random vector
    while np.linalg.matrix_rank(basis) < d:  # while rank of basis is less than self.P.d
        if basis is not None:  # if we already have computed basis points
            if basis.shape[1] == d:
                # if this line is reached then it means that there is numerical instability
                print('Numerical Issues!')
                _, _, V = np.linalg.svd(basis[:, :-1], full_matrices=True)
                return list(range(n))
            orth_basis = null_space(basis.T)  # get the orthant of basis
            v = orth_basis[:, 0] if orth_basis.ndim > 1 else orth_basis  # set v to be the first column of basis
        Q = np.dot(P, v.T)  # get the dot product of each row of self.P.P and v
        if len(basis_points) > 0:  # if there are already chosen points, then their dot product is depricated
            Q[basis_points] = np.nan

        p_alpha = np.nanargmax(np.dot(P, v.T))  # get the index of row with largest non nan dot product value
        p_beta = np.nanargmin(np.dot(P, v.T)) # get the index of row with smallest non nan dot product value
        v = np.expand_dims(P[p_beta, :] - P[p_alpha, :], 1)  # let v be the substraction between the
                                                                           # row of the largest dot product and the
                                                                           # point with the smallest dot product
        if basis is None:  # if no basis was computed
            basis = v / np.linalg.norm(v)
        else:  # add v to the basis
            basis = np.hstack((basis, v / np.linalg.norm(v, 2)))

        basis_points.append(p_alpha)  # add the index of the point with largest dot product
        basis_points.append(p_beta) # add the index of the point with smallest dot product

    return basis_points


def computemahalanobisDistance(Q, ellip):
    """
    This function is used for computing the distance between the rows of Q and ellip using the Mahalanobis
    loss function.

    :param ellip: A numpy array representing a p.s.d matrix (an ellipsoid)
    :return: The Mahalanobis distance between each row in self.P.P to ellip.
    """
    s = np.einsum("ij,ij->i", np.dot(Q, ellip), Q)  # compute the distance efficiently

    return s


def computeEllipsoid(P, weights):
    """
    This function computes the ellipsoid which is the MVEE of self.P.
    :param weights: a numpy of array of weights with respest to the rows of self.P.P.

    :return:
            - The MVEE of self.P.P in a p.s.d. matrix form.
            - The center of the MVEE of self.P.P.
    """
    if weights.ndim == 1:  # make sure that the weights are not flattened
        weights = np.expand_dims(weights, 1)

    c = np.dot(P.T, weights)  # attain the center of the MVEE

    d = P.shape[1]

    Q = P[np.where(weights.flatten() > 0.0)[0], :]  # get all the points with positive weights
    weights2 = weights[np.where(weights.flatten() > 0.0)[0], :]  # get all the positive weights

    # compute a p.s.d matrix which will represent the ellipsoid
    ellipsoid = 1.0 / d * np.linalg.inv(np.dot(np.multiply(Q, weights2).T, Q)
                                                - np.multiply.outer(c.T.ravel(), c.T.ravel()))
    return ellipsoid, c


def enlargeByTol(ellipsoid):
    """
    The function at hand enlarges the MVEE (the ellipsoid) by a fact or (1 + Utils.TOL).

    :param ellipsoid: A numpy matrix represent a p.s.d matrix
    :return: An enlarged version of ellipsoid.
    """
    return ellipsoid / (1 + Utils.TOL) ** 2.0


def getContainedEllipsoid(ellipsoid):
    """
    This function returns a dialtion of E such that it will be contained in the convex hull of self.P.P.

    :param ellipsoid: A p.s.d matrix which represents the MVEE of self.P.P
    :return: A dilated version of the MVEE of self.P.P such that it will be contained in the convex hull
             of self.P.P.
    """
    return ellipsoid * ellipsoid.shape[1] ** 2 * (1 + Utils.TOL) ** 2  # get inscribed ellipsoid


def computeEllipInHigherDimension(Q, weights):
    """
    The function at hand computes the ellipsoid in a self.P.d + 1 dimensional space (with respect to the
    lifted points) which is centered at the origin.

    :param weights: A numpy array of weights with respect to each lifter point in self.Q
    :return:
    """
    idxs = np.where(weights > 0.0)[0]  # get all indices of points with positive weights
    weighted_Q = np.multiply(Q[idxs, :], np.expand_dims(np.sqrt(weights[idxs]), 1))  # multiply the postive
                                                                                          # weights with their
                                                                                          # corresponding points
    delta = np.sum(np.einsum('bi,bo->bio', weighted_Q, weighted_Q), axis=0)  # compute an ellipsoid which is
                                                                             # centered at the origin
    return delta


def optimalityCondition(d, Q, ellip, weights):
    """
    This function checks if the MVEE of P is found in the context of Michael J. Todd and E. Alper Yıldırım
    algorithm.

    :param ellip: A numpy array representing a p.s.d matrix.
    :param weights: A numpy array of weights with respect to the rows of P.
    :return: A boolean value whether the desired MVEE has been achieved or not.
    """
    pos_weights_idxs = np.where(weights > 0)[0]  # get the indices of all the points with positive weights
    current_dists = computemahalanobisDistance(Q, ellip)  # compute the Mahalanobis distance between ellip and
                                                            # the rows of P
    # check if all the distance are at max (1 + self.tol) * (self.P.d +1) and the distances of the points
    # with positive weights are at least (1.0 - self.tol) * (self.P.d + 1)
    return np.all(current_dists <= (1.0 + Utils.TOL) * (d + 1)) and \
        np.all(current_dists[pos_weights_idxs] >= (1.0 - Utils.TOL) * (d + 1)), current_dists


def yilidrimAlgorithm(P):
    """
    This is our implementation of Algorithm 4.2 at the paper "On Khachiyan’s Algorithm for te Computation of Minimum
    Volume Enclosing Ellipsoids" by Michael J. Todd and E. Alper Yıldırım. It serves to compute an MVEE of self.P.P
    faster than Khachiyan's algorithm.

    :return: The MVEE ellipsoid of self.P.P.
    """
    n, d = P.shape
    Q = np.hstack((P, np.ones((n, 1))))

    chosen_indices = volumeApproximation(P)  # compute an initial set of points which will give the initial
                                             # ellipsoid
    if len(chosen_indices) == n:  # if all the points were chosen then simply run Khachiyan's algorithm.
                                  # Might occur due to numerical instabilities.
        return khachiyanAlgorithm(P)
    weights = np.zeros((n, 1)).flatten()  # initial the weights to zeros
    weights[chosen_indices] = 1.0 / len(chosen_indices)  # all the chosen indices of points by the
                                                         # volume Approximation algorithm are given uniform weights
    ellip = np.linalg.inv(computeEllipInHigherDimension(Q, weights))  # compute the initial ellipsoid

    while True:  # run till conditions are fulfilled
        stop_flag, distances = optimalityCondition(d, Q, ellip, weights)  # check if current ellipsoid is desired
                                                                         # MVEE, and get the distance between rows
                                                                         # of self.P.P to current ellipsoid
        pos_weights_idx = np.where(weights > 0)[0]  # get indices of points with positive weights
        if stop_flag:  # if desired MVEE is achieved
            break
        j_plus = np.argmax(distances)  # index of maximal distance from the ellipsoid
        k_plus = distances[j_plus]  # maximal distance from the ellipsoid
        j_minus = pos_weights_idx[np.argmin(distances[pos_weights_idx])]  # get the the index of the points with
                                                                          # positive weights which also have the
                                                                          # smallest distance from the current
                                                                          # ellipsoid
        k_minus = distances[j_minus]  # the smallest distance of the point among the points with positive weights
        eps_plus = k_plus / (d + 1.0) - 1.0
        eps_minus = 1.0 - k_minus / (d + 1.0)
        if eps_plus > eps_minus:  # new point is found and it is important
            beta_current = (k_plus - d - 1.0) / ((d + 1) * (k_plus - 1.0))
            weights = (1.0 - beta_current) * weights
            weights[j_plus] = weights[j_plus] + beta_current
        else:  # a point which was already found before, yet has large impact on the ellipsoid
            beta_current = min((d + 1.0 - k_minus) / ((d + 1.0) * (k_minus - 1.0)),
                               weights[j_minus]/(1 - weights[j_minus]))
            weights = weights * (1 + beta_current)
            weights[j_minus] = weights[j_minus] - beta_current

        weights[weights < 0.0] = 0.0  # all negative weights are set to zero
        ellip = np.linalg.inv(computeEllipInHigherDimension(weights))  # recompute the ellipsoid

    return computeEllipsoid(P, weights)


def khachiyanAlgorithm(P):
    """
    This is our implementation of Algorithm 3.1 at the paper "On Khachiyan’s Algorithm for te Computation of Minimum
    Volume Enclosing Ellipsoids" by Michael J. Todd and E. Alper Yıldırım. It serves to compute an MVEE of self.P.P
    using Khachiyan's algorithm.

    :return: The MVEE ellipsoid of self.P.P.
    """
    err = 1
    count = 1  # used for debugging purposes
    n, d = P.shape
    u = np.ones((n, 1)) / n  # all points have uniform weights
    Q = np.hstack((P, np.ones((n, 1))))

    while err > Utils.TOL:  # while the approximation of the ellipsoid is higher than desired
        X = np.dot(np.multiply(Q, u).T, Q)  # compute ellipsoid
        M = computemahalanobisDistance(Q, np.linalg.inv(X))  # get Mahalanobis distances between rows of self.P.P
                                                               # and current ellipsoid
        j = np.argmax(M)  # index of point with maximal distance from current ellipsoid
        max_val = M[j]  # the maximal Mahalanobis distance from the rows of self.P.P and the current ellipsoid
        step_size = (max_val - d - 1) / ((d + 1) * (max_val - 1))
        new_u = (1 - step_size) * u  # update weights
        new_u[j, 0] += step_size
        count += 1
        err = np.linalg.norm(new_u - u)  # set err to be the change between updated weighted and current weights
        u = new_u

    return computeEllipsoid(P, u)


def computeMVEE(P, alg_type=1):
    """
    This function is responsible for running the desired algorithm chosen by the user (or by default value) for
    computing the MVEE of P.

    :param alg_type: An algorithm type indicator where 1 stands for yilidrim and 0 stands kachaiyan.
    :return:
            - The inscribed version of MVEE of P.
            - The center of the MVEE of P.
            - The vertices of the inscribed ellipsoid.
    """
    global ax

    if alg_type == 1: # yilidrim is chosen or by default
        E, C = yilidrimAlgorithm(P)
    else:  # Kachaiyan, slower yet more numerically stable
        E, C = khachiyanAlgorithm(P)

    # self.plotEllipsoid(self.E, self.C, self.computeAxesPoints())
    contained_ellipsoid = getContainedEllipsoid(E)  # get inscribed ellipsoid

    return contained_ellipsoid, C, computeAxesPoints(contained_ellipsoid, C)



################################################## ApproximateCenterProblems ###########################################
def computeLINFCoresetKOne(P):
    """
    The function at hand computes an L∞ coreset for the matrix vector multiplication or the dot product, with
    respect to the weighted set of points P.

    :return:
            - C: the coreset points, which are a subset of the rows of P
            - idx_in_P: the indices with respect to the coreset points C in P.
            - an upper bound on the approximation which our L∞ coreset is associated with.
    """
    global max_time
    r = matrix_rank(P[:, :-1])  # get the rank of P or the dimension of the span of P
    d = P.shape[1]
    if r < (d - 1) or True:  # if the span of P is a subspace in REAL^d
        svd = TruncatedSVD(n_components=d-2)  # an instance of TruncatedSVD
        Q = svd.fit_transform(P[:, :-1])  # reduce the dimensionality of P by taking their dot product by the
                                          # subspace which spans P
        Q = np.hstack((Q, np.expand_dims(P[:, -1], 1)))  # concatenate the indices to their respected "projected"
                                                         # points
    else:  # if the span of P is REAL^d where d is the dimension of P
        Q = P

    start_time = time.time()  # start counting the time here
    if r > 1:  # if the dimension of the "projected points" is not on a line
        if Q.shape[1] - 1 >= Q.shape[0]:
            return Q, np.arange(Q.shape[0]).astype(np.int), Utils.UPPER_BOUND(r)
        else:
            _, _, S = computeMVEE(Q[:, :-1], alg_type=0)  # compute the MVEE of Q

    else:  # otherwise
        # get the index of the maximal and minimal point on the line, i.e., both its ends
        idx_in_P = np.unique([np.argmin(Q[:, :-1]).astype(np.int),
                                   np.argmax(Q[:, :-1]).astype(np.int)]).tolist()
        return Q[idx_in_P], idx_in_P, Utils.UPPER_BOUND(r)

    C = []
    # idx_in_P_list = []
    # C_list = []
    # ts = time.time()
    # for q in S: # for each boundary points along the axis of the MVEE of Q
    #     K = attainCaratheodorySet(P[:, :-1], q)  # get d+1 indices of points from Q where q is their convex
    #                                                       # combination
    #     idx_in_P_list += [int(idx) for idx in K]  # get the indices of the coreset point in Q
    #     C_list += [int(Q[idx, -1]) for idx in K]  # the actual coreset points
    # # print('Time for list {}'.format(time.time() - ts))


    idx_in_P = np.empty((2*(Utils.J + 1) ** 2, )).astype(np.int)
    C = np.empty((2*(Utils.J + 1) ** 2, )).astype(np.int)
    idx = 0
    # ts = time.time()
    for q in S: # for each boundary points along the axis of the MVEE of Q
        K = attainCaratheodorySet(Q[:, :-1], q)  # get d+1 indices of points from Q where q is their convex
                                                 # combination
        idx_in_P[idx:idx+K.shape[0]] = K.astype(np.int)  # get the indices of the coreset point in Q
        C[idx:idx+K.shape[0]] = Q[idx_in_P[idx:idx+K.shape[0]], -1].astype(np.int)
        idx += K.shape[0]
    # print('Time for numpy {}'.format(time.time() - ts))

    return np.unique(C[:idx]), np.unique(idx_in_P[:idx]), Utils.UPPER_BOUND(r)


####################################################### Bicriteria #####################################################

def attainClosestPointsToSubspaces(P, W, flats, indices):
    """
    This function returns the closest n/2 points among all of the n points to a list of flats.

    :param flats: A list of flats where each flat is represented by an orthogonal matrix and a translation vector.
    :param indices: A list of indices of points in self.P.P
    :return: The function returns the closest n/2 points to flats.
    """
    dists = np.empty((P[indices, :].shape[0], ))
    N = indices.shape[0]
    if not Utils.ACCELERATE_BICRETERIA:
        for i in range(N):
            dists[i] = np.min([
                Utils.computeDistanceToSubspace(P[np.array([indices[i]]), :], flats[j][0], flats[j][1])
                for j in range(len(flats))])
    else:
        dists = Utils.computeDistanceToSubspace(P[indices, :], flats[0], flats[1])
        idxs = np.argpartition(dists, N // 2)[:N//2]
        return idxs.tolist()

    return np.array(indices)[np.argsort(dists).astype(np.int)[:int(N / 2)]].tolist()



def sortDistancesToSubspace(P, X, v, points_indices):
    """
    The function at hand sorts the distances in an ascending order between the points and the flat denoted by (X,v).

    :param X: An orthogonal matrix which it's span is a subspace.
    :param v: An numpy array denoting a translation vector.
    :param points_indices: a numpy array of indices for computing the distance to a subset of the points.

    :return: sorted distances between the subset points addressed by points_indices and the flat (X,v).
    """
    dists = Utils.computeDistanceToSubspace(P[points_indices, :], X, v)  # compute the distance between the subset
                                                                         # of points towards
                                                                         # the flat which is represented by (X,v)
    return np.array(points_indices)[np.argsort(dists).astype(np.int)].tolist()  # return sorted distances


def computeSubOptimalFlat(P, weights):
    """
    This function computes the sub optimal flat with respect to l2^2 loss function, which relied on computing the
    SVD factorization of the set of the given points, namely P.

    :param P: A numpy matrix which denotes the set of points.
    :param weights: A numpy array of weightes with respect to each row (point) in P.
    :return: A flat which best fits P with respect to the l2^2 loss function.
    """
    v = np.average(P, axis=0, weights=weights)  # compute the weighted mean of the points
    svd = TruncatedSVD(algorithm='randomized', n_iter=1, n_components=Utils.J).fit(P-v)
    V = svd.components_
    return V, v  # return a flat denoted by an orthogonal matrix and a translation vector


def clusterIdxsBasedOnKSubspaces(P, B):
    """
    This functions partitions the points into clusters a list of flats.

    :param B: A list of flats
    :return: A numpy array such each entry contains the index of the flat to which the point which is related to the
             entry is assigned to.
    """
    n = P.shape[0]
    idxs = np.arange(n)  # a numpy array of indices
    centers = np.array(B)  # a numpy array of the flats

    dists = np.apply_along_axis(lambda x: Utils.computeDistanceToSubspace(P[idxs, :], x[0], x[1]), 1, centers)  # compute the
                                                                                                # distance between
                                                                                                # each point and
                                                                                                # each flat
    idxs = np.argmin(dists, axis=0)

    return idxs  # return the index of the closest flat to each point in self.P.P


def addFlats(P, W, S, B):
    """
    This function is responsible for computing a set of all possible flats which passes through j+1 points.

    :param S: list of j+1 subsets of points.
    :return: None (Add all the aforementioned flats into B).
    """
    indices = [np.arange(S[i].shape[0]) for i in range(len(S))]

    points = np.meshgrid(*indices)                    # compute a mesh grid using the duplicated coefs
    points = np.array([p.flatten() for p in points])  # flatten each point in the meshgrid for computing the
                                                      # all possible ordered sets of j+1 points
    idx = len(B)
    for i in range(points.shape[1]):
        A = [S[j][points[j, i]][0] for j in range(points.shape[0])]
        P_sub, W_sub = P[A, :], W[A]
        B.append(computeSubOptimalFlat(P_sub, W_sub))

    return np.arange(idx, len(B)), B


def computeBicriteria(P, W):
    """
    The function at hand is an implemetation of Algorithm Approx-k-j-Flats(P, k, j) at the paper
    "Bi-criteria Linear-time Approximations for Generalized k-Mean/Median/Center". The algorithm returns an
    (2^j, O(log(n) * (jk)^O(j))-approximation algorithm for the (k,j)-projective clustering problem using the l2^2
    loss function.

    :return: A (2^j, O(log(n) * (jk)^O(j)) approximation solution towards the optimal solution.
    """
    n = P.shape[0]
    Q = np.arange(0, n, 1)
    t = 1
    B = []
    tol_sample_size = Utils.K * (Utils.J + 1)
    sample_size = (lambda t: int(np.ceil(Utils.K * (Utils.J + 1) * (2 + np.log(Utils.J + 1) +
                                                                         np.log(Utils.K) +
                                                                         min(t, np.log(np.log(n)))))))

    while np.size(Q) >= tol_sample_size:  # run we have small set of points
        S = []
        for i in range(0, Utils.J+1):  # Sample j + 1 subsets of the points in an i.i.d. fashion
            random_sample = np.random.choice(Q, size=sample_size(t))
            S.append(random_sample[:, np.newaxis])

        if not Utils.ACCELERATE_BICRETERIA:
            F = addFlats(P, W, S, B)
        else:
            S = np.unique(np.vstack(S).flatten())
            F = computeSubOptimalFlat(P[S, :], W[S])
            B.append(F)

        sorted_indices = attainClosestPointsToSubspaces(P, W, F, Q)
        Q = np.delete(Q, sorted_indices)
        t += 1

    if not Utils.ACCELERATE_BICRETERIA:
        _, B = addFlats(P, W, [Q for i in range(Utils.J + 1)], B)
    else:
        F = computeSubOptimalFlat(P[Q.flatten(), :], W[Q.flatten()])
        B.append(F)

    return B


################################################### L1Coreset ##########################################################

def applyBiCriterea(P, W):
    """
    The function at hand runs a bicriteria algorithm, which then partition the rows of P into clusters.

    :return:
            - B: The set of flats which give the bicriteria algorithm, i.e., O((jk)^{j+1}) j-flats which attain 2^j
                 approximation towards the optimal (k,j)-projective clustering problem involving self.P.P.
            - idxs: The set of indices where each entry is with respect to a point in P and contains
                    index of the flat in B which is assigned to respected point in P.
    """
    B = computeBicriteria(P,W)  # compute the set of flats which bi-cirteria algorithm returns
    idxs = clusterIdxsBasedOnKSubspaces(P, B)  # compute for each point which flat fits it best
    return B, idxs


def initializeSens(P, B, idxs):
    """
    This function initializes the sensitivities using the bicriteria algorithm, to be the distance between each
    point to it's closest flat from the set of flats B divided by the sum of distances between self.P.P and B.

    :param B: A set of flats where each flat is represented by an orthogonal matrix and a translation vector.
    :param idxs: A numpy array which represents the clustering which B imposes on self.P.P
    :return: None.
    """
    centers_idxs = np.unique(idxs)  # number of clusters imposed by B
    sensitivity_additive_term = np.zeros((P.shape[0], ))

    for center_idx in centers_idxs:  # go over each cluster of points from self.P.P
        cluster_per_center = np.where(idxs == center_idx)[0]  # get all points in certain cluster

        # compute the distance of each point in the cluster to its respect flat
        cost_per_point_in_cluster = Utils.computeDistanceToSubspace(P[cluster_per_center, :-1],
                                                                    B[center_idx][0], B[center_idx][1])

        # ost_per_point_in_cluster = np.apply_along_axis(lambda x:
        #                                                Utils.computeDistanceToSubspace(x, B[center_idx][0],
        #                                                                                B[center_idx][1]), 1,
        #                                                self.set_P.P[cluster_per_center, :-1])

        # set the sensitivity to the distance of each point from its respected flat divided by the total distance
        # between cluster points and the respected flat
        if Utils.J < 10:
            phi = 2 ** Utils.J
        else:
            phi = 1.0
        sensitivity_additive_term[cluster_per_center] = phi * \
                                                             np.nan_to_num(cost_per_point_in_cluster /
                                                                           np.sum(cost_per_point_in_cluster))

    return sensitivity_additive_term


def Level(P, k, V, desired_eps=0.01):
    """
    The algorithm is an implementation of Algorithm 7 of "Coresets for Gaussian Mixture Models of Any shapes" by Zahi
    Kfir and Dan Feldman.

    :param P: A Pointset object, i.e., a weighted set of points.
    :param k: The number of $j$-subspaces which defines the (k,j)-projective clustering problem.
    :param V: A set of numpy arrays
    :param desired_eps: An approximation error, default value is set to 0.01.
    :return: A list "C" of subset of points of P.P.
    """
    t = V.shape[0]  # numnber of points in V
    d = P.shape[1] - 1  # exclude last entry of each point for it is the concatenated index
    # C = [[]] #np.empty((P.shape[0] + Utils.J ** (2 * Utils.K), P.shape[1]))  # initialize list of coresets
    # U = [[]] #np.empty((P.shape[0] + Utils.J ** (2 * Utils.K), P.shape[1]))  # list of each point in V \setminus V_0 minus its
                                                            # projection onto a specific affine subspace, see below
    C = np.zeros((P.shape[0], ), dtype="bool")
    D = np.zeros((P.shape[0], ), dtype="bool")

    if k <= 1 or t-1 >= Utils.J:
        return np.array([])


    # ts = time.time()
    A, v = Utils.computeAffineSpan(V)
    # print('Affine took {}'.format(time.time() - ts))
    dists_from_P_to_A = Utils.computeDistanceToSubspace(P[:, :-1], A.T, v)
    non_zero_idxs = np.where(dists_from_P_to_A > 1e-11)[0]

    d_0 = 0 if len(non_zero_idxs) < 1 else np.min(dists_from_P_to_A[non_zero_idxs])
    c = 1 / d ** (1.5 * (d + 1))
    M = np.max(np.abs(P[:, :-1]))
    on_j_subspace = np.where(dists_from_P_to_A <= 1e-11)[0]

    B = [[]]

    if on_j_subspace.size != 0:
        B[0] = P[on_j_subspace, :]
        if B[0].shape[0] >= Utils.J ** (2 * k):
            indices_in_B = B[0][:, -1]
            Q = np.hstack((B[0][:,:-1], np.arange(B[0].shape[0])[:, np.newaxis]))
            temp = computeLInfCoreset(B[0], k-1)
            C[indices_in_B[temp].astype(np.int)] = True
        else:
            C[B[0][:, -1].astype(np.int)] = True

        # current_point += temp.shape[0]
    # D = [P[C]]
    # print('Bound is {}'.format(int(np.ceil(8 * np.log(M) + np.log(1.0/c)) + 1)))
    if d_0 > 0:
        for i in range(1, int(np.ceil(8 * np.log(M) + np.log(1.0/c)) + 1)):
            B.append(P[np.where(np.logical_and(2 ** (i-1) * d_0 <= dists_from_P_to_A,
                                                 dists_from_P_to_A <= 2 ** i * d_0))[0], :])
            if len(B[i]) > 0:
                if len(B[i]) >= Utils.J ** (2 * k):
                    indices_B = B[i][:, -1]
                    Q_B = np.hstack((B[i][:, :-1], np.arange(B[i].shape[0])[:, np.newaxis]))
                    temp = computeLInfCoreset(Q_B, k-1)
                    if temp.size > 0:
                        C[indices_B[temp].astype(np.int)] = True
                else:
                    C[B[i][:, -1].astype(np.int)] = True
                    temp = np.arange(B[i].shape[0]).astype(np.int)
                list_of_coresets = [x for x in B if len(x) > 0]
                Q = np.vstack(list_of_coresets)
                indices_Q = Q[:, -1]
                Q = np.hstack((Q[:, :-1], np.arange(Q.shape[0])[:, np.newaxis]))
                if temp.size > 0:
                    for point in B[i][temp, :]:
                        indices = Level(Q, k-1, np.vstack((V, point[np.newaxis, :-1])))
                        if indices.size > 0:
                            D[indices_Q[indices].astype(np.int)] = True
                        # D.extend(Level(Q, k-1, np.vstack((V, point[np.newaxis, :-1]))))

    return np.where(np.add(C, D))[0]


def computeLInfCoreset(P, k):
    """
    This function is our main L_\infty coreset method, as for k = 1 it runs our fast algorithm for computing the
    L_\infty coreset. When k > 1, it runs a recursive algorithm for computing a L_\infty coreset for the
    (k,j)-projective clustering problem.

    This algorithm is a variant of Algorithm 6 of "Coresets for Gaussian Mixture Models of Any shapes" by Zahi
    Kfir and Dan Feldman.

    :param P: A PointSet object, i.e., a weighted set of points.
    :param k: The number of $j$-subspaces which defines the (k,j)-projective clustering problem.
    :return: A PointSet object which contains a subset of P which serves as a L_\infty coreset for the
             (k,j)-projective clustering problem.
    """
    C = []
    if k == 1:  # if subspace clustering problem
        _, idxs_in_Q, upper_bound = computeLINFCoresetKOne(P)  # Compute our L_\infty coreset for P
        return idxs_in_Q
    elif k < 1:  # should return None here
        return np.array([])
    else:  # If k > 1
        temp = computeLInfCoreset(P, k-1)  # call recursively till k == 1
        C = np.zeros((P.shape[0], ), dtype="bool")
        C[P[temp, -1].astype(np.int)] = True
        # Q = np.empty((P.shape[0] + Utils.J ** (2 * Utils.K), P.shape[1]))
        # Q[:C_0.shape[0], :] = C_0
        for p in P[temp, :]:  # for each point in coreset
            # print('K = {}'.format(k))
            recursive_core = Level(P, k, p[np.newaxis, :-1])  # compute a coreset for (k,j)-projective clustering
                                                              # problem using a coreset for (k-1,j)-projective
                                                              # clustering problem
            if recursive_core.size > 0:  # if the coreset for the (k,j)-projective clustering problem is not empty
                C[P[recursive_core, -1].astype(np.int)] = True

            if np.where(C == False)[0].size < 1:
                return np.where(C)[0]
        return np.where(C)[0]  # return a L_\infty coreset for (k,j)-projective clustering problem


def computeSensitivity(P, W):
    """
    The function at hand computes the sensitivity of each point using a reduction from L_\infty to L1.

    :return: None
    """
    P = np.hstack((P, np.arange(P.shape[0])[:, np.newaxis]))
    w_min = np.min(W)
    remove_only_one = False
    if np.min(W) != np.max(W):
        remove_only_one = True
        h = np.ceil(P.shape[1] * W / w_min)
    else:
        h = 1
    s = np.zeros((P.shape[0], ))
    Utils.J = (P.shape[1] - 1)
    i = 1
    remaining_idxs = list(range(P.shape[0]))
    N = len(remaining_idxs)
    while N > 1:
        Q = copy.deepcopy(P[remaining_idxs])

        idxs_of_Q = computeLInfCoreset(np.hstack((Q[:, :-1], np.arange(Q.shape[0])[:, np.newaxis])), 1)
        if remove_only_one:
            q_i, idx_q_i = np.min(h[Q[idxs_of_Q,-1].astype(np.int)]), np.argmin(h[Q[idxs_of_Q,-1].astype(np.int)])
            h[Q[idxs_of_Q,-1].astype(np.int)] -= q_i
            m = i + q_i - 1
            if i == 1:
                s[Q[idxs_of_Q,-1].astype(np.int)] += P.shape[1] * (np.log(q_i) + 1.0 / (2.0 * q_i))
            else:
                s[Q[idxs_of_Q, -1].astype(np.int)] += P.shape[1] ** 2 * (np.log(m / (i-1.0)) +
                                                                         1.0/(2.0 * m) - 1.0/(2.0 * (i-1)) + 1.0/(i-1.0)**2)
            remaining_idxs.remove(int(Q[idxs_of_Q[idx_q_i],-1]))
        else:
            s[Q[idxs_of_Q, -1].astype(np.int)] += P.shape[1] ** 1.5 * (1.0/i)
            for x in Q[idxs_of_Q, -1].astype(np.int):
                remaining_idxs.remove(x)
        N = len(remaining_idxs)
        i += 1
        sys.stdout.write('\r%d left!' % (N))
        sys.stdout.flush()


    Utils.J = None
    return s


def generateEpsCoreset(P,W, s, sample_size):
    t = np.sum(s)  # sum of sensitivity
    p = s / t  # sampling probabilities

    cnt = 0
    S = []

    hist = np.random.multinomial(sample_size, p).flatten()
    # print('hist {}'.format(hist))
    # print('hist.shape {}'.format(hist.shape))
    indxs = np.nonzero(hist)[0]

    # if indxs.shape[0] == 0:
    #     return self.computeCoreset(P, sensitivity, sampleSize, weights)
    # Select the indices.

    weights = np.asarray(np.multiply(W[indxs], hist[indxs]), dtype=float).flatten()

    # TESTING WEIGHTS = 1.
    # weights = np.asarray(weights[indxs])

    weights = np.multiply(weights, 1.0 / (p[indxs] * sample_size))
    # print("X: {} , Y: {}, W: {}, hist: {}".format(self.S[:,:-1].shape, self.S[:,-1].shape, self.weights.shape, indxs.shape))
    # timeTaken = time.time() - startTime


    return indxs, weights





if __name__ == '__main__':
    P = np.random.randn(10000, 5)
    P = np.hstack((P, np.arange(10000)[:, np.newaxis]))
    W = np.random.rand(P.shape[0], 1).flatten()
    s = computeSensitivity(P, W)
