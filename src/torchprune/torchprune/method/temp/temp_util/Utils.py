
import scipy as sp
import numpy as np
import time
import pathlib
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from scipy.linalg import null_space
from scipy.stats import ortho_group
from scipy.spatial import Delaunay
import copy
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import TruncatedSVD
# from wpca import WPCA


################################################## General constants ###################################################
Epsilon = 1e-9  # used for approximated the gradient of a loss function

TOL = 0.1  # defines the approximation with respect to the minimum volume enclosing ellipsoid
J = None  # the dimension of the desired subspace
K = 1  # number of j dimensional subspaces
Z = 2  # the power on the distance function
STEPS_FOR_EM = 6  # number of max steps for EM-like algorithm
NUM_INIT_FOR_EM = 1000
LAMBDA = 1  # the regularization parameter

UPPER_BOUND = (lambda x: 2.0 ** (Z+1) * x ** (Z * 1.5))  # defines the factor of the upper bounds of our approximation
PARALLELIZE = False  # whether to apply the experimental results in a parallel fashion
ACCELERATE_BICRETERIA = True
ACCELERATE_SENSE = True
PREPROCESS = True
svd = TruncatedSVD(n_components=J)

# M estimator loss functions supported by our framework
M_ESTIMATOR_FUNCS = {
    'lp': (lambda x: np.abs(x) ** Z / Z),
    'huber': (lambda x: np.where(np.abs(x) <= LAMBDA, x ** 2 / 2, LAMBDA * (np.abs(x) - LAMBDA / 2))),
    'cauchy': (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
    'geman_McClure': (lambda x: x ** 2 / (2 * (1 + x ** 2))),
    'welsch': (lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-x ** 2 / LAMBDA ** 2))),
    'tukey': (lambda x: np.where(np.abs(x) <= LAMBDA, LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3),
                                 LAMBDA**2 / 6)),
    'L1-2': (lambda x: 2 * (np.sqrt(1 + x ** 2 / 2) - 1)),
    'fair': (lambda x: LAMBDA ** 2 * (np.abs(x) / LAMBDA - np.log(1 + np.abs(x) / LAMBDA))),
    'logit': (lambda x: 1.0 / (1 + np.exp(-x))),
    'relu': (lambda x: np.max(x, 0)),
    'leaky-relu': (lambda x: x if x > 0 else 0.01 * x)
}


############################################# Data option constants ####################################################
SYNTHETIC_DATA = 0  # use synthetic data
REAL_DATA = 1  # use real data
DATA_TYPE = REAL_DATA  # distinguishes between the use of real vs synthetic data
FILENAME = None

########################################### Experimental results constants #############################################
# colors for our graphs
color_matching = {'Our coreset': 'red',
                  'Uniform sampling': 'blue',
                  'All data': 'black'}

REPS = 22  # number of repetitions for sampling a coreset
SEED = np.random.randint(1, int(1e7), REPS)  # Seed for each repetition
NUM_SAMPLES = 10  # number of coreset sizes
METHOD = 'lp'
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS[METHOD]  # the objective function which we want to generate a coreset for
x0 = None  # initial solution for hueristical solver


def readFileIntoNumpy(file_path):
    """
    The function at hand reads the desired dataset chosen by the user

    :param file_path: A string containing the physical path of dataset chosen to be used by the user.
    :return: A numpy matrix containing a dataset chosen by the user.
    """
    if file_path == '':
        if DATA_TYPE == 0:  # if the chosen dataset is synthetic
            return readSyntheticRegressionData()
        else:  # real-world Resources
            return readRealData(problemType=0)
    else:
        return readRealData(file_path)


def determineUpperBound(d=None):
    """
    This function defines the upper bound on the approximation which our coreset attains.

    :param d: A positive integer denoting the dimension of the points
    :return: An upper bound on the approximation of our coreset
    """
    global J, UPPER_BOUND
    if d is None:
        return UPPER_BOUND(J)
    else:
        return UPPER_BOUND(d)


def getIdxsOfEllements(P, idxs):
    P_idxs = np.array([Q[1] for Q in P])
    return np.where(P_idxs == idxs)[0]


def getAllElementsWhichAreNotInIdx(P, idxs):
    x = np.array(P)
    mask = np.ones(len(P), dtype=bool)
    mask[idxs] = False
    return x[mask]


def computeSuboptimalSubspace(P):
    """
    This function computes a suboptimal subspace in case of having the generalized K-means objective function.

    :param P: A weighted set, namely, an object of PointSet.
    :return: A tuple of a basis of J dimensional spanning subspace, namely, X and a translation vector denoted by v.
    """
    global J

    start_time = time.time()
    if False:
        v = np.average(P.P, axis=0, weights=P.W)  # weighted mean of the point
        _, _, V = np.linalg.svd(P.P - v, full_matrices=False)  # computing the spanning subspace
        return V[:J, :], v, time.time() - start_time

    else:
        v = np.average(P.P, axis=0, weights=P.W)
        svd = TruncatedSVD(n_components=J).fit(np.multiply(np.sqrt(P.W[:, np.newaxis]), (P.P - v)))
        # wpcaa = WPCA(n_components=J)
        # W = np.ones(P.P.shape)* P.W[:, np.newaxis]
        # wpcaa.fit(X=P.P, weights=W)
        return svd.components_, v, time.time()-start_time




def EMLikeAlg(P, j, k, steps=STEPS_FOR_EM):
    """
    The function at hand, is an EM-like algorithm which is hueristic in nature. It finds a suboptimal solution for the
    (K,J)-projective clustering problem with respect to a user chosen

    :param P: A weighted set, namely, a PointSet object
    :param j: An integer denoting the desired dimension of the flat (affine subspace)
    :param k: An integer denoting the number of j-flats
    :param steps: An ingteger denoting the max number of the allowed steps
    :return: A list of k j-flats which locally optimize the cost functio
    """
    global OBJECTIVE_LOSS, NUM_INIT_FOR_EM
    start_time = time.time()

    max_norm = np.max(np.linalg.norm(P.P, axis=1))
    min_vs = None
    min_Vs = None
    optimal_cost = np.inf
    for iter in range(NUM_INIT_FOR_EM):  # run EM for 10 random initializations
        # np.random.seed()
        vs = P.P[np.random.choice(np.arange(P.n), size=k, replace=False), :]
        Vs = np.empty((k, j, P.d))
        idxs = np.arange(P.n)
        np.random.shuffle(idxs)
        idxs = np.array_split(idxs, k)
        for i in range(k):  # initialize k random orthogonal matrices
            Vs[i, :, :], _, _ = computeSuboptimalSubspace(P.attainSubset(idxs[i]))
            # Vs[i, :, :] = ortho_group.rvs(dim=P.d)[:j, :]

        for i in range(steps):  # find best k j-flats which can attain local optimum
            dists = np.empty((P.n, k))  # distance of point to each one of the k j-flats
            for l in range(k):
                _, dists[:, l] = computeCost(P, Vs[l, :, :], vs[l, :])

            cluster_indices = np.argmin(dists, 1)  # determine for each point, the closest flat to it
            unique_idxs = np.unique(cluster_indices)  # attain the number of clusters

            for idx in unique_idxs:  # recompute better flats with respect to the updated cluster matching
                temp, vs[idx, :], _ = \
                    computeSuboptimalSubspace(P.attainSubset(np.where(cluster_indices == idx)[0]))
                if temp.shape[0] < j:
                    null_temp = null_space(temp)[:, :j - temp.shape[0]].T
                    temp = np.vstack((temp, null_temp if null_temp.ndim > 1 else null_temp[np.newaxis, :]))

                Vs[idx, :, :] = temp

        current_cost = computeCost(P, Vs, vs)[0]
        if current_cost < optimal_cost:
            min_Vs = copy.deepcopy(Vs)
            min_vs = copy.deepcopy(vs)
            optimal_cost = current_cost

    return min_Vs, min_vs, time.time()-start_time


def computeKSuboptimalSubspace(P):
    """
    The function at hand attains a suboptimal subspace with respect to the generalized k-means when the queries are
    flats instead of points. If k == 1 then it applies the method computeSuboptimalSubspace and EMLikeAlg otherwise.

    :param P: A weighted set, namely, a PointSet object.
    :return: k j-flats which locally optimize our desired cost function
    """
    global J,K
    if K == 1:
        return computeSuboptimalSubspace(P)
    else:
        return EMLikeAlg(P, J, K)


def computeCost(P, X, v=None):
    """
    This function represents our cost function which is a generalization of k-means where the means are now J-flats.

    :param P: A weighed set, namely, a PointSet object.
    :param X: A numpy matrix of J x d which defines the basis of the subspace which we would like to compute the
              distance to.
    :param v: A numpy array of d entries which defines the translation of the J-dimensional subspace spanned by the
              rows of X.
    :return: The sum of weighted distances of each point to the affine J dimensional flat which is denoted by (X,v)
    """

    if X.ndim == 2:
        dist_per_point = OBJECTIVE_LOSS(computeDistanceToSubspace(P.P, X, v))
        cost_per_point = np.multiply(P.W, dist_per_point)
        # cost_per_point = np.multiply(P.W, np.apply_along_axis(lambda x:
        #                                                       OBJECTIVE_LOSS(computeDistanceToSubspace(x, X, v)),
        #                                                       axis=1, arr=P.P))
    else:
        temp_cost_per_point = np.empty((P.n, X.shape[0]))
        for i in range(X.shape[0]):
            # temp_cost_per_point[:, i] = np.multiply(P.W, np.apply_along_axis(lambda x: OBJECTIVE_LOSS(
            #     computeDistanceToSubspace(x, X[i, :, :], v[i, :])), axis=1, arr=P.P))
            temp_cost_per_point[:, i] = \
                np.multiply(P.W, OBJECTIVE_LOSS(computeDistanceToSubspace(P.P, X[i, :, :], v[i, :])))

        cost_per_point = np.min(temp_cost_per_point, 1)

    return np.sum(cost_per_point), cost_per_point


def optimizer(P):
    global x0
    start_time = time.time()
    user_options = {'disp': True, 'maxiter': int(1e13), 'gtol':1e-12, "iprint": 0,
                    'eps': 1e-12, 'maxls': int(1e5), 'maxcor': int(1e3),
                    'ftol': np.finfo(float).eps}

    bounds = np.tile([-np.inf, np.inf], (x0.shape[0], 1))
    bounds[-1][0] = np.finfo(np.float64).eps * 10
    res = sp.optimize.minimize(CaucyLossAndGrad, x0[0, :].flatten(), method='BFGS', args=P, jac=True,
                               options=user_options, bounds=bounds)
    tries = 1

    while res.status == 2:
        print('Damn Bro!')
        # x1 = np.random.rand(P.d -1, )
        res = sp.optimize.minimize(CaucyLossAndGrad, x0[tries, :].flatten(), method='CG', args=P, jac=True,
                                   options=user_options, bounds=bounds)
        tries += 1
        if tries >= x0.shape[0] - 1:
            raise ValueError('Optimization did not converge. The problem is ill-defined!')
    w = res.x

    X = np.multiply(P.P[:, :-1], np.sqrt(P.W)[:, np.newaxis])
    y = np.multiply(P.P[:, -1], np.sqrt(P.W))
    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), -y)

    # huber = HuberRegressor(fit_intercept=False, tol=1e-12, max_iter=1e13).fit(P.P[:, :-1], -P.P[:, -1], P.W)
    # w = huber.coef_

    return CaucyLossAndGrad(w, P)[0], w, time.time() - start_time


def cauchyLossPerPoint(point, Y, v):
    dist = computeDistanceToSubspace(point, Y, v)
    return np.log(1 + dist ** 2.0)


def welschLossPerPoint(point, Y, v):
    dist = computeDistanceToSubspace(point, Y, v)
    return LAMBDA ** 2 / 1 * (1 - np.exp(- dist ** 2 / LAMBDA ** 2))


def gemanMcClureLossPerPoint(point, Y, v):
    dist = computeDistanceToSubspace(point, Y, v)
    return dist ** 2 / (2 * (1 + dist ** 2))


def linearReg(w, P):
    cost_func = lambda x: np.sum(np.multiply(P.W, np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))
    grad = approx_fprime(w, cost_func, Epsilon)  # * P.n
    return cost_func(w), grad

    # loss = np.multiply(np.sqrt(P.W), np.dot(P.P[:, :-1], w) + P.P[:, -1])
    # grad =
    # return np.multiply(np.sqrt(P.W), np.dot(P.P[:, :-1], w) + P.P[:, -1])


def HuberLossAndGrad(w, P):
    cost_func = lambda w: HuberLoss(w, P)
    grad = approx_fprime(w, cost_func, Epsilon) # * P.n
    return cost_func(w), grad


def HuberLoss(w, P):
    vals = np.dot(P.P[:, :-1], w) + P.P[:, -1]
    idxs = np.where(np.abs(vals) >= 1.35)[0]
    vals2 = vals ** 2 / 2
    vals2[idxs] = 1.35 * np.abs(vals[idxs]) - 0.5 * 1.35 ** 2
    return np.sum(np.multiply(P.W, vals2)) + 0.00001 * np.sum(P.W) / P.n * np.linalg.norm(w) ** 2


def CaucyLossAndGrad(w, P):
    # loss = npSum(npmultiply(P.W, npLog(1.0 + npSquare(npDot(P.P[:, :-1], w) + P.P[:, -1]))))
    # loss = np.sum(np.multiply(P.W, np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))
    # grad_subterm_A = np.multiply(P.P[:, :-1], np.expand_dims(np.dot(P.P[:, :-1], w) + P.P[:, -1], 1))
    # grad_subterm_B = np.expand_dims(2/(1.0 + np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])), 1)
    # grad = np.sum(np.multiply(np.expand_dims(P.W, 1), np.multiply(grad_subterm_A, grad_subterm_B)), 0)

    cost_func = lambda w: np.sum(np.multiply(P.W, np.log(1.0 +
                                                         np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))) +\
                          1 * np.sum(P.W) / P.n * np.linalg.norm(w) ** 2
    # cost_func = lambda w: np.sum(np.multiply(P.W, np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))
    grad2 = approx_fprime(w, cost_func, Epsilon)

    return cost_func(w), grad2


def getObjectiveFunction():
    global PROBLEM_DEF, OBJECTIVE_FUNC, GRAD_FUNC

    if PROBLEM_DEF == 1:
        OBJECTIVE_FUNC = (lambda P, w: np.sum(np.multiply(P.W,np.log(1.0 + np.square(np.dot(P.P[:, :-1], w) + P.P[:, -1])))))
        GRAD_FUNC = (lambda P, w: np.sum(
            np.multiply(P.W, np.multiply(np.expand_dims(2/(1.0 + np.square(np.dot(P.P[:, :-1], w) - P.P[:, -1])), 1),
                                       np.multiply(P.P[:, :-1], np.expand_dims(np.dot(P.P[:, :-1], w) + P.P[:, -1], 1)), 0))))


def generateSampleSizes(n):
    """
    The function at hand, create a list of samples which denote the desired coreset sizes.

    :param n: An integer which denotes the number of points in the dataset.
    :return: A list of coreset sample sizes.
    """
    global NUM_SAMPLES

    min_val = 100  #int(10 * np.log(n) ** 2)  # minimum sample size
    max_val = 1000  #int(max(n / 7, n ** 0.6))  # maximal sample size
    samples = np.geomspace([min_val], [max_val], NUM_SAMPLES)  # a list of samples
    return samples


def readSyntheticRegressionData():
    data = np.load('SyntheticRegDataDan.npz')
    X = data['X']
    y = data['y']
#     P = PointSet(np.hstack((X[:, np.newaxis], -y[:, np.newaxis])))
    P = np.hstack((X[:, np.newaxis], -y[:, np.newaxis]))
    return P


def plotPointsBasedOnSens():
    sens = np.load('sens.npy')
    data = np.load('SyntheticRegDataDan.npz')
    X = data['X']
    y = data['y']
    getObjectiveFunction()
    P = np.hstack((X[:, np.newaxis], -y[:, np.newaxis]))

    colorbars = ['bwr']#, 'seismic', 'coolwarm', 'jet', 'rainbow', 'gist_rainbow', 'hot', 'autumn']

    for i in range(len(colorbars)):
        plt.style.use('classic')
        min_, max_ = np.min(sens), np.max(sens)

        plt.scatter(P[:, 0], P[:, 1], c=sens, marker='o', s=50, cmap=colorbars[i])
        plt.clim(min_, max_)

        ax = plt.gca()
        cbar = plt.colorbar(pad=-0.1, fraction=0.046)
        cbar.ax.get_yaxis().labelpad = 24
        cbar.set_label('Sensitivity', rotation=270, size='xx-large', weight='bold')
        cbar.ax.tick_params(labelsize=24)
        plt.ylabel('y')
        plt.xlabel('x')
        plt.axis('off')
        figure = plt.gcf()
        figure.set_size_inches(20, 13)
        plt.savefig('Sens{}.pdf'.format(i), bbox_inches='tight', pad_inches=0)


def createRandomRegressionData(n=2e4, d=2):
    X, y = sklearn.datasets.make_regression(n_samples=int(n), n_features=d, random_state=0, noise=4.0,
                           bias=100.0)

    # X = np.random.randn(int(n),d)
    # y = np.random.rand(y.shape[0], )

    X = np.vstack((X, 1000 * np.random.rand(20, d)))
    y = np.hstack((y, 10000 * np.random.rand(20, )))

    np.savez('SyntheticRegData', X=X, y=y)


def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=True, cageColor='r', cageAlpha=1):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(np.array([x[i, j], y[i, j], z[i, j]]), rotation) + center.flatten()

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        print('Axis are: ', axes)
        # print(axes + center.flatten())

        # plot axes
        print('Whole points are: ')
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 2) + center[0]
            Y3 = np.linspace(-p[1], p[1], 2) + center[1]
            Z3 = np.linspace(-p[2], p[2], 2) + center[2]
            ax.plot3D(X3, Y3, Z3, color='m')
            PP = np.vstack((X3, Y3, Z3)).T
            print(PP)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)


def createSyntheticDan(n = 10000, d=2):
    P = np.hstack((np.random.randn(n, 1) * 1e4, np.random.randn(n, 1) / 1e6))
    Q = np.hstack((np.random.rand(5, 1) * 1e4, 20 + (np.random.randn(5, 1) / 1e3)))
    P = np.vstack((P, Q))

    X = P[:, 0]
    y = P[:, 1]

    np.savez('SyntheticRegDataDan', X=X, y=y)


##################################################### READ DATASETS ####################################################
def findDelimiter(path):
    reader = pd.read_csv(path, sep = None, iterator = True, header=None, nrows=2)
    return reader._engine.data.dialect.delimiter

def readRealData(datafile=r'hour.csv', problem_type=1):
    """
    This function, given a physical path towards an csv file, reads the data into a weighted set.

    :param datafile: A string containing the physical path on the machine towards the dataset which the user desires
                     to use.
    :param problemType: A integer defining whether the dataset is used for regression or clustering.
    :return: A weighted set, namely, a PointSet object containing the dataset.
    """
    global FILENAME, PREPROCESS
    full_path = r'datasets\\' + datafile
    FILENAME = datafile if '.' not in datafile else datafile.split('.')[0]
    delim = findDelimiter(full_path)
    dataset = pd.read_csv(full_path, sep=delim) if '.npz' not in full_path else np.load(full_path) # read csv file
    if '.npz' in full_path:
        P = np.hstack((dataset['X'][:, np.newaxis], dataset['y'][:, np.newaxis]))
    else:
        P = np.hstack((dataset.values[:, :7].astype(np.float), dataset.values[:, 8:].astype(np.float)))  # get the data which the csv file has

    P = np.around(P, 6)  # round the dataset to avoid numerical instabilities
    if problem_type == 0:  # if the problem is an instance of regression problem
        P[:, -1] = -P[:, -1]

    if PREPROCESS:
        P = preProccessing(P)

#     return PointSet(P)
    return P


def preProccessing(P, standaraztion=False, normalization=True):
    Q = copy.deepcopy(P)
    if standaraztion:
        Q = StandardScaler().fit_transform(Q)

    if normalization:
        Q = Normalizer().fit_transform(Q)
    return Q

################################################# Auxuilary methods ####################################################
def unifyLists(list1, list2):
    """
    This function performs union between two lists.

    :param list1: A list.
    :param list2: A list.
    :return: the union of list1 and list2.
    """

    return [x for x in list1 + list2 if len(x) > 0]
    # return list(set(list1) | set(list2))


def attainAllButSpecifiedIndices(P, indices):
    """
    This function serves to get all points whose index is not present in indices.

    :param P: A PointSet object, which is a weighted set of points.
    :param indices: A numpy array of indices with respect to P.
    :return: A boolean array which contains True at each i entry where i is NOT in indices, and False's elsewhere.
    """
    n = P.shape[0]
    _, idxs_of_intersection, _ = np.intersect1d(P[:, -1], indices, return_indices=True)  # get indices of intersection
    mask = np.ones((n, ), dtype=bool)  # initialize a mask of true vales
    mask[idxs_of_intersection] = False  # all entries whose respected index is in indices will have False value
    return mask

def checkIfFileExists(file_path):
    """
    The function at hand checks if a file at given path exists.

    :param file_path: A string which contains a path of file.
    :return: A boolean variable which counts for the existence of a file at a given path.
    """
    file = pathlib.Path(file_path)
    return file.exists()


def createRandomInitialVector(d):
    """
    This function create a random orthogonal matrix which each column can be use as an initial vector for
    regression problems.

    :param d: A scalar denoting a desired dimension.
    :return: None (using global we get A random orthogonal matrix).
    """
    global x0
    x0 = np.random.randn(d,d)  # random dxd matrix
    [x0, r] = np.linalg.qr(x0)  # attain an orthogonal matrix

############################################## Computational Geometry tools ############################################
def computeAffineSpan(vectors):
    """
    This function computes the affine span of set of vectors. This is done by substracting one point from the set of
    vectors such that the origin is present in the "vectors - point" which then the affine span is equivalent to the
    the span of the resulted substracted set with the translation of the point.

    :param vectors: A list of vectors.
    :return: A basis for the set of "vectors - p" where p is in vectors, along with a translation vector p.
    """
    global svd
    j = len(vectors)
    v = vectors[0]
    vectors = vectors[1:]
    if len(vectors) > 0:
        A = np.vstack(vectors)
        A -= v  # make sure that the origin will be in A - A[0, :]
        # svd.fit(A)
        _, _, V = np.linalg.svd(A)  # compute the basis which A lies in the span of
        return V[:, :j], v  # return the basis, and a translation vector
        # return svd.components_.T, v  # return the basis, and a translation vector
    else:
        return np.zeros((v.shape[0],v.shape[0])), v


def gram_schmidt(vectors):
    """
    The function at hand computes a basis with respect to a given set of vectors, using the notorious Gram Schmidt
    approach.

    :param vectors: A numpy array of vectors.
    :return: A basis which the set of vectors are in it's span.
    """
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


def computeDistanceToSubspace(point, X, v=None):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.

    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    global OBJECTIVE_LOSS
    if point.ndim > 1:
        return OBJECTIVE_LOSS(np.linalg.norm(np.dot(point - v[np.newaxis, :], null_space(X)), ord=2, axis=1))
    return OBJECTIVE_LOSS(np.linalg.norm(np.dot(point-v if v is not None else point, null_space(X))))


def computeRectangle(V, side_length=1):
    """
    This function computes a rectangle by receiving an "orthogonal matrix", by simply adding its negate.

    :param V: A list of vectors which are columns of an orthogonal matrix.
    :param side_length: The desired length of each rectangle edge.
    :return: set of vectors representing the sides of the rectangle.
    """
    sides = copy.deepcopy(V)  # copy the orthogonal matrix
    sides.append([-x for x in sides])  # add its negate
    return side_length * np.vstack(sides)  # make sure that the sides have desired length


def computeRectangles(rectangle, desired_eps, t, v):
    """
    The function at hand, given a specific rectangle computes 1/desired_eps^|rectangle| sub rectangles.

    :param rectangle: A list of vectors which are the edges of a rectangle.
    :param desired_eps: A scalar which makes up for the length of the sides of each sub rectangle.
    :param t: A scalar which counts for the number of sides of the rectangle.
    :param v: A translation vector.
    :return: A list of the subretangles such that the union of this list is the input rectangle itself.
    """
    coefs = np.arange(desired_eps, 1+desired_eps, desired_eps)  # create an array of coefficentes for which dissect
                                                                # the length of each side of side of the rectangle into
                                                                # 1/desired_eps equal parts
    all_coefs = np.hstack((-np.flip(coefs), coefs))  # add also the negate of coefs for counting to other half of each
                                                     # side of the rectangle
    end_points = [all_coefs for i in range(rectangle.shape[0] // 2)]  # duplicate the coefs for each positive side of
                                                                      # the rectangle
    points = np.meshgrid(*end_points)  # compute a mesh grid using the duplicated coefs
    points = np.array([p.flatten() for p in points])  # flatten each point in the meshgrid for computing the
                                                      # rectrangle parts
    Rs = []  # Initialize an empty list

    # using points, we create here the subrectangle parts
    for j in range(points.shape[1]):  # move along combination of coefficients
        if j + 1 < points.shape[1] - 1:
            for i in range(points[:, j].shape[0]):  # for each combination of coefts, we create an appropriate rectangle
                Rs.append(np.vstack((np.array([points[i, j] * rectangle[i, :]]),
                                     np.array([points[i, j+1] * rectangle[i, :]]))) + v[np.newaxis, :])

    return Rs


def computePointsInRectangle(P, rectangle, v):
    """
    The function at hand returns all the points inside a given rectangle.

    :param P: A PointSet object, which is a weighted set of points.
    :param rectangle: A numpy array containing the side edges of a rectangle
    :param v: A translation vector with respect to the rectangle
    :return: A tuple containing a weighted set of the points which are inside the rectangle, and their indices in the
             PointSet P.
    """
    if rectangle.shape[0] >= 2 * rectangle.shape[1]:  # A requirement for using ConvexHull like methods
        polyhedra = Delaunay(rectangle)  # compute a "polyhedra" which is the rectangle
        idxs = np.where(np.apply_along_axis(lambda p: polyhedra.find_simplex(p) >= 0, 1, P.P[:, :-1]))[0]  # compute all
                                                                                                           # indices of
                                                                                                           # points in
                                                                                                           # the
                                                                                                           # rectangle
        return P.attainSubset(idxs), idxs
    else:  # This case accounts for dealing with 1 dimensional rectangle - bounded line
        orth_line = null_space((rectangle[0, :] - v)[:, np.newaxis].T)  # compute the orthant of rectangle
        Q = np.linalg.norm((P.P[:, :-1]-v).dot(orth_line), axis=1)  # get the distance of each point from the rectangle
        idxs = np.where(Q == 0)[0]  # attain all points are on the span of the rectangle (along the line)
        if idxs.size == 0: # if there is no points on span of the rectangle
            return None, None
        else:
            # for each point compute whether it lies on the rectangle (bouned range on the line) or not
            Q = np.where(np.apply_along_axis(lambda p: np.all(np.logical_and(np.greater_equal(p, rectangle[0, :]),
                                                                             np.less_equal(p, rectangle[1, :]))),
                                             arr=P.P[idxs, :-1], axis=1))[0]
            if Q.size == 0:  # If there is no such point
                return None, None
            else: # otherwise
                return P.attainSubset(idxs[Q]), idxs[Q]


#################################################### Embeddings ########################################################
def stabledevd(p):
    # we set beta = 0 by analogy with the normal and cauchy case
    # in order to get symmetric stable distributions.
    np.random.seed()
    theta=np.pi*(np.random.uniform(0,1) - 0.5)
    W = -np.log(np.random.uniform(0,1))  # takes natural log

    left = np.sin(p*theta)/np.pow(np.cos(theta), 1.0/p)
    right= np.pow(np.cos(theta*(1.0 - p))/W, ((1.0-p)/p))
    holder=left*right
    return holder


def computeSparseEmbeddingMatrix(d):
    global Z, J
    s = Z ** 3 / 0.5
    m = int(J ** 2 / 0.5 ** Z)

    S = np.zeros((d, m))

    for i in range(m):
        S[np.random.choice(np.arange(d), size=s, replace=False),i] = np.random.choice([-1, 1], size=s) / np.sqrt(s)

    return S

def computeDistortionEmbeddingLp(n,d):
    global Z
    omega = 10
    s = int(omega * d ** 5 * np.log(d) ** 5)
    e = np.eye(N=s, M=1, dtype=np.float)
    S = np.zeros((s, n))
    np.random.seed()
    idxs = np.random.choice(a=np.arange(s), size=(n, ))
    D = np.zeros((1,n))
    for i in range(n):
        S[:, i] = np.roll(e, idxs[i])
        D[i,0] = stabledevd(Z)

    return np.multiply(S,D)

# def ConstApproxLp(A):
#     global J
#     n, d = A.shape
#
#     R = computeSparseEmbeddingMatrix(d)
#     X = computeU()
#     return X








if __name__ == '__main__':
    # createSyntheticDan()
    # createRandomRegressionData()
    # testCauchy()
    # plotPointsBasedOnSens()
    # readRealData()
    computeRectangles()
