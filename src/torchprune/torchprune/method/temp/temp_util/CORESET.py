import numpy as np
import copy
import time
from scipy import optimize
from datetime import datetime
from RegressionCoresets import computeSensitivity
import AproxMVEE

R = 10


def obtainSensitivity(X, w, approxMVEE=False):
    if not approxMVEE:
        return computeSensitivity(X, w)
    else:
        cost_func = lambda x: np.linalg.norm(np.dot(X, x), ord=1)
        mvee = AproxMVEE.MVEEApprox(X, cost_func, 10)
        ellipsoid, center = mvee.compute_approximated_MVEE()
        # G = np.linalg.pinv(ellipsoid)
        U = X.dot(ellipsoid)
        return np.linalg.norm(U, ord=1, axis=1)


def generateCoreset(X, y, sensitivity, sample_size, weights=None, SEED=1):
    if weights is None:
        weights = np.ones((X.shape[0], 1)).flatten()

    # Compute the sum of sensitivities.
    t = np.sum(sensitivity)

    # The probability of a point prob(p_i) = s(p_i) / t
    probability = sensitivity.flatten() / t

    startTime = time.time()

    # initialize new seed
    np.random.seed()

    # Multinomial Distribution
    # hist = np.random.multinomial(sample_size, probability.flatten()).flatten()
    # indxs = np.nonzero(hist)[0]
    hist = np.random.choice(np.arange(probability.shape[0]), size=sample_size, replace=False, p=probability.flatten())
    indxs, counts = np.unique(hist, return_counts=True)
    S = X[indxs]
    labels = y[indxs]

    # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
    weights = np.asarray(np.multiply(weights[indxs], counts), dtype=float).flatten()

    weights = np.multiply(weights, 1.0 / (probability[indxs] * sample_size))
    timeTaken = time.time() - startTime

    return indxs, S, labels, weights, timeTaken