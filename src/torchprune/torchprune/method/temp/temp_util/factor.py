import numpy as np
from .testEM2 import computeCost, computeDistanceToSubspace, EMLikeAlg


def getProjectiveClustering(
    P, j, k, verbose=True, steps=15, NUM_INIT_FOR_EM=10
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


def getCost(P, flats):
    n = P.shape[0]
    w = np.ones(n)  # unit weights

    cost = computeCost(P, w, flats)[0]
    return cost
