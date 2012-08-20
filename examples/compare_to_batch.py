#!/usr/bin/env python
"""
Batch LS vs. RLS comparison example
===================================

"""
import numpy as np

import rlspy

def generate_random_truth_data(order = 3, sigma = 1):
    return np.random.normal(0, sigma, [order, 1])


def generate_random_coupling_matrix(shape = [4, 3]):
    return np.random.normal(0, 1, shape)


def generate_noisy_measurements(A, x, sigma):
    return np.dot(A, x) + np.random.normal(0, sigma)


def example(order = 3, morder = 4, N = 20):
    x = generate_random_truth_data(order, 1)
    A = [generate_random_coupling_matrix([morder, order]) for i in xrange(N)]
    sm = 1e-2 * np.ones(morder).reshape(-1, 1)
    V = np.diag(sm.ravel()**2)
    b = [generate_noisy_measurements(Ai, x, sm) for Ai in A]

    x0 = np.ones(order).reshape(-1, 1)
    P0 = np.identity(order)
    rlsi = rlspy.data_matrix.Estimator(x0, P0)

    # preallocate some arrays to track the evolution of the estimate
    xest = np.empty([order, N + 1])
    Pest = np.empty([order, order, N + 1])

    xest[:,0] = x0.ravel()
    Pest[:,:,0] = P0
    # run the RLS identification
    for i, (Ai, bi) in enumerate(zip(A, b)):
        rlsi.update(Ai, bi, V)
        xest[:, i + 1] = rlsi.x.ravel()
        Pest[:, :, i + 1] = rlsi.P
    xerr = x - xest

    # organize the coupling matrices and data for batch LS
    Ab = np.vstack(A)
    bb = np.vstack(b)
    bxest, residue, rankA, singA = np.linalg.lstsq(Ab, bb)
    bxerr = x - bxest
    bPest = None # TODO A inv(V) A.T, needs full data V

    return xest, Pest, xerr, bxest, bPest, bxerr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    order = 2
    morder = 2
    N = 200
    x, P, r, bx, bp, br = example(order, morder, N)
    plt.semilogy(np.abs(r.T))
    plt.semilogy(np.array([0,N]), np.outer(np.ones(order), np.abs(br.T)), lw=2)
    plt.grid(True)
    plt.ylabel('abs(estimation error)')
    plt.xlabel('iteration')
    plt.show()
