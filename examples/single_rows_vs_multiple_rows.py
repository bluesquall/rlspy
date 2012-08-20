#!/usr/bin/env python
"""
Simple RLS identification example
=================================

"""
import numpy as np

import rlspy

def generate_random_truth_data(order = 3, sigma = 1):
    return np.random.normal(0, sigma, [order, 1])


def generate_random_coupling_matrix(shape = [4, 3]):
    return np.random.normal(0, 1, shape)


def generate_noisy_measurements(A, x, sigma):
    return np.dot(A, x) + np.random.normal(0, sigma)


def set_up_rlsi(order = 3, morder = 4, N = 200):
    x = generate_random_truth_data(order, 1)
    A = [generate_random_coupling_matrix([morder, order]) for i in xrange(N)]
    v = 1e-2
    sm = v * np.ones(morder).reshape(-1, 1)
    V = np.diag(sm.ravel()**2)
    b = [generate_noisy_measurements(Ai, x, sm) for Ai in A]

    x0 = np.ones(order).reshape(-1, 1)
    P0 = np.identity(order)
    rlsi = rlspy.data_matrix.Estimator(x0, P0)
    return rlsi, A, x, b, V, v, x0, P0


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    order = 3
    morder = 4
    N = int(2e3)

    rlsi, A, x, b, V, v, x0, P0 = set_up_rlsi(order, morder, N)
    #TODO some sort of timeit timing, just use ipython magic for now...
 

    # reset, track, and plot the consecutive update
    rlsi, A, x, b, V, v, x0, P0 = set_up_rlsi(order, morder, N)
    
    # preallocate some arrays to track the evolution of the estimate
    xest = np.empty([order, N + 1])
    Pest = np.empty([order, order, N + 1])

    xest[:,0] = x0.ravel()
    Pest[:,:,0] = P0

    for i, (Ai, bi) in enumerate(zip(A, b)):
        rlsi.consecutive_update(Ai, bi, v)
        xest[:, i + 1] = rlsi.x.ravel()
        Pest[:, :, i + 1] = rlsi.P

    r = x - xest # residual estimation error

    plt.semilogy(np.abs(r.T))
    plt.grid(True)
    plt.ylabel('abs(estimation error)')
    plt.xlabel('iteration')
    plt.show()

