#!/usr/bin/env python
"""
Example of RLS estiumation using sparse matrices
================================================

note :: This is lazy in the way it handles sparsity--it just uses 
        the scipy.sparse package. It does NOT do any kind of 
        special factorizations or anything.

"""

import itertools
import numpy
import scipy.sparse

import rlspy

N = 10
M = 20
rho = 0.7
sigma_x = 1
sigma_m = 0.1
count = 10


x0 = scipy.sparse.csc_matrix(numpy.zeros((N,1)))
P0 = scipy.sparse.identity(N)
Q = scipy.sparse.identity(M) * sigma_m


x = scipy.sparse.csr_matrix(numpy.random.normal(0, sigma_x, (N, 1)))
As = [scipy.sparse.rand(M, N, rho, format='csr') for i in range(count)]
bs = [numpy.dot(A, x) for A in As]
for b in bs: print b

rlsi = rlspy.data_matrix_sparse.Estimator(x0, P0)
for A, b in itertools.izip(As, bs):
    rlsi.consecutive_update(A, b, sigma_m)

print rlsi.x - x
