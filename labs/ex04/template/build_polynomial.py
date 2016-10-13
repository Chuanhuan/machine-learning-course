# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    basis_matrix = np.zeros(shape=(len(x), degree+1))
    for i in range(0,len(x)):
        for j in range(0,degree+1):
            basis_matrix[i,j] = x[i]**j 
    return basis_matrix
