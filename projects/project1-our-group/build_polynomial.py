# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    num_samples = len(x)
    degrees = [x**i for i in range(2, degree+1)]
    poly = np.vstack(degrees).transpose()

    return poly
