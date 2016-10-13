# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

def ridge_regression(y, tx, lamb, degree):
    """implement ridge regression."""
    #same as before, we just added +lamb*np.identity
    #remember: dimensionality is M+1!
    w = np.dot(np.linalg.inv(np.dot(tx.T,tx)+lamb*np.identity(degree+1)) , np.dot(tx.T,y))
    return w
