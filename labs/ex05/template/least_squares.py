# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    N = len(y)
    w = np.dot(np.linalg.inv(np.dot(tx.T,tx)) , np.dot(tx.T,y))
    mse = 1/(2*N)*np.sum((y- np.dot(tx,w))**2)
    # Print the results
    print("Least Squares: loss*={l}, w0*={w0}, w1*={w1}".format(
            l=mse, w0=w[0], w1=w[1]))
    return mse, w

