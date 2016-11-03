import numpy as np

def compute_loss_MSE(y, tx, w):
    """Calculate the mean squared error"""

    N = len(y)
    e = y - tx @ w
    e = (1/(2*N)) * e.T @ e

    return e

def compute_loss_MAE(y, tx, w):
    """Calculate the mean squared error"""

    N = len(y)
    e = y - tx @ w
    e = (1/N) * np.sum(np.abs(e))

    return e

def compute_loss_LOG(y, tx, w):
    """compute the cost by negative log likelihood."""
    xw = tx @ w
    # was log(1 + exp(xw)) but as xw goes large fast we
    # can estimate it as xw.
    return np.sum(xw - y * xw)

def compute_loss_RLOG(y, tx, w, lambda_):
    """compute the cost by negative log likelihood and then penalizes it."""
    return compute_loss_LOG(y, tx, w) + lambda_ * w.T @ w