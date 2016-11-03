import numpy as np

def compute_gradient_MSE(y, tx, w):
    """Compute the gradient for MSE."""
    N = len(y)
    e = y - (tx @ w)
    gradient = (-1/N) * (tx.T @ e)

    return gradient

def compute_gradient_MAE(y, tx, w):
    """Compute the gradient for MAE."""
    N = len(y)
    e = y - (tx @ w)

    gradient = (-1/N) * np.dot(np.sign(e),tx)

    return gradient

def compute_gradient_LOG(y, tx, w):
    """compute the gradient for logistic regression."""
    xw = tx @ w
    return tx.T @ (np.apply_along_axis(sigmoid, 0, xw) - y)

def compute_gradient_RLOG(y, tx, w, lambda_):
    """compute the cost by negative log likelihood and then penalizes it."""
    return compute_gradient_LOG(y, tx, w) + 2 * lambda_ * w

def sigmoid(t):
    """applies sigmoid function on t."""
    et = np.exp(-t)
    return 1/(1 + et)
