import numpy as np
from helpers import batch_iter, sample_data_iterator
from costs import *
from gradients import *

#TODO: these functions now take additional arguments to the cost and gradient function. Need to be removed to submit

def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, 
                     compute_gradient, reg=False, lambda_=0):
    """implementation of the gradient descent algorithm."""
    
    threshold = 1e-8
    w = initial_w

    prev_loss = np.nan
    for n_iter in range(max_iters):

        # compute gradient for current w and current batch
        if not reg:
            gradient = compute_gradient(y, tx, w);
            loss = compute_loss(y, tx, w)
        else:
            gradient = compute_gradient(y, tx, w, lambda_);
            loss = compute_loss(y, tx, w, lambda_)

        # adjust weights
        w = w - gamma*gradient

        if not np.isnan(prev_loss) and np.abs(prev_loss - loss) < threshold:
            break
        prev_loss = loss

    return w, loss

def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, 
                                compute_gradient, size_sample=200, seed=1, 
                                reg=False, lambda_=0):
    """implementation of the stochastic gradient descent algorithm."""
    threshold = 1e-8
    w = initial_w
    iter_samples = sample_data_iterator(y, tx, seed, size_sample, max_iters)

    prev_loss = np.nan
    for n_iter in range(max_iters):

        # get new batch
        s_y, s_tx = next(iter_samples)

        # compute gradient for current w and current batch
        if not reg:
            gradient = compute_gradient(s_y, s_tx, w);
            loss = compute_loss(s_y, s_tx, w)
        else:
            gradient = compute_gradient(s_y, s_tx, w, lambda_);
            loss = compute_loss(s_y, s_tx, w, lambda_)
         
        # adjust weights
        w = w - gamma*gradient

        if not np.isnan(prev_loss) and np.abs(prev_loss - loss) < threshold:
            break
        prev_loss = loss
            
    if not reg:
        loss = compute_loss(y, tx, w)
    else:
        loss = compute_loss(y, tx, w, lambda_)

    return w, loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_MSE, 
                            compute_gradient_MSE)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, size_sample=50, seed=1):
    
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, 
                                       compute_loss_MSE, compute_gradient_MSE)

def least_squares(y, tx, compute_loss=compute_loss_MSE):
    """implementation of the least squares solution using normal equations"""

    w = (np.linalg.inv(tx.T @ tx) @ tx.T @ y)

    loss = compute_loss(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_, compute_loss=compute_loss_MSE):
    """implementation of ridge regression using normal equations"""

    w = np.linalg.inv(tx.T @ tx + lambda_*np.identity(tx.shape[1])) @ tx.T @ y

    loss = compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implementation of logistic regression using GD"""
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_LOG, 
                            compute_gradient_LOG)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implementation of regularized logistic regression using GD"""
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss_RLOG, 
                            compute_gradient_RLOG, reg=True, lambda_=lambda_)
