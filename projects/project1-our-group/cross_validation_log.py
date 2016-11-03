import numpy as np
from plots import cross_validation_visualization
from costs import compute_loss_RLOG

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    np.random.seed(seed)
    
    num_row = y.shape[0]
    indices = np.random.permutation(num_row)
    
    interval = int(num_row / k_fold)
    
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)

def find_best_lambda(lambdas, rmse_te, rmse_tr):
    id_min = np.argmin(rmse_te - rmse_tr)
    return lambdas[id_min]

def cross_validation(y, x, k_indices, k, initial_w, max_iters, _lamb, 
                     regression_function, loss_function):
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    
    x_tr = np.delete(x, k_indices[k], axis = 0)
    y_tr = np.delete(y, k_indices[k])

    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************

    w, loss = regression_function(y_tr, x_tr, _lamb, initial_w, max_iters, 1e-06)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    loss_tr = np.sqrt(2 * loss/len(y_tr))
    loss_te = np.sqrt(2 * compute_loss_RLOG(y_te, x_te, w, _lamb)/len(y_te))
    print("Cross validation with lamb={l}, loss tr={l_tr}, loss test={l_te}".format(
              l=_lamb, l_tr=loss_tr, l_te=loss_te, w=w))

    return loss_tr, loss_te

def cross_validation_demo(y, x, regression_function, loss_function, initial_w, 
                          max_iters, k_fold, seed, logmin=-2, logmax=4):
    
    lambdas = np.logspace(logmin, logmax, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # ***************************************************   
    for _lambda in lambdas:
        total_loss_tr = 0
        total_loss_te = 0
        for k in np.arange(k_fold):
            loss_tr, loss_te = cross_validation(y, x, k_indices, k, initial_w, 
                                                max_iters, _lambda, 
                                                regression_function, 
                                                loss_function)
            total_loss_tr = total_loss_tr + loss_tr
            total_loss_te = total_loss_te + loss_te
            
        losses_tr.append(total_loss_tr/k_fold)
        losses_te.append(total_loss_te/k_fold)
      
    cross_validation_visualization(lambdas, losses_tr, losses_te)
    
    return find_best_lambda(lambdas, np.array(losses_te), np.array(losses_tr))
