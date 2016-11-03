
# Project 1 of the PCML course !

# Useful starting lines

import numpy as np

# Import data

import helpers

ids, x_tr, y_tr = helpers.load_data('data/train.csv')
ids_test, x_te, _ = helpers.load_data('data/test.csv')


# concat the features

x_tot = np.concatenate((x_tr, x_te))


# Clean up missing data

# The missing values (-999) are filled with the mean of 
# the column.

import clean_data

x_tot, y_tr, cols = clean_data.clean_data_by_mean(x_tot, y_tr, 0.3)


# Feature Engineering

from build_polynomial import build_poly

x_tot_plus = build_poly(x_tot.T, 4)
x_tot = np.concatenate((x_tot, x_tot_plus), axis=1)


# initialization

# Standardizing the data
x_stdize, mean_x, std_x = helpers.standardize(x_tot)
# Building the model
y, tx_tot = helpers.build_model_data(x_stdize, y_tr)
# Replacement of -1 to 0
y[y==-1] = 0


# Separation

tx_te = tx_tot[250000:,:]
tx = tx_tot[:250000,:]

print("Shape of testing set",tx_te.shape)
print("Shape of training set",tx.shape)


# Machine Learning !

# Algorithm parameters and initialization

from implementations import *
from costs import *
from gradients import *

# Parameters

max_iters = 1000000
initial_w = np.zeros(tx.shape[1])
gamma = 1.1e-03

# Regularized logistic regression

def stoch_reg_logistic_regression(y, tx, lambda_, initial_w, 
                                  max_iters, gamma):
    """implementation of regularized logistic regression using GD"""
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, 
                                       gamma, compute_loss_RLOG, 
                                       compute_gradient_RLOG, 
                                       reg=True, lambda_=lambda_)


w, _ = stoch_reg_logistic_regression(y, tx, 0, initial_w, 
                                        max_iters , gamma)


# verification

yPred = helpers.predict_labels(w, tx)

y[y == 0] = -1
pred = np.count_nonzero(yPred == y) / len(y)
print("percentage of good predicion in training set :", pred)

# Output to file

OUTPUT_PATH = 'submissions/' + 'out.csv'
y_pred = helpers.predict_labels(w, tx_te)
helpers.create_csv_submission(ids_test, y_pred, OUTPUT_PATH)