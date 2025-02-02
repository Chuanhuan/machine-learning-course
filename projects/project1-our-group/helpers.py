# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv

def load_data(data_path, sub_sample=False):
    """Load data and convert it to the metric system."""
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return ids, input_data, yb

def sample_data_iterator(y, x, seed, size_samples, num_samples):
    """sample from dataset."""
    np.random.seed(seed)
    for i in range(num_samples):
        num_observations = y.shape[0]
        random_indices = (np.random.sample(size_samples) * num_observations).astype(int)
        y_out = y[random_indices]
        x_out = x[random_indices]
        yield y_out, x_out

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = x.shape[0]
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def outputFile(weights, removed_features, cols_rem, name='out.csv',
               filter_PCA=False, std=False, mean_x=0, std_x=1):
    DATA_TEST_PATH = 'data/test.csv'

    ids_test, tX_test, _ = load_data(DATA_TEST_PATH)
    tX_test = np.delete(tX_test, removed_features, axis = 1)
    if filter_PCA:
        tX_test = np.delete(tX_test, cols_rem, axis = 1)
    if std:
        tX_test = (tX_test - mean_x) / std_x
    ids_test, tX_test = build_model_data(tX_test, ids_test)
    OUTPUT_PATH = 'submissions/' + name
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
