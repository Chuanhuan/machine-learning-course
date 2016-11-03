# The Higgs boson machine learning challenge
## Mini-project 1 - CS-433 - Pattern Classification and Machine Learning

run.py contains the implementation of our best result.

!IMPORTANT: the code assumes you have a folder called data at the root of the project with the project training and testing data and a folder called submissions where after executing run.py the output will be stored as 'out.csv'.

In Proj1.ipynb you can walk through all steps of the data wrangling process: importing the data, cleaning the data and building the polynomial. After that you can try each of the implemented machine learning methods.

The machine learning methods are imported from the implementations.py file. Following algorithms are implemented:
- least_squares_GD(y, tx, initial w, max iters, gamma)
- least_squares_SGD(y, tx, initial w, max iters, gamma)
- least_squares(y, tx)
- ridge_regression(y, tx, lambda_)
- logistic_regression(y, tx, initial_w, max_iters, gamma)
- reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)

The algorithms make use of cost and gradient calculations using MSE, MAE and log likelihood. All variants are implemented in costs.py and gradients.py

The rest of the files contain helper functions:
- helpers.py for importing the data, standardizing, building model data, making predictions and writing output
- build_polynomial.py	for generating a polynomial of the data for a given degree
- clean_data.py	contains two different functions for dealing with missing data

PCA.py and outliers.py respectively contain the code for running Principle Component Analysis and Outlier removal. PCA can be run on a data set by using PCA(data, K) where K is the number of features you want to keep and the outlier detection can be run by calling MD_removeOutliers(x, y, threshold) on the data x and y. Here the lower the threshold value, the more samples are removed. The according notebooks can be used to play around with these filters.

IMPORTANT: We had troubles merging our group together so we appear in the list twice (53 and AlwaysDivergent)
