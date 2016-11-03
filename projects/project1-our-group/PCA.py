import numpy as np
import matplotlib.pyplot as plt

def PCA(data, K=15):
    #computing the d-dimensional mean vector
    D = len(data[0])
    mean_vec = np.mean(data, axis=0)
    print(len(mean_vec), "elements in mean-dimensional vector:")
    print(mean_vec)

    #computing covariance matrix
    cov_mat = np.cov(data, rowvar=False)
    print(np.shape(cov_mat), "size of covariance matrix")

    #Computing eigenvectors and corresponding eigenvalues
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    print(len(eig_val) , "eigen values:")
    print(eig_val)
    print(len(eig_vec), "eigen vectors")

    #Sorting the eigenvectors by decreasing eigenvalues
    for ev in eig_vec:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (dimension, eigenvalue, eigenvector) tuples
    eig_pairs = [(i, np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

    # Sort the (dims, eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[1], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print("List of most signigicant dimensions, and their eigenvalues:")
    for eig in eig_pairs:
        print(eig[0], eig[1])

    #Choosing k eigenvectors with the largest eigenvalues
    #we are combining the two eigenvectors with the highest eigenvalues
    #to construct our d×k k-dimensional eigenvector matrix_w.
    #print(eig_pairs[2])
    eig_vec_filtered = []
    for k in range(K):
        eig_vec_filtered.append(eig_vec[k])
    matrix_w = np.column_stack(eig_vec_filtered)

    filtered_dims_ids = []
    for k in range(K,D):
        filtered_dims_ids.append(eig_pairs[k][0])

    print(np.shape(matrix_w), "shape of filtered eigen vectors matrix")

    data_transformed = matrix_w.T.dot(data.T)
    print("removed dimensions: ", filtered_dims_ids)
    print(np.shape(data_transformed.T), "shape of transformed final data with K most-meaningful dimensions")

    return data_transformed.T, filtered_dims_ids
