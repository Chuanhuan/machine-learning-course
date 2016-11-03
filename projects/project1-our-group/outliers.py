import numpy as np
import matplotlib.pyplot as plt

def MahalanobisDist(data, preds):
    N = len(preds)
    D = len(data[0])
    xy = np.zeros((N,D+1))
    for i in range(N):
        for j in range(D):
            xy[i][j] = data[i][j]
    for i in range(N):
        xy[i][D] = preds[i]
    #print(np.shape(xy), "size of xy matrix" )

    cov_mat = np.cov(xy, rowvar=False)
    #print(np.shape(cov_mat), "size of covariance matrix")
    cov_map_inv = np.linalg.inv(cov_mat)
    #print(np.shape(cov_map_inv), "size of covariance matrix inverse")
    
    #Center each value by the mean by subtracting the mean from i in array x and y.
    xy_mean = np.mean(xy, axis=0)
    diff_xy = np.zeros((N,D+1))
    for i in range(N):
        for j in range(D):
            diff_xy[i][j] = xy[i][j] - xy_mean[j]
    #print(np.shape(diff_xy), "size of diff_xy arrays")
    
    #calculate mahalanobis distance
    mahalanobis = []
    for i in range(N):
        mahalanobis.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),cov_map_inv),diff_xy[i])))
    #print(np.shape(mahalanobis), "shape of mahalanobis distance matrix")
    return mahalanobis

def MD_removeOutliers(data, preds, threshold_scale = 1.5):
    MD = MahalanobisDist(data, preds)
    threshold = np.mean(MD) * threshold_scale # adjust 1.5 accordingly
    #print("threshold for MahalanobisDist:", threshold)
    nx, ny, outliers = [], [], []
    for i in range(len(MD)):
        if MD[i] <= threshold:
            nx.append(data[i])
            ny.append(preds[i])
        else:
            outliers.append(i) # position of removed pair
    print(len(outliers), " outliers")
    return (np.array(nx), np.array(ny), np.array(outliers))

#How to call:
#nx, ny, outliers_ids = MD_removeOutliers(data,preds,threshold_scale=1.5)
