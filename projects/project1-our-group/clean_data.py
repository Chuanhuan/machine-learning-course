import numpy as np

def clean_data_by_removing(x, y, f_threshold):
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # remove features with more than f_threshold percent missing values
    removed_features = []

    for f in range(n_features):
        percentage_missing = np.count_nonzero(x[:, f]==-999)/n_samples
        if percentage_missing > f_threshold:
            removed_features.append(f)

    x_feature_filtered = np.delete(x, removed_features, axis = 1)

    # remove samples with missing values
    removed_samples = []
    for s in range(n_samples):
        missing_values = np.count_nonzero(x_feature_filtered[s, :]==-999) > 0
        if missing_values:
            removed_samples.append(s)

    x_filtered = np.delete(x_feature_filtered, removed_samples, axis = 0)
    y_filtered = np.delete(y, removed_samples, axis = 0)

    return x_filtered, y_filtered, removed_features

def clean_data_by_mean(x, y, f_threshold):
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # remove features with more than f_threshold percent missing values
    #removed_features = []

    #for f in range(n_features):
    #    percentage_missing = np.count_nonzero(x[:, f]==-999)/n_samples
    #    if percentage_missing > f_threshold:
    #        removed_features.append(f)

    x_feature_filtered = x #np.delete(x, removed_features, axis = 1)

    # calculate mean per column without -999 values
    n_features = x_feature_filtered.shape[1]
    x_filtered = x_feature_filtered

    for f in range(n_features):
        mean_f = (x_filtered[x_filtered[:,f]!=-999,f]).mean()
        x_filtered[x_filtered[:,f]==-999,f] = [mean_f]

    return x_filtered, y, []#removed_features
