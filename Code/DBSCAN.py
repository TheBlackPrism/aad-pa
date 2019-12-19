import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN

#Found on: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
def __dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign clean label to y_new
                y_new[j] = 1
                break

    return y_new

def dbscan(training_vectors, clean_vectors, anomalous_vectors, eps=0.3, min_samples=3):
    print("Starting DB-Scan Fitting...")

    #Building the clustering model
    """eps is the max. distance between two neighbouring datapoints.
    min_samples gives the minimum number of samples that must be found to form a cluster.
    Both parameters must be chosen carefully, depending on the dataset.
    """
    dbscan = DBSCAN(eps = eps, min_samples=min_samples)

    print("Fitting with Parameters: ", dbscan.get_params())
    model = dbscan.fit(training_vectors)

    print("Training done! Switch to testing.")
    print("Start prediction...")
    
    result_training = __dbscan_predict(model, training_vectors)
    result_clean = __dbscan_predict(model, clean_vectors)
    result_anomalous = __dbscan_predict(model, anomalous_vectors)

    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous, result_training
