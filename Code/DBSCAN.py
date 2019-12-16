import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
import logfileparser as parser
from NGram import *
import outlier

#inspired (aka copied) from: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
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
    #eps is the radius of the cluster
    """eps is the radius of the cluster.
    min_samples gives the minimum number of samples that must be found within
    eps to form a cluster.
    Both parameters must be chosen carefully, depending on the dataset.
    """
    dbscan = DBSCAN(eps = eps, min_samples=min_samples)
    model = dbscan.fit(training_vectors)

    print("Training done! Switch to testing.")
    print("Start prediction...")
    
    result_training = __dbscan_predict(model, training_vectors)
    result_clean = __dbscan_predict(model, clean_vectors)
    result_anomalous = __dbscan_predict(model, anomalous_vectors)

    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous, result_training

def main():
    print("**************************")
    print("Reading Data...")

    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    training_data = parser.append_parameter_to_request(training_data)
    test_clean = parser.append_parameter_to_request(test_clean)
    test_anomalous = parser.append_parameter_to_request(test_anomalous)

    print("Done!")
    print("**************************")
    print("Starting Feature Extraction...")

    # Training the N-Gramm extractor
    ng_url = NGram()
    ng_url.fit(training_data,True)

    ng_param = NGram()
    ng_param.fit(training_data,False)


    # Getting Feature Vectors
    training_vectors_parameter, ngrams_training_parameter = ng_param.get_feature_vectors_multidimensional(training_data)
    test_vectors_clean_parameter, ngrams_clean_parameter = ng_param.get_feature_vectors_multidimensional(test_clean)
    test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_param.get_feature_vectors_multidimensional(test_anomalous)
    
    training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(training_data)
    test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_clean)
    test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_anomalous)

    print("N-Gramms extracted!")

    eps = 0.1
    min_samples = 3

    result_clean_param, result_anomalous_param = dbscan(training_vectors_parameter, test_vectors_clean_parameter, test_vectors_anomalous_parameter, eps, min_samples)
    result_clean_url, result_anomalous_url = dbscan(training_vectors_url, test_vectors_clean_url, test_vectors_anomalous_url, eps, min_samples)

    # Merge the two result lists
    result_clean = merge_results(result_clean_param, result_clean_url)
    result_anomalous = merge_results(result_anomalous_param, result_anomalous_url)

    outlier.evaluate_detection(result_clean, result_anomalous)

if __name__ == "__main__":
    main()