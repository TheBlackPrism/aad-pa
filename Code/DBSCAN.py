import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
import logfileparser as parser
from NGram import *

#inspired (aka copied) from: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        
        
        for i, x_core in enumerate(dbscan_model.components_):
            x_new_reshaped = x_new.reshape(11,1)
            x_core_reshaped = x_core.reshape(83,1)
            if metric(x_new_reshaped, x_core_reshaped,w=None) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

def main():
    print("Reading Data...")
    print("**************************")

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
    print("**************************")
    print("Starting DB-Scan Fitting...")
    print("\n**************************")
    print("Training model:")
    print("eps = 0.3")
    print("minimum number of samples needed for a cluster: 3")

    #Building the clustering model
    #eps is the radius of the cluster
    """eps is the radius of the cluster.
    min_samples gives the minimum number of samples that must be found within
    eps to form a cluster.
    Both parameters must be chosen carefully, depending on the dataset.
    """
    dbscan = DBSCAN(eps = 0.3, min_samples=3)
    model_url = dbscan.fit(training_vectors_url)
    model_param = dbscan.fit(training_vectors_parameter)

            
    #test clean data
    print("Training done! Switch to testing.")
    print("**************************")
    print("Start prediction...")


    result_clean_url = dbscan_predict(model_url, test_vectors_clean_url)
    result_anomalous_url = dbscan_predict(model_url, test_vectors_anomalous_url)
    result_clean_param = dbscan_predict(model_param, test_vectors_clean_parameter)
    result_anomalous_param = dbscan_predict(model_param, test_vectors_anomalous_parameter)

    # Merge the two result lists
    result_clean = merge_results(result_clean_param, result_clean_url)
    result_anomalous = merge_results(result_anomalous_param, result_anomalous_url)

    print("Predicting successful!")    
    print("**************************")
    print("Start evaluation...")

    accuracy_anomalous = (float(np.count_nonzero(result_anomalous==-1)))/len(result_anomalous) * 100
    accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean)*100

    print("Results: ")
    print("True Positive %.f %%" % accuracy_anomalous)
    print("False Positive: %.f %%" % (100 - accuracy_clean))
    print("Accuracy: %.f %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))


if __name__ == "__main__":
    main()