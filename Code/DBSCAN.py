import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import logfileparser as parser
from NGram import *

print("Reading Data...")
print("**************************")

# Reading Data
training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')
print(training_data)

print("Done!")
print("**************************")
print("Starting Feature Extraction...")

# Training the N-Gramm extractor
ng = NGram()
training_data = ng.fit(training_data,True)

# Getting Feature Vectors
training_vectors= ng.get_feature_vectors(training_data)
test_vectors_clean = ng.get_feature_vectors(test_clean)
test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)
    
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
model = dbscan.fit(training_vectors)


            
#test clean data
print("Training done! Switch to testing.")
print("**************************")
print("Start prediction...")

result_clean = dbscan_predict(model,training_vectors_clean)
result_anomalous = dbscan_predict(model, training_vectors_anomalous)

print("Predicting successful!")    
print("**************************")
print("Start evaluation...")

accuracy_anomalous = (float(np.count_nonzero(result_anomalous==-1)))/len(result_anomalous) * 100
accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean)*100

print("Results: ")
print("True Positive %.f %%" % accuracy_anomalous)
print("False Positive: %.f %%" % (100 - accuracy_clean))
print("Accuracy: %.f %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))


#inspired (aka copied) from: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new



"""Visualisation of clusters depends on the data in the dataset.

"""


