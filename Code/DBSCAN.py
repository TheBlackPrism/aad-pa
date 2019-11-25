import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import logfileparser as parser
from NGram import *

# Reading Data
training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

# Training the N-Gramm extractor
ng = NGram()
training_data_url = ng.fit(training_data,True)
training_data_param = ng.fit(training_data, False)
    
print("N-Gramms extracted!")
print("**************************")
print("Starting DB-Scan Fitting...")




# Getting Feature Vectors
training_vectors_url = ng.get_feature_vectors(training_data_url)
training_vectors_param = ng.get_feature_vectors(traing_data_param)

#test_vectors_clean = ng.get_feature_vectors(test_clean)
#test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)


print("\n**************************")
print("Training model:")
print("eps = 0.1")
print("minimum number of samples needed for a cluster: 3")

#Building the clustering model
#eps is the radius of the cluster
"""eps is the radius of the cluster.
min_samples gives the minimum number of samples that must be found within
eps to form a cluster.
Both parameters must be chosen carefully, depending on the dataset.
"""
dbscan = DBSCAN(eps = 0.1, min_samples=3)
model_url = dbscan.fit(training_vectors_url)
model_param = dbscan.fit(training_vectors_param)
y_pred_url = dbscan.fit_predict(training_vectors_url)
y_pred_param = dbscan.fit_predict(training_vectors_param)
labels_url = model_url.labels_
labels_param = model_param.labels_

#Identify cores
#cores = np.zeros_like(labels, dtype=bool)
#cores[dbscan.core_sample_indices_]=True

#compute number of clusters
#nbr_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#print(nbr_of_clusters)

plt.scatter(training_vectors_url[:,0], training_vectors_url[:,1], c = y_pred_url, cmap='Paired')
plt.scatter(training_vectors_param[:,0], training_vectors_param[:,1], c = y_pred_param, cmap='Paired')



plt.show()




"""Visualisation of clusters depends on the data in the dataset.

"""


