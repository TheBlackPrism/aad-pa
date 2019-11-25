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
training_data = ng.fit(training_data,True)
    
print("N-Gramms extracted!")
print("**************************")
print("Starting DB-Scan Fitting...")




# Getting Feature Vectors
training_vectors= ng.get_feature_vectors(training_data)

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
model = dbscan.fit(training_vectors)


#Identify cores
#cores = np.zeros_like(labels, dtype=bool)
#cores[dbscan.core_sample_indices_]=True

#compute number of clusters
#nbr_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#print(nbr_of_clusters)

plt.scatter(training_vectors[:,0], training_vectors[:,1],  s=100, color = "c", alpha = 0.5, label = "Training Datapoints")
plt.show()




"""Visualisation of clusters depends on the data in the dataset.

"""


