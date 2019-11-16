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
ng.fit(training_data)
    
print("N-Gramms extracted!")
print("**************************")
print("Starting K-Means Fitting...")




# Getting Feature Vectors
training_vectors = ng.get_feature_vectors(training_data)
test_vectors_clean = ng.get_feature_vectors(test_clean)
test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)


print("\n**************************")
print("Training model:")
print("eps = 0.5")
print("minimum number of samples needed for a cluster: 3")

#Building the clustering model
#eps is the radius of the cluster
"""eps is the radius of the cluster.
min_samples gives the minimum number of samples that must be found within
eps to form a cluster.
Both parameters must be chosen carefully, depending on the dataset.
"""
dbscan = DBSCAN(eps = 0.5, min_samples=3).fit(training_vectors)
labels = dbscan.labels_

"""Visualisation of clusters depends on the data in the dataset.

"""


