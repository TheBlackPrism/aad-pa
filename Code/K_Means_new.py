import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
import NGramm as ng


class K_Means_new(object):
    """description of class"""

    def __init__(self, k=1, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def main():
        # Reading Data
        training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
        test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
        test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')
        training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')

        # Training the N-Gramm extractor
        ng = NGramm()
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
        print("k = %d" %k)

        #fit trainings data to obtain clusters
        kmeans = KMeans(k)
        clusters = kmeans.fit(training_vectors)


        for i in range(len(clusters)):
            if cluster[i] == 1:
                cluster1 = plt.scatter(request_lengths[i], c = 'r', marker = 'x')
            elif clusters[i] == 0:
                cluster2 = plt.scatter(request_lengths[i], c = 'g', marker = 'o')
            

        #test clean data
        print("Training done! Switch to testing.")
        print("**************************")
        print("Testing normal traffic:")
  
        result_clean = kmeans.predict(test_vectors_clean)
        result_anomalous = kmeans.predict(test_vectors_anomalous)







