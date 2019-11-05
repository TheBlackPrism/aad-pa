import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
from NGramm import *


class K_Means_new():
    """description of class"""

    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

def main():
    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

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

    for i in range(1,21):

        print("\n**************************")
        print("Training model:")
        print("k = %d" % i)
        #print("k = %d" %self.k)

        #fit trainings data to obtain clusters
        kmeans = KMeans(i)
        clusters = kmeans.fit(training_vectors)

            

        #test clean data
        print("Training done! Switch to testing.")
        print("**************************")
        print("Testing normal traffic:")
  
        result_clean = kmeans.predict(test_vectors_clean)
        result_anomalous = kmeans.predict(test_vectors_anomalous)

        print("Predicting successful!")    
        print("**************************")
        print("Results:")

        # Evaluation
        accuracy_anomalous = np.count_nonzero(result_anomalous == -1) / len(result_anomalous) * 100
        accuracy_clean = np.count_nonzero(result_clean == 1) / len(result_clean) * 100

        print("True Positiv: %d %%" % accuracy_anomalous)
        print("False Positiv: %d %%" % (100 - accuracy_clean))
        print("Accuracy: %d %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))
    
    # Plotting Vectors
    fig, ax = plt.subplots()
    samples = 300
    ax.scatter(training_vectors[:samples,0], training_vectors[:samples,1], s=200,color = "g", alpha = 0.5, label = "Trainings Data")
    ax.scatter(test_vectors_clean[:samples,0], test_vectors_clean[:samples,1], s=150, color = "b", alpha = 0.5, label = "Clean Data")
    ax.scatter(test_vectors_anomalous[:samples,0], test_vectors_anomalous[:samples,1], s=100, color = "r", alpha = 0.5, label = "Anomalous Data")
    plt.xlabel("Probability of the Request")
    plt.ylabel("Number of N-Gramms Occurences")
    plt.title("Distribution of Feature Vectors")
    ax.legend()
    plt.show()

if __name__ == "__main__":
        main()






