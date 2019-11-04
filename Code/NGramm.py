import re
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib as matplot
import matplotlib.pyplot as plt
import logfileparser as parser

class NGramm():

    def __init__(self, n=2):
        self.n = n
        self.ngramms = {}
        self.total_number_ngramms = 0
        self.ngramms_probability = {}

    def fit(self, data):
        """Reads a set of requests and stores probabilities and occurences in the class
        """
        self.ngramms = self.get_ngramms_for_all(data)
        self.total_number_ngramms = sum(self.ngramms.values())

        for ngramm in self.ngramms:
            self.ngramms_probability[ngramm] = float(self.ngramms[ngramm]) / self.total_number_ngramms

    def get_probability_of_ngrammset(self, ngramms):
        """Returns the probability from a set of ngramms
        """
        total_probability = 0
        for ngramm in ngramms:
            probability_ngramm = self.ngramms_probability.get(ngramm, 0)
            total_probability += probability_ngramm

        return total_probability / len(ngramms)


    def get_ngramms_for_all(self, data):
        """Returns a set of N-Gramms from a set of requests
        """
        ngramms = {}
        normalized_requests = []

        for request in data:
            normalized_requests.append(normalize_request(request['Request']))

        for request in normalized_requests:
            for i in range(len(request)):
                ngramm = request[i:i + self.n] # Split a requests into the n-gramms for the length of n
                if ngramm in ngramms:
                    ngramms[ngramm] += 1
                else:
                    ngramms[ngramm] = 1

        return ngramms

    def get_ngramms_for_request(self, request):
        """Returns a set of N-Gramms for a single request
        """
        ngramms = {}
        normalized_request = normalize_request(request['Request'])

        for i in range(len(normalized_request)):
            ngramm = normalized_request[i:i + self.n]
            if ngramm in ngramms:
                ngramms[ngramm] += 1
            else:
                ngramms[ngramm] = 1

        return ngramms
    
    def get_feature_vectors(self, data):
        """Get a set of two dimensional feature vectors
        with probability as one axis and the occurences of ngramms as the other axis.
        """
        vectors = []
        for request in data:
            ngramms = self.get_ngramms_for_request(request)
            probability = self.get_probability_of_ngrammset(ngramms)
            occurences = sum(ngramms.values())
            vectors.append([probability, occurences])
            
        return np.asarray(vectors)

def normalize_request(request):
    """Normalizes a request by replacing all alphanumeric characters with @
    and removes all newline characters
    """
    regex = re.compile(r"[a-zA-Z0-9]+")
    replaced = re.sub(regex, '@',request)
    regex = re.compile(r"\n")
    replaced = re.sub(regex, '', replaced)
    return replaced

def main():
    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    print("**************************")
    print("Extracting N-Gramms...")

    # Training the N-Gramm extractor
    ng = NGramm()
    ng.fit(training_data)
    
    print("N-Gramms extracted!")
    print("**************************")
    print("Starting Local Outlier Fitting...")


    # Getting Feature Vectors
    training_vectors = ng.get_feature_vectors(training_data)
    test_vectors_clean = ng.get_feature_vectors(test_clean)
    test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)

    # Fitting model for novel predictions
    km = LocalOutlierFactor(novelty = True).fit(training_vectors)
    
    print("Fitting successful!")    
    print("**************************")
    print("Starting Prediction...")

    # Predict returns 1 for inlier and -1 for outlier
    result_clean = km.predict(test_vectors_clean)
    result_anomalous = km.predict(test_vectors_anomalous)
    
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