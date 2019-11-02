import re
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
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
            
        return vectors

def normalize_request(request):
    """Normalizes a request by replacing all alphanumeric characters with @
    """
    regex = re.compile(r"[a-zA-Z0-9]+")
    replaced = re.sub(regex, '@',request)
    regex = re.compile(r"\n")
    replaced = re.sub(regex, '', replaced)
    return replaced

def main():
    #read data here
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')

    print("\n**************************")
    print("Extracting N-Gramms...")

    #fit data
    ng = NGramm()
    ng.fit(training_data)
    
    print("N-Gramms extracted!")

    """print("\n**************************")
    print("All N-Gramms:")
    print(ng.ngramms)
    print("\n**************************")
    print("N-Gramms probabilities:")
    print(ng.ngramms_probability)
    print("\n**************************")
    print("Total N-Gramms:")
    print(ng.total_number_ngramms)
    """

    print("\n**************************")
    print("Starting K-Means...")

    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    training_vectors = ng.get_feature_vectors(training_data)
    test_vectors_clean = ng.get_feature_vectors(test_clean)
    test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)

    km = LocalOutlierFactor(novelty = True).fit(training_vectors)

    # predict returns 1 for inlier and -1 for outlier
    result_clean = km.predict(test_vectors_clean)
    result_anomalous = km.predict(test_vectors_anomalous)
    print(result_anomalous)


if __name__ == "__main__":
    main()