import re
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import logfileparser as parser
import outlier

class NGram():

    def __init__(self, n=2):
        self.n = n
        self.ngrams = {}
        self.total_number_ngrams = 0
        self.ngrams_probability = {}

    def fit(self, data):
        """Reads a set of requests and stores probabilities and occurences in the class
        """
        self.ngrams = self.get_ngrams_for_all(data)
        self.total_number_ngrams = sum(self.ngrams.values())

        for ngram in self.ngrams:
            self.ngrams_probability[ngram] = float(self.ngrams[ngram]) / self.total_number_ngrams

    def get_probability_of_ngramset(self, ngrams):
        """Returns the probability from a set of ngrams
        """
        total_probability = 0
        for ngram in ngrams:
            probability_ngram = self.ngrams_probability.get(ngram, 0)
            total_probability += probability_ngram

        return total_probability / len(ngrams)


    def get_ngrams_for_all(self, data):
        """Returns a set of N-Grams from a set of requests
        """
        ngrams = {}
        normalized_requests = []

        for request in data:
            normalized_requests.append(normalize_request(request['Request']))

        for request in normalized_requests:
            for i in range(len(request)):
                ngram = request[i:i + self.n] # Split a requests into the n-grams for the length of n
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

        return ngrams

    def get_ngrams_for_request(self, request):
        """Returns a set of N-Grams for a single request
        """
        ngrams = {}
        normalized_request = normalize_request(request['Request'])

        for i in range(len(normalized_request)):
            ngram = normalized_request[i:i + self.n]
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

        return ngrams
    
    def get_feature_vectors(self, data):
        """Get a set of two dimensional feature vectors
        with probability as one axis and the occurences of ngrams as the other axis.
        """
        vectors = []
        for request in data:
            ngrams = self.get_ngrams_for_request(request)
            probability = self.get_probability_of_ngramset(ngrams)
            occurences = sum(ngrams.values())
            vectors.append([probability, occurences])
            
        return np.asarray(vectors)

def normalize_request(request):
    """Normalizes a request by replacing all alphanumeric characters with @
    and removes all newline characters
    """
    regex = re.compile(r"[a-zA-Z0-9]")
    replaced = re.sub(regex, '@',request)
    return replaced

def main():
    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    training_data = parser.append_parameter_to_request(training_data)
    test_clean = parser.append_parameter_to_request(test_clean)
    test_anomalous = parser.append_parameter_to_request(test_anomalous)
    print("**************************")
    print("Extracting N-Grams...")

    # Training the N-Gram extractor
    ng = NGram()
    ng.fit(training_data)
    
    print("N-Grams extracted!")

    # Getting Feature Vectors
    training_vectors = ng.get_feature_vectors(training_data)
    test_vectors_clean = ng.get_feature_vectors(test_clean)
    test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)

    outlier.local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous)
    outlier.one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous)

    # Plotting Vectors
    fig, ax = plt.subplots()
    samples = 3000
    ax.scatter(training_vectors[:samples,0], training_vectors[:samples,1], s=200,color = "g", alpha = 0.3, label = "Trainings Data")
    ax.scatter(test_vectors_clean[:samples,0], test_vectors_clean[:samples,1], s=150, color = "b", alpha = 0.3, label = "Clean Data")
    ax.scatter(test_vectors_anomalous[:samples,0], test_vectors_anomalous[:samples,1], s=100, color = "r", alpha = 0.3, label = "Anomalous Data")
    plt.xlabel("Probability of the Request")
    plt.ylabel("Number of N-Grams Occurences")
    plt.title("Distribution of Feature Vectors")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()