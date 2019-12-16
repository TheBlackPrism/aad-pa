import re
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import logfileparser as parser
import outlier
import pandas as pd

class NGram():

    def __init__(self):
        self.n = 0
        self.ngrams = {}
        self.total_number_ngrams = 0
        self.ngrams_probability = {}
        self.is_url = False

    def fit(self, data, is_url, n = 2, normalize = True):
        """Reads a set of requests and stores probabilities and occurences in the class
        """
        self.is_url = is_url
        self.n = n
        self.normalize = normalize

        self.ngrams = self.get_ngrams_for_all(data)
        self.total_number_ngrams = sum(self.ngrams.values())

        for ngram in self.ngrams:
            self.ngrams_probability[ngram] = float(self.ngrams[ngram]) / self.total_number_ngrams
        self.ngrams["Remainder"] = 0

    def get_probability_of_ngramset(self, ngrams):
        """Returns the probability from a set of N-Grams
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
            normalized = self.normalize_request(request['Request'])
            normalized_requests.append(normalized)

        for request in normalized_requests:
            for i in range(len(request)):
                ngram = request[i:i + self.n] # Split a requests into the N-Grams for the length of n
                if ngram in ngrams:
                    ngrams[ngram] += 1
                else:
                    ngrams[ngram] = 1

        return ngrams

    def get_ngrams_for_request(self, request):
        """Returns a set of N-Grams for a single request
        """
        ngrams = {}
        normalized = self.normalize_request(request['Request'])

        for i in range(len(normalized) - self.n + 1): 
            ngram = normalized[i:i + self.n]
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

        return ngrams
    
    def get_feature_vectors(self, data):
        """Get a set of two dimensional feature vectors from a set of requests
        with probability as one axis and the occurences of N-Grams as the other axis.
        """
        vectors = []
        for request in data:
            ngrams = self.get_ngrams_for_request(request)
            probability = self.get_probability_of_ngramset(ngrams)
            occurences = sum(ngrams.values())
            vectors.append([probability, occurences])
            
        return np.asarray(vectors)

    def get_feature_vectors_multidimensional(self, data):
        """Get a set of a multidimensional feature vectors from a set of requests.
        One dimension for each N-Gram of the training set, containing the frequency of the N-Gram in the according request
        """
        vectors = []
        ngrams_per_request = []

        for request in data:
            ngrams = self.get_ngrams_for_request(request)
            ngram_frequency = self.get_ngram_frequency_ngram_dictionary(ngrams)

            # Compare if an N-Gram of the request is not in all of the N-Grams
            if not set(ngrams.keys()).issubset(set(self.ngrams.keys())):
                #print(set(ngrams.keys()).intersection(set(self.ngrams.keys())))
                remainder = 1 - sum(ngram_frequency.values())
            else:
                remainder = 0

            ngram_frequency["Remainder"] = remainder

            vectors.append(list(ngram_frequency.values()))
            ngrams_per_request.append(ngrams)
            
        return np.asarray(vectors), ngrams_per_request

    def get_ngram_frequency_ngram_dictionary(self, ngrams_request):
        """Get the frequency of ngrams in a request for ngrams in the training set
        """
        ngrams_frequency = {}
        total_ngrams = sum(ngrams_request.values())

        for ngram in self.ngrams:
            if ngram in ngrams_request:
                ngrams_frequency[ngram] = ngrams_request[ngram] / total_ngrams
            else:
                ngrams_frequency[ngram] = 0
        return ngrams_frequency

    def normalize_request(self, request):
        """Normalizes a request by replacing all alphabet characters with a and numeric characters with 1
        """
        if self.normalize:
            regex = re.compile(r"[a-zA-Z]")
            replaced = re.sub(regex, 'a',request)
            regex = re.compile(r"[0-9]")
            replaced = re.sub(regex, '1',replaced)

        else:
            replaced = request.lower()
            if self.is_url:
                replaced = extract_url(replaced)
            else:
                replaced = extract_parameter(replaced)
        return replaced

def extract_parameter(request):
    """Extracts the parameter values of a request
    if the request has no parameter return an empty string
    """
    if request.find("?") == -1:
        return ""

    # Remove url part
    regex = re.compile(r"^([^\\?]*\?)")
    replaced = re.sub(regex, '',request)

    # Remove parameter names
    regex = re.compile(r"^([a-zA-Z0-9]*=)|(&[a-zA-Z0-9]*=)")
    replaced = re.sub(regex, '',replaced)
    return replaced

def extract_url(request):
    """Extracts the URL of a request
    """
    
    # Remove normalized localhost
    regex = re.compile(r"^(aaaa://aaaaaaaaa:1111)|^(http://localhost:8080)")
    replaced = re.sub(regex, '',request)

    if request.find("?") == -1:
        return replaced

    # Remove Parameters
    regex = re.compile(r"(\?.*)$")
    replaced = re.sub(regex, '',replaced)

    return replaced

def merge_results(list1, list2):
    """Merges two result lists into one.
    If an entry in one of the lists is -1 the according result entry will be -1 too
    """
    result = []
    for i in range(len(list1)):
        if list1[i] == -1 or list2[i] == -1:
            result.append(-1)
        else:
            result.append(1)

    return np.asarray(result)
