import re
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import logfileparser as parser
import outlier
import pandas as pd

class NGram():

    def __init__(self, n=2):
        self.n = n
        self.ngrams = {}
        self.total_number_ngrams = 0
        self.ngrams_probability = {}
        self.is_url = False

    def fit(self, data, is_url):
        """Reads a set of requests and stores probabilities and occurences in the class
        """
        self.is_url = is_url

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
        regex = re.compile(r"[a-zA-Z]")
        replaced = re.sub(regex, 'a',request)
        regex = re.compile(r"[0-9]")
        replaced = re.sub(regex, '1',replaced)
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

def main():
    """
    training_url = '../Logfiles/Labeled/data-ngram-paramvalues-mp/featuresNormalTraining1ReducedLocalDecoded.csv'
    test_clean_url = '../Logfiles/Labeled/data-ngram-paramvalues-mp/featuresNormalTest1ReducedLocalDecoded.csv'
    test_anomalous_url = '../Logfiles/Labeled/data-ngram-paramvalues-mp/featuresAnomalousTest1ReducedLocalDecoded.csv'
    """

    training_url = '../Logfiles/Labeled/rema-short/normalTraining'
    test_clean_url = '../Logfiles/Labeled/rema-short/normalTest'
    test_anomalous_url = '../Logfiles/Labeled/rema-short/anomalousTest'
    
    csv = re.compile(r"(csv)$")
    if re.search(csv, training_url):
        print("**************************")
        print("Reading Feature-Vectors from CSV...")
        training_data = read_csv(training_url)
        test_clean = read_csv(test_clean_url)
        test_anomalous = read_csv(test_anomalous_url)

        result_clean_parameter, result_anomalous_parameter = outlier.one_class_svm(training_data, test_clean, test_anomalous)
        #result_clean_parameter, result_clean_parameter = outlier.local_outlier_detection(training_data, test_clean, test_anomalous)
 
    else:
        # Reading Data
        training_data = parser.read_data(training_url)
        test_clean = parser.read_data(test_clean_url)
        test_anomalous = parser.read_data(test_anomalous_url)

        training_data = parser.append_parameter_to_request(training_data, True)
        test_clean = parser.append_parameter_to_request(test_clean, True)
        test_anomalous = parser.append_parameter_to_request(test_anomalous, True)

        print("**************************")
        print("Extracting N-Grams...")

        # Training the N-Gram extractor
        ng_parameter = NGram()
        ng_parameter.fit(training_data, False)

        ng_url = NGram()
        ng_url.fit(training_data, True)
    
        print("N-Grams extracted!")

        # Getting Feature Vectors
        training_vectors_parameter, ngrams_training_parameter = ng_parameter.get_feature_vectors_multidimensional(training_data)
        test_vectors_clean_parameter, ngrams_clean_parameter = ng_parameter.get_feature_vectors_multidimensional(test_clean)
        test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_parameter.get_feature_vectors_multidimensional(test_anomalous)
    
        training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(training_data)
        test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_clean)
        test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_anomalous)

        result_clean_parameter, result_anomalous_parameter = outlier.local_outlier_detection(training_vectors_parameter, test_vectors_clean_parameter, test_vectors_anomalous_parameter)
        result_clean_url, result_anomalous_url = outlier.local_outlier_detection(training_vectors_url, test_vectors_clean_url, test_vectors_anomalous_url)
        # outlier.one_class_svm(training_vectors, test_vectors_clean,
        # test_vectors_anomalous)

        # Merge the two result lists
        result_clean = merge_results(result_clean_parameter, result_clean_url)
        result_anomalous = merge_results(result_anomalous_parameter, result_anomalous_url)

        outlier.evaluate_detection(result_clean, result_anomalous)

        # Write Results to file
        f = open(str(ng_parameter.n) + "Gram_Result.txt", "w", encoding="utf-8")
    
        accuracy_anomalous = (float(np.count_nonzero(result_anomalous == -1))) / len(result_anomalous) * 100
        accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean) * 100
   
        f.write(str(ng_parameter.n) + "-Gram rema logs")
        f.write("\nEvaluation:")
        f.write("\nTrue Positive: %.4f %%" % accuracy_anomalous)
        f.write("\nFalse Positive: %.4f %%" % (100 - accuracy_clean))
        f.write("\nAccuracy: %.4f %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))

        f.write("\n*************************\nClean Data:\n")
        f.write("\nFeature Vector Index URL:" + str(ng_url.ngrams.keys()))
        f.write("\nFeature Vector Index Parameter:" + str(ng_parameter.ngrams.keys()))
        for i in range(len(test_clean)):
            request = test_clean[i]
            f.write("\n\nRequest:\n" + request["Request"])   

            f.write("\nN-Grams URL:\n")
            for keys,values in ngrams_clean_url[i].items():
                f.write("{" + keys + " " + str(values) + "}")
            f.write("\nFeature Vector URL:\n" + np.array2string(test_vectors_clean_url[i]))

            f.write("\nN-Grams Parameter:\n")
            for keys,values in ngrams_clean_parameter[i].items():
                f.write("{" + keys + " " + str(values) + "}")
            f.write("\nFeature Vector Parameter:\n" + np.array2string(test_vectors_clean_parameter[i]))

            f.write("\nResult:\n" + np.array2string(result_clean[i]))
    
        f.write("\n\n\n\n\n\n\n\n*************************\nAnomalous Data:\n")
        for i in range(len(test_anomalous)):
            request = test_anomalous[i]
            f.write("\n\nRequest:\n" + request["Request"])

            f.write("\nN-Grams URL:\n")
            for keys,values in ngrams_anomalous_url[i].items():
                f.write("{" + keys + " " + str(values) + "}")
            f.write("\nFeature Vector URL:\n" + np.array2string(test_vectors_anomalous_url[i]))

            f.write("\nN-Grams Parameter:\n")
            for keys,values in ngrams_anomalous_parameter[i].items():
                f.write("{" + keys + " " + str(values) + "}")
            f.write("\nFeature Vector Parameter:\n" + np.array2string(test_vectors_anomalous_parameter[i]))

            f.write("\nResult:\n" + np.array2string(result_anomalous[i]))

        f.close()

    # Plotting Vectors
    """
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
    """

def read_csv(url):
    """Reads a CSV file and returns the feature vectors as an np array
    """
    dict = pd.read_csv(url).to_dict('index')
    ngrams = list(dict.values())
    features = []
    for tuple in ngrams:
        feature = np.asarray(list(tuple.values()))
        features.append(list(map(float, feature[:-2]))) # This is a hack to remove the last element of the featurevectors and convert
                                                        # the rest to floats
    return np.asarray(features)

if __name__ == "__main__":
    main()