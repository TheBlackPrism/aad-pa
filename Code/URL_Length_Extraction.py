import logfileparser as parser
import numpy as np
import outlier


class URL_Length_Extraction(object):
    """description of class"""

    def __init__(self):
        pass

    def get_urls(self, parsed_dataset):
        url_list = []

        for i in range(len(parsed_dataset)):
            url = parsed_dataset[i].get('Request')
            url_list.append(url)

        return url_list

    def get_url_lengths(self, url_list):
        url_lengths = []

        for i in range(len(url_list)):
            length = len(url_list[i])
            url_lengths.append(length)

        return url_lengths

    def build_feature_vector(self, url_lengths):
        feature_vector = []
        

        for i in range(len(url_lengths)):
            lengths = []
            lengths.append(url_lengths[i])
            feature_vector.append(lengths.copy())
            lengths.clear()

        return feature_vector

    def extract_feature(self,data):
        data_urls = self.get_urls(data)
        url_lengths = self.get_url_lengths(data_urls)
        feature_vector = self.build_feature_vector(url_lengths)

        return feature_vector

        





               




def main():


    print("**************************")
    print("Reading data...")

    # Reading Data
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
    test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

    training_data = parser.append_parameter_to_request(training_data)
    test_clean = parser.append_parameter_to_request(test_clean)
    test_anomalous = parser.append_parameter_to_request(test_anomalous)
    print("**************************")
    print("Extracting URL Length...")
    
    urlLength = URL_Length_Extraction()
    training_vectors = urlLength.extract_feature(training_data)
    test_vectors_clean = urlLength.extract_feature(test_clean)
    test_vectors_anomalous = urlLength.extract_feature(test_anomalous)



    
    outlier.local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous)
    outlier.one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous)
    
    
    print("Done.")
    print("**************************")


if __name__ == "__main__":
    main()








