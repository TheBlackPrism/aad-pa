import logfileparser as parser
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import outlier
import NGram
import URL_Length_Extraction

class Anomaly_Detection():
    """description of class"""
    def __init__(self):
        pass

    def reading_data_from_file(self,path):
        if(path == ''):
            print('Invalid path name. Please enter valid path...')
            return
        training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
        test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
        test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

        training_data = parser.append_parameter_to_request(training_data)
        test_clean = parser.append_parameter_to_request(test_clean)
        test_anomalous = parser.append_parameter_to_request(test_anomalous)

        return training_data, test_clean, test_anomalous

    def feature_extraction(self, extraction_name, trainging_data, test_data_clean, test_data_anomalous):

        if(extraction_name == 'ngram parameter'):
            print("**************************")
            print("Extracting N-Grams...")

            # Training the N-Gram extractor
            ng_parameter = NGram()
            ng_parameter.fit(training_data, False)

            # Getting Feature Vectors
            training_vectors_parameter, ngrams_training_parameter = ng_parameter.get_feature_vectors_multidimensional(trainging_data)
            test_vectors_clean_parameter, ngrams_clean_parameter = ng_parameter.get_feature_vectors_multidimensional(test_data_clean)
            test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_parameter.get_feature_vectors_multidimensional(test_data_anomalous)

            return training_vectors_parameter, test_vectors_clean_parameter, test_vectors_anomalous_parameter

        elif(extraction_name == 'ngram url'):

            ng_url = NGram()
            ng_url.fit(training_data, True)

            #Getting Feature Vectors
            training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(trainging_data)
            test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_data_clean)
            test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_data_anomalous)

            return training_vectors_url, test_vectors_clean_url, test_vectors_anomalous_url

        elif(extraction_name == 'url length'):
            ul = URL_Length_Extraction()
            training_vectors = ul.extract_feature(trainging_data)
            test_vectors_clean = ul.extract_feature(test_data_clean)
            test_vectors_anomalous = ul.extract_feature(test_data_anomalous)
            
            return training_vectors, test_vectors_clean, test_vectors_anomalous

        else:
            print('Extraction Method not found.')
            return 



def main():
    ad = Anomaly_Detection()
    print('Please enter the path of the logfiles...')
    path = str(input())
    print("**************************")
    print('Reading data...')

    # Reading Data
    training_data, test_clean, test_anomalous = ad.reading_data_from_file(path)

    print("**************************")
    print('Data read!')
    print('Please enter the feature extraction you would like to use...')

    feature_extraction = str(input())

    
        

    print('Please enter the algorithm you would like to use...')

    alg_name = str(input())

    if alg_name == 'ol':
        ol = outlier()
        ol.local_outlier_detection()


if __name__ == "__main__":
    main()


