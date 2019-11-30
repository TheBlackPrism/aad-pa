import logfileparser as parser
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
from outlier import *
from NGram import *
from URL_Length_Extraction import *

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

    def feature_extraction(self, extraction_name, training_data, test_data_clean, test_data_anomalous):

        if(extraction_name == 'ngram parameter'):
            print("**************************")
            print("Extracting N-Grams...")

            # Training the N-Gram extractor
            ng_parameter = NGram()
            ng_parameter.fit(training_data, False)

            # Getting Feature Vectors
            training_vectors_parameter, ngrams_training_parameter = ng_parameter.get_feature_vectors_multidimensional(training_data)
            test_vectors_clean_parameter, ngrams_clean_parameter = ng_parameter.get_feature_vectors_multidimensional(test_data_clean)
            test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_parameter.get_feature_vectors_multidimensional(test_data_anomalous)

            return training_vectors_parameter, test_vectors_clean_parameter, test_vectors_anomalous_parameter

        elif(extraction_name == 'ngram url'):

            ng_url = NGram()
            ng_url.fit(training_data, True)

            #Getting Feature Vectors
            training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(training_data)
            test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_data_clean)
            test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_data_anomalous)

            return training_vectors_url,test_vectors_clean_url,test_vectors_anomalous_url

        elif(extraction_name == 'url length'):
            ul = URL_Length_Extraction()
            training_vectors = ul.extract_feature(training_data)
            test_vectors_clean = ul.extract_feature(test_data_clean)
            test_vectors_anomalous = ul.extract_feature(test_data_anomalous)
            
            return training_vectors,test_vectors_clean,test_vectors_anomalous

        else:
            print('Extraction Method not found.')
            return 

    def apply_algorithm(self, alg_name, training_vectors, test_vectors_clean, test_vectors_anomalous):
        if(alg_name == 'ol'):
            ol = outlier()
            result_clean, result_anomalous = ol.local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        elif(alg_name == 'svm'):
            svm = outlier()
            result_clean, result_anomalous = ol.one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        else:
            print('Algorithm Name not found.')
            return
           





def main():
    ad = Anomaly_Detection()
    print('Please enter the path of the logfiles...')
    path = str(input())
    print("**************************")
    print('Reading data...')

    # Reading Data
    training_data,test_clean,test_anomalous = ad.reading_data_from_file(path)

    print("**************************")
    print('Data read!')
    print('Please enter the feature extraction you would like to use...')
    print('ngram url = N-Grams using the URL\n ngram parameter = N-Grams using the parameter values\n url length = Length of the URLs')


    feature_extraction = str(input())
    training_vectors,test_vectors_clean,test_vectors_anomalous = ad.feature_extraction(feature_extraction, training_data, test_clean, test_anomalous)

    print("**************************")
    print('Feature Extracted!')
    print('Please enter the algorithm you would like to use...')
    print('ol = Local Outlier Detection\n svm = One Class Support Vector Maching\n')

    alg_name = str(input())

    result_clean,result_anomalous = ad.apply_algorithm(alg_name, training_vectors, test_vectors_clean, test_vectors_anomalous)

    print("**************************")
    print('Done!')
    print('Starting evaluation...')

    accuracy_anomalous = (float(np.count_nonzero(result_anomalous == -1))) / len(result_anomalous) * 100
    accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean) * 100

    print("\nEvaluation:")
    print("\nTrue Positive: %.4f %%" % accuracy_anomalous)
    print("\nFalse Positive: %.4f %%" % (100 - accuracy_clean))
    print("\nAccuracy: %.4f %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))




    


if __name__ == "__main__":
    main()


