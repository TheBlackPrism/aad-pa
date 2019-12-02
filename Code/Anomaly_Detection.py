import logfileparser as parser
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import outlier
from NGram import *
from URL_Length_Extraction import *
from pathlib import Path
import DBSCAN

class Anomaly_Detection():
    """description of class"""
    def __init__(self):
        pass

    def reading_data_from_file(self,path):
        
        if(path == ''):
            print('Invalid path name. Please enter valid path...')
            return
        training_data = parser.read_data(path / "normalTrafficTraining.txt")
        test_clean = parser.read_data( path / "normalTrafficTest.txt")
        test_anomalous = parser.read_data(path / "anomalousTrafficTest.txt")

        training_data = parser.append_parameter_to_request(training_data)
        test_clean = parser.append_parameter_to_request(test_clean)
        test_anomalous = parser.append_parameter_to_request(test_anomalous)

        return training_data, test_clean, test_anomalous


    def apply_algorithm(self, alg_name, training_vectors, test_vectors_clean, test_vectors_anomalous):
        if(alg_name == 'lof'):
            result_clean, result_anomalous = outlier.local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        elif(alg_name == 'svm'):
            result_clean, result_anomalous = outlier.one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        elif(alg_name =='dbscan'):
            result_clean,result_anomalous = DBSCAN.dbscan(training_vectors,test_vectors_clean,test_vectors_anomalous)

            return result_clean,result_anomalous

        else:
            print('Algorithm Name not found.')
            return

    def merge_results(self,list1, list2):
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
    ad = Anomaly_Detection()
    print('Please enter the path of the logfiles...')
    path = str(input())
    path = Path(path)
    print("**************************")
    print('Reading data...')

    # Reading Data
    training_data,test_clean,test_anomalous = ad.reading_data_from_file(path)

    print("**************************")
    print('Data read!')
    print('Starting feature extraction...')
    print("**************************")


    # Training the N-Gram extractor
    ng_parameter = NGram()
    ng_parameter.fit(training_data, False)

    ng_url = NGram()
    ng_url.fit(training_data, True)

    # Getting Feature Vectors
    training_vectors_parameter, ngrams_training_parameter = ng_parameter.get_feature_vectors_multidimensional(training_data)
    test_vectors_clean_parameter, ngrams_clean_parameter = ng_parameter.get_feature_vectors_multidimensional(test_clean)
    test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_parameter.get_feature_vectors_multidimensional(test_anomalous)

    training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(training_data)
    test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_clean)
    test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_anomalous)

    #URL Length Extraction
    ul = URL_Length_Extraction()
    training_vectors_url_length = ul.extract_feature(training_data)
    test_vectors_clean_url_length = ul.extract_feature(test_clean)
    test_vectors_anomalous_url_length = ul.extract_feature(test_anomalous)



    print("**************************")
    print('Feature extraction successful!')
    print('Please enter the algorithm you would like to use...')
    print('lof = Local Outlier Detection\n svm = One Class Support Vector Maching\n dbscan = DBSCAN\n')

    alg_name = str(input())

    result_clean_ng_url,result_anomalous_ng_url = ad.apply_algorithm(alg_name,training_vectors_url,test_vectors_clean_url,test_vectors_anomalous_url)
    result_clean_ng_param,result_anomalous_ng_param = ad.apply_algorithm(alg_name,training_vectors_parameter,test_vectors_clean_parameter,test_vectors_anomalous_parameter)

    result_clean_ng = ad.merge_results(result_clean_ng_param,result_clean_ng_url)
    result_anomalous_ng = ad.merge_results(result_anomalous_ng_param,result_anomalous_ng_url)

    result_clean_url_length,result_anomalous_url_length = ad.apply_algorithm(alg_name,training_vectors_url_length,test_vectors_clean_url_length,test_vectors_anomalous_url_length)
    

    print("**************************")
    print('Done!')
    print('Starting evaluation...')

    #Evaluate N-Grams
    accuracy_anomalous_ng = (float(np.count_nonzero(result_anomalous_ng == -1))) / len(result_anomalous_ng) * 100
    accuracy_clean_ng = (float(np.count_nonzero(result_clean_ng == 1))) / len(result_clean_ng) * 100

    print("\nEvaluation using N-Grams:")
    print("\nTrue Positive: %.4f %%" % accuracy_anomalous_ng)
    print("\nFalse Positive: %.4f %%" % (100 - accuracy_clean_ng))
    print("\nAccuracy: %.4f %%" % ((accuracy_anomalous_ng * len(result_anomalous_ng) + accuracy_clean_ng * len(result_clean_ng)) / (len(result_clean_ng) + len(result_anomalous_ng))))


    #Evaluate URL-Length
    accuracy_anomalous_url_length = (float(np.count_nonzero(result_anomalous_url_length == -1))) / len(result_anomalous_url_length) * 100
    accuracy_clean_url_length = (float(np.count_nonzero(result_clean_url_length == 1))) / len(result_clean_url_length) * 100

    print("\nEvaluation using URL-Length:")
    print("\nTrue Positive: %.4f %%" % accuracy_anomalous_url_length)
    print("\nFalse Positive: %.4f %%" % (100 - accuracy_clean_url_length))
    print("\nAccuracy: %.4f %%" % ((accuracy_anomalous_url_length * len(result_anomalous_url_length) + accuracy_clean_url_length * len(result_clean_url_length)) / (len(result_clean_url_length) + len(result_anomalous_url_length))))



if __name__ == "__main__":
    main()


