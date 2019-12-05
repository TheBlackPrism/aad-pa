import logfileparser as parser
import numpy as np
import matplotlib as matplot
import matplotlib.pyplot as plt
import outlier
from NGram import *
from URL_Length_Extraction import *
from pathlib import Path
import DBSCAN
import os

class Anomaly_Detection():
    """description of class"""
    def __init__(self):
        pass

    def reading_data_from_file(self,path):
        
        if(path == ''):
            print('Invalid path name. Please enter valid path...')
            return
        training_data = parser.read_data(path / "normalTrafficTraining")
        test_clean = parser.read_data(path / "normalTrafficTest")
        test_anomalous = parser.read_data(path / "anomalousTrafficTest")

        training_data = parser.append_parameter_to_request(training_data, True)
        test_clean = parser.append_parameter_to_request(test_clean, True)
        test_anomalous = parser.append_parameter_to_request(test_anomalous, True)

        return training_data, test_clean, test_anomalous


    def apply_algorithm(self, alg_name, training_vectors, test_vectors_clean, test_vectors_anomalous):
        if(alg_name == 'lof'):
            result_clean, result_anomalous = outlier.local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        elif(alg_name == 'svm'):
            result_clean, result_anomalous = outlier.one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous)

            return result_clean, result_anomalous

        elif(alg_name == 'dbscan'):
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

def evaluate_detection(result_clean, result_anomalous):
    """Evaluates the detection rate of a model and prints it
    """
    accuracy_anomalous = (float(np.count_nonzero(result_anomalous == -1))) / len(result_anomalous) * 100
    accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean) * 100

    print("Anomalous Samples: %d" % len(result_anomalous))
    print("Clean Samples: %d\n" % len(result_clean))

    print("True Positive: %.2f%%" % accuracy_anomalous)
    print("False Positive: %.2f%%" % (100 - accuracy_clean))


def main():
    ad = Anomaly_Detection()
    print('Please enter name of the dataset...')
    path = '../Logfiles/'
    print(os.listdir(path))
    path += str(input('Logfiles/'))
    path = Path(path)

    print("**************************")
    print('Please enter the algorithm you would like to use...')
    print('lof = Local Outlier Detection\nsvm = One Class Support Vector Machine\ndbscan = DBSCAN')

    alg_name = str(input('Algorithm: ')).lower()

    print("**************************")
    print('Reading data...')

    # Reading Data
    training_data,test_clean,test_anomalous = ad.reading_data_from_file(path)

    print('Data read!')
    print('Starting feature extraction...')


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

    #1-Grams Extraction
    onegram_parameter = NGram(n=1)
    onegram_parameter.fit(training_data,False)

    onegram_url = NGram(n=1)
    onegram_url.fit(training_data,True)

    training_vectors_one_gram_paramteter, onegram_training_parameter = onegram_parameter.get_feature_vectors_multidimensional(training_data)
    test_vectors_one_gram_clean_parameter, onegram_clean_parameter = onegram_parameter.get_feature_vectors_multidimensional(test_clean)
    test_vectors_one_gram_anomalous_parameter, onegram_anomalous_parameter = onegram_parameter.get_feature_vectors_multidimensional(test_anomalous)

    training_vectors_one_gram_url, onegram_training_url = onegram_url.get_feature_vectors_multidimensional(training_data)
    test_vectors_one_gram_clean_url, onegram_clean_url = onegram_url.get_feature_vectors_multidimensional(test_clean)
    test_vectors_one_gram_anomalous_url, onegram_anomalous_url = onegram_url.get_feature_vectors_multidimensional(test_anomalous)

    print('Feature extraction successful!')
    print("**************************")
    print("Analysing URL N-Grams:")

    # Workaround that prevents svm applied to url as features are not spread in the url and points on the same place 
    # cannot form a cluster in svm
    if alg_name == 'svm':
        url_alg = 'lof'
    else:
        url_alg = alg_name

    result_clean_ng_url,result_anomalous_ng_url = ad.apply_algorithm(url_alg,training_vectors_url,test_vectors_clean_url,test_vectors_anomalous_url)
    
    
    print("Analysing Parameter N-Grams:")
    result_clean_ng_param,result_anomalous_ng_param = ad.apply_algorithm(alg_name,training_vectors_parameter,test_vectors_clean_parameter,test_vectors_anomalous_parameter)

    result_clean_ng = ad.merge_results(result_clean_ng_param,result_clean_ng_url)
    result_anomalous_ng = ad.merge_results(result_anomalous_ng_param,result_anomalous_ng_url)

    print("Analysing Parameter 1-Grams:")
    result_clean_onegram_param,result_anomalous_onegram_param = ad.apply_algorithm(alg_name,training_vectors_one_gram_paramteter,test_vectors_one_gram_clean_parameter,test_vectors_one_gram_anomalous_parameter)
    result_clean_onegram_url,result_anomalous_onegram_url = ad.apply_algorithm(alg_name,training_vectors_one_gram_url,test_vectors_one_gram_clean_url,test_vectors_one_gram_anomalous_url)

    result_clean_onegram = ad.merge_results(result_clean_onegram_param,result_clean_onegram_url)
    result_anomalous_onegram = ad.merge_results(result_anomalous_onegram_param,result_anomalous_onegram_url)
    
    print("Analysing URL Length:")
    result_clean_url_length,result_anomalous_url_length = ad.apply_algorithm(alg_name,training_vectors_url_length,test_vectors_clean_url_length,test_vectors_anomalous_url_length)

    result_clean_onegram_ngram = ad.merge_results(result_clean_onegram,result_clean_ng)
    result_anomalous_onegram_ngram = ad.merge_results(result_anomalous_onegram, result_anomalous_ng)

    result_overall_clean = ad.merge_results(result_clean_onegram_ngram, result_clean_url_length)
    result_overall_anomalous = ad.merge_results(result_anomalous_onegram_ngram,result_anomalous_url_length)


    print('Starting evaluation...')

    #Evaluate N-Grams
    print("\nN-Gram Evaluation")
    evaluate_detection(result_clean_ng, result_anomalous_ng)

    #Evaluate 1-Grams
    print("\n1-Gram Evaluation")
    evaluate_detection(result_clean_onegram, result_anomalous_onegram)

    #Evaluate URL-Length
    print("\nURL Length Evaluation")
    evaluate_detection(result_clean_url_length, result_anomalous_url_length)

    
    print("\n**************************")
    #overall evaluation
    print("Overall Evaluation " + alg_name.upper() + "\n")
    evaluate_detection(result_overall_clean, result_overall_anomalous)
    print()

if __name__ == "__main__":
    main()


