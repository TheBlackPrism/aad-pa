import logfileparser as parser
import numpy as np
import NGram
import outlier
import K_Means_new
import matplotlib as matplot
import matplotlib.pyplot as plt

class Anomaly_Detection():
    """description of class"""
    def __init__(self):
        pass

    def reading_data_from_file(self,path):
        if(path == null):
            print('Invalid path name. Please enter valid path...')
            return
        training_data = parser.read_data(path + '/normalTrafficTraining.txt')
        test_clean = parser.read_data(path + '/normalTraffic.txt')
        test_anomalous = parser.read_data(path + '/anomalousTrafficTest.txt')

        training_data = parser.append_parameter_to_request(training_data)
        test_clean = parser.append_parameter_to_request(test_clean)
        test_anomalous = parser.append_parameter_to_request(test_anomalous)

        return training_data, test_clean, test_anomalous

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
        print('Please enter the algorithm you would like to use...')

if __name__ == "__main__":
    main()


