import logfileparser as parser
import numpy as np
import NGram
import outlier
import K_Means_new
import matplotlib as matplot
import matplotlib.pyplot as plt

class Anomaly_Detection(object):
    """description of class"""
    def __init__(self):
        pass

    def main():
        # Reading Data
        training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
        test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
        test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

        training_data = parser.append_parameter_to_request(training_data)
        test_clean = parser.append_parameter_to_request(test_clean)
        test_anomalous = parser.append_parameter_to_request(test_anomalous)
        print("**************************")

if __name__ == "__main__":
    main()


