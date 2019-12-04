from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

anomalous = "Anomalous"
clean = "Clean"

def local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous, contamination='auto', n_neighbors=20):
    """Predicting outliers using Local Outlier Detection
    """
    print("Starting Local Outlier Fitting...")

    # Fitting model for novel predictions
    km = LocalOutlierFactor(novelty = True, contamination = contamination, n_neighbors = n_neighbors).fit(training_vectors)
    
    print("Fitting successful!")
    print("Starting Prediction...")

    # Predict returns 1 for inlier and -1 for outlier
    result_clean = km.predict(test_vectors_clean)
    result_anomalous = km.predict(test_vectors_anomalous)
    
    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous

def one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous, gamma = 'scale', nu = 0.5):
    """Predicting Outlier using a one Class SVM
    """
    print("Starting One Class SVM...")

    # Fitting model for novel predictions
    svm = OneClassSVM(gamma = gamma, nu = nu).fit(training_vectors)
    
    print("Fitting successful!")    
    print("Starting Prediction...")

    # Predict returns 1 for inlier and -1 for outlier
    result_clean = svm.predict(test_vectors_clean)
    result_anomalous = svm.predict(test_vectors_anomalous)
    
    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous

def split_anomalous_clean(test_vectors, result):
    """Splits anomalous and clean identified logs into the according dictionaries
    """
    dict = {}
    list_clean = []
    list_anomalous = []
    
    for i in range(len(test_vectors)):
        if result[i] == 1:  
            list_clean.append(test_vectors[i])
        else:
            list_anomalous.append(test_vectors[i])
    dict[clean] = np.asarray(list_clean)
    dict[anomalous] = np.asarray(list_anomalous)
    return dict