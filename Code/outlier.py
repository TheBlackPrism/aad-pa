from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

anomalous = "Anomalous"
clean = "Clean"

def local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous):
    """Predicting outliers using Local Outlier Detection
    """
    print("Starting Local Outlier Fitting...")

    # Fitting model for novel predictions
    lof = LocalOutlierFactor(novelty = True, contamination = 'auto', algorithm = 'auto', n_neighbors = 20, n_jobs = -1)
    print("Fitting with Parameters: ", lof.get_params())
    lof.fit(training_vectors)
    result_training = lof.predict(training_vectors)

    print("Fitting successful!")
    print("Starting Prediction...")
    # Predict returns 1 for inlier and -1 for outlier
    result_clean = lof.predict(test_vectors_clean)
    result_anomalous = lof.predict(test_vectors_anomalous)
    
    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous, result_training

def one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous):
    """Predicting Outlier using a one Class SVM
    """
    print("Starting One Class SVM...")

    # Fitting model for novel predictions
    svm = OneClassSVM(gamma = 'auto', kernel = 'rbf', nu = 0.05)
    print("Fitting with Parameters: ", svm.get_params())
    result_training = svm.fit_predict(training_vectors)

    print("Fitting successful!")    
    print("Starting Prediction...")

    # Predict returns 1 for inlier and -1 for outlier
    result_clean = svm.predict(test_vectors_clean)
    result_anomalous = svm.predict(test_vectors_anomalous)

    print("Predicting successful!")    
    print("**************************")

    return result_clean, result_anomalous, result_training

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
