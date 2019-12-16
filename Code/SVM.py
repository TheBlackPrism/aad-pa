from sklearn.svm import OneClassSVM

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

