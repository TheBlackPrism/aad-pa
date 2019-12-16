from sklearn.neighbors import LocalOutlierFactor

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
