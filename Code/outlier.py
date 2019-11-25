from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

anomalous = "Anomalous"
clean = "Clean"

def local_outlier_detection(training_vectors, test_vectors_clean, test_vectors_anomalous, plot=False):
    """Predicting outliers using Local Outlier Detection
    """
    print("**************************")
    print("Starting Local Outlier Fitting...")

    # Fitting model for novel predictions
    km = LocalOutlierFactor(novelty = True, contamination = "auto").fit(training_vectors)
    
    print("Fitting successful!")    
    print("**************************")
    print("Starting Prediction...")
    # Predict returns 1 for inlier and -1 for outlier
    result_clean = km.predict(test_vectors_clean)
    result_anomalous = km.predict(test_vectors_anomalous)
    
    print("Predicting successful!")    
    print("**************************")

    evaluate_detection(result_clean, result_anomalous)
    if plot:
        plot_clustering(split_anomalous_clean(test_vectors_anomalous, result_anomalous), training_vectors)

    return result_clean, result_anomalous

def one_class_svm(training_vectors, test_vectors_clean, test_vectors_anomalous, plot = False):
    """Predicting Outlier using a one Class SVM
    """
    print("**************************")
    print("Starting One Class SVM...")

    # Fitting model for novel predictions
    svm = OneClassSVM(gamma = "scale").fit(training_vectors)
    
    print("Fitting successful!")    
    print("**************************")
    print("Starting Prediction...")

    # Predict returns 1 for inlier and -1 for outlier
    result_clean = svm.predict(test_vectors_clean)
    result_anomalous = svm.predict(test_vectors_anomalous)
    
    print("Predicting successful!")    
    print("**************************")

    evaluate_detection(result_clean, result_anomalous)
    if plot:
        plot_clustering(split_anomalous_clean(test_vectors_anomalous, result_anomalous), training_vectors)

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

def plot_clustering(vectors_dict, training_vectors):
    """Plots a dictionary of clean and anomalous identified vectors over the training vector
    """
    fig, ax = plt.subplots()
    anomalous_vectors = vectors_dict[anomalous]
    clean_vectors = vectors_dict[clean]
    ax.scatter(training_vectors[:,0], training_vectors[:,1], s=100, color = "c", alpha = 0.5, label = "Training Datapoints")
    ax.scatter(anomalous_vectors[:,0], anomalous_vectors[:,1], s=20, color = "r", alpha = 0.5, label = "Anomalous Identified Datapoints")
    ax.scatter(clean_vectors[:,0], clean_vectors[:,1], s=20, color = "g", alpha = 0.5, label = "Clean Identified Datapoints")
    plt.xlabel("Probability of the Request")
    plt.ylabel("Number of N-Grams Occurences")
    plt.title("Clean and Anomalous Identified Datapoints")
    ax.legend()
    plt.show()

def evaluate_detection(result_clean, result_anomalous):
    """Evaluates the detection rate of a model and prints it
    """
    accuracy_anomalous = (float(np.count_nonzero(result_anomalous == -1))) / len(result_anomalous) * 100
    accuracy_clean = (float(np.count_nonzero(result_clean == 1))) / len(result_clean) * 100
    
    print("Results:")
    print("True Positive: %.2f %%" % accuracy_anomalous)
    print("False Positive: %.2f %%" % (100 - accuracy_clean))
    print("Accuracy: %.2f %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))
