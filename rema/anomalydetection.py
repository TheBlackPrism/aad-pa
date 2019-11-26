"""
Takes three input files consisting of feature vectors of normal training data,
normal test data, and anomalous test data. Trains an ML model using unsupervised
learning and returns performance data of the model with respect to anomaly 
sdetection. The parameter standard/minmax/none defines the scaler to be used 
and ee|ocs|is|lof|dbs allow to choose the ML algorithm.
"""

import sys  
import os
import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import preprocessing

def main():
    if (len(sys.argv) != 6):
        usage()
        sys.exit()
    in_filename_normal_training = sys.argv[1]
    in_filename_normal_test = sys.argv[2]
    in_filename_anomalous_test = sys.argv[3]
    scaler_type = sys.argv[4]
    model_type = sys.argv[5]
    if not os.path.isfile(in_filename_normal_training):
        print("File {} does not exist. Exiting...".format(in_filename_normal_training))
        sys.exit()
    if not os.path.isfile(in_filename_normal_test):
        print("File {} does not exist. Exiting...".format(in_filename_normal_test))
        sys.exit()
    if not os.path.isfile(in_filename_anomalous_test):
        print("File {} does not exist. Exiting...".format(in_filename_anomalous_test))
        sys.exit()

    ## Read normal training data, train model and show results
    df = pd.read_csv(in_filename_normal_training, encoding="ISO-8859-1")
    print("Shape of normal training data dataframe: " + str(df.shape))
    X_normal_train = df.iloc[:,0:len(df.columns)-2]
    if (scaler_type == "standard"):
        scaler = preprocessing.StandardScaler().fit(X_normal_train)
    elif (scaler_type == "minmax"):
        scaler = preprocessing.MinMaxScaler().fit(X_normal_train)
    X_normal_train_norm = X_normal_train
    if (scaler_type != "none"):
        X_normal_train_norm = scaler.transform(X_normal_train)
    if (model_type == "ee"):
        model = EllipticEnvelope(contamination=0.01)
    elif (model_type == "ocs"):
        model = OneClassSVM(gamma='scale', nu=0.01)
    elif (model_type == "if"):
        model = IsolationForest(contamination=0.01, behaviour="new")
    elif (model_type == "lof"):
        model = LocalOutlierFactor(novelty=True, n_neighbors=100, contamination=0.01)
    elif (model_type == "dbs"):
        model = DBSCAN(eps=0.3)
    elif (model_type == "km"):
        model = KMeans()
    model.fit(X_normal_train_norm)
    if (model_type == "ee" or model_type == "ocs" or model_type == "if" or model_type == "lof" or model_type == "km"):
        ## predict returns 1 if it's an inlier (normal) and -1 if it's an outlier (anomalous)
        y_normal_train_pred = model.predict(X_normal_train_norm)
    elif (model_type == "dbs"):
        y_normal_train_pred = dbscan_predict(model, X_normal_train_norm)
    num_errors = sum(y_normal_train_pred == -1)
    print('Number of errors in normal training data: {} of {}, {:.3f}'.
    format(num_errors, len(y_normal_train_pred), num_errors/len(y_normal_train_pred)))
    num_correct = sum(y_normal_train_pred != -1)
    print('Number of correctly predicted training data: {} of {}, {:.3f}'.
    format(num_correct, len(y_normal_train_pred), num_correct/len(y_normal_train_pred)))

    ## Read normal test data and show results
    df = pd.read_csv(in_filename_normal_test, encoding="ISO-8859-1")
    print("Shape of normal test data dataframe: " + str(df.shape))
    X_normal_test = df.iloc[:,0:len(df.columns)-2]
    X_normal_test_norm = X_normal_test
    if (scaler_type != "none"):
       X_normal_test_norm = scaler.transform(X_normal_test)
    if (model_type == "ee" or model_type == "ocs" or model_type == "if" or model_type == "lof" or model_type == "km"):
        ## predict returns 1 if it's an inlier (normal) and -1 if it's an outlier (anomalous)
        y_normal_test_pred = model.predict(X_normal_test_norm)
    elif (model_type == "dbs"):
        y_normal_test_pred = dbscan_predict(model, X_normal_test_norm)
    num_errors = sum(y_normal_test_pred == -1)
    print('Number of errors in normal test data: {} of {}, {:.3f}'.
    format(num_errors, len(y_normal_test_pred), num_errors/len(y_normal_test_pred)))
    num_correct = sum(y_normal_test_pred != -1)
    print('Number of correctly predicted normal test data: {} of {}, {:.3f}'.
    format(num_correct, len(y_normal_test_pred), num_correct/len(y_normal_test_pred)))

    ## Read anomalous test data and show results
    df = pd.read_csv(in_filename_anomalous_test, encoding="ISO-8859-1")
    print("Shape of anomalous test data dataframe: " + str(df.shape))
    X_anomalous_test = df.iloc[:,0:len(df.columns)-2]
    X_anomalous_test_norm = X_anomalous_test
    if (scaler_type != "none"):
       X_anomalous_test_norm = scaler.transform(X_anomalous_test)
    if (model_type == "ee" or model_type == "ocs" or model_type == "if" or model_type == "lof" or model_type == "km"):
        ## predict returns 1 if it's an inlier (normal) and -1 if it's an outlier (anomalous)
        y_anomalous_test_pred = model.predict(X_anomalous_test_norm)
    elif (model_type == "dbs"):
        y_anomalous_test_pred = dbscan_predict(model, X_anomalous_test_norm)
    num_errors = sum(y_anomalous_test_pred != -1)
    print('Number of errors in anomalous test data: {} of {}, {:.3f}'.
    format(num_errors, len(y_anomalous_test_pred), num_errors/len(y_anomalous_test_pred)))
    num_correct = sum(y_anomalous_test_pred == -1)
    print('Number of correctly predicted anomalous test data: {} of {}, {:.3f}'.
    format(num_correct, len(y_anomalous_test_pred), num_correct/len(y_anomalous_test_pred)))

def dbscan_predict(model, X):
    ## Found here: https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan
    nr_samples = X.shape[0]
    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)
        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
    return y_new

def usage():
    print("Usage: python clusterer.py in-file-normal-training in-file-normal-test in-file-anomalous-test standard|minmax|none ee|ocs|is|lof|dbs")

if __name__ == '__main__':  
   main()