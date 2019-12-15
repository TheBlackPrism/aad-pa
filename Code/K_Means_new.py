import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
from NGram import *
from URL_Length_Extraction import *

def k_means(training_vectors,test_vectors_clean, test_vectors_anomalous):

    kmeans = KMeans(n_clusters = 3, init='random', 
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
                )
    clusters = kmeans.fit_predict(training_vectors)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    clusters_radii = dict()

    """sort each datapoint to its corresponding cluster
    """

    cluster0 = []
    cluster1 = []
    cluster2 = []
    for j in range(len(labels)):
        if(labels[j] == 0.0):
            cluster0.append(training_vectors[j])
        elif(labels[j] == 1.0):
            cluster1.append(training_vectors[j])
        elif(labels[j]== 2.0):
            cluster2.append(training_vectors[j])



        

    """get the radius for each cluster.
    In lack of a better method just compute the datapoint farthest away from the centroid and take the
    distance as radius.
    """

    centroid = centroids[0]
    max_val = 0
    for i in range(len(cluster0)):
        val = np.linalg.norm(centroid-cluster0[i])
        if val > max_val:
            max_val = val
    clusters_radii[0] = max_val

    centroid = centroids[1]
    max_val = 0
    for i in range(len(cluster1)):
        val = np.linalg.norm(centroid-cluster1[i])
        if val > max_val:
            max_val = val
    clusters_radii[1] = max_val

    centroid = centroids[2]
    max_val = 0
    for i in range(len(cluster2)):
        val = np.linalg.norm(centroid - cluster2[i])
        if val > max_val:
            max_val = val
    clusters_radii[2] = max_val

            
    #test clean data
    print("Training done! Switch to testing.")
    print("**************************")
    print("Start prediction...")

    """To see if the test data belongs to one of the obtained clusters we test,
    if the distance between the centroid and the datapoint is larger than the radius for said cluster.
    """

    #could_not_be_assigned_clean_test_vector = [] #would be bad. The vectors should belong to a cluster
    #could_be_assigned_clean_test_vector = [] #would be good. The vectors should belong to a cluster
    result_clean = []    
    result_training = []

    r0 = clusters_radii[0]
    c0 = centroids[0]


    r1 = clusters_radii[1]
    c1 = centroids[1]


    r2 = clusters_radii[2]
    c2 = centroids[2]



    dist0 = 0
    dist1 = 0
    dist2 = 0
    for i in range(len(test_vectors_clean)):

        dist0 = np.linalg.norm(c0-test_vectors_clean[i])
        dist1 = np.linalg.norm(c1-test_vectors_clean[i])
        dist2 = np.linalg.norm(c2-test_vectors_clean[i])
        if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
            result_clean.append(1)
            #could_be_assigned_clean_test_vector.append(test_vectors_clean[i])
        else:
            #result_clean.append(test_vectors_clean[i])
            result_clean.append(-1)
            

   
        dist0 = 0
        dist1 = 0
        dist2 = 0

    dist0 = 0
    dist1 = 0
    dist2 = 0
    for i in range(len(training_vectors)):

        dist0 = np.linalg.norm(c0-training_vectors[i])
        dist1 = np.linalg.norm(c1-training_vectors[i])
        dist2 = np.linalg.norm(c2-training_vectors[i])
        if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
            result_training.append(1)
        else:
            #result_clean.append(test_vectors_clean[i])
            result_training.append(-1)
            

   
        dist0 = 0
        dist1 = 0
        dist2 = 0

    #detected_anomalies_in_anomalous_test_vector = [] #would be good. The vectors shouldn't belong to any cluster
    #undetected_anomalies_in_anomalous_test_vector = [] #would be bad. The vectors shouldn't belong to any cluster
    result_anomalous =[]


    dist0 = 0
    dist1 = 0
    dist2 = 0
    for i in range(len(test_vectors_anomalous)):
        dist0 = np.linalg.norm(c0 - test_vectors_anomalous[i])
        dist1 = np.linalg.norm(c1 - test_vectors_anomalous[i])
        dist2 = np.linalg.norm(c2 - test_vectors_anomalous[i])
    

        if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
            #undetected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous[i])
            result_anomalous.append(1)
        else:
            #result_anomalous.append(test_vectors_anomalous[i]) 
            result_anomalous.append(-1)

        dist0 = 0
        dist1 = 0
        dist2 = 0



    print("Predicting successful!")    
    print("**************************")
    
    return np.asarray(result_clean), np.asarray(result_anomalous), np.asarray(result_training)
    
