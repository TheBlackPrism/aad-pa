import numpy as np
from sklearn.cluster import KMeans

def k_means(training_vectors,test_vectors_clean, test_vectors_anomalous, k = 10):
    kmeans = KMeans(n_clusters = k, init='k-means++', 
                n_init=100, max_iter=300, 
                tol=1e-04, random_state=0
                )

    print("Fitting with Parameters: ", kmeans.get_params())
    labels = kmeans.fit_predict(training_vectors)
    centroids = kmeans.cluster_centers_

    clusters = __sort_to_cluster(k, labels, training_vectors)
    clusters_radii = __get_radius_for_clusters(k, clusters, centroids)
            
    print("Training done! Switch to testing.")
    print("**************************")
    print("Start prediction...")

    result_clean = __predict_outliers(k, clusters_radii, centroids, test_vectors_clean)
    result_anomalous = __predict_outliers(k, clusters_radii, centroids, test_vectors_anomalous)

    print("Predicting successful!")    
    print("**************************")
    
    return np.asarray(result_clean), np.asarray(result_anomalous), np.asarray([])
    
def __sort_to_cluster(k, labels, vectors):
    """sort each datapoint to its corresponding cluster
    """
    clusters = [[] for i in range(k)]
    for i in range(len(labels)):
        clusters[labels[i]].append(vectors[i])
    return clusters

def __get_radius_for_clusters(k, clusters, centroids):
    """get the radius for each cluster.
    In lack of a better method just compute the datapoint farthest away from the centroid and take the
    distance as radius.
    """
    clusters_radii = []

    for i in range(k):
        centroid = centroids[i]
        cluster = clusters[i]
        max_val = 0

        for j in range(len(cluster)):
            val = np.linalg.norm(centroid-cluster[j])
            if val > max_val:
                max_val = val

        clusters_radii.append(max_val)

    return clusters_radii

def __predict_outliers(k, clusters_radii, centroids, vectors):
    """To see if the test data belongs to one of the obtained clusters we test,
    if the distance between the centroid and the datapoint is larger than the radius for said cluster.
    """
    distances = np.zeros(k)
    result = []
    
    for i in range(len(vectors)):
        distances = np.zeros(k)
        for j in range(k):
            distances[j] = np.linalg.norm(centroids[j] - vectors[i])

        if any(r >= d for r, d in zip(clusters_radii, distances)):
            result.append(1)
        else:
            result.append(-1)
    return result

