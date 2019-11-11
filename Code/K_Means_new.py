import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
from NGram import *
from collections import Counter, defaultdict


#def main():
# Reading Data
training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

# Training the N-Gramm extractor
ng = NGram()
ng.fit(training_data)
    
print("N-Gramms extracted!")
print("**************************")
print("Starting K-Means Fitting...")




# Getting Feature Vectors
training_vectors = ng.get_feature_vectors(training_data)
test_vectors_clean = ng.get_feature_vectors(test_clean)
test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)

#for i in range(1,21):

print("\n**************************")
print("Training model:")
print("k = 3")

"""Initialize K-Means, fit training data and predict which cluster it belongs to
"""
kmeans = KMeans(n_clusters = 3, init='random', 
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
                )
clusters = kmeans.fit_predict(training_vectors)
centroids = kmeans.cluster_centers_
np.savetxt('centroids.txt',centroids)
labels = kmeans.labels_
clusters_radii = dict()
clusters_datapoints = defaultdict(list)

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

print(len(cluster0))
print(len(cluster1))
print(len(cluster2))

        

"""get a dictionary with the cluster index as key and the radiuses for clusters
as value
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
    if val > maxmax_val:
        max_val = val
clusters_radii[1] = max_val

centroid = centroids[2]
max_val = 0
for i in range(len(cluster2)):
    val = np.linalg.norm(centroid - cluster2[i])
    if val > max_val:
        max_val = val
clusters_radii[2] = max_val

print("radii")
print(clusters_radii)
    

for cluster in cluster_centroids:
    max_val = 0
    for i in zip(training_vectors[clusters==cluster,0], training_vectors[clusters == cluster,1]):
        c = cluster_centroids[cluster]
        #for i in range
        val =np.linalg.norm(np.subtract(i,c))
        if val > max_val:
            max_val = val
    clusters_radii[cluster] = max_val


    

            
#test clean data
print("Training done! Switch to testing.")
print("**************************")
print("Testing normal traffic:")

"""To see if the test data belongs to one of the obtained clusters we test,
if the distance between the centroid and the datapoint is larger than the radius for said cluster.
"""

malicious_query_in_clean_test_vector = [] #would be bad. The vectors should belong to a cluster
could_be_assigned_clean_test_vector = [] #would be good. The vectors should belong to a cluster
for cluster in clusters_radii:
    r = clusters_radii[cluster]
    c = clusters_datapoints[cluster]
    
    for i in range(len(test_vector_clean)):
        dist = 0
        dist = c - test_vector_clean[i]
        if dist > r:
            malicious_query_in_clean_test_vector.append(test_vector_clean[i])
        else:
            could_be_assigned_clean_test_vector.append(test_vector_clean[i])

malicious_query_in_anomalous_test_vector = [] #would be good. The vectors shouldn't belong to any cluster
could_be_assigned_anomalous_test_vector = [] #would be bad. The vectors shouldn't belong to any cluster
for cluster in clusters_radii:
    r = clusters_radii[cluster]
    c = clusters_datapoints

    for i in range(len(test_vectors_anomalous)):
        dist = 0
        dist = c - test_vectors_anomalous[i]
        if dist > r:
            malicious_query_in_anomalous_test_vector.append(test_vectors_anomalous[i])
        else:
            could_be_assigned_anomalous_test_vector.append(test_vectors_anomalous[i])

            



print("Predicting successful!")    
print("**************************")
print("Results:")

# Evaluation
accuracy_anomalous = np.count_nonzero(malicious_query_in_anomalous_test_vector == -1) / len(test_vectors_anomalous) * 100
accuracy_clean = np.count_nonzero(could_be_assigned_clean_test_vector == 1) / len(test_vectors_clean) * 100

print("True Positiv: %d %%" % accuracy_anomalous)
print("False Positiv: %d %%" % (100 - accuracy_clean))
print("Accuracy: %d %%" % ((accuracy_anomalous * len(result_anomalous) + accuracy_clean * len(result_clean)) / (len(result_clean) + len(result_anomalous))))
    
# Plotting Vectors
#fig, ax = plt.subplot(2,1,1)
plt.subplot(2,1,1)
samples = 300
plt.scatter(training_vectors[:samples,0], training_vectors[:samples,1], s=200,color = "g", alpha = 0.5, label = "Trainings Data")
plt.scatter(test_vectors_clean[:samples,0], test_vectors_clean[:samples,1], s=150, color = "b", alpha = 0.5, label = "Clean Data")
plt.scatter(test_vectors_anomalous[:samples,0], test_vectors_anomalous[:samples,1], s=100, color = "r", alpha = 0.5, label = "Anomalous Data")
plt.xlim(0.02,0.1)
plt.ylim(0, 500)
plt.title("Distribution of Feature Vectors")
plt.legend()
plt.grid()
#plt.show()

"""Visualize clusters
"""
plt.subplot(2,1,2)
plt. scatter(
    training_vectors[clusters==0,0], training_vectors[clusters==0,1],
    s = 50, c = 'green',
    marker= 's', edgecolor= 'black',
    label = 'cluster 1'
    )

plt.scatter(
    training_vectors[clusters==1,0], training_vectors[clusters == 1,1],
    s = 50, c ='orange',
    marker = 'o', edgecolor='black',
    label = 'cluster2'
    )

plt.scatter(
    training_vectors[clusters == 2,0], training_vectors[clusters==2,1],
    s = 50, c='blue',
    marker = 'v', edgecolor='black',
    label = 'cluster 3'
    )

"""plot the centroids
"""
plt.scatter(
    kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
    s = 250, marker = '*',
    c = 'red', edgecolor='black',
    label='centroids'
    )

plt.xlim(0.02,0.1)
plt.ylim(0, 500)
plt.legend(scatterpoints=1)
plt.xlabel("Probability of the Request")
plt.ylabel("Number of N-Gramms Occurences")
plt.title("Visualisation of clusters")
plt.grid()
plt.show()