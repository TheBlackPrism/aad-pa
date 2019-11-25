import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
from NGram import *
#from collections import Counter, defaultdict


# Reading Data
training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

# Training the N-Gramm extractor
ng_url = NGram()
ng_url.fit(training_data, True)

ng_parameter = NGram()
ng_parameter.fit(training_data,False)
    
print("N-Gramms extracted!")
print("**************************")
print("Starting K-Means Fitting...")



"""
# Getting Feature Vectors
training_vectors = ng.get_feature_vectors(training_data)
test_vectors_clean = ng.get_feature_vectors(test_clean)
test_vectors_anomalous = ng.get_feature_vectors(test_anomalous)

"""


# Getting Feature Vectors
training_vectors_parameter, ngrams_training_parameter = ng_parameter.get_feature_vectors_multidimensional(training_data)
test_vectors_clean_parameter, ngrams_clean_parameter = ng_parameter.get_feature_vectors_multidimensional(test_clean)
test_vectors_anomalous_parameter, ngrams_anomalous_parameter = ng_parameter.get_feature_vectors_multidimensional(test_anomalous)
    
training_vectors_url, ngrams_training_url = ng_url.get_feature_vectors_multidimensional(training_data)
test_vectors_clean_url, ngrams_clean_url = ng_url.get_feature_vectors_multidimensional(test_clean)
test_vectors_anomalous_url, ngrams_anomalous_url = ng_url.get_feature_vectors_multidimensional(test_anomalous)



print("\n**************************")
print("Training model with URL:")
print("k = 3")

"""Initialize K-Means, fit training data and predict which cluster it belongs to
"""
kmeans = KMeans(n_clusters = 3, init='random', 
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
                )



clusters = kmeans.fit_predict(training_vectors_url)
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
            cluster0.append(training_vectors_url[j])
        elif(labels[j] == 1.0):
            cluster1.append(training_vectors_url[j])
        elif(labels[j]== 2.0):
            cluster2.append(training_vectors_url[j])



        

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
print("Testing normal traffic:")

"""To see if the test data belongs to one of the obtained clusters we test,
if the distance between the centroid and the datapoint is larger than the radius for said cluster.
"""

could_not_be_assigned_clean_test_vector = [] #would be bad. The vectors should belong to a cluster
could_be_assigned_clean_test_vector = [] #would be good. The vectors should belong to a cluster

r0 = clusters_radii[0]
c0 = centroids[0]

r1 = clusters_radii[1]
c1 = centroids[1]

r2 = clusters_radii[2]
c2 = centroids[2]



for i in range(len(test_vectors_clean_url)):

    dist0 = np.linalg.norm(c0-test_vectors_clean_url[i])
    dist1 = np.linalg.norm(c1-test_vectors_clean_url[i])
    dist2 = np.linalg.norm(c2-test_vectors_clean_url[i])
    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        could_be_assigned_clean_test_vector.append(test_vectors_clean_url[i])
    else:
        could_not_be_assigned_clean_test_vector.append(test_vectors_clean_url[i])


detected_anomalies_in_anomalous_test_vector = [] #would be good. The vectors shouldn't belong to any cluster
undetected_anomalies_in_anomalous_test_vector = [] #would be bad. The vectors shouldn't belong to any cluster

dist0 = 0
dist1 = 0
dist2 = 0
for i in range(len(test_vectors_anomalous_url)):
    dist0 = np.linalg.norm(c0 - test_vectors_anomalous_url[i])
    dist1 = np.linalg.norm(c1 - test_vectors_anomalous_url[i])
    dist2 = np.linalg.norm(c2 - test_vectors_anomalous_url[i])

    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        undetected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous_url[i])
    else:
        detected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous_url[i]) 


print("Predicting successful!")    
print("**************************")
print("Results:")

# Evaluation
accuracy_anomalous = len(detected_anomalies_in_anomalous_test_vector ) / len(test_vectors_anomalous_url) * 100
accuracy_clean = len(could_be_assigned_clean_test_vector) / len(test_vectors_clean_url) * 100

print("True Positiv: %f %%" % accuracy_anomalous)
print("False Positiv: %f %%" % (100 - accuracy_clean))
print("Accuracy: %f %%" % ((accuracy_anomalous * len(test_vectors_anomalous_url) + accuracy_clean * len(test_vectors_clean_url)) / (len(test_vectors_clean_url) + len(test_vectors_anomalous_url))))
    
# Plotting Vectors
plt.subplot(2,1,1)
samples = 300
plt.scatter(training_vectors_url[:samples,0], training_vectors_url[:samples,1], s=200,color = "g", alpha = 0.5, label = "Trainings Data")
plt.scatter(test_vectors_clean_url[:samples,0], test_vectors_clean_url[:samples,1], s=150, color = "b", alpha = 0.5, label = "Clean Data")
plt.scatter(test_vectors_anomalous_url[:samples,0], test_vectors_anomalous_url[:samples,1], s=100, color = "r", alpha = 0.5, label = "Anomalous Data")
plt.xlim(0.02,0.1)
plt.ylim(0, 500)
plt.title("Distribution of Feature Vectors")
plt.legend()
plt.grid()


"""Visualize clusters
"""
plt.subplot(2,1,2)
plt. scatter(
    training_vectors_url[clusters==0,0], training_vectors_url[clusters==0,1],
    s = 50, c = 'green',
    marker= 's', edgecolor= 'black',
    label = 'cluster 1'
    )

plt.scatter(
    training_vectors_url[clusters==1,0], training_vectors_url[clusters == 1,1],
    s = 50, c ='orange',
    marker = 'o', edgecolor='black',
    label = 'cluster2'
    )

plt.scatter(
    training_vectors_url[clusters == 2,0], training_vectors_url[clusters==2,1],
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



print("\n**************************")
print("Training model with Parameters:")
print("k = 3")

"""Initialize K-Means, fit training data and predict which cluster it belongs to
"""
kmeans = KMeans(n_clusters = 3, init='random', 
                n_init=10, max_iter=300, 
                tol=1e-04, random_state=0
                )



clusters = kmeans.fit_predict(training_vectors_parameter)
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
            cluster0.append(training_vectors_parameter[j])
        elif(labels[j] == 1.0):
            cluster1.append(training_vectors_parameter[j])
        elif(labels[j]== 2.0):
            cluster2.append(training_vectors_parameter[j])



        

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
print("Testing normal traffic:")

"""To see if the test data belongs to one of the obtained clusters we test,
if the distance between the centroid and the datapoint is larger than the radius for said cluster.
"""

could_not_be_assigned_clean_test_vector = [] #would be bad. The vectors should belong to a cluster
could_be_assigned_clean_test_vector = [] #would be good. The vectors should belong to a cluster

r0 = clusters_radii[0]
c0 = centroids[0]

r1 = clusters_radii[1]
c1 = centroids[1]

r2 = clusters_radii[2]
c2 = centroids[2]



for i in range(len(test_vectors_clean_parameter)):

    dist0 = np.linalg.norm(c0-test_vectors_clean_parameter[i])
    dist1 = np.linalg.norm(c1-test_vectors_clean_parameter[i])
    dist2 = np.linalg.norm(c2-test_vectors_clean_parameter[i])
    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        could_be_assigned_clean_test_vector.append(test_vectors_clean_parameter[i])
    else:
        could_not_be_assigned_clean_test_vector.append(test_vectors_clean_parameter[i])


detected_anomalies_in_anomalous_test_vector = [] #would be good. The vectors shouldn't belong to any cluster
undetected_anomalies_in_anomalous_test_vector = [] #would be bad. The vectors shouldn't belong to any cluster

dist0 = 0
dist1 = 0
dist2 = 0
for i in range(len(test_vectors_anomalous_parameter)):
    dist0 = np.linalg.norm(c0 - test_vectors_anomalous_parameter[i])
    dist1 = np.linalg.norm(c1 - test_vectors_anomalous_parameter[i])
    dist2 = np.linalg.norm(c2 - test_vectors_anomalous_parameter[i])

    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        undetected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous_parameter[i])
    else:
        detected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous_parameter[i]) 


print("Predicting successful!")    
print("**************************")
print("Results:")

# Evaluation
accuracy_anomalous = len(detected_anomalies_in_anomalous_test_vector ) / len(test_vectors_anomalous_parameter) * 100
accuracy_clean = len(could_be_assigned_clean_test_vector) / len(test_vectors_clean_parameter) * 100

print("True Positiv: %f %%" % accuracy_anomalous)
print("False Positiv: %f %%" % (100 - accuracy_clean))
print("Accuracy: %f %%" % ((accuracy_anomalous * len(test_vectors_anomalous_parameter) + accuracy_clean * len(test_vectors_clean_parameter)) / (len(test_vectors_clean_parameter) + len(test_vectors_anomalous_parameter))))
    