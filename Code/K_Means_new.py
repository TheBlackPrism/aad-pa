import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import logfileparser as parser
from NGram import *
from URL_Length_Extraction import *
#from collections import Counter, defaultdict

print("Reading Data...")
print("**************************")


# Reading Data
training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')
test_clean = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
test_anomalous = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')

print("Done!")
print("**************************")
print("Starting Feature Extraction...")


# Training the N-Gramm extractor
#ng = NGram()
#ng.fit(training_data)
urlLength = URL_Length_Extraction()

    
print("URL Lengths extracted!")
print("**************************")
print("Starting K-Means Fitting...")




# Getting Feature Vectors
training_vectors = urlLength.extract_feature(training_data)
test_vectors_clean = urlLength.extract_feature(test_clean)
test_vectors_anomalous = urlLength.extract_feature(test_anomalous)


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



dist0 = 0
dist1 = 0
dist2 = 0
for i in range(len(test_vectors_clean)):

    dist0 = np.linalg.norm(c0-test_vectors_clean[i])
    dist1 = np.linalg.norm(c1-test_vectors_clean[i])
    dist2 = np.linalg.norm(c2-test_vectors_clean[i])
    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        could_be_assigned_clean_test_vector.append(test_vectors_clean[i])
    else:
        could_not_be_assigned_clean_test_vector.append(test_vectors_clean[i])

   
    dist0 = 0
    dist1 = 0
    dist2 = 0







detected_anomalies_in_anomalous_test_vector = [] #would be good. The vectors shouldn't belong to any cluster
undetected_anomalies_in_anomalous_test_vector = [] #would be bad. The vectors shouldn't belong to any cluster

dist0 = 0
dist1 = 0
dist2 = 0
for i in range(len(test_vectors_anomalous)):
    dist0 = np.linalg.norm(c0 - test_vectors_anomalous[i])
    dist1 = np.linalg.norm(c1 - test_vectors_anomalous[i])
    dist2 = np.linalg.norm(c2 - test_vectors_anomalous[i])
    

    if dist0 <= r0 or dist1 <= r1 or dist2 <= r2:
        undetected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous[i])
    else:
        detected_anomalies_in_anomalous_test_vector.append(test_vectors_anomalous[i]) 
    dist0 = 0
    dist1 = 0
    dist2 = 0



print("Predicting successful!")    
print("**************************")
print("Results:")

# Evaluation
accuracy_anomalous = (len(detected_anomalies_in_anomalous_test_vector ) / len(test_vectors_anomalous)) * 100 #TP
accuracy_clean = (len(could_be_assigned_clean_test_vector) / len(test_vectors_clean)) * 100

print("True Positiv: %f %%" % accuracy_anomalous)
print("False Positiv: %f %%" % (100 - accuracy_clean))
print("Accuracy: %f %%" % ((accuracy_anomalous * len(test_vectors_anomalous) + accuracy_clean * len(test_vectors_clean)) / (len(test_vectors_clean) + len(test_vectors_anomalous))))
    
# Plotting Vectors
"""plt.subplot(2,1,1)
samples = 300"""
#plt.scatter(training_vectors[:samples,0], training_vectors[:samples,1], s=200,color = "g", alpha = 0.5, label = "Trainings Data")
#plt.scatter(test_vectors_clean[:samples,0], test_vectors_clean[:samples,1], s=150, color = "b", alpha = 0.5, label = "Clean Data")
#plt.scatter(test_vectors_anomalous[:samples,0], test_vectors_anomalous[:samples,1], s=100, color = "r", alpha = 0.5, label = "Anomalous Data")
#plt.xlim(0.02,0.1)
#plt.ylim(0, 500)
#plt.title("Distribution of Feature Vectors")
#plt.legend()
#plt.grid()



"""Visualize clusters
"""
#plt.subplot(2,1,2)

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