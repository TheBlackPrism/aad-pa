import math
import numpy as np
import pandas as pd
import matplotlib as matplot
import matplotlib.pyplot as plt
import logfileparser as parser
import numbers

class K_Means:
    """description of class"""
    
    def __init__(self, k=1, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = np.zeros([k])
    
    def fit(self, data):

        clusters = {}
        self.data = data

        #initialize centroids, using first k elements in the dataset
        for i in range(self.k):
            self.centroids[i] = data[i]

        #main iterations
        for i in range(self.max_iterations):
            self.classes = {}

            for j in range(self.k):
                self.classes[j] = []
            
            for features in data:
                distances = [Euclidean_distance(features, centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
        
            #re-calculate centroids
            previous = self.centroids

            for i in range(len(self.classes)):
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)
            
            #check if clusters are optimal
            isOptimal = True

            for i in range(len(self.centroids)):
                original_centroid = previous[i]
                curr = self.centroids[i]

                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            if isOptimal:
                
                #clusters = dict(self.classes)
                #for classification in self.classes:
                    #clusters[classification] = self.centroids[classification]
                break 
        return self.centroids

    def get_radius(self, attr):
        radius = {}
        
        for i in range(len(self.centroids)):
                distances = [Euclidean_distance(feature, self.centroids[i]) for feature in self.classes[i]]
                radius[self.centroids[i]] = np.max(distances)
                #radius[cent] = ((np.absolute(classellipsis[cent] +
                #1)/np.absolute(classes[cent])) * max_Dist)
        return radius
        
#Compute euclidean distance for two features p and q
def Euclidean_distance(p,q):
    distance = 0
    if isinstance(p, numbers.Number) and isinstance(q, numbers.Number):
        distance = abs(p - q)
    else:
        squared_distance = 0

        if len(p) == len(q):
            return 0
    
        for i in range(len(p)):
            squared_distance += (p[i] - q[i]) ** 2

        distance = squared_distance(squared_distance)
    return distance

def get_malicious_requests(data, radius):
    malicious_requests = []
    for i in range(len(data)):
        counter = 0
        for m in radius:
            dist = Euclidean_distance(data[i],m)
            if dist > radius[m]:
                counter += 1
                if counter == len(radius):
                    malicious_requests.append(data[i])
    return malicious_requests

def main():
    clusters = {}
    means = []

    #read data here
    training_data = parser.read_data('../Logfiles/Labeled/normalTrafficTraining.txt')

    #Test feature could be length of request.
    #Parse data, extract requests and put their lengths in a list
    requests = ["" for x in range(len(training_data))]
    request_lengths = np.zeros([len(training_data)])
    for i in range(len(training_data)):
        tmp = training_data[i]
        requests[i] = tmp.get('Request', None)
        req = requests[i]
        request_lengths[i] = len(req)

    print("\n**************************")
    print("Training model:")
    #fit data
    km = K_Means(3)
    means = km.fit(request_lengths)
    radius = km.get_radius('attr')
    print("Centeroid: Radius =", radius)

    #test clean data
    print("**************************")
    print("Testing normal traffic:")
    test_data = parser.read_data('../Logfiles/Labeled/normalTrafficTest.txt')
    test_requests = []
    test_request_lengths = []
    dist = 0
    malicious_requests = []

    test_requests = ["" for x in range(len(training_data))]
    test_request_lengths = np.zeros([len(training_data)])
    for i in range(len(test_data)):
        tmp = test_data[i]
        test_requests[i] = tmp.get('Request', None) #returns None if key doesn't exist
        test_req = test_requests[i]
        test_request_lengths[i] = len(test_req)

        
    malicious_requests = get_malicious_requests(test_request_lengths, radius)

    if len(malicious_requests) == 0:
        print("No malicious requests detected.")
    else:
        accuracy = 100 - (len(malicious_requests) / len(test_data) * 100)
        print("Number of test requests: %d" % len(test_data))
        print("Number of malicious requests detected: %d" % len(malicious_requests))
        print("Accuracy: %.3f%%" % accuracy)

    #test anomalious data
    print("**************************")
    print("Testing anomalous traffic:")
    test_data = parser.read_data('../Logfiles/Labeled/anomalousTrafficTest.txt')
    test_requests = []
    test_request_lengths = []
    dist = 0
    malicious_requests = []

    test_requests = ["" for x in range(len(training_data))]
    test_request_lengths = np.zeros([len(training_data)])
    for i in range(len(test_data)):
        tmp = test_data[i]
        test_requests[i] = tmp.get('Request', None) #returns None if key doesn't exist
        test_req = test_requests[i]
        test_request_lengths[i] = len(test_req)

    malicious_requests = get_malicious_requests(test_request_lengths, radius)

    if len(malicious_requests) == 0:
        print("No malicious requests detected.")
    else:
        accuracy = (len(malicious_requests) / len(test_data) * 100)
        print("Number of test requests: %d" % len(test_data))
        print("Number of malicious requests detected: %d" % len(malicious_requests))
        print("Accuracy: %.3f%%" % accuracy)
    print("**************************\n")

    #Plot
    colors = 10 * ["r","g","c","b","k"]
    if isinstance(km.centroids[0], numbers.Number):
        for classification in km.classes:
            num_points = 0
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features, 1,color = color, s = 30)
                num_points += 1
                if num_points > 50:
                    break
        for centroid in km.centroids:
            plt.scatter(centroid, 1,s = 130, marker = "x")
        plt.show()
    else:
        for centroid in km.centroids:
            plt.scatter(km.centroids[centroid][0],km.centroids[centroid][1],s = 130, marker = "x")

        for classification in km.classes:
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features[0], features[1],color = color, s = 30)
        plt.show()

if __name__ == "__main__":
    main()